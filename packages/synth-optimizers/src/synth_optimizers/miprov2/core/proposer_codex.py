"""Codex workspace proposer backend for MIPROv2.

Runs a Codex app-server session against a materialized workspace directory
to generate instruction patches, then imports the proposal manifest back into
the compiled search space.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synth_optimizers.codex_runtime import (
    AppServerCodexSession,
    CodexAppServerLaunchSpec,
    CodexAppServerStdioClient,
    CodexProfile,
    ExecutionConfig,
    ExecutionProfile,
    ExecutionResultStatus,
    ExecutionSessionSpec,
    ParticipantRole,
    SandboxProfile,
    WorkerHostKind,
    build_codex_app_server_command,
)
from synth_optimizers.miprov2.core.program_compiler import (
    CompiledMiproSpace,
    register_instruction_candidate,
    register_stage_candidate,
)
from synth_optimizers.miprov2.core.proposer_openenv import (
    MiproInstructionPatch,
    MiproOpenEnvProposerContext,
    MiproOpenEnvProposerOutcome,
    summarize_compiled_space,
)

_PROPOSAL_SCHEMA_VERSION = "mipro_codex_proposer_proposal_v1"


@dataclass(frozen=True, slots=True)
class MiproCodexProposerConfig:
    model: str = "gpt-5.4-mini"
    openai_base_url: str = ""
    api_key_env: str = ""
    copy_host_auth: bool = True
    turn_timeout_seconds: float = 300.0
    shutdown_timeout_seconds: float = 30.0
    workspace_root: str = ""


def materialize_mipro_codex_workspace(
    compiled_space: CompiledMiproSpace,
    proposer_context: MiproOpenEnvProposerContext,
    workspace_dir: Path,
) -> None:
    """Write all workspace files Codex needs to produce a proposal manifest."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    state_dir = workspace_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    proposal_dir = workspace_dir / "proposal"
    proposal_dir.mkdir(parents=True, exist_ok=True)

    # state/candidates.json — current instruction options per module
    space_summary = summarize_compiled_space(compiled_space)
    (state_dir / "candidates.json").write_text(
        json.dumps(space_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # state/instruction_components.json — full base text for each option
    instruction_components: dict[str, Any] = {}
    for component_key, option_map in compiled_space.instruction_base_lookup.items():
        instruction_components[component_key] = {
            option_id: text for option_id, text in option_map.items()
        }
    (state_dir / "instruction_components.json").write_text(
        json.dumps(instruction_components, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # state/run_metadata.json — proposer context metadata
    run_metadata: dict[str, Any] = {
        "objective": proposer_context.objective,
        "round_idx": proposer_context.round_idx,
        "recent_failures": list(proposer_context.recent_failures),
        "recent_successes": list(proposer_context.recent_successes),
        "candidate_summary_counts": dict(proposer_context.candidate_summary_counts),
        "current_best_candidate_id": proposer_context.current_best_candidate_id,
        "baseline_candidate_id": proposer_context.baseline_candidate_id,
        "grounding_payload": dict(proposer_context.grounding_payload),
        "run_metadata": dict(proposer_context.run_metadata),
        "read_model_payload": dict(proposer_context.read_model_payload),
    }
    (state_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # proposal/PROPOSAL_SCHEMA.md — instructions for the model
    stages = compiled_space.program_template.stages
    has_stages = len(stages) > 1 or (len(stages) == 1 and stages[0].stage_id != "stage_0")
    if has_stages:
        stages_desc = ""
        for stage in stages:
            module_lines = "\n".join(f'        "{m.module_id}": "<new instruction text>"' for m in stage.modules)
            stages_desc += f'    {{\n      "stage_id": "{stage.stage_id}",\n      "modules": {{\n{module_lines}\n      }}\n    }},\n'
        stage_names_desc = "\n".join(
            f"  - `{s.stage_id}`" + (f" ({s.stage_name})" if s.stage_name else "")
            + ": " + ", ".join(f"`{m.module_id}`" for m in s.modules)
            for s in stages
        )
        schema_doc = f"""# MIPROv2 Codex Proposer — Proposal Schema

Write `proposal/manifest.json` using schema_version `{_PROPOSAL_SCHEMA_VERSION}`.

This is a **multi-stage pipeline**. Each patch covers one pipeline stage and must supply
updated instruction text for every module in that stage.

## Required fields

```json
{{
  "schema_version": "{_PROPOSAL_SCHEMA_VERSION}",
  "patches": [
{stages_desc.rstrip()}
  ]
}}
```

## Stages and their modules

{stage_names_desc}

## Rules

- Each patch must have `stage_id` (string) and `modules` (object mapping module_id → instruction_text).
- Supply ALL modules for the stage — partial updates are rejected.
- You may propose multiple patches per stage (different ideas for the same stage).
- Do not reuse instruction text that already appears verbatim in `state/instruction_components.json`.
- Read `state/run_metadata.json` for context on what has worked and what has failed.
- Write the manifest as strict JSON to `proposal/manifest.json`.
"""
    else:
        modules_desc = "\n".join(
            f"  - `{m.module_id}`"
            for m in sorted(compiled_space.program_template.modules, key=lambda x: x.module_id)
        )
        schema_doc = f"""# MIPROv2 Codex Proposer — Proposal Schema

Write `proposal/manifest.json` using schema_version `{_PROPOSAL_SCHEMA_VERSION}`.

## Required fields

```json
{{
  "schema_version": "{_PROPOSAL_SCHEMA_VERSION}",
  "patches": [
    {{
      "module_id": "<module_id>",
      "instruction_text": "<new instruction text>"
    }}
  ]
}}
```

## Modules available

{modules_desc}

## Rules

- Each patch must include `module_id` (string) and `instruction_text` (non-empty string).
- You may propose multiple patches — one per module per idea.
- Do not reuse instruction text that already appears verbatim in `state/instruction_components.json`.
- Read `state/run_metadata.json` for context on what has worked and what has failed.
- Write the manifest as strict JSON to `proposal/manifest.json`.
"""
    (proposal_dir / "PROPOSAL_SCHEMA.md").write_text(schema_doc, encoding="utf-8")

    # proposal/manifest.json — empty template so the path exists
    empty_manifest: dict[str, Any] = {
        "schema_version": _PROPOSAL_SCHEMA_VERSION,
        "patches": [],
    }
    manifest_path = proposal_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(empty_manifest, indent=2),
            encoding="utf-8",
        )

    # README.md — top-level orientation
    readme = f"""# MIPROv2 Codex Proposer Workspace

Your task is to propose new instruction candidates for the MIPROv2 optimizer.

## Workflow

1. Read `state/run_metadata.json` to understand the current optimization round,
   recent failures, and recent successes.
2. Read `state/instruction_components.json` to see existing instruction text for
   each module — do NOT propose duplicates.
3. Read `state/candidates.json` for a compact summary of the search space.
4. Read `proposal/PROPOSAL_SCHEMA.md` for the exact JSON schema you must produce.
5. Write your proposal to `proposal/manifest.json`.

The optimizer will import your manifest and register new instruction candidates
for the next evaluation round.
"""
    (workspace_dir / "README.md").write_text(readme, encoding="utf-8")


def import_mipro_codex_proposal(
    workspace_dir: Path,
    compiled_space: CompiledMiproSpace,
) -> list[str]:
    """Read proposal/manifest.json and register new instruction candidates.

    Returns a list of newly registered candidate option IDs.
    """
    manifest_path = workspace_dir / "proposal" / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Codex proposer did not write proposal manifest: {manifest_path}"
        )
    raw = manifest_path.read_text(encoding="utf-8")
    manifest = _parse_last_json_object(raw)
    if not isinstance(manifest, dict):
        raise RuntimeError(
            f"Codex proposer manifest is not a JSON object: {manifest_path}"
        )
    schema_version = str(manifest.get("schema_version") or "").strip()
    if schema_version != _PROPOSAL_SCHEMA_VERSION:
        raise RuntimeError(
            f"Codex proposer manifest has unexpected schema_version={schema_version!r}; "
            f"expected {_PROPOSAL_SCHEMA_VERSION!r}: {manifest_path}"
        )
    patches = manifest.get("patches")
    if not isinstance(patches, list):
        raise RuntimeError(
            f"Codex proposer manifest missing 'patches' list: {manifest_path}"
        )
    new_candidate_ids: list[str] = []
    for patch in patches:
        if not isinstance(patch, dict):
            continue

        stage_id = str(patch.get("stage_id") or "").strip()
        if stage_id:
            # Stage patch: {"stage_id": "...", "modules": {"mod_id": "text", ...}}
            raw_modules = patch.get("modules")
            if not isinstance(raw_modules, dict):
                continue
            module_instructions = {
                str(k).strip(): str(v).strip()
                for k, v in raw_modules.items()
                if str(k).strip() and str(v).strip()
            }
            if not module_instructions:
                continue
            registered, _stage_oid = register_stage_candidate(
                compiled_space=compiled_space,
                stage_id=stage_id,
                module_instructions=module_instructions,
            )
            if registered:
                # Collect the per-module option IDs that were just registered
                stage_key = f"stage:{stage_id}:instruction"
                stage_lookup = compiled_space.stage_instruction_lookup.get(stage_key, {})
                per_module_ids = stage_lookup.get(_stage_oid, {}).get(
                    "__per_module_option_ids__", {}
                )
                for opt_id in per_module_ids.values():
                    if opt_id not in new_candidate_ids:
                        new_candidate_ids.append(opt_id)
        else:
            # Legacy per-module patch: {"module_id": "...", "instruction_text": "..."}
            module_id = str(patch.get("module_id") or "").strip()
            instruction_text = str(patch.get("instruction_text") or "").strip()
            if not module_id or not instruction_text:
                continue
            registered, option_id, _component_key = register_instruction_candidate(
                compiled_space=compiled_space,
                module_id=module_id,
                instruction_text=instruction_text,
            )
            if registered:
                new_candidate_ids.append(option_id)

    return new_candidate_ids


async def run_codex_workspace_proposer(
    compiled_space: CompiledMiproSpace,
    proposer_context: MiproOpenEnvProposerContext,
    config: MiproCodexProposerConfig,
    workspace_root: Path,
) -> MiproOpenEnvProposerOutcome:
    """Orchestrate workspace materialization, Codex session, and proposal import.

    Returns a MiproOpenEnvProposerOutcome populated with the new candidates and
    token usage from the Codex session.
    """
    workspace_dir = workspace_root.resolve()
    materialize_mipro_codex_workspace(compiled_space, proposer_context, workspace_dir)

    # Clear any stale manifest from a previous round
    manifest_path = workspace_dir / "proposal" / "manifest.json"
    empty_manifest: dict[str, Any] = {
        "schema_version": _PROPOSAL_SCHEMA_VERSION,
        "patches": [],
    }
    manifest_path.write_text(json.dumps(empty_manifest, indent=2), encoding="utf-8")

    model = str(config.model or "gpt-5.4-mini").strip() or "gpt-5.4-mini"

    # Build env for the Codex process
    env = os.environ.copy()
    if config.api_key_env and env.get(config.api_key_env):
        env["OPENAI_API_KEY"] = env[config.api_key_env]
    if config.openai_base_url:
        env["OPENAI_BASE_URL"] = str(config.openai_base_url).strip()

    # Prepare CODEX_HOME (copy host auth if requested)
    codex_home = workspace_dir / ".codex_home"
    codex_home.mkdir(parents=True, exist_ok=True)
    if config.copy_host_auth:
        source_home_raw = str(env.get("CODEX_HOME") or Path.home() / ".codex").strip()
        if source_home_raw:
            source_home = Path(source_home_raw).expanduser().resolve()
            if source_home.is_dir():
                for filename in ("auth.json", "installation_id", "version.json", "models_cache.json"):
                    source = source_home / filename
                    if source.is_file():
                        destination = codex_home / filename
                        shutil.copy2(source, destination)
    env["CODEX_HOME"] = str(codex_home)

    command = build_codex_app_server_command(binary="codex")
    launch_spec = CodexAppServerLaunchSpec(
        command=command,
        working_dir=workspace_dir,
        env=env,
    )
    client = CodexAppServerStdioClient(launch_spec)
    session = AppServerCodexSession(client)

    instructions = _codex_instructions(workspace_dir)
    session_id = f"mipro_codex_proposer_r{proposer_context.round_idx}"
    spec = ExecutionSessionSpec(
        run_id=session_id,
        session_id=session_id,
        instructions=instructions,
        execution_config=ExecutionConfig(
            participant_role=ParticipantRole.WORKER,
            host_kind=WorkerHostKind.LOCAL,
            profile=ExecutionProfile(
                profile_id="mipro_codex_proposer",
                codex=CodexProfile(
                    profile_id="mipro_codex_proposer_model",
                    model=model,
                    approval_policy="never",
                ),
                sandbox=SandboxProfile(
                    profile_id="mipro_codex_proposer_workspace",
                    sandbox_mode="workspace-write",
                ),
            ),
        ),
    )

    turn_timeout_seconds = max(1.0, float(config.turn_timeout_seconds))
    shutdown_timeout_seconds = max(1.0, float(config.shutdown_timeout_seconds))

    await session.start(spec)
    snapshot = await session.snapshot()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + turn_timeout_seconds
    wait_task = asyncio.create_task(session.wait())
    manifest_cutoff = False
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise RuntimeError(
                    f"Codex proposer turn timed out after {turn_timeout_seconds:.1f}s "
                    f"without writing proposal manifest: {manifest_path}"
                )
            poll = min(2.0, remaining)
            result = None
            timed_out = False
            try:
                result = await asyncio.wait_for(asyncio.shield(wait_task), timeout=poll)
            except asyncio.TimeoutError:
                timed_out = True
            snapshot = await session.snapshot()
            if timed_out:
                # Check if manifest has been written with actual patches
                if _manifest_has_patches(manifest_path):
                    manifest_cutoff = True
                    wait_task.cancel()
                    break
                continue
            # Turn completed
            if not manifest_cutoff:
                if result is not None and result.status != ExecutionResultStatus.COMPLETED:
                    raise RuntimeError(
                        f"Codex proposer session failed: status={result.status.value}, "
                        f"reason={result.failure_reason or ''}, "
                        f"diagnostics={result.diagnostics}"
                    )
            break
    finally:
        if not wait_task.done():
            wait_task.cancel()
        try:
            await asyncio.wait_for(session.cancel(), timeout=shutdown_timeout_seconds)
        except asyncio.TimeoutError:
            pass

    new_candidate_ids = import_mipro_codex_proposal(workspace_dir, compiled_space)

    usage = snapshot.usage or {}
    instruction_patches: list[MiproInstructionPatch] = []
    _seen_patches: set[tuple[str, str]] = set()
    for module in compiled_space.program_template.modules:
        component_key = f"module:{module.module_id}:instruction"
        base_lookup = compiled_space.instruction_base_lookup.get(component_key, {})
        for option_id in new_candidate_ids:
            patch_key = (module.module_id, option_id)
            if option_id in base_lookup and patch_key not in _seen_patches:
                _seen_patches.add(patch_key)
                instruction_patches.append(
                    MiproInstructionPatch(
                        module_id=module.module_id,
                        component_key=component_key,
                        option_id=option_id,
                        instruction_text=base_lookup[option_id],
                    )
                )

    stop_reason = "codex_workspace_completed" if not manifest_cutoff else "codex_workspace_manifest_cutoff"

    return MiproOpenEnvProposerOutcome(
        compiled_space=compiled_space,
        instruction_patches=instruction_patches,
        demo_patches=[],
        transcript=[],
        action_counts={"codex_workspace": 1},
        read_action_count=0,
        patch_action_count=len(new_candidate_ids),
        duplicate_patch_count=0,
        ignored_patch_count=0,
        policy_violation_count=0,
        grounding_read_action_count=0,
        evidence_read_action_count=0,
        read_tools_used=(),
        stop_reason=stop_reason,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        cached_prompt_tokens=int(usage.get("cached_prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
        model_turn_count=1,
        tool_call_count=0,
        archive_spill_count=0,
        archived_message_count=0,
        archive_path=None,
        queue_state={},
        memory_state={},
        plugin_state={},
        runbook_warnings=[],
        runbook_violation_count=0,
        runbook_summary={},
    )


def _parse_last_json_object(raw: str) -> dict | None:
    """Return the last top-level JSON object found in raw text.

    Codex sometimes appends to rather than overwrites manifest.json, producing
    two concatenated JSON objects.  We want the last complete one.
    """
    raw = raw.strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    last_obj: dict | None = None
    pos = 0
    while pos < len(raw):
        try:
            obj, pos = decoder.raw_decode(raw, pos)
            if isinstance(obj, dict):
                last_obj = obj
        except json.JSONDecodeError:
            pos += 1
    return last_obj


def _manifest_has_patches(manifest_path: Path) -> bool:
    if not manifest_path.is_file():
        return False
    raw = manifest_path.read_text(encoding="utf-8").strip()
    if not raw:
        return False
    data = _parse_last_json_object(raw)
    if not isinstance(data, dict):
        return False
    patches = data.get("patches")
    return isinstance(patches, list) and len(patches) > 0


def _codex_instructions(workspace_dir: Path) -> str:
    return (
        "You are the MIPROv2 Codex proposer. Work only inside this workspace.\n\n"
        "Your task is to propose new instruction candidates for a prompt optimization run.\n\n"
        "Steps:\n"
        "1. Read `README.md` for an overview of the workspace layout.\n"
        "2. Read `proposal/PROPOSAL_SCHEMA.md` for the exact JSON schema you must produce.\n"
        "3. Read `state/run_metadata.json` to understand the current round, objective, "
        "recent failures, and recent successes.\n"
        "4. Read `state/instruction_components.json` to see existing instruction text — "
        "do NOT propose text that is already registered.\n"
        "5. Read `state/candidates.json` for a compact search space summary.\n"
        "6. Write your proposal as strict JSON to `proposal/manifest.json`, following "
        f"the schema_version `{_PROPOSAL_SCHEMA_VERSION}`.\n\n"
        "Constraints:\n"
        "- Do not propose duplicate instruction text.\n"
        "- Each patch must have a valid `module_id` and non-empty `instruction_text`.\n"
        "- Propose at least one patch that directly addresses the recent failures.\n"
        "- Keep proposals concise and targeted; do not pad with filler text.\n"
        "- Do not invent module IDs; use only those listed in `state/candidates.json`.\n"
    )
