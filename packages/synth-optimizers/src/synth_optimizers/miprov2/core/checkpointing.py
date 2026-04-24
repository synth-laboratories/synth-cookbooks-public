"""Checkpoint helpers for proposer-only MIPROv2 replay evals."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from synth_optimizers.miprov2.core.instruction_transforms import InstructionTransform
from synth_optimizers.miprov2.core.optimizer import MiproTrialResult
from synth_optimizers.miprov2.core.program_compiler import (
    CompiledMiproSpace,
    compile_search_space,
    decode_config,
)
from synth_optimizers.miprov2.core.program_model import MiproProgramCandidate, MiproProgramTemplate, demo_from_dict
from synth_optimizers.miprov2.core.proposer_openenv import (
    MiproOpenEnvProposerConfig,
    MiproOpenEnvProposerContext,
    MiproOpenEnvProposerOutcome,
    MiproOpenEnvProposerVariant,
    MiproOpenEnvReactAgent,
    proposer_outcome_summary,
    run_openenv_react_proposer,
)
from synth_optimizers.miprov2.core.run_ledger import SQLiteMiproRunLedger

CHECKPOINT_STAGE_BEFORE_PROPOSER = "before_proposer"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def trial_to_dict(trial: MiproTrialResult) -> dict[str, Any]:
    return {
        "config": dict(trial.config),
        "score": float(trial.score),
        "details": dict(trial.details),
        "cost_proxy": float(trial.cost_proxy),
        "latency_ms": float(trial.latency_ms),
        "candidate_id": trial.candidate_id,
        "lever_bundle_hash": trial.lever_bundle_hash,
        "timestamp": float(trial.timestamp),
    }


def trial_from_dict(payload: Mapping[str, Any]) -> MiproTrialResult:
    return MiproTrialResult(
        config=dict(payload.get("config") or {}),
        score=float(payload.get("score") or 0.0),
        details=dict(payload.get("details") or {}),
        cost_proxy=float(payload.get("cost_proxy") or 0.0),
        latency_ms=float(payload.get("latency_ms") or 0.0),
        candidate_id=str(payload["candidate_id"]) if payload.get("candidate_id") else None,
        lever_bundle_hash=(
            str(payload["lever_bundle_hash"]) if payload.get("lever_bundle_hash") else None
        ),
        timestamp=float(payload.get("timestamp") or time.time()),
    )


def compiled_space_to_snapshot(compiled_space: CompiledMiproSpace) -> dict[str, Any]:
    return {
        "program_template": compiled_space.program_template.to_dict(),
        "search_space": {key: list(value) for key, value in compiled_space.search_space.items()},
        "instruction_lookup": {
            key: dict(value) for key, value in compiled_space.instruction_lookup.items()
        },
        "instruction_base_lookup": {
            key: dict(value) for key, value in compiled_space.instruction_base_lookup.items()
        },
        "instruction_transforms": {
            key: {transform_id: transform.to_dict() for transform_id, transform in value.items()}
            for key, value in compiled_space.instruction_transforms.items()
        },
        "demo_lookup": {
            key: {option_id: demo.to_dict() for option_id, demo in value.items()}
            for key, value in compiled_space.demo_lookup.items()
        },
        "component_order": list(compiled_space.component_order),
        "instruction_metadata": {
            key: {option_id: dict(payload) for option_id, payload in value.items()}
            for key, value in compiled_space.instruction_metadata.items()
        },
        "instruction_base_metadata": {
            key: {option_id: dict(payload) for option_id, payload in value.items()}
            for key, value in compiled_space.instruction_base_metadata.items()
        },
        "demo_metadata": {
            key: {option_id: dict(payload) for option_id, payload in value.items()}
            for key, value in compiled_space.demo_metadata.items()
        },
    }


def compiled_space_from_snapshot(snapshot: Mapping[str, Any]) -> CompiledMiproSpace:
    template = MiproProgramTemplate.from_dict(dict(snapshot.get("program_template") or {}))
    compiled = compile_search_space(template)

    search_space = snapshot.get("search_space")
    if isinstance(search_space, Mapping):
        compiled.search_space.clear()
        compiled.search_space.update(
            {
                str(key): [str(item) for item in value]
                for key, value in search_space.items()
                if isinstance(value, list)
            }
        )

    for attr in (
        "instruction_lookup",
        "instruction_base_lookup",
        "instruction_metadata",
        "instruction_base_metadata",
        "demo_metadata",
    ):
        raw = snapshot.get(attr)
        if not isinstance(raw, Mapping):
            continue
        target = getattr(compiled, attr)
        target.clear()
        target.update(
            {
                str(key): {
                    str(option_id): (dict(value) if isinstance(value, Mapping) else str(value))
                    for option_id, value in option_map.items()
                }
                for key, option_map in raw.items()
                if isinstance(option_map, Mapping)
            }
        )

    raw_transforms = snapshot.get("instruction_transforms")
    if isinstance(raw_transforms, Mapping):
        compiled.instruction_transforms.clear()
        compiled.instruction_transforms.update(
            {
                str(key): {
                    str(transform_id): InstructionTransform.from_dict(dict(transform_payload))
                    for transform_id, transform_payload in transform_map.items()
                    if isinstance(transform_payload, Mapping)
                }
                for key, transform_map in raw_transforms.items()
                if isinstance(transform_map, Mapping)
            }
        )

    raw_demos = snapshot.get("demo_lookup")
    if isinstance(raw_demos, Mapping):
        compiled.demo_lookup.clear()
        compiled.demo_lookup.update(
            {
                str(key): {
                    str(option_id): demo_from_dict(dict(demo_payload))
                    for option_id, demo_payload in option_map.items()
                    if isinstance(demo_payload, Mapping)
                }
                for key, option_map in raw_demos.items()
                if isinstance(option_map, Mapping)
            }
        )

    return compiled


def proposer_context_to_dict(context: MiproOpenEnvProposerContext) -> dict[str, Any]:
    return {
        "objective": context.objective,
        "round_idx": int(context.round_idx),
        "recent_failures": list(context.recent_failures),
        "recent_successes": list(context.recent_successes),
        "grounding_payload": dict(context.grounding_payload),
        "run_metadata": dict(context.run_metadata),
        "candidate_summary_counts": dict(context.candidate_summary_counts),
        "current_best_candidate_id": context.current_best_candidate_id,
        "baseline_candidate_id": context.baseline_candidate_id,
        "delta_digest_paths": dict(context.delta_digest_paths),
        "workspace_locations": dict(context.workspace_locations),
        "read_model_payload": dict(context.read_model_payload),
    }


def proposer_context_from_dict(payload: Mapping[str, Any]) -> MiproOpenEnvProposerContext:
    return MiproOpenEnvProposerContext(
        objective=str(payload.get("objective") or ""),
        round_idx=int(payload.get("round_idx") or 0),
        recent_failures=tuple(str(item) for item in list(payload.get("recent_failures") or [])),
        recent_successes=tuple(str(item) for item in list(payload.get("recent_successes") or [])),
        grounding_payload=dict(payload.get("grounding_payload") or {}),
        run_metadata=dict(payload.get("run_metadata") or {}),
        candidate_summary_counts=dict(payload.get("candidate_summary_counts") or {}),
        current_best_candidate_id=(
            str(payload["current_best_candidate_id"])
            if payload.get("current_best_candidate_id")
            else None
        ),
        baseline_candidate_id=(
            str(payload["baseline_candidate_id"]) if payload.get("baseline_candidate_id") else None
        ),
        delta_digest_paths=dict(payload.get("delta_digest_paths") or {}),
        workspace_locations=dict(payload.get("workspace_locations") or {}),
        read_model_payload=dict(payload.get("read_model_payload") or {}),
    )


def write_proposer_checkpoint(
    *,
    checkpoint_dir: str | Path,
    run_id: str,
    round_idx: int,
    compiled_space: CompiledMiproSpace,
    observations: list[MiproTrialResult],
    best_train_candidate: MiproProgramCandidate | None,
    best_train_score: float | None,
    baseline_candidate: MiproProgramCandidate,
    baseline_train_score: float | None,
    heldout_baseline_score: float | None,
    proposer_context: MiproOpenEnvProposerContext,
    train_read_model: Mapping[str, Any],
    sampled_train_rows: list[dict[str, Any]],
    recent_trial_rows: list[dict[str, Any]],
    tabu_hashes: set[str],
    config: Mapping[str, Any],
    ledger_path: str,
) -> dict[str, Any]:
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve() / f"before_proposer_round_{int(round_idx):03d}"
    checkpoint_id = f"{run_id}:before_proposer:{int(round_idx)}"
    payload = {
        "checkpoint_id": checkpoint_id,
        "stage": CHECKPOINT_STAGE_BEFORE_PROPOSER,
        "run_id": run_id,
        "round_idx": int(round_idx),
        "created_at": float(time.time()),
        "ledger_path": str(ledger_path),
        "compiled_space": compiled_space_to_snapshot(compiled_space),
        "optimizer_observations": [trial_to_dict(trial) for trial in observations],
        "best_train_candidate": (
            best_train_candidate.to_dict() if best_train_candidate is not None else None
        ),
        "best_train_score": best_train_score,
        "baseline_candidate": baseline_candidate.to_dict(),
        "baseline_train_score": baseline_train_score,
        "heldout_baseline_score": heldout_baseline_score,
        "tabu_hashes": sorted(str(item) for item in tabu_hashes if str(item).strip()),
        "proposer_context": proposer_context_to_dict(proposer_context),
        "train_read_model": dict(train_read_model),
        "sampled_train_rows": list(sampled_train_rows),
        "recent_trial_rows": list(recent_trial_rows),
        "config": dict(config),
        "artifact_refs": {
            "checkpoint_dir": str(checkpoint_root),
            "checkpoint": str(checkpoint_root / "checkpoint.json"),
            "proposer_context": str(checkpoint_root / "proposer_context.json"),
            "read_model": str(checkpoint_root / "read_model.json"),
        },
    }
    _write_json(checkpoint_root / "checkpoint.json", payload)
    _write_json(checkpoint_root / "proposer_context.json", payload["proposer_context"])
    _write_json(checkpoint_root / "read_model.json", dict(train_read_model))
    return payload


def load_proposer_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if str(payload.get("stage") or "") != CHECKPOINT_STAGE_BEFORE_PROPOSER:
        raise ValueError("checkpoint is not a before_proposer checkpoint")
    return dict(payload)


async def run_proposer_from_checkpoint(
    *,
    checkpoint: Mapping[str, Any],
    agent: MiproOpenEnvReactAgent,
    config: MiproOpenEnvProposerConfig | None = None,
    variant: MiproOpenEnvProposerVariant | Mapping[str, Any] | None = None,
) -> MiproOpenEnvProposerOutcome:
    compiled_space = compiled_space_from_snapshot(dict(checkpoint.get("compiled_space") or {}))
    context = proposer_context_from_dict(dict(checkpoint.get("proposer_context") or {}))
    return await run_openenv_react_proposer(
        compiled_space=compiled_space,
        agent=agent,
        context=context,
        config=config,
        variant=variant,
        queue_state=dict(context.read_model_payload.get("rollout_queue_state") or {}),
        memory_state=dict(context.read_model_payload.get("proposer_memory_state") or {}),
    )


def proposed_candidates_from_outcome(
    *,
    checkpoint: Mapping[str, Any],
    outcome: MiproOpenEnvProposerOutcome,
) -> list[MiproProgramCandidate]:
    before = compiled_space_from_snapshot(dict(checkpoint.get("compiled_space") or {}))
    after = outcome.compiled_space
    best_payload = checkpoint.get("best_train_candidate") or checkpoint.get("baseline_candidate")
    if isinstance(best_payload, Mapping):
        base_config = dict(best_payload.get("source_config") or {})
    else:
        base_config = {}
    if not base_config:
        baseline = checkpoint.get("baseline_candidate")
        base_config = dict(baseline.get("source_config") or {}) if isinstance(baseline, Mapping) else {}
    candidates: list[MiproProgramCandidate] = []
    seen_hashes: set[str] = set()
    for component_key, after_options in sorted(after.search_space.items()):
        before_options = set(before.search_space.get(component_key) or [])
        for option_id in after_options:
            if option_id in before_options:
                continue
            candidate_config = {**base_config, component_key: str(option_id)}
            try:
                candidate = decode_config(after, candidate_config)
            except Exception:
                continue
            digest = str(candidate.lever_bundle_hash or "")
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            candidates.append(candidate)
    return candidates


def export_candidate_train_scores_from_ledger(
    ledger: SQLiteMiproRunLedger,
    *,
    task_id: str,
    output_path: str | Path | None = None,
    limit: int = 10000,
) -> dict[str, Any]:
    candidates = ledger.query_candidates(limit=limit)
    rollouts = ledger.query_rollouts(split="train", limit=limit)
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for rollout in rollouts:
        summary = rollout.get("rollout_summary")
        trace = rollout.get("trace_payload")
        summary_map = dict(summary) if isinstance(summary, Mapping) else {}
        trace_map = dict(trace) if isinstance(trace, Mapping) else {}
        candidate_id = str(rollout.get("candidate_id") or "")
        rows_by_candidate.setdefault(candidate_id, []).append(
            {
                "row_id": str(rollout.get("task_row_ref") or rollout.get("rollout_id") or ""),
                "rollout_id": rollout.get("rollout_id"),
                "split": "train",
                "index": summary_map.get("index"),
                "seed": rollout.get("seed"),
                "expected": summary_map.get("expected"),
                "prediction": summary_map.get("prediction"),
                "reward": float(rollout.get("score") or 0.0),
                "correct": summary_map.get("correct"),
                "trace": trace_map,
            }
        )
    payload = {
        "run_id": ledger.run_id,
        "task_id": task_id,
        "train_rows": max((len(rows) for rows in rows_by_candidate.values()), default=0),
        "candidates": [
            {
                "candidate_id": row.get("candidate_id"),
                "parent_candidate_id": row.get("parent_candidate_id"),
                "train_score": row.get("average_score"),
                "candidate": row.get("candidate"),
                "source_config": row.get("source_config"),
                "per_task_scores": rows_by_candidate.get(str(row.get("candidate_id") or ""), []),
            }
            for row in candidates
        ],
    }
    if output_path is not None:
        _write_json(Path(output_path), payload)
    return payload


def proposer_replay_summary(
    outcome: MiproOpenEnvProposerOutcome,
) -> dict[str, Any]:
    return {
        "proposer_summary": proposer_outcome_summary(outcome),
        "instruction_patches": [asdict(item) for item in outcome.instruction_patches],
        "demo_patches": [asdict(item) for item in outcome.demo_patches],
        "transcript": list(outcome.transcript),
    }


__all__ = [
    "CHECKPOINT_STAGE_BEFORE_PROPOSER",
    "compiled_space_from_snapshot",
    "compiled_space_to_snapshot",
    "export_candidate_train_scores_from_ledger",
    "load_proposer_checkpoint",
    "proposed_candidates_from_outcome",
    "proposer_context_from_dict",
    "proposer_context_to_dict",
    "proposer_replay_summary",
    "run_proposer_from_checkpoint",
    "trial_from_dict",
    "trial_to_dict",
    "write_proposer_checkpoint",
]
