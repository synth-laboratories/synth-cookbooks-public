"""Phase-3B runner: grounded OpenEnv proposer + dedup-aware train loop."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast
from uuid import uuid4

from synth_optimizers.miprov2.core.optimizer import (
    DiscreteMiproOptimizer,
    MiproTrialResult,
)
from synth_optimizers.miprov2.core.checkpointing import (
    CHECKPOINT_STAGE_BEFORE_PROPOSER,
    compiled_space_to_snapshot,
    write_proposer_checkpoint,
)
from synth_optimizers.miprov2.core.phase2_runner import (
    EvaluateCandidateFn,
    EvaluateCandidateOutcome,
    MiproHeldoutSnapshot,
    build_baseline_config,
)
from synth_optimizers.miprov2.core.program_compiler import (
    CompiledMiproSpace,
    decode_config,
)
from synth_optimizers.miprov2.core.program_model import (
    MiproProgramCandidate,
    demo_from_dict,
)
from synth_optimizers.miprov2.core.proposer_openenv import (
    MiproOpenEnvProposerConfig,
    MiproOpenEnvProposerContext,
    MiproOpenEnvProposerOutcome,
    MiproOpenEnvReactAgent,
    clone_compiled_space,
    proposer_outcome_summary,
    run_openenv_react_proposer,
    sync_optimizer_search_space,
)
from synth_optimizers.miprov2.core.proposer_environment import (
    MiproProposerEnvironment,
    tool_state_from_dict,
)
from synth_optimizers.miprov2.core.proposer_memory import (
    MiproRolloutLabel,
    MiproRolloutLabelAssignmentSource,
    MiproRolloutLabelDefinitionStatus,
    normalize_memory_state,
    proposer_memory_summary,
    rollout_label_id_for,
)
from synth_optimizers.miprov2.core.proposer_tools import tool_category
from synth_optimizers.miprov2.core.run_ledger import (
    SQLiteMiproRunLedger,
    load_resume_state,
    open_sqlite_run_ledger,
)
from synth_optimizers.miprov2.core.rollout_queue import (
    MiproCandidateInterventionRef,
    MiproQueuedRollout,
    MiproRolloutQueue,
    queue_id_for,
    rollout_id_for,
)

SampleTrainRowsOutcome: TypeAlias = list[Mapping[str, Any]]
SampleTrainRowsFn: TypeAlias = Callable[
    [int, int],
    SampleTrainRowsOutcome | Awaitable[SampleTrainRowsOutcome],
]
SummarizeRecentTrialsOutcome: TypeAlias = Mapping[str, Any]
SummarizeRecentTrialsFn: TypeAlias = Callable[
    [list[MiproTrialResult], int],
    SummarizeRecentTrialsOutcome | Awaitable[SummarizeRecentTrialsOutcome],
]
EvaluateQueuedRolloutFn: TypeAlias = Callable[
    [MiproProgramCandidate, MiproQueuedRollout],
    EvaluateCandidateOutcome | Awaitable[EvaluateCandidateOutcome],
]
LabelCompletedRolloutsOutcome: TypeAlias = list[Mapping[str, Any]]
LabelCompletedRolloutsFn: TypeAlias = Callable[
    [MiproTrialResult, MiproProgramCandidate, list[Mapping[str, Any]]],
    LabelCompletedRolloutsOutcome | Awaitable[LabelCompletedRolloutsOutcome],
]


@dataclass(slots=True, frozen=True)
class MiproPhase3Config:
    proposer_rounds: int = 4
    train_rounds_per_proposer_round: int = 2
    bootstrap_train_rounds: int = 0
    top_k: int = 1
    max_concurrency: int = 1
    seed_with_baseline: bool = True
    heldout_interval: int | None = None
    compute_final_heldout: bool = True
    max_recent_failures: int = 3
    max_recent_successes: int = 3
    proposer_read_model_limit: int = 50
    candidate_delta_example_limit: int = 5
    proposer_trace_dir: str | None = ".out/miprov2/proposer_traces"
    write_proposer_trace_json: bool = True
    proposer_config: MiproOpenEnvProposerConfig | None = None
    checkpoint_policy: str = "none"
    checkpoint_dir: str | None = None
    proposer_control: str = "auto"
    interactive_session_root: str | None = None
    interactive_resume_session_id: str | None = None

    def __post_init__(self) -> None:
        if int(self.proposer_rounds) < 0:
            raise ValueError("MiproPhase3Config.proposer_rounds must be >= 0")
        if int(self.train_rounds_per_proposer_round) < 0:
            raise ValueError(
                "MiproPhase3Config.train_rounds_per_proposer_round must be >= 0"
            )
        if int(self.bootstrap_train_rounds) < 0:
            raise ValueError("MiproPhase3Config.bootstrap_train_rounds must be >= 0")
        if int(self.top_k) <= 0:
            raise ValueError("MiproPhase3Config.top_k must be > 0")
        if int(self.max_concurrency) <= 0:
            raise ValueError("MiproPhase3Config.max_concurrency must be > 0")
        if self.heldout_interval is not None and int(self.heldout_interval) <= 0:
            raise ValueError(
                "MiproPhase3Config.heldout_interval must be > 0 when provided"
            )
        if int(self.max_recent_failures) < 0:
            raise ValueError("MiproPhase3Config.max_recent_failures must be >= 0")
        if int(self.max_recent_successes) < 0:
            raise ValueError("MiproPhase3Config.max_recent_successes must be >= 0")
        if int(self.proposer_read_model_limit) <= 0:
            raise ValueError("MiproPhase3Config.proposer_read_model_limit must be > 0")
        if int(self.candidate_delta_example_limit) <= 0:
            raise ValueError(
                "MiproPhase3Config.candidate_delta_example_limit must be > 0"
            )
        checkpoint_policy = str(self.checkpoint_policy or "none").strip()
        if checkpoint_policy not in {"none", "before_each_proposer"}:
            raise ValueError(
                "MiproPhase3Config.checkpoint_policy must be 'none' or 'before_each_proposer'"
            )
        proposer_control = str(self.proposer_control or "auto").strip()
        if proposer_control not in {"auto", "interactive_pause"}:
            raise ValueError(
                "MiproPhase3Config.proposer_control must be 'auto' or 'interactive_pause'"
            )
        if self.interactive_resume_session_id and proposer_control != "interactive_pause":
            raise ValueError(
                "MiproPhase3Config.interactive_resume_session_id requires proposer_control='interactive_pause'"
            )


class MiproGroundingHooksLike(Protocol):
    sample_train_rows: SampleTrainRowsFn | None
    summarize_recent_trials: SummarizeRecentTrialsFn | None


@dataclass(slots=True, frozen=True)
class MiproGroundingHooks:
    sample_train_rows: SampleTrainRowsFn | None = None
    summarize_recent_trials: SummarizeRecentTrialsFn | None = None


@dataclass(slots=True)
class MiproPhase3Outcome:
    train_observations: list[MiproTrialResult] = field(default_factory=list)
    best_train_candidate: MiproProgramCandidate | None = None
    best_train_score: float | None = None
    baseline_train_score: float | None = None
    heldout_baseline_score: float | None = None
    heldout_best_score: float | None = None
    heldout_lift: float | None = None
    heldout_snapshots: list[MiproHeldoutSnapshot] = field(default_factory=list)
    proposer_sessions: list[dict[str, Any]] = field(default_factory=list)
    proposer_round_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    proposer_diagnostics_aggregate: dict[str, Any] = field(default_factory=dict)
    proposer_trace_paths: list[str] = field(default_factory=list)
    stop_reason_frequency: dict[str, int] = field(default_factory=dict)
    tabu_hash_count: int = 0
    skipped_tabu_candidates: int = 0
    run_id: str | None = None
    ledger_path: str | None = None
    run_status: str = "running"
    pending_interactive_session: dict[str, Any] | None = None
    consumed_interactive_session: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class MiproProposerStepInput:
    round_idx: int
    recent_failures: tuple[str, ...]
    recent_successes: tuple[str, ...]
    sampled_train_rows: list[dict[str, Any]]
    recent_trial_rows: list[dict[str, Any]]
    recent_trial_summary: dict[str, Any]
    train_read_model: dict[str, Any]
    proposer_context: MiproOpenEnvProposerContext


def _aggregate_proposer_diagnostics(
    rounds: list[dict[str, Any]],
    stop_reason_frequency: Mapping[str, int],
) -> dict[str, Any]:
    model_turns = sum(int(item.get("model_turn_count") or 0) for item in rounds)
    tool_calls = sum(int(item.get("tool_call_count") or 0) for item in rounds)
    patch_actions = sum(int(item.get("patch_action_count") or 0) for item in rounds)
    patches_added = sum(int(item.get("patches_added") or 0) for item in rounds)
    read_actions = sum(int(item.get("read_action_count") or 0) for item in rounds)
    archive_spills = sum(int(item.get("archive_spill_count") or 0) for item in rounds)
    archived_messages = sum(
        int(item.get("archived_message_count") or 0) for item in rounds
    )
    duplicate_patches = sum(
        int(item.get("duplicate_patch_count") or 0) for item in rounds
    )
    ignored_patches = sum(int(item.get("ignored_patch_count") or 0) for item in rounds)
    policy_violations = sum(
        int(item.get("policy_violation_count") or 0) for item in rounds
    )
    grounding_reads = sum(
        int(item.get("grounding_read_action_count") or 0) for item in rounds
    )
    evidence_reads = sum(
        int(item.get("evidence_read_action_count") or 0) for item in rounds
    )
    distinct_tools_union: set[str] = set()
    for item in rounds:
        raw_tools = item.get("read_tools_used")
        if isinstance(raw_tools, list):
            for tool in raw_tools:
                text = str(tool).strip()
                if text:
                    distinct_tools_union.add(text)
    return {
        "round_count": len(rounds),
        "model_turn_count_total": int(model_turns),
        "tool_call_count_total": int(tool_calls),
        "read_action_count_total": int(read_actions),
        "patch_action_count_total": int(patch_actions),
        "patches_added_total": int(patches_added),
        "archive_spill_count_total": int(archive_spills),
        "archived_message_count_total": int(archived_messages),
        "duplicate_patch_count_total": int(duplicate_patches),
        "ignored_patch_count_total": int(ignored_patches),
        "policy_violation_count_total": int(policy_violations),
        "grounding_read_action_count_total": int(grounding_reads),
        "evidence_read_action_count_total": int(evidence_reads),
        "distinct_read_tools": sorted(distinct_tools_union),
        "patch_yield_total": (float(patches_added) / float(patch_actions))
        if patch_actions > 0
        else 0.0,
        "stop_reason_frequency": {
            str(k): int(v) for k, v in stop_reason_frequency.items()
        },
    }


def _safe_file_stem(value: str) -> str:
    text = "".join(
        ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(value)
    )
    compact = text.strip("._")
    return compact or "mipro_run"


def _write_proposer_trace_json(
    *,
    run_id: str | None,
    trace_dir: str | None,
    round_idx: int,
    payload: Mapping[str, Any],
) -> str | None:
    if trace_dir is None:
        return None
    folder = str(trace_dir).strip()
    if not folder:
        return None
    os.makedirs(folder, exist_ok=True)
    stem = _safe_file_stem(str(run_id or "mipro_run"))
    out_path = os.path.abspath(
        os.path.join(folder, f"{stem}_proposer_round_{int(round_idx):03d}.json")
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    return out_path


def _read_json_file(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))


def _read_jsonl_file(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        parsed = json.loads(text)
        if isinstance(parsed, Mapping):
            rows.append(dict(parsed))
    return rows


def _interactive_session_root(
    *,
    cfg: MiproPhase3Config,
    checkpoint_root: str,
) -> str:
    if cfg.interactive_session_root is not None:
        configured = str(cfg.interactive_session_root).strip()
        if configured:
            return configured
    return str(Path(checkpoint_root).expanduser().resolve().parent / "proposer_sessions")


def _interactive_outcome_from_committed_session(
    *,
    session_root: str,
    session_id: str,
    pending: Mapping[str, Any] | None,
) -> tuple[MiproOpenEnvProposerOutcome, dict[str, Any]]:
    environment = MiproProposerEnvironment.load(
        session_root=session_root,
        session_id=session_id,
    )
    session = environment.session
    if session.status != "committed":
        raise ValueError(
            f"interactive proposer session '{session_id}' must be committed, got {session.status!r}"
        )
    if session.committed_state_ref is None:
        raise ValueError(f"interactive proposer session '{session_id}' has no committed_state_ref")
    committed_state_path = Path(session.committed_state_ref).expanduser()
    if not committed_state_path.exists():
        raise FileNotFoundError(str(committed_state_path))

    if pending is not None:
        expected_run_id = str(pending.get("run_id") or "").strip()
        if expected_run_id and str(session.run_id or "") != expected_run_id:
            raise ValueError(
                f"interactive proposer session run_id mismatch: expected {expected_run_id}, got {session.run_id}"
            )
        expected_round_idx = int(pending.get("round_idx") or 0)
        if expected_round_idx and int(session.round_idx) != expected_round_idx:
            raise ValueError(
                f"interactive proposer session round mismatch: expected {expected_round_idx}, got {session.round_idx}"
            )
        expected_checkpoint_id = str(pending.get("checkpoint_id") or "").strip()
        if expected_checkpoint_id and str(session.source_ref or "") != expected_checkpoint_id:
            raise ValueError(
                "interactive proposer session checkpoint mismatch: "
                f"expected {expected_checkpoint_id}, got {session.source_ref}"
            )

    state = tool_state_from_dict(_read_json_file(committed_state_path))
    events = _read_jsonl_file(session.event_log_path)
    tool_events = [
        item
        for item in events
        if str(item.get("event_type") or "") == "tool_call"
        and str(item.get("tool_name") or "").strip()
    ]
    action_counts: dict[str, int] = {}
    read_tools: set[str] = set()
    patch_action_count = 0
    read_action_count = 0
    stop_reason = "interactive_commit"
    for event in tool_events:
        tool_name = str(event.get("tool_name") or "")
        action_counts[tool_name] = int(action_counts.get(tool_name, 0)) + 1
        category = tool_category(tool_name)
        if category == "evidence":
            read_action_count += 1
            read_tools.add(tool_name)
        if category == "search_space":
            result = dict(event.get("result") or {})
            if bool(result.get("state_mutated")):
                patch_action_count += 1
        result = dict(event.get("result") or {})
        if bool(result.get("stop_session")) and tool_name == "finish":
            stop_reason = "interactive_finish"

    outcome = MiproOpenEnvProposerOutcome(
        compiled_space=state.compiled_space,
        instruction_patches=list(state.instruction_patches),
        demo_patches=list(state.demo_patches),
        transcript=events,
        action_counts=action_counts,
        read_action_count=read_action_count,
        patch_action_count=patch_action_count,
        read_tools_used=tuple(sorted(read_tools)),
        stop_reason=stop_reason,
        tool_call_count=len(tool_events),
        queue_state=dict(state.queue_state),
        memory_state=dict(state.memory_state),
    )
    summary = {
        "session_id": session.session_id,
        "session_root": str(Path(session_root).expanduser().resolve()),
        "session_dir": session.session_dir,
        "run_id": session.run_id,
        "round_idx": int(session.round_idx),
        "source_ref": session.source_ref,
        "committed_state_ref": str(committed_state_path),
        "event_log_path": session.event_log_path,
        "event_count": int(session.event_count),
        "current_version": int(session.current_version),
        "status": session.status,
        "queue_state": dict(state.queue_state),
        "memory_state": dict(state.memory_state),
        "memory_summary": proposer_memory_summary(state.memory_state),
    }
    return outcome, summary


def _candidate_intervention_ref(candidate: MiproProgramCandidate) -> MiproCandidateInterventionRef:
    return MiproCandidateInterventionRef(
        candidate_id=str(candidate.candidate_id or ""),
        parent_candidate_id=candidate.parent_candidate_id,
        lever_bundle_hash=str(candidate.lever_bundle_hash or ""),
        source_config=dict(candidate.source_config),
        plugin_kind=str(candidate.active_execution_mode or "prompt"),
        plugin_id=candidate.active_model_transform_id,
        prompt_intervention={
            "selected_instructions": dict(candidate.selected_instructions),
            "selected_instruction_base_option_ids": dict(
                candidate.selected_instruction_base_option_ids
            ),
            "selected_instruction_transform_ids": {
                key: list(value)
                for key, value in candidate.selected_instruction_transform_ids.items()
            },
            "selected_demos": {
                module_id: {
                    slot_id: demo.to_dict() for slot_id, demo in slot_map.items()
                }
                for module_id, slot_map in candidate.selected_demos.items()
            },
        },
        sft_intervention={
            "active_finetune_ref": candidate.active_finetune_ref,
            "active_model_transform_id": candidate.active_model_transform_id,
        },
        metadata={"program_id": candidate.program_id},
    )


def _row_ref(row: Mapping[str, Any], idx: int) -> tuple[str, int | None, str | None]:
    row_id = str(
        row.get("row_id")
        or row.get("id")
        or row.get("task_instance_id")
        or row.get("seed")
        or f"row_{idx:04d}"
    )
    seed_value = row.get("seed")
    seed = int(seed_value) if seed_value is not None and str(seed_value).strip() else None
    task_instance_id = (
        str(row["task_instance_id"]) if row.get("task_instance_id") is not None else None
    )
    return row_id, seed, task_instance_id


async def _build_rollout_queue(
    *,
    optimizer: DiscreteMiproOptimizer,
    compiled_space: CompiledMiproSpace,
    run_id: str,
    round_idx: int,
    sampled_rows: list[dict[str, Any]],
    top_k: int,
    split: str = "train",
    created_by: str = "tpe",
    queue_kind: str = "tentative",
    suffix: str = "",
) -> MiproRolloutQueue:
    configs = await optimizer.preview_suggest(top_k=max(1, int(top_k)))
    queue_id = queue_id_for(
        run_id=run_id,
        round_idx=int(round_idx),
        kind=queue_kind,
        suffix=suffix or created_by,
    )
    candidates: list[MiproCandidateInterventionRef] = []
    rollouts: list[MiproQueuedRollout] = []
    rows = list(sampled_rows or [{}])
    for candidate_index, trial_config in enumerate(configs):
        candidate = decode_config(compiled_space, trial_config)
        candidate_ref = _candidate_intervention_ref(candidate)
        candidates.append(candidate_ref)
        for row_index, row in enumerate(rows):
            row_id, seed, task_instance_id = _row_ref(row, row_index)
            rollout_id = rollout_id_for(
                queue_id=queue_id,
                candidate_id=str(candidate.candidate_id or ""),
                row_id=row_id,
                index=(candidate_index * len(rows)) + row_index,
            )
            rollouts.append(
                MiproQueuedRollout(
                    rollout_id=rollout_id,
                    candidate_id=str(candidate.candidate_id or ""),
                    candidate_interventions=[candidate_ref],
                    split=split,
                    row_id=row_id,
                    seed=seed,
                    task_instance_id=task_instance_id,
                    evaluator_config={
                        "row": dict(row),
                        "row_index": row_index,
                        "candidate_index": candidate_index,
                    },
                    priority=float(len(rollouts)),
                    created_by=created_by,
                )
            )
    return MiproRolloutQueue(
        queue_id=queue_id,
        queue_kind=queue_kind,
        task_id=None,
        split=split,
        created_by=created_by,
        candidates=candidates,
        rollouts=rollouts,
        metadata={
            "round_idx": int(round_idx),
            "top_k": int(top_k),
            "sampled_row_count": len(rows),
        },
    )


def _coerce_eval_outcome(
    outcome: EvaluateCandidateOutcome,
) -> tuple[float, dict[str, Any]]:
    if isinstance(outcome, tuple):
        score, details = outcome
        return float(score), dict(details)
    return float(outcome), {}


async def _evaluate_candidate_once(
    *,
    evaluate: EvaluateCandidateFn,
    candidate: MiproProgramCandidate,
    semaphore: asyncio.Semaphore,
) -> tuple[float, dict[str, Any], float]:
    start = time.perf_counter()
    async with semaphore:
        outcome = evaluate(candidate)
        if inspect.isawaitable(outcome):
            outcome = await cast(Awaitable[EvaluateCandidateOutcome], outcome)
    score, details = _coerce_eval_outcome(outcome)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return score, details, latency_ms


def _recent_candidate_ids(
    observations: list[MiproTrialResult],
    *,
    limit: int,
    reverse: bool,
) -> tuple[str, ...]:
    if limit <= 0:
        return ()
    seen: set[str] = set()
    output: list[str] = []
    for trial in sorted(observations, key=lambda item: item.score, reverse=reverse):
        candidate_id = str(trial.candidate_id or "").strip()
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        output.append(candidate_id)
        if len(output) >= limit:
            break
    return tuple(output)


async def _resolve_train_rows(
    hooks: MiproGroundingHooksLike,
    *,
    round_idx: int,
    limit: int,
) -> list[dict[str, Any]]:
    callback = hooks.sample_train_rows
    if callback is None:
        return []
    raw = callback(int(round_idx), int(limit))
    if inspect.isawaitable(raw):
        raw = await cast(Awaitable[SampleTrainRowsOutcome], raw)
    if not isinstance(raw, list):
        raise ValueError("MiproGroundingHooks.sample_train_rows must return a list")
    return [dict(item) for item in raw if isinstance(item, Mapping)]


async def _resolve_recent_trial_summary(
    hooks: MiproGroundingHooksLike,
    *,
    observations: list[MiproTrialResult],
    limit: int,
) -> dict[str, Any]:
    callback = hooks.summarize_recent_trials
    if callback is None:
        return {}
    raw = callback(list(observations), int(limit))
    if inspect.isawaitable(raw):
        raw = await cast(Awaitable[SummarizeRecentTrialsOutcome], raw)
    if not isinstance(raw, Mapping):
        raise ValueError(
            "MiproGroundingHooks.summarize_recent_trials must return a mapping"
        )
    return dict(raw)


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _sanitize_rollout_component(value: Any) -> str:
    text = "".join(
        ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(value or "")
    ).strip("._")
    return text or "item"


def _fallback_rollout_id(
    *,
    split: str,
    candidate_id: str,
    round_idx: int,
    ordinal: int,
) -> str:
    return (
        f"{_sanitize_rollout_component(split)}_"
        f"{_sanitize_rollout_component(candidate_id)}_"
        f"r{int(round_idx):03d}_"
        f"{int(ordinal):03d}_{uuid4().hex[:8]}"
    )


def _normalize_trace_payload(
    trial_details: Mapping[str, Any],
    row_details: Mapping[str, Any],
) -> dict[str, Any]:
    raw_trace = row_details.get("trace_payload")
    if isinstance(raw_trace, Mapping):
        payload = dict(raw_trace)
    else:
        payload = {}
    for key in (
        "trace",
        "prompt_excerpt",
        "assistant_response_excerpt",
        "ideal_excerpt",
        "reasoning_trace",
    ):
        value = row_details.get(key)
        if value in (None, ""):
            value = trial_details.get(key)
        if value not in (None, ""):
            payload.setdefault(
                "trace_text" if key == "trace" else str(key), str(value)
            )
    if not payload:
        payload = {
            "trace_text": json.dumps(
                {k: v for k, v in row_details.items() if k not in {"artifacts", "evidence_artifacts"}},
                sort_keys=True,
                ensure_ascii=True,
            )[:4000]
        }
    return payload


def _normalize_score_components(
    trial: MiproTrialResult,
    row_details: Mapping[str, Any],
) -> dict[str, Any]:
    raw_components = row_details.get("score_components")
    if isinstance(raw_components, Mapping):
        payload = dict(raw_components)
    else:
        payload = {}
    for key in ("harm_penalty", "rubric_count", "reward", "cost_proxy"):
        value = row_details.get(key)
        if value is None:
            value = trial.details.get(key)
        if value is not None:
            payload[key] = value
    payload.setdefault("trial_score", float(trial.score))
    return payload


def _normalize_verifier_verdict(
    trial_details: Mapping[str, Any],
    row_details: Mapping[str, Any],
) -> dict[str, Any]:
    for key in ("verifier_verdict", "verdict", "judge_verdict"):
        value = row_details.get(key)
        if value is None:
            value = trial_details.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {
        "status": "unsupported",
        "reason": "verifier verdict not provided by evaluator",
    }


def _normalize_artifacts(
    trial_details: Mapping[str, Any],
    row_details: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw = row_details.get("evidence_artifacts")
    if raw is None:
        raw = row_details.get("artifacts")
    if raw is None:
        raw = trial_details.get("evidence_artifacts")
    if raw is None:
        raw = trial_details.get("artifacts")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _trial_rollout_rows(
    *,
    trial: MiproTrialResult,
    candidate: MiproProgramCandidate,
    split: str,
    round_idx: int,
) -> list[dict[str, Any]]:
    trial_details = dict(trial.details)
    raw_rows = trial_details.get("seed_details")
    rows = (
        [dict(item) for item in raw_rows if isinstance(item, Mapping)]
        if isinstance(raw_rows, list)
        else []
    )
    if not rows:
        rows = [trial_details]
    payload: list[dict[str, Any]] = []
    for ordinal, row_details in enumerate(rows, start=1):
        rollout_id = _string_or_none(row_details.get("rollout_id")) or _fallback_rollout_id(
            split=split,
            candidate_id=str(candidate.candidate_id or ""),
            round_idx=round_idx,
            ordinal=ordinal,
        )
        prompt_id = _string_or_none(row_details.get("prompt_id"))
        seed = _int_or_none(row_details.get("seed"))
        task_row_ref = (
            _string_or_none(row_details.get("task_row_ref"))
            or prompt_id
            or (str(seed) if seed is not None else None)
            or rollout_id
        )
        item_score = row_details.get("score")
        try:
            normalized_score = float(item_score) if item_score is not None else float(trial.score)
        except (TypeError, ValueError):
            normalized_score = float(trial.score)
        trace_payload = _normalize_trace_payload(trial_details, row_details)
        summary_payload = {
            "rollout_id": rollout_id,
            "candidate_id": str(candidate.candidate_id or ""),
            "parent_candidate_id": str(candidate.parent_candidate_id or "").strip() or None,
            "parent_candidate_ids": list(candidate.parent_candidate_ids),
            "lever_bundle_hash": str(candidate.lever_bundle_hash or ""),
            "split": str(split),
            "round_idx": int(round_idx),
            "task_row_ref": task_row_ref,
            "seed": seed,
            "prompt_id": prompt_id,
            "score": normalized_score,
            "trial_score": float(trial.score),
            "system_prompt": trial_details.get("system_prompt"),
            "selected_instruction_base_option_ids": dict(
                candidate.selected_instruction_base_option_ids
            ),
            "selected_instruction_transform_ids": {
                module_id: list(transform_ids)
                for module_id, transform_ids in candidate.selected_instruction_transform_ids.items()
            },
            "prompt_excerpt": row_details.get("prompt_excerpt"),
            "assistant_response_excerpt": row_details.get("assistant_response_excerpt"),
            "ideal_excerpt": row_details.get("ideal_excerpt"),
        }
        payload.append(
            {
                "rollout_id": rollout_id,
                "candidate_id": str(candidate.candidate_id or ""),
                "lever_bundle_hash": str(candidate.lever_bundle_hash or ""),
                "split": str(split),
                "round_idx": int(round_idx),
                "task_row_ref": task_row_ref,
                "seed": seed,
                "prompt_id": prompt_id,
                "score": normalized_score,
                "score_components": _normalize_score_components(trial, row_details),
                "rollout_summary": summary_payload,
                "trace_payload": trace_payload,
                "verifier_verdict": _normalize_verifier_verdict(trial_details, row_details),
                "evidence_artifacts": _normalize_artifacts(trial_details, row_details),
            }
        )
    return payload


def _persist_candidate_trial(
    *,
    ledger: SQLiteMiproRunLedger,
    trial: MiproTrialResult,
    candidate: MiproProgramCandidate,
    split: str,
    round_idx: int,
    candidate_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ledger.upsert_candidate(
        candidate=candidate,
        round_idx=round_idx,
        candidate_metadata=dict(candidate_metadata or {}),
    )
    persisted: list[dict[str, Any]] = []
    for row in _trial_rollout_rows(
        trial=trial,
        candidate=candidate,
        split=split,
        round_idx=round_idx,
    ):
        workspace_entry = ledger.append_rollout(
            rollout_id=str(row["rollout_id"]),
            candidate_id=str(row["candidate_id"]),
            lever_bundle_hash=str(row["lever_bundle_hash"]),
            split=str(row["split"]),
            round_idx=int(row["round_idx"]),
            task_row_ref=_string_or_none(row["task_row_ref"]),
            seed=_int_or_none(row["seed"]),
            prompt_id=_string_or_none(row["prompt_id"]),
            score=float(row["score"]),
            score_components=dict(row["score_components"]),
            rollout_summary=dict(row["rollout_summary"]),
            trace_payload=row["trace_payload"],
            verifier_verdict=dict(row["verifier_verdict"]),
            evidence_artifacts=list(row["evidence_artifacts"]),
        )
        persisted.append({**row, "workspace": workspace_entry})
    return persisted


def _rollout_key(row: Mapping[str, Any]) -> str:
    task_row_ref = _string_or_none(row.get("task_row_ref"))
    prompt_id = _string_or_none(row.get("prompt_id"))
    seed = _int_or_none(row.get("seed"))
    if task_row_ref is not None:
        return f"task:{task_row_ref}"
    if prompt_id is not None and seed is not None:
        return f"prompt_seed:{prompt_id}:{seed}"
    if prompt_id is not None:
        return f"prompt:{prompt_id}"
    if seed is not None:
        return f"seed:{seed}"
    return f"rollout:{_string_or_none(row.get('rollout_id')) or uuid4().hex}"


def _motif_key(verdict: Mapping[str, Any]) -> str:
    status = _string_or_none(verdict.get("status")) or "unknown"
    reason = _string_or_none(verdict.get("reason"))
    labels = verdict.get("labels")
    if isinstance(labels, list) and labels:
        label_text = ",".join(sorted(str(item) for item in labels[:3]))
        return f"{status}:{label_text}"
    if reason:
        return f"{status}:{reason}"
    return status


def _evidence_refs_for_rollout(
    evidence_by_rollout: Mapping[str, list[dict[str, Any]]],
    rollout_id: str,
) -> list[dict[str, Any]]:
    return [
        {
            "kind": item.get("kind"),
            "path": item.get("path"),
            "relative_path": item.get("relative_path"),
        }
        for item in evidence_by_rollout.get(str(rollout_id), [])
    ]


def _build_verdict_digest_summary(
    *,
    candidate_id: str,
    split: str,
    rollouts: list[dict[str, Any]],
    evidence_by_rollout: Mapping[str, list[dict[str, Any]]],
    example_limit: int,
) -> dict[str, Any]:
    motif_counts: dict[str, int] = {}
    unsupported = 0
    for row in rollouts:
        verdict = row.get("verifier_verdict")
        verdict_map = dict(verdict) if isinstance(verdict, Mapping) else {}
        if str(verdict_map.get("status") or "") == "unsupported":
            unsupported += 1
        motif = _motif_key(verdict_map)
        motif_counts[motif] = int(motif_counts.get(motif, 0)) + 1
    average_score = (
        sum(float(row.get("score") or 0.0) for row in rollouts) / len(rollouts)
        if rollouts
        else None
    )
    representative = []
    for row in sorted(rollouts, key=lambda item: float(item.get("score") or 0.0))[
        : max(1, int(example_limit))
    ]:
        representative.append(
            {
                "rollout_id": row.get("rollout_id"),
                "score": row.get("score"),
                "prompt_id": row.get("prompt_id"),
                "seed": row.get("seed"),
                "verifier_verdict": row.get("verifier_verdict"),
                "evidence_refs": _evidence_refs_for_rollout(
                    evidence_by_rollout, str(row.get("rollout_id") or "")
                ),
            }
        )
    return {
        "candidate_id": str(candidate_id),
        "split": str(split),
        "rollout_count": len(rollouts),
        "average_score": average_score,
        "unsupported_verdict_count": unsupported,
        "supported_verdict_count": max(0, len(rollouts) - unsupported),
        "repeated_failure_motifs": [
            {"motif": motif, "count": count}
            for motif, count in sorted(
                motif_counts.items(),
                key=lambda item: (-int(item[1]), str(item[0])),
            )[: max(1, int(example_limit))]
        ],
        "representative_rollouts": representative,
    }


def _build_delta_summary(
    *,
    candidate_id: str,
    compare_to_candidate_id: str,
    comparison_kind: str,
    split: str,
    candidate_rollouts: list[dict[str, Any]],
    compare_rollouts: list[dict[str, Any]],
    evidence_by_rollout: Mapping[str, list[dict[str, Any]]],
    example_limit: int,
) -> dict[str, Any]:
    compare_by_key = {_rollout_key(row): row for row in compare_rollouts}
    disagreements: list[dict[str, Any]] = []
    motif_counts: dict[str, int] = {}
    for row in candidate_rollouts:
        compare_row = compare_by_key.get(_rollout_key(row))
        if compare_row is None:
            continue
        delta = float(row.get("score") or 0.0) - float(compare_row.get("score") or 0.0)
        verdict_map = dict(row.get("verifier_verdict") or {})
        motif = _motif_key(verdict_map)
        motif_counts[motif] = int(motif_counts.get(motif, 0)) + 1
        disagreements.append(
            {
                "task_row_ref": row.get("task_row_ref"),
                "prompt_id": row.get("prompt_id"),
                "seed": row.get("seed"),
                "candidate_rollout_id": row.get("rollout_id"),
                "compare_rollout_id": compare_row.get("rollout_id"),
                "candidate_score": row.get("score"),
                "compare_score": compare_row.get("score"),
                "delta": delta,
                "candidate_evidence_refs": _evidence_refs_for_rollout(
                    evidence_by_rollout, str(row.get("rollout_id") or "")
                ),
                "compare_evidence_refs": _evidence_refs_for_rollout(
                    evidence_by_rollout, str(compare_row.get("rollout_id") or "")
                ),
            }
        )
    disagreements.sort(key=lambda item: float(item.get("delta") or 0.0), reverse=True)
    compared_count = len(disagreements)
    aggregate_lift = (
        sum(float(item.get("delta") or 0.0) for item in disagreements) / compared_count
        if compared_count > 0
        else None
    )
    top_improved = disagreements[: max(1, int(example_limit))]
    top_regressed = sorted(
        disagreements,
        key=lambda item: float(item.get("delta") or 0.0),
    )[: max(1, int(example_limit))]
    return {
        "candidate_id": str(candidate_id),
        "compare_to_candidate_id": str(compare_to_candidate_id),
        "comparison_kind": str(comparison_kind),
        "split": str(split),
        "candidate_rollout_count": len(candidate_rollouts),
        "compare_rollout_count": len(compare_rollouts),
        "compared_rollout_count": compared_count,
        "aggregate_lift": aggregate_lift,
        "per_task_seed_disagreements": disagreements[: max(3, int(example_limit) * 3)],
        "top_improved_rollouts": top_improved,
        "top_regressed_rollouts": top_regressed,
        "repeated_verifier_failure_motifs": [
            {"motif": motif, "count": count}
            for motif, count in sorted(
                motif_counts.items(),
                key=lambda item: (-int(item[1]), str(item[0])),
            )[: max(1, int(example_limit))]
        ],
    }


def _refresh_candidate_read_model(
    *,
    ledger: SQLiteMiproRunLedger,
    split: str,
    example_limit: int,
) -> dict[str, Any]:
    rollouts = ledger.query_rollouts(split=split, limit=10_000)
    evidence_files = ledger.query_evidence_files(limit=20_000)
    evidence_by_rollout: dict[str, list[dict[str, Any]]] = {}
    for item in evidence_files:
        evidence_by_rollout.setdefault(str(item.get("rollout_id") or ""), []).append(
            item
        )
    rollouts_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rollouts:
        rollouts_by_candidate.setdefault(str(row.get("candidate_id") or ""), []).append(
            row
        )
    candidates = ledger.query_candidates(limit=10_000)
    for candidate_row in candidates:
        candidate_id = str(candidate_row.get("candidate_id") or "")
        candidate_rollouts = rollouts_by_candidate.get(candidate_id, [])
        if candidate_rollouts:
            ledger.upsert_candidate_verdict_digest(
                candidate_id=candidate_id,
                split=split,
                summary=_build_verdict_digest_summary(
                    candidate_id=candidate_id,
                    split=split,
                    rollouts=candidate_rollouts,
                    evidence_by_rollout=evidence_by_rollout,
                    example_limit=example_limit,
                ),
            )
    refreshed_digests = ledger.query_candidate_verdict_digests(split=split, limit=10_000)
    refreshed_deltas: list[dict[str, Any]] = []
    baseline_candidate_ids = [
        str(item.get("candidate_id") or "")
        for item in candidates
        if bool((item.get("candidate_metadata") or {}).get("phase3_baseline"))
    ]
    baseline_candidate_id = baseline_candidate_ids[0] if baseline_candidate_ids else None
    for candidate_row in candidates:
        candidate_id = str(candidate_row.get("candidate_id") or "")
        candidate_rollouts = rollouts_by_candidate.get(candidate_id, [])
        parent_candidate_id = _string_or_none(candidate_row.get("parent_candidate_id"))
        if (
            baseline_candidate_id is not None
            and candidate_id != baseline_candidate_id
            and candidate_rollouts
        ):
            compare_rollouts = rollouts_by_candidate.get(baseline_candidate_id, [])
            if compare_rollouts:
                refreshed_deltas.append(
                    ledger.upsert_candidate_delta(
                        candidate_id=candidate_id,
                        compare_to_candidate_id=baseline_candidate_id,
                        comparison_kind="baseline",
                        split=split,
                        summary=_build_delta_summary(
                            candidate_id=candidate_id,
                            compare_to_candidate_id=baseline_candidate_id,
                            comparison_kind="baseline",
                            split=split,
                            candidate_rollouts=candidate_rollouts,
                            compare_rollouts=compare_rollouts,
                            evidence_by_rollout=evidence_by_rollout,
                            example_limit=example_limit,
                        ),
                    )
                )
        if (
            parent_candidate_id is not None
            and parent_candidate_id != candidate_id
            and candidate_rollouts
        ):
            parent_rollouts = rollouts_by_candidate.get(parent_candidate_id, [])
            if parent_rollouts:
                refreshed_deltas.append(
                    ledger.upsert_candidate_delta(
                        candidate_id=candidate_id,
                        compare_to_candidate_id=parent_candidate_id,
                        comparison_kind="parent",
                        split=split,
                        summary=_build_delta_summary(
                            candidate_id=candidate_id,
                            compare_to_candidate_id=parent_candidate_id,
                            comparison_kind="parent",
                            split=split,
                            candidate_rollouts=candidate_rollouts,
                            compare_rollouts=parent_rollouts,
                            evidence_by_rollout=evidence_by_rollout,
                            example_limit=example_limit,
                        ),
                    )
                )
    return {
        "baseline_candidate_id": baseline_candidate_id,
        "candidates": ledger.query_candidates(limit=10_000),
        "rollouts": ledger.query_rollouts(split=split, limit=10_000),
        "evidence_files": ledger.query_evidence_files(limit=20_000),
        "candidate_deltas": ledger.query_candidate_rollout_deltas(
            split=split, limit=10_000
        ),
        "verdict_digests": refreshed_digests,
    }


def _recent_trial_rows_from_read_model(
    *,
    read_model_payload: Mapping[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    raw_rollouts = read_model_payload.get("rollouts")
    if not isinstance(raw_rollouts, list):
        return []
    rows: list[dict[str, Any]] = []
    for rollout in raw_rollouts:
        if not isinstance(rollout, Mapping):
            continue
        if str(rollout.get("split") or "") != "train":
            continue
        summary = rollout.get("rollout_summary")
        trace_payload = rollout.get("trace_payload")
        verifier_verdict = rollout.get("verifier_verdict")
        rows.append(
            {
                "candidate_id": rollout.get("candidate_id"),
                "rollout_id": rollout.get("rollout_id"),
                "lever_bundle_hash": rollout.get("lever_bundle_hash"),
                "score": rollout.get("score"),
                "details": {
                    "rollout_summary": dict(summary) if isinstance(summary, Mapping) else {},
                    "trace_payload": dict(trace_payload) if isinstance(trace_payload, Mapping) else trace_payload,
                    "verifier_verdict": dict(verifier_verdict) if isinstance(verifier_verdict, Mapping) else {},
                },
                "trace": (
                    str((trace_payload or {}).get("trace_text") or "")
                    if isinstance(trace_payload, Mapping)
                    else str(trace_payload or "")
                ),
            }
        )
        if len(rows) >= max(1, int(limit)):
            break
    return rows


async def run_phase3_loop(
    *,
    compiled_space: CompiledMiproSpace,
    optimizer: DiscreteMiproOptimizer,
    agent: MiproOpenEnvReactAgent,
    evaluate_train: EvaluateCandidateFn,
    evaluate_queued_rollout: EvaluateQueuedRolloutFn | None = None,
    label_completed_rollouts: LabelCompletedRolloutsFn | None = None,
    evaluate_heldout: EvaluateCandidateFn | None = None,
    grounding_hooks: MiproGroundingHooksLike | None = None,
    config: MiproPhase3Config | None = None,
    run_id: str | None = None,
    ledger_path: str | None = None,
    resume: bool = False,
) -> MiproPhase3Outcome:
    """Run proposer + train loop with evidence-rich rollout state and digests."""

    cfg = config or MiproPhase3Config()
    hooks = grounding_hooks or MiproGroundingHooks()
    semaphore = asyncio.Semaphore(max(1, int(cfg.max_concurrency)))
    out = MiproPhase3Outcome()
    working_space = clone_compiled_space(compiled_space)
    tabu_lever_bundle_hashes: set[str] = set()
    ledger = open_sqlite_run_ledger(
        program_id=compiled_space.program_template.program_id,
        mode="phase3",
        run_id=run_id,
        ledger_path=ledger_path,
        resume=resume,
    )
    out.run_id = ledger.run_id
    out.ledger_path = ledger.ledger_path
    observation_seq = 1
    heldout_seq = 1
    proposer_round_seq = 1
    train_rounds_completed = 0
    proposer_rounds_completed = 0
    latest_train_read_model: dict[str, Any] = {}
    proposer_memory_state: dict[str, Any] = normalize_memory_state({})
    pending_interactive_proposer: dict[str, Any] | None = None
    interactive_resume_session_id = str(cfg.interactive_resume_session_id or "").strip()

    def persist_compiled_space_snapshot() -> None:
        ledger.upsert_state(
            key="compiled_space_snapshot",
            value=compiled_space_to_snapshot(working_space),
        )

    async def refresh_train_read_model() -> dict[str, Any]:
        nonlocal latest_train_read_model
        latest_train_read_model = _refresh_candidate_read_model(
            ledger=ledger,
            split="train",
            example_limit=int(cfg.candidate_delta_example_limit),
        )
        return latest_train_read_model

    async def refresh_heldout_read_model() -> dict[str, Any]:
        return _refresh_candidate_read_model(
            ledger=ledger,
            split="heldout",
            example_limit=int(cfg.candidate_delta_example_limit),
        )

    def persist_common_state() -> None:
        ledger.upsert_state(key="train_rounds_completed", value=train_rounds_completed)
        ledger.upsert_state(
            key="proposer_rounds_completed", value=proposer_rounds_completed
        )
        ledger.upsert_state(
            key="skipped_tabu_candidates", value=out.skipped_tabu_candidates
        )
        ledger.upsert_state(
            key="stop_reason_frequency", value=dict(out.stop_reason_frequency)
        )
        ledger.upsert_state(key="proposer_sessions", value=list(out.proposer_sessions))
        ledger.upsert_state(
            key="proposer_round_diagnostics", value=list(out.proposer_round_diagnostics)
        )
        ledger.upsert_state(
            key="proposer_diagnostics_aggregate",
            value=dict(out.proposer_diagnostics_aggregate),
        )
        ledger.upsert_state(
            key="proposer_trace_paths", value=list(out.proposer_trace_paths)
        )
        ledger.upsert_state(key="tabu_hashes", value=sorted(tabu_lever_bundle_hashes))
        ledger.upsert_state(key="tabu_hash_count", value=len(tabu_lever_bundle_hashes))
        ledger.upsert_state(key="latest_proposer_memory_state", value=dict(proposer_memory_state))
        ledger.upsert_state(key="baseline_train_score", value=out.baseline_train_score)
        ledger.upsert_state(
            key="heldout_baseline_score", value=out.heldout_baseline_score
        )
        ledger.upsert_state(key="heldout_best_score", value=out.heldout_best_score)
        ledger.upsert_state(key="heldout_lift", value=out.heldout_lift)
        ledger.upsert_state(key="best_train_score", value=out.best_train_score)
        if out.best_train_candidate is not None:
            ledger.upsert_state(
                key="best_train_candidate", value=out.best_train_candidate.to_dict()
            )
        ledger.upsert_state(
            key="heldout_snapshots_count", value=len(out.heldout_snapshots)
        )
        persist_compiled_space_snapshot()

    async def evaluate_candidate_trial(
        *,
        evaluate: EvaluateCandidateFn,
        trial_config: dict[str, str],
        candidate: MiproProgramCandidate,
        round_idx_value: int,
        split: str,
        extra_details: Mapping[str, Any] | None = None,
    ) -> MiproTrialResult:
        score, details, latency = await _evaluate_candidate_once(
            evaluate=evaluate,
            candidate=candidate,
            semaphore=semaphore,
        )
        detail_payload = {
            **dict(details),
            **dict(extra_details or {}),
            "candidate_id": candidate.candidate_id,
            "lever_bundle_hash": candidate.lever_bundle_hash,
            "round_idx": round_idx_value,
            "split": split,
        }
        return MiproTrialResult(
            config=trial_config,
            score=score,
            details=detail_payload,
            latency_ms=latency,
            candidate_id=candidate.candidate_id,
            lever_bundle_hash=candidate.lever_bundle_hash,
        )

    async def persist_train_trial(
        *,
        seq: int,
        round_idx_value: int,
        trial: MiproTrialResult,
        candidate: MiproProgramCandidate,
        phase3_baseline: bool = False,
        proposer_round_idx: int | None = None,
    ) -> int:
        ledger.append_observation(seq=seq, round_idx=round_idx_value, trial=trial)
        _persist_candidate_trial(
            ledger=ledger,
            trial=trial,
            candidate=candidate,
            split="train",
            round_idx=round_idx_value,
            candidate_metadata={
                "phase3_baseline": bool(phase3_baseline),
                "latest_split": "train",
                "latest_round_idx": int(round_idx_value),
                "proposer_round_idx": proposer_round_idx,
            },
        )
        return seq + 1

    async def persist_heldout_trial(
        *,
        trial: MiproTrialResult,
        candidate: MiproProgramCandidate,
        round_idx_value: int,
        phase3_baseline: bool = False,
    ) -> None:
        _persist_candidate_trial(
            ledger=ledger,
            trial=trial,
            candidate=candidate,
            split="heldout",
            round_idx=round_idx_value,
            candidate_metadata={
                "phase3_baseline": bool(phase3_baseline),
                "latest_split": "heldout",
                "latest_round_idx": int(round_idx_value),
            },
        )

    def active_label_definitions() -> list[Mapping[str, Any]]:
        normalized = normalize_memory_state(proposer_memory_state)
        return [
            dict(item)
            for item in normalized["label_definitions"].values()
            if str(item.get("status") or "active")
            == MiproRolloutLabelDefinitionStatus.ACTIVE.value
        ]

    async def maybe_label_completed_rollouts(
        *,
        trial: MiproTrialResult,
        candidate: MiproProgramCandidate,
    ) -> list[dict[str, Any]]:
        nonlocal proposer_memory_state
        definitions = active_label_definitions()
        if label_completed_rollouts is None or not definitions:
            return []
        raw = label_completed_rollouts(trial, candidate, definitions)
        if inspect.isawaitable(raw):
            raw = await cast(Awaitable[LabelCompletedRolloutsOutcome], raw)
        normalized = normalize_memory_state(proposer_memory_state)
        known_definitions = {
            str(item.get("label_id") or ""): dict(item)
            for item in normalized["label_definitions"].values()
        }
        accepted: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        for item in list(raw or []):
            if not isinstance(item, Mapping):
                warnings.append({"reason": "label payload is not an object", "payload": str(item)})
                continue
            payload = dict(item)
            label_id = str(payload.get("label_id") or "").strip()
            definition = known_definitions.get(label_id)
            if definition is None:
                warnings.append({"reason": "unknown label_id", "label_id": label_id})
                continue
            if (
                str(definition.get("status") or "active")
                != MiproRolloutLabelDefinitionStatus.ACTIVE.value
            ):
                warnings.append({"reason": "label definition is not active", "label_id": label_id})
                continue
            value = str(payload.get("value") or "").strip()
            allowed_values = [str(value) for value in list(definition.get("allowed_values") or [])]
            if value not in allowed_values:
                warnings.append(
                    {
                        "reason": "value is not allowed for label definition",
                        "label_id": label_id,
                        "value": value,
                        "allowed_values": allowed_values,
                    }
                )
                continue
            label_payload = {
                **payload,
                "label_id": label_id,
                "value": value,
                "rollout_id": str(payload.get("rollout_id") or trial.candidate_id or ""),
                "candidate_id": str(payload.get("candidate_id") or candidate.candidate_id or ""),
                "assignment_source": str(
                    payload.get("assignment_source")
                    or MiproRolloutLabelAssignmentSource.LABELLER.value
                ),
            }
            label_payload["rollout_label_id"] = str(
                payload.get("rollout_label_id") or rollout_label_id_for(label_payload)
            )
            try:
                label = MiproRolloutLabel.from_dict(label_payload)
            except ValueError as exc:
                warnings.append({"reason": str(exc), "label_id": label_id})
                continue
            normalized["rollout_labels"][label.rollout_label_id] = label.to_dict()
            accepted.append(label.to_dict())
        proposer_memory_state = normalized
        if accepted or warnings:
            trial.details["rollout_labels"] = list(trial.details.get("rollout_labels") or []) + accepted
            trial.details["rollout_label_warnings"] = (
                list(trial.details.get("rollout_label_warnings") or []) + warnings
            )
        return accepted

    async def run_train_wave(
        *,
        wave_round_idx: int,
        proposer_round_idx: int | None,
        rollout_queue: Mapping[str, Any] | None = None,
    ) -> None:
        nonlocal observation_seq
        queue_payload = dict(rollout_queue or {})
        queue_candidate_rows = [
            dict(item)
            for item in list(queue_payload.get("candidates") or [])
            if isinstance(item, Mapping)
        ]
        if queue_candidate_rows:
            configs = [
                {
                    str(key): str(value)
                    for key, value in dict(item.get("source_config") or {}).items()
                }
                for item in queue_candidate_rows
            ]
        else:
            configs = await optimizer.suggest(top_k=int(cfg.top_k))
        if not configs:
            return
        decoded_pairs: list[tuple[dict[str, str], MiproProgramCandidate]] = []
        for trial_config in configs:
            candidate = decode_config(working_space, trial_config)
            candidate_hash = str(candidate.lever_bundle_hash or "").strip()
            if candidate_hash and candidate_hash in tabu_lever_bundle_hashes:
                out.skipped_tabu_candidates += 1
                continue
            decoded_pairs.append((trial_config, candidate))
        if not decoded_pairs:
            return
        queue_rollouts = [
            MiproQueuedRollout.from_dict(dict(item))
            for item in list(queue_payload.get("rollouts") or [])
            if isinstance(item, Mapping)
        ]

        async def evaluate_pair(
            trial_cfg: dict[str, str],
            candidate: MiproProgramCandidate,
        ) -> MiproTrialResult:
            candidate_rollouts = [
                item
                for item in queue_rollouts
                if str(item.candidate_id) == str(candidate.candidate_id or "")
            ]
            if evaluate_queued_rollout is None or not candidate_rollouts:
                return await evaluate_candidate_trial(
                    evaluate=evaluate_train,
                    trial_config=trial_cfg,
                    candidate=candidate,
                    round_idx_value=wave_round_idx,
                    split="train",
                    extra_details={
                        "phase3_tabu_prefilter": True,
                        "proposer_round_idx": proposer_round_idx,
                        "rollout_queue_id": queue_payload.get("queue_id"),
                        "queued_rollout_ids": [
                            str(item.rollout_id) for item in candidate_rollouts
                        ],
                    },
                )
            rollout_results: list[dict[str, Any]] = []
            latencies: list[float] = []
            for queued_rollout in candidate_rollouts:
                start = time.perf_counter()
                async with semaphore:
                    raw = evaluate_queued_rollout(candidate, queued_rollout)
                    if inspect.isawaitable(raw):
                        raw = await cast(Awaitable[EvaluateCandidateOutcome], raw)
                score, details = _coerce_eval_outcome(raw)
                latencies.append((time.perf_counter() - start) * 1000.0)
                rollout_results.append(
                    {
                        "rollout_id": queued_rollout.rollout_id,
                        "candidate_id": queued_rollout.candidate_id,
                        "row_id": queued_rollout.row_id,
                        "seed": queued_rollout.seed,
                        "score": float(score),
                        "details": dict(details),
                    }
                )
            scores = [float(item["score"]) for item in rollout_results]
            return MiproTrialResult(
                config=trial_cfg,
                score=(sum(scores) / len(scores)) if scores else 0.0,
                details={
                    "candidate_id": candidate.candidate_id,
                    "lever_bundle_hash": candidate.lever_bundle_hash,
                    "round_idx": wave_round_idx,
                    "split": "train",
                    "proposer_round_idx": proposer_round_idx,
                    "rollout_queue_id": queue_payload.get("queue_id"),
                    "queued_rollout_results": rollout_results,
                },
                latency_ms=sum(latencies),
                candidate_id=candidate.candidate_id,
                lever_bundle_hash=candidate.lever_bundle_hash,
            )

        evaluated = await asyncio.gather(
            *[evaluate_pair(trial_cfg, candidate) for trial_cfg, candidate in decoded_pairs]
        )
        await optimizer.observe_batch(list(evaluated))
        out.train_observations.extend(list(evaluated))
        for trial, (_, candidate) in zip(evaluated, decoded_pairs, strict=False):
            await maybe_label_completed_rollouts(trial=trial, candidate=candidate)
            observation_seq = await persist_train_trial(
                seq=observation_seq,
                round_idx_value=wave_round_idx,
                trial=trial,
                candidate=candidate,
                proposer_round_idx=proposer_round_idx,
            )
            candidate_hash = str(
                candidate.lever_bundle_hash or trial.lever_bundle_hash or ""
            ).strip()
            if candidate_hash:
                tabu_lever_bundle_hashes.add(candidate_hash)
            if out.best_train_score is None or trial.score > out.best_train_score:
                out.best_train_score = float(trial.score)
                out.best_train_candidate = candidate
        if queue_payload.get("queue_id"):
            ledger.upsert_rollout_queue(
                queue_id=str(queue_payload["queue_id"]),
                round_idx=wave_round_idx,
                queue_kind=str(queue_payload.get("queue_kind") or "committed"),
                queue_payload={
                    **queue_payload,
                    "execution": {
                        "wave_round_idx": wave_round_idx,
                        "proposer_round_idx": proposer_round_idx,
                        "candidate_scores": {
                            str(trial.candidate_id or ""): float(trial.score)
                            for trial in evaluated
                        },
                    },
                },
            )
        await refresh_train_read_model()
        if active_label_definitions():
            ledger.upsert_proposer_memory(
                memory_state=proposer_memory_state,
                round_idx=proposer_round_idx,
            )
        persist_common_state()

    try:
        if resume:
            resume_state = load_resume_state(ledger)
            observation_seq = int(resume_state.next_observation_seq)
            heldout_seq = int(resume_state.next_heldout_seq)
            proposer_round_seq = int(resume_state.next_proposer_round_seq)
            train_rounds_completed = int(
                resume_state.run_state.get("train_rounds_completed") or 0
            )
            proposer_rounds_completed = int(
                resume_state.run_state.get("proposer_rounds_completed") or 0
            )
            out.skipped_tabu_candidates = int(
                resume_state.run_state.get("skipped_tabu_candidates") or 0
            )
            snapshot_payload = resume_state.run_state.get("compiled_space_snapshot")
            if isinstance(snapshot_payload, Mapping):
                search_space = snapshot_payload.get("search_space")
                instruction_lookup = snapshot_payload.get("instruction_lookup")
                instruction_base_lookup = snapshot_payload.get("instruction_base_lookup")
                instruction_transforms = snapshot_payload.get("instruction_transforms")
                instruction_metadata = snapshot_payload.get("instruction_metadata")
                instruction_base_metadata = snapshot_payload.get("instruction_base_metadata")
                demo_metadata = snapshot_payload.get("demo_metadata")
                if isinstance(search_space, Mapping):
                    working_space.search_space.clear()
                    working_space.search_space.update({
                        str(key): [str(item) for item in value]
                        for key, value in search_space.items()
                        if isinstance(value, list)
                    })
                if isinstance(instruction_lookup, Mapping):
                    working_space.instruction_lookup.clear()
                    working_space.instruction_lookup.update({
                        str(key): {
                            str(option_id): str(text)
                            for option_id, text in value.items()
                        }
                        for key, value in instruction_lookup.items()
                        if isinstance(value, Mapping)
                    })
                if isinstance(instruction_base_lookup, Mapping):
                    working_space.instruction_base_lookup.clear()
                    working_space.instruction_base_lookup.update({
                        str(key): {
                            str(option_id): str(text)
                            for option_id, text in value.items()
                        }
                        for key, value in instruction_base_lookup.items()
                        if isinstance(value, Mapping)
                    })
                if isinstance(instruction_transforms, Mapping):
                    from synth_optimizers.miprov2.core.instruction_transforms import InstructionTransform

                    working_space.instruction_transforms.clear()
                    working_space.instruction_transforms.update({
                        str(key): {
                            str(transform_id): InstructionTransform.from_dict(dict(payload))
                            for transform_id, payload in value.items()
                            if isinstance(payload, Mapping)
                        }
                        for key, value in instruction_transforms.items()
                        if isinstance(value, Mapping)
                    })
                if isinstance(instruction_metadata, Mapping):
                    working_space.instruction_metadata.clear()
                    working_space.instruction_metadata.update({
                        str(key): {
                            str(option_id): dict(payload)
                            for option_id, payload in value.items()
                            if isinstance(payload, Mapping)
                        }
                        for key, value in instruction_metadata.items()
                        if isinstance(value, Mapping)
                    })
                if isinstance(instruction_base_metadata, Mapping):
                    working_space.instruction_base_metadata.clear()
                    working_space.instruction_base_metadata.update({
                        str(key): {
                            str(option_id): dict(payload)
                            for option_id, payload in value.items()
                            if isinstance(payload, Mapping)
                        }
                        for key, value in instruction_base_metadata.items()
                        if isinstance(value, Mapping)
                    })
                if isinstance(demo_metadata, Mapping):
                    working_space.demo_metadata.clear()
                    working_space.demo_metadata.update({
                        str(key): {
                            str(option_id): dict(payload)
                            for option_id, payload in value.items()
                            if isinstance(payload, Mapping)
                        }
                        for key, value in demo_metadata.items()
                        if isinstance(value, Mapping)
                    })
                demo_lookup = snapshot_payload.get("demo_lookup")
                if isinstance(demo_lookup, Mapping):
                    working_space.demo_lookup.clear()
                    working_space.demo_lookup.update({
                        str(key): {
                            str(option_id): demo_from_dict(dict(demo_payload))
                            for option_id, demo_payload in value.items()
                            if isinstance(demo_payload, Mapping)
                        }
                        for key, value in demo_lookup.items()
                        if isinstance(value, Mapping)
                    })
                await sync_optimizer_search_space(
                    optimizer=optimizer, compiled_space=working_space
                )
            for entry in resume_state.observations:
                await optimizer.observe(entry.trial)
                out.train_observations.append(entry.trial)
                digest = str(entry.trial.lever_bundle_hash or "").strip()
                if digest:
                    tabu_lever_bundle_hashes.add(digest)
            out.heldout_snapshots = [
                MiproHeldoutSnapshot(
                    round_idx=entry.snapshot.round_idx,
                    best_candidate_id=entry.snapshot.best_candidate_id,
                    best_score=entry.snapshot.best_score,
                    baseline_score=entry.snapshot.baseline_score,
                    lift=entry.snapshot.lift,
                )
                for entry in resume_state.heldout_snapshots
            ]
            out.baseline_train_score = (
                float(resume_state.run_state["baseline_train_score"])
                if resume_state.run_state.get("baseline_train_score") is not None
                else None
            )
            out.heldout_baseline_score = (
                float(resume_state.run_state["heldout_baseline_score"])
                if resume_state.run_state.get("heldout_baseline_score") is not None
                else None
            )
            out.best_train_score = (
                float(resume_state.run_state["best_train_score"])
                if resume_state.run_state.get("best_train_score") is not None
                else None
            )
            out.heldout_best_score = (
                float(resume_state.run_state["heldout_best_score"])
                if resume_state.run_state.get("heldout_best_score") is not None
                else None
            )
            out.heldout_lift = (
                float(resume_state.run_state["heldout_lift"])
                if resume_state.run_state.get("heldout_lift") is not None
                else None
            )
            out.best_train_candidate = SQLiteMiproRunLedger.candidate_from_state(
                resume_state.run_state.get("best_train_candidate")
            )
            if out.best_train_candidate is None and out.train_observations:
                best_trial = max(out.train_observations, key=lambda item: item.score)
                out.best_train_candidate = decode_config(working_space, best_trial.config)
                out.best_train_score = float(best_trial.score)
            restored_tabu = resume_state.run_state.get("tabu_hashes")
            if isinstance(restored_tabu, list):
                for value in restored_tabu:
                    text = str(value).strip()
                    if text:
                        tabu_lever_bundle_hashes.add(text)
            restored_memory = resume_state.run_state.get("latest_proposer_memory_state")
            if isinstance(restored_memory, Mapping):
                proposer_memory_state = normalize_memory_state(dict(restored_memory))
            restored_sessions = resume_state.run_state.get("proposer_sessions")
            if isinstance(restored_sessions, list):
                out.proposer_sessions = [
                    dict(item) for item in restored_sessions if isinstance(item, Mapping)
                ]
            restored_round_diag = resume_state.run_state.get("proposer_round_diagnostics")
            if isinstance(restored_round_diag, list):
                out.proposer_round_diagnostics = [
                    dict(item) for item in restored_round_diag if isinstance(item, Mapping)
                ]
            restored_stop_freq = resume_state.run_state.get("stop_reason_frequency")
            if isinstance(restored_stop_freq, Mapping):
                out.stop_reason_frequency = {
                    str(k): int(v) for k, v in restored_stop_freq.items()
                }
            restored_agg = resume_state.run_state.get("proposer_diagnostics_aggregate")
            if isinstance(restored_agg, Mapping):
                out.proposer_diagnostics_aggregate = dict(restored_agg)
            restored_trace_paths = resume_state.run_state.get("proposer_trace_paths")
            if isinstance(restored_trace_paths, list):
                out.proposer_trace_paths = [
                    str(item) for item in restored_trace_paths if str(item).strip()
                ]
            restored_pending = resume_state.run_state.get("pending_interactive_proposer")
            if isinstance(restored_pending, Mapping):
                pending_interactive_proposer = dict(restored_pending)
                out.pending_interactive_session = dict(pending_interactive_proposer)
            restored_consumed = resume_state.run_state.get("consumed_interactive_proposer")
            if isinstance(restored_consumed, Mapping):
                out.consumed_interactive_session = dict(restored_consumed)
            await refresh_train_read_model()

        baseline_config = build_baseline_config(working_space)
        baseline_candidate = decode_config(working_space, baseline_config)
        baseline_hash = str(baseline_candidate.lever_bundle_hash or "").strip()
        baseline_candidate_id = str(baseline_candidate.candidate_id or "")

        if cfg.seed_with_baseline and not out.train_observations:
            baseline_trial = await evaluate_candidate_trial(
                evaluate=evaluate_train,
                trial_config=baseline_config,
                candidate=baseline_candidate,
                round_idx_value=0,
                split="train",
                extra_details={"phase3_baseline": True},
            )
            await optimizer.observe(baseline_trial)
            out.train_observations.append(baseline_trial)
            out.baseline_train_score = float(baseline_trial.score)
            out.best_train_candidate = baseline_candidate
            out.best_train_score = float(baseline_trial.score)
            if baseline_hash:
                tabu_lever_bundle_hashes.add(baseline_hash)
            observation_seq = await persist_train_trial(
                seq=observation_seq,
                round_idx_value=0,
                trial=baseline_trial,
                candidate=baseline_candidate,
                phase3_baseline=True,
            )
            await refresh_train_read_model()
            persist_common_state()

        if evaluate_heldout is not None and out.heldout_baseline_score is None:
            heldout_baseline_trial = await evaluate_candidate_trial(
                evaluate=evaluate_heldout,
                trial_config=baseline_config,
                candidate=baseline_candidate,
                round_idx_value=0,
                split="heldout",
                extra_details={"phase3_baseline": True},
            )
            out.heldout_baseline_score = float(heldout_baseline_trial.score)
            await persist_heldout_trial(
                trial=heldout_baseline_trial,
                candidate=baseline_candidate,
                round_idx_value=0,
                phase3_baseline=True,
            )
            await refresh_heldout_read_model()
            persist_common_state()

        for _ in range(int(cfg.bootstrap_train_rounds)):
            train_rounds_completed += 1
            await run_train_wave(
                wave_round_idx=train_rounds_completed,
                proposer_round_idx=None,
            )

        for proposer_round_idx in range(
            proposer_rounds_completed + 1, int(cfg.proposer_rounds) + 1
        ):
            skipped_before_round = int(out.skipped_tabu_candidates)
            recent_successes = _recent_candidate_ids(
                out.train_observations,
                limit=int(cfg.max_recent_successes),
                reverse=True,
            )
            recent_failures = _recent_candidate_ids(
                out.train_observations,
                limit=int(cfg.max_recent_failures),
                reverse=False,
            )
            grounding_limit = max(
                1, int(cfg.max_recent_failures) + int(cfg.max_recent_successes)
            )
            sampled_rows = await _resolve_train_rows(
                hooks,
                round_idx=proposer_round_idx,
                limit=grounding_limit,
            )
            tentative_queue = await _build_rollout_queue(
                optimizer=optimizer,
                compiled_space=working_space,
                run_id=str(out.run_id or run_id or ledger.run_id),
                round_idx=proposer_round_idx,
                sampled_rows=sampled_rows,
                top_k=int(cfg.top_k),
                created_by="tpe",
                queue_kind="tentative",
            )
            tentative_queue_payload = tentative_queue.to_dict()
            ledger.upsert_rollout_queue(
                queue_id=tentative_queue.queue_id,
                round_idx=proposer_round_idx,
                queue_kind=tentative_queue.queue_kind,
                queue_payload=tentative_queue_payload,
            )
            rollout_queue_state = {
                "tentative_queue_id": tentative_queue.queue_id,
                "active_queue_id": tentative_queue.queue_id,
                "queues": {tentative_queue.queue_id: tentative_queue_payload},
                "overrides": [],
            }
            recent_trial_summary = await _resolve_recent_trial_summary(
                hooks,
                observations=out.train_observations,
                limit=grounding_limit,
            )
            train_read_model = await refresh_train_read_model()
            recent_trial_rows = _recent_trial_rows_from_read_model(
                read_model_payload=train_read_model,
                limit=grounding_limit,
            )
            summary_scores = [item.score for item in out.train_observations]
            score_mean = (
                sum(summary_scores) / len(summary_scores) if summary_scores else None
            )
            candidate_rows = cast(list[dict[str, Any]], train_read_model.get("candidates") or [])
            best_candidate_id = (
                str(out.best_train_candidate.candidate_id)
                if out.best_train_candidate is not None
                else None
            )
            best_delta_paths = [
                str(item.get("artifact_path"))
                for item in cast(list[dict[str, Any]], train_read_model.get("candidate_deltas") or [])
                if str(item.get("candidate_id") or "") == str(best_candidate_id or "")
            ]
            best_verdict_paths = [
                str(item.get("artifact_path"))
                for item in cast(list[dict[str, Any]], train_read_model.get("verdict_digests") or [])
                if str(item.get("candidate_id") or "") == str(best_candidate_id or "")
            ]
            grounding_payload: dict[str, Any] = {
                "summary_stats": {
                    "observation_count": len(out.train_observations),
                    "score_mean": score_mean,
                    "score_min": min(summary_scores) if summary_scores else None,
                    "score_max": max(summary_scores) if summary_scores else None,
                    "tabu_hash_count": len(tabu_lever_bundle_hashes),
                    "skipped_tabu_candidates": out.skipped_tabu_candidates,
                    "best_train_score": out.best_train_score,
                    "heldout_baseline_score": out.heldout_baseline_score,
                    "hook_summary": recent_trial_summary,
                },
                "sampled_train_rows": sampled_rows,
                "recent_trial_rows": recent_trial_rows,
                "proposer_memory_summary": proposer_memory_summary(proposer_memory_state),
            }
            proposer_context = MiproOpenEnvProposerContext(
                objective=(
                    "Improve train metric while preserving output constraints "
                    "and heldout generalization."
                ),
                round_idx=proposer_round_idx,
                recent_failures=recent_failures,
                recent_successes=recent_successes,
                grounding_payload=grounding_payload,
                run_metadata={
                    "run_id": out.run_id,
                    "mode": "phase3",
                    "train_rounds_completed": train_rounds_completed,
                    "proposer_rounds_completed": proposer_rounds_completed,
                    "best_train_score": out.best_train_score,
                    "heldout_baseline_score": out.heldout_baseline_score,
                    "observation_count": len(out.train_observations),
                    "tabu_hash_count": len(tabu_lever_bundle_hashes),
                    "skipped_tabu_candidates": out.skipped_tabu_candidates,
                    "tentative_rollout_queue_id": tentative_queue.queue_id,
                    "proposer_memory_summary": proposer_memory_summary(proposer_memory_state),
                },
                candidate_summary_counts={
                    "candidate_count": len(candidate_rows),
                    "train_rollout_count": len(
                        [
                            item
                            for item in cast(list[dict[str, Any]], train_read_model.get("rollouts") or [])
                            if str(item.get("split") or "") == "train"
                        ]
                    ),
                    "delta_digest_count": len(
                        cast(list[dict[str, Any]], train_read_model.get("candidate_deltas") or [])
                    ),
                    "verdict_digest_count": len(
                        cast(list[dict[str, Any]], train_read_model.get("verdict_digests") or [])
                    ),
                },
                current_best_candidate_id=best_candidate_id,
                baseline_candidate_id=str(
                    train_read_model.get("baseline_candidate_id") or baseline_candidate.candidate_id
                ),
                delta_digest_paths={
                    "best_candidate_delta_paths": best_delta_paths,
                    "best_candidate_verdict_paths": best_verdict_paths,
                },
                workspace_locations={
                    "workspace_root": str(ledger.workspace_root),
                    "ledger_path": str(ledger.ledger_path),
                },
                read_model_payload={
                    **dict(train_read_model),
                    "sampled_train_rows": sampled_rows,
                    "recent_trial_rows": recent_trial_rows,
                    "rollout_queue_state": rollout_queue_state,
                    "proposer_memory_state": proposer_memory_state,
                    "proposer_memory_summary": proposer_memory_summary(proposer_memory_state),
                },
            )
            run_identifier = out.run_id or run_id or ledger.run_id
            checkpoint_payload: dict[str, Any] | None = None
            checkpoint_root = (
                cfg.checkpoint_dir
                if cfg.checkpoint_dir is not None
                else str(Path(ledger.workspace_root) / "checkpoints")
            )
            if (
                cfg.checkpoint_policy == "before_each_proposer"
                or cfg.proposer_control == "interactive_pause"
            ):
                checkpoint_payload = write_proposer_checkpoint(
                    checkpoint_dir=checkpoint_root,
                    run_id=str(run_identifier),
                    round_idx=proposer_round_idx,
                    compiled_space=working_space,
                    observations=list(out.train_observations),
                    best_train_candidate=out.best_train_candidate,
                    best_train_score=out.best_train_score,
                    baseline_candidate=baseline_candidate,
                    baseline_train_score=out.baseline_train_score,
                    heldout_baseline_score=out.heldout_baseline_score,
                    proposer_context=proposer_context,
                    train_read_model=train_read_model,
                    sampled_train_rows=sampled_rows,
                    recent_trial_rows=recent_trial_rows,
                    tabu_hashes=tabu_lever_bundle_hashes,
                    config={
                        "phase3": {
                            "proposer_rounds": int(cfg.proposer_rounds),
                            "train_rounds_per_proposer_round": int(
                                cfg.train_rounds_per_proposer_round
                            ),
                            "bootstrap_train_rounds": int(cfg.bootstrap_train_rounds),
                            "top_k": int(cfg.top_k),
                            "max_concurrency": int(cfg.max_concurrency),
                            "checkpoint_policy": str(cfg.checkpoint_policy),
                        },
                        "proposer_config": (
                            asdict(cfg.proposer_config)
                            if cfg.proposer_config is not None
                            else None
                        ),
                    },
                    ledger_path=str(ledger.ledger_path),
                )
                artifact_refs = dict(checkpoint_payload.get("artifact_refs") or {})
                ledger.upsert_checkpoint(
                    checkpoint_id=str(checkpoint_payload["checkpoint_id"]),
                    stage=CHECKPOINT_STAGE_BEFORE_PROPOSER,
                    round_idx=proposer_round_idx,
                    path=str(artifact_refs.get("checkpoint") or ""),
                    metadata={
                        "best_train_score": out.best_train_score,
                        "heldout_baseline_score": out.heldout_baseline_score,
                        "candidate_count": len(candidate_rows),
                    },
                )

            if cfg.proposer_control == "interactive_pause":
                if checkpoint_payload is None:
                    raise RuntimeError("interactive proposer requires a checkpoint payload")
                artifact_refs = dict(checkpoint_payload.get("artifact_refs") or {})
                checkpoint_path = str(artifact_refs.get("checkpoint") or "").strip()
                checkpoint_id = str(checkpoint_payload.get("checkpoint_id") or "").strip()
                session_root = _interactive_session_root(
                    cfg=cfg,
                    checkpoint_root=checkpoint_root,
                )
                if interactive_resume_session_id:
                    if not isinstance(pending_interactive_proposer, Mapping):
                        raise ValueError(
                            "interactive_resume_session_id was provided but no pending_interactive_proposer "
                            "state exists for this run"
                        )
                    proposer_outcome, consumed_session = _interactive_outcome_from_committed_session(
                        session_root=session_root,
                        session_id=interactive_resume_session_id,
                        pending=pending_interactive_proposer,
                    )
                    out.consumed_interactive_session = dict(consumed_session)
                    out.pending_interactive_session = None
                    pending_interactive_proposer = None
                    interactive_resume_session_id = ""
                    ledger.upsert_state(key="pending_interactive_proposer", value=None)
                    ledger.upsert_state(
                        key="consumed_interactive_proposer",
                        value=dict(consumed_session),
                    )
                else:
                    environment = MiproProposerEnvironment.from_checkpoint(
                        checkpoint_payload,
                        session_root=session_root,
                        source_ref=checkpoint_id,
                        config=cfg.proposer_config,
                        queue_state=rollout_queue_state,
                        memory_state=proposer_memory_state,
                    )
                    pending_interactive_proposer = {
                        "run_id": str(run_identifier),
                        "round_idx": proposer_round_idx,
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_path": checkpoint_path,
                        "session_id": environment.session.session_id,
                        "session_root": str(Path(session_root).expanduser().resolve()),
                        "session_dir": environment.session.session_dir,
                        "event_log_path": environment.session.event_log_path,
                        "tentative_rollout_queue_id": tentative_queue.queue_id,
                        "train_rounds_completed": train_rounds_completed,
                        "proposer_rounds_completed": proposer_rounds_completed,
                        "ledger_path": str(ledger.ledger_path),
                    }
                    out.pending_interactive_session = dict(pending_interactive_proposer)
                    out.run_status = "paused_for_interactive_proposer"
                    ledger.upsert_state(
                        key="pending_interactive_proposer",
                        value=dict(pending_interactive_proposer),
                    )
                    persist_common_state()
                    ledger.set_status(status="paused_for_interactive_proposer")
                    return out
            else:
                proposer_outcome = await run_openenv_react_proposer(
                    compiled_space=working_space,
                    agent=agent,
                    context=proposer_context,
                    config=cfg.proposer_config,
                    queue_state=rollout_queue_state,
                    memory_state=proposer_memory_state,
                )
            stop_reason = str(proposer_outcome.stop_reason)
            out.stop_reason_frequency[stop_reason] = (
                int(out.stop_reason_frequency.get(stop_reason, 0)) + 1
            )
            working_space = proposer_outcome.compiled_space
            await sync_optimizer_search_space(
                optimizer=optimizer,
                compiled_space=working_space,
            )
            persist_compiled_space_snapshot()
            proposer_summary = proposer_outcome_summary(proposer_outcome)
            proposer_memory_state = normalize_memory_state(proposer_outcome.memory_state)
            proposer_memory_artifact = ledger.upsert_proposer_memory(
                memory_state=proposer_memory_state,
                round_idx=proposer_round_idx,
            )
            ledger.upsert_state(
                key="latest_proposer_memory_state",
                value=dict(proposer_memory_state),
            )
            proposer_trace_payload = {
                "run_id": run_identifier,
                "round_idx": proposer_round_idx,
                "objective": str(proposer_context.objective),
                "recent_failures": list(recent_failures),
                "recent_successes": list(recent_successes),
                "grounding_payload": dict(grounding_payload),
                "run_metadata": dict(proposer_context.run_metadata),
                "candidate_summary_counts": dict(proposer_context.candidate_summary_counts),
                "workspace_locations": dict(proposer_context.workspace_locations),
                "delta_digest_paths": dict(proposer_context.delta_digest_paths),
                "proposer_summary": proposer_summary,
                "proposer_memory_artifact": dict(proposer_memory_artifact),
                "transcript": list(proposer_outcome.transcript),
            }
            trace_path: str | None = None
            if bool(cfg.write_proposer_trace_json):
                trace_path = _write_proposer_trace_json(
                    run_id=run_identifier,
                    trace_dir=cfg.proposer_trace_dir,
                    round_idx=proposer_round_idx,
                    payload=proposer_trace_payload,
                )
                if trace_path is not None:
                    out.proposer_trace_paths.append(trace_path)
            proposer_session = {
                "round_idx": proposer_round_idx,
                "recent_failures": list(recent_failures),
                "recent_successes": list(recent_successes),
                "grounding_row_count": len(sampled_rows),
                "proposer_summary": proposer_summary,
                "proposer_memory_artifact": dict(proposer_memory_artifact),
                "proposer_trace_json": trace_path,
            }
            if out.consumed_interactive_session is not None:
                proposer_session["interactive_session"] = dict(
                    out.consumed_interactive_session
                )
            out.proposer_sessions.append(proposer_session)

            queue_state = dict(proposer_outcome.queue_state or {})
            queue_map = {
                str(key): dict(value)
                for key, value in dict(queue_state.get("queues") or {}).items()
                if isinstance(value, Mapping)
            }
            committed_queue_id = str(queue_state.get("committed_queue_id") or "").strip()
            committed_queue_payload = (
                queue_map.get(committed_queue_id) if committed_queue_id else None
            )
            if committed_queue_payload is None:
                default_queue = await _build_rollout_queue(
                    optimizer=optimizer,
                    compiled_space=working_space,
                    run_id=str(run_identifier),
                    round_idx=proposer_round_idx,
                    sampled_rows=sampled_rows,
                    top_k=int(cfg.top_k),
                    created_by="tpe",
                    queue_kind="committed",
                    suffix="post_proposer_default",
                )
                committed_queue_payload = default_queue.to_dict()
                committed_queue_id = default_queue.queue_id
                queue_state = {
                    **queue_state,
                    "committed_queue_id": committed_queue_id,
                    "commit": {
                        "commit_id": f"commit_{committed_queue_id}",
                        "queue_id": tentative_queue.queue_id,
                        "committed_queue_id": committed_queue_id,
                        "accept_tpe_defaults": True,
                        "proposer_override_refs": [],
                        "reason": "auto-committed TPE default queue",
                    },
                    "queues": {
                        **queue_map,
                        committed_queue_id: committed_queue_payload,
                    },
                }
                proposer_outcome.queue_state = dict(queue_state)
                proposer_summary["committed_rollout_queue_id"] = committed_queue_id
            ledger.upsert_rollout_queue(
                queue_id=committed_queue_id,
                round_idx=proposer_round_idx,
                queue_kind=str(committed_queue_payload.get("queue_kind") or "committed"),
                queue_payload={
                    **committed_queue_payload,
                    "commit": (
                        dict(cast(Mapping[str, Any], queue_state.get("commit")))
                        if isinstance(queue_state.get("commit"), Mapping)
                        else {}
                    ),
                    "original_queue": tentative_queue_payload,
                },
            )
            ledger.upsert_state(
                key="latest_rollout_queue_state",
                value=dict(queue_state),
            )

            for wave_index in range(int(cfg.train_rounds_per_proposer_round)):
                train_rounds_completed += 1
                await run_train_wave(
                    wave_round_idx=train_rounds_completed,
                    proposer_round_idx=proposer_round_idx,
                    rollout_queue=committed_queue_payload if wave_index == 0 else None,
                )
                should_snapshot = (
                    evaluate_heldout is not None
                    and cfg.heldout_interval is not None
                    and out.best_train_candidate is not None
                    and str(out.best_train_candidate.candidate_id or "")
                    != baseline_candidate_id
                    and train_rounds_completed % int(cfg.heldout_interval) == 0
                )
                if should_snapshot:
                    assert evaluate_heldout is not None
                    assert out.best_train_candidate is not None
                    best_train_candidate = out.best_train_candidate
                    heldout_trial = await evaluate_candidate_trial(
                        evaluate=evaluate_heldout,
                        trial_config=best_train_candidate.source_config,
                        candidate=best_train_candidate,
                        round_idx_value=train_rounds_completed,
                        split="heldout",
                    )
                    baseline_score = out.heldout_baseline_score
                    lift = (
                        float(heldout_trial.score) - baseline_score
                        if baseline_score is not None
                        else None
                    )
                    snapshot = MiproHeldoutSnapshot(
                        round_idx=train_rounds_completed,
                        best_candidate_id=str(best_train_candidate.candidate_id),
                        best_score=float(heldout_trial.score),
                        baseline_score=baseline_score,
                        lift=lift,
                    )
                    out.heldout_snapshots.append(snapshot)
                    ledger.append_heldout_snapshot(seq=heldout_seq, snapshot=snapshot)
                    heldout_seq += 1
                    await persist_heldout_trial(
                        trial=heldout_trial,
                        candidate=best_train_candidate,
                        round_idx_value=train_rounds_completed,
                    )
                    await refresh_heldout_read_model()
                    persist_common_state()

            patches_total = int(proposer_summary.get("instruction_patches") or 0) + int(
                proposer_summary.get("demo_patches") or 0
            )
            patch_actions = int(proposer_summary.get("patch_action_count") or 0)
            patch_yield = (patches_total / patch_actions) if patch_actions > 0 else 0.0
            round_diagnostics = {
                "round_idx": proposer_round_idx,
                "stop_reason": stop_reason,
                "action_mix": dict(proposer_summary.get("action_counts") or {}),
                "model_turn_count": int(proposer_summary.get("model_turn_count") or 0),
                "tool_call_count": int(proposer_summary.get("tool_call_count") or 0),
                "read_action_count": int(proposer_summary.get("read_action_count") or 0),
                "patch_action_count": patch_actions,
                "patches_added": patches_total,
                "archive_spill_count": int(
                    proposer_summary.get("archive_spill_count") or 0
                ),
                "archived_message_count": int(
                    proposer_summary.get("archived_message_count") or 0
                ),
                "archive_path": proposer_summary.get("archive_path"),
                "duplicate_patch_count": int(proposer_summary.get("duplicate_patch_count") or 0),
                "ignored_patch_count": int(proposer_summary.get("ignored_patch_count") or 0),
                "policy_violation_count": int(proposer_summary.get("policy_violation_count") or 0),
                "grounding_read_action_count": int(
                    proposer_summary.get("grounding_read_action_count") or 0
                ),
                "evidence_read_action_count": int(
                    proposer_summary.get("evidence_read_action_count") or 0
                ),
                "read_tools_used": list(proposer_summary.get("read_tools_used") or []),
                "patch_yield": patch_yield,
                "skipped_tabu_delta": int(out.skipped_tabu_candidates) - skipped_before_round,
            }
            out.proposer_round_diagnostics.append(round_diagnostics)
            proposer_rounds_completed = proposer_round_idx
            out.proposer_diagnostics_aggregate = _aggregate_proposer_diagnostics(
                out.proposer_round_diagnostics,
                out.stop_reason_frequency,
            )
            ledger.append_proposer_round(
                seq=proposer_round_seq,
                round_idx=proposer_round_idx,
                summary=proposer_session,
                diagnostics=round_diagnostics,
                stop_reason=stop_reason,
                skipped_tabu_delta=int(out.skipped_tabu_candidates) - skipped_before_round,
            )
            proposer_round_seq += 1
            persist_common_state()

        should_run_final_heldout = (
            evaluate_heldout is not None
            and cfg.compute_final_heldout
            and out.best_train_candidate is not None
        )
        if should_run_final_heldout:
            assert evaluate_heldout is not None
            assert out.best_train_candidate is not None
            best_train_candidate = out.best_train_candidate
            best_train_candidate_id = str(best_train_candidate.candidate_id or "")
            if best_train_candidate_id == baseline_candidate_id:
                out.heldout_best_score = out.heldout_baseline_score
                out.heldout_lift = 0.0 if out.heldout_baseline_score is not None else None
                persist_common_state()
                ledger.upsert_state(key="heldout_best_score", value=out.heldout_best_score)
                ledger.upsert_state(key="heldout_lift", value=out.heldout_lift)
                out.run_status = "completed"
                ledger.set_status(status="completed")
                return out
            latest_snapshot_round = (
                out.heldout_snapshots[-1].round_idx if out.heldout_snapshots else None
            )
            if latest_snapshot_round != train_rounds_completed:
                heldout_trial = await evaluate_candidate_trial(
                    evaluate=evaluate_heldout,
                    trial_config=best_train_candidate.source_config,
                    candidate=best_train_candidate,
                    round_idx_value=train_rounds_completed,
                    split="heldout",
                )
                baseline_score = out.heldout_baseline_score
                lift = (
                    float(heldout_trial.score) - baseline_score
                    if baseline_score is not None
                    else None
                )
                snapshot = MiproHeldoutSnapshot(
                    round_idx=train_rounds_completed,
                    best_candidate_id=str(best_train_candidate.candidate_id),
                    best_score=float(heldout_trial.score),
                    baseline_score=baseline_score,
                    lift=lift,
                )
                out.heldout_snapshots.append(snapshot)
                ledger.append_heldout_snapshot(seq=heldout_seq, snapshot=snapshot)
                heldout_seq += 1
                await persist_heldout_trial(
                    trial=heldout_trial,
                    candidate=best_train_candidate,
                    round_idx_value=train_rounds_completed,
                )
                await refresh_heldout_read_model()

        out.tabu_hash_count = len(tabu_lever_bundle_hashes)
        if out.heldout_snapshots:
            final_snapshot = out.heldout_snapshots[-1]
            out.heldout_best_score = final_snapshot.best_score
            out.heldout_lift = final_snapshot.lift

        persist_common_state()
        ledger.upsert_state(key="heldout_best_score", value=out.heldout_best_score)
        ledger.upsert_state(key="heldout_lift", value=out.heldout_lift)
        out.run_status = "completed"
        ledger.set_status(status="completed")
        return out
    except Exception:
        ledger.set_status(status="failed")
        raise
    finally:
        ledger.close()


__all__ = [
    "SampleTrainRowsOutcome",
    "SampleTrainRowsFn",
    "SummarizeRecentTrialsOutcome",
    "SummarizeRecentTrialsFn",
    "EvaluateQueuedRolloutFn",
    "LabelCompletedRolloutsOutcome",
    "LabelCompletedRolloutsFn",
    "MiproPhase3Config",
    "MiproGroundingHooksLike",
    "MiproGroundingHooks",
    "MiproPhase3Outcome",
    "run_phase3_loop",
]
