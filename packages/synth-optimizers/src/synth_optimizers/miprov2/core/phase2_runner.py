"""Phase-2 async rollout driver for candidate-native MIPRO optimization."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

from synth_optimizers.miprov2.core.optimizer import (
    DiscreteMiproOptimizer,
    MiproTrialResult,
)
from synth_optimizers.miprov2.core.program_compiler import (
    CompiledMiproSpace,
    decode_config,
)
from synth_optimizers.miprov2.core.program_model import MiproProgramCandidate
from synth_optimizers.miprov2.core.run_ledger import (
    SQLiteMiproRunLedger,
    load_resume_state,
    open_sqlite_run_ledger,
)

EvaluateCandidateWithDetails: TypeAlias = tuple[float, Mapping[str, Any]]
EvaluateCandidateOutcome: TypeAlias = float | EvaluateCandidateWithDetails
EvaluateCandidateFn: TypeAlias = Callable[
    [MiproProgramCandidate],
    EvaluateCandidateOutcome | Awaitable[EvaluateCandidateOutcome],
]


@dataclass(slots=True, frozen=True)
class MiproPhase2Config:
    rounds: int = 8
    top_k: int = 1
    max_concurrency: int = 1
    seed_with_baseline: bool = True
    heldout_interval: int | None = None
    compute_final_heldout: bool = True

    def __post_init__(self) -> None:
        if int(self.rounds) < 0:
            raise ValueError("MiproPhase2Config.rounds must be >= 0")
        if int(self.top_k) <= 0:
            raise ValueError("MiproPhase2Config.top_k must be > 0")
        if int(self.max_concurrency) <= 0:
            raise ValueError("MiproPhase2Config.max_concurrency must be > 0")
        if self.heldout_interval is not None and int(self.heldout_interval) <= 0:
            raise ValueError("MiproPhase2Config.heldout_interval must be > 0 when provided")


@dataclass(slots=True, frozen=True)
class MiproHeldoutSnapshot:
    round_idx: int
    best_candidate_id: str
    best_score: float
    baseline_score: float | None
    lift: float | None


@dataclass(slots=True)
class MiproPhase2Outcome:
    train_observations: list[MiproTrialResult] = field(default_factory=list)
    best_train_candidate: MiproProgramCandidate | None = None
    best_train_score: float | None = None
    baseline_train_score: float | None = None
    heldout_baseline_score: float | None = None
    heldout_best_score: float | None = None
    heldout_lift: float | None = None
    heldout_snapshots: list[MiproHeldoutSnapshot] = field(default_factory=list)
    run_id: str | None = None
    ledger_path: str | None = None


def build_baseline_config(compiled_space: CompiledMiproSpace) -> dict[str, str]:
    """Deterministic baseline config from first option in each component."""

    baseline: dict[str, str] = {}
    for component in compiled_space.component_order:
        options = compiled_space.search_space.get(component, [])
        if not options:
            raise ValueError(f"component '{component}' has no options")
        baseline[component] = str(options[0])
    return baseline


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


async def run_train_loop(
    *,
    compiled_space: CompiledMiproSpace,
    optimizer: DiscreteMiproOptimizer,
    evaluate_train: EvaluateCandidateFn,
    evaluate_heldout: EvaluateCandidateFn | None = None,
    config: MiproPhase2Config | None = None,
    run_id: str | None = None,
    ledger_path: str | None = None,
    resume: bool = False,
) -> MiproPhase2Outcome:
    """Run phase-2 suggest/decode/evaluate/observe loop."""

    cfg = config or MiproPhase2Config()
    semaphore = asyncio.Semaphore(max(1, int(cfg.max_concurrency)))
    out = MiproPhase2Outcome()
    ledger: SQLiteMiproRunLedger | None = None
    observation_seq = 1
    heldout_seq = 1
    rounds_completed = 0

    if run_id is not None or ledger_path is not None or resume:
        ledger = open_sqlite_run_ledger(
            program_id=compiled_space.program_template.program_id,
            mode="phase2",
            run_id=run_id,
            ledger_path=ledger_path,
            resume=resume,
        )
        out.run_id = ledger.run_id
        out.ledger_path = ledger.ledger_path
        ledger.append_event(
            event_type="run_started",
            payload={
                "mode": "phase2",
                "resume": bool(resume),
                "program_id": compiled_space.program_template.program_id,
            },
        )

    try:
        if ledger is not None and resume:
            resume_state = load_resume_state(ledger)
            observation_seq = int(resume_state.next_observation_seq)
            heldout_seq = int(resume_state.next_heldout_seq)
            rounds_completed = int(resume_state.run_state.get("rounds_completed") or 0)
            for entry in resume_state.observations:
                await optimizer.observe(entry.trial)
                out.train_observations.append(entry.trial)
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
                out.best_train_candidate = decode_config(compiled_space, best_trial.config)
                out.best_train_score = float(best_trial.score)
            ledger.append_event(
                event_type="run_resumed",
                payload={
                    "observations_loaded": len(resume_state.observations),
                    "heldout_snapshots_loaded": len(resume_state.heldout_snapshots),
                    "rounds_completed": rounds_completed,
                },
            )

        baseline_config = build_baseline_config(compiled_space)
        baseline_candidate = decode_config(compiled_space, baseline_config)

        if cfg.seed_with_baseline and not out.train_observations:
            baseline_score, baseline_details, baseline_latency = await _evaluate_candidate_once(
                evaluate=evaluate_train,
                candidate=baseline_candidate,
                semaphore=semaphore,
            )
            baseline_trial = MiproTrialResult(
                config=baseline_config,
                score=baseline_score,
                details={
                    **baseline_details,
                    "candidate_id": baseline_candidate.candidate_id,
                    "lever_bundle_hash": baseline_candidate.lever_bundle_hash,
                    "phase2_baseline": True,
                    "round_idx": 0,
                },
                latency_ms=baseline_latency,
                candidate_id=baseline_candidate.candidate_id,
                lever_bundle_hash=baseline_candidate.lever_bundle_hash,
            )
            await optimizer.observe(baseline_trial)
            out.train_observations.append(baseline_trial)
            out.baseline_train_score = baseline_trial.score
            out.best_train_candidate = baseline_candidate
            out.best_train_score = baseline_trial.score
            if ledger is not None:
                ledger.upsert_candidate(candidate=baseline_candidate, round_idx=0)
                ledger.append_observation(seq=observation_seq, round_idx=0, trial=baseline_trial)
                ledger.append_event(
                    event_type="baseline_observed",
                    payload={
                        "score": baseline_trial.score,
                        "candidate_id": baseline_candidate.candidate_id,
                    },
                    round_idx=0,
                    candidate_id=baseline_candidate.candidate_id,
                )
                observation_seq += 1
                ledger.upsert_state(key="baseline_train_score", value=out.baseline_train_score)
                ledger.upsert_state(
                    key="best_train_candidate", value=out.best_train_candidate.to_dict()
                )
                ledger.upsert_state(key="best_train_score", value=out.best_train_score)
                ledger.upsert_state(key="rounds_completed", value=rounds_completed)

        if evaluate_heldout is not None and out.heldout_baseline_score is None:
            heldout_baseline, _baseline_details, _baseline_latency = await _evaluate_candidate_once(
                evaluate=evaluate_heldout,
                candidate=baseline_candidate,
                semaphore=semaphore,
            )
            out.heldout_baseline_score = heldout_baseline
            if ledger is not None:
                ledger.upsert_state(
                    key="heldout_baseline_score", value=out.heldout_baseline_score
                )

        for round_idx in range(rounds_completed, int(cfg.rounds)):
            configs = await optimizer.suggest(top_k=int(cfg.top_k))
            if not configs:
                break

            decoded_pairs: list[tuple[dict[str, str], MiproProgramCandidate]] = [
                (trial_config, decode_config(compiled_space, trial_config)) for trial_config in configs
            ]
            round_idx_value = round_idx + 1

            async def evaluate_pair(
                trial_config: dict[str, str],
                candidate: MiproProgramCandidate,
                round_idx: int,
            ) -> tuple[MiproTrialResult, MiproProgramCandidate]:
                score, details, latency = await _evaluate_candidate_once(
                    evaluate=evaluate_train,
                    candidate=candidate,
                    semaphore=semaphore,
                )
                trial = MiproTrialResult(
                    config=trial_config,
                    score=score,
                    details={
                        **details,
                        "candidate_id": candidate.candidate_id,
                        "lever_bundle_hash": candidate.lever_bundle_hash,
                        "round_idx": round_idx,
                    },
                    latency_ms=latency,
                    candidate_id=candidate.candidate_id,
                    lever_bundle_hash=candidate.lever_bundle_hash,
                )
                return trial, candidate

            evaluated = await asyncio.gather(
                *[
                    evaluate_pair(cfg_map, cand, round_idx_value)
                    for cfg_map, cand in decoded_pairs
                ]
            )
            batch_trials = [item[0] for item in evaluated]
            await optimizer.observe_batch(batch_trials)
            out.train_observations.extend(batch_trials)
            if ledger is not None:
                for trial in batch_trials:
                    ledger.append_observation(
                        seq=observation_seq,
                        round_idx=round_idx + 1,
                        trial=trial,
                    )
                    ledger.append_event(
                        event_type="trial_observed",
                        payload={"score": trial.score, "details": dict(trial.details)},
                        round_idx=round_idx + 1,
                        candidate_id=trial.candidate_id,
                    )
                    observation_seq += 1

            for trial, candidate in evaluated:
                if ledger is not None:
                    ledger.upsert_candidate(
                        candidate=candidate,
                        round_idx=round_idx_value,
                        candidate_metadata={
                            "train_score": float(trial.score),
                            "details": dict(trial.details),
                        },
                    )
                if out.best_train_score is None or trial.score > out.best_train_score:
                    out.best_train_score = trial.score
                    out.best_train_candidate = candidate

            rounds_completed = round_idx + 1
            if ledger is not None:
                ledger.upsert_state(key="rounds_completed", value=rounds_completed)
                if out.best_train_candidate is not None:
                    ledger.upsert_state(
                        key="best_train_candidate", value=out.best_train_candidate.to_dict()
                    )
                ledger.upsert_state(key="best_train_score", value=out.best_train_score)

            should_snapshot = (
                evaluate_heldout is not None
                and cfg.heldout_interval is not None
                and out.best_train_candidate is not None
                and rounds_completed % int(cfg.heldout_interval) == 0
            )
            if should_snapshot:
                assert evaluate_heldout is not None
                assert out.best_train_candidate is not None
                best_train_candidate = out.best_train_candidate
                heldout_best, _heldout_details, _heldout_latency = await _evaluate_candidate_once(
                    evaluate=evaluate_heldout,
                    candidate=best_train_candidate,
                    semaphore=semaphore,
                )
                baseline_score = out.heldout_baseline_score
                lift = (
                    heldout_best - baseline_score
                    if baseline_score is not None
                    else None
                )
                snapshot = MiproHeldoutSnapshot(
                    round_idx=rounds_completed,
                    best_candidate_id=str(best_train_candidate.candidate_id),
                    best_score=heldout_best,
                    baseline_score=baseline_score,
                    lift=lift,
                )
                out.heldout_snapshots.append(snapshot)
                if ledger is not None:
                    ledger.append_heldout_snapshot(seq=heldout_seq, snapshot=snapshot)
                    ledger.append_event(
                        event_type="heldout_snapshot",
                        payload={
                            "best_score": snapshot.best_score,
                            "baseline_score": snapshot.baseline_score,
                            "lift": snapshot.lift,
                        },
                        round_idx=snapshot.round_idx,
                        candidate_id=snapshot.best_candidate_id,
                    )
                    heldout_seq += 1

        should_run_final_heldout = (
            evaluate_heldout is not None
            and cfg.compute_final_heldout
            and out.best_train_candidate is not None
        )
        if should_run_final_heldout:
            assert evaluate_heldout is not None
            assert out.best_train_candidate is not None
            best_train_candidate = out.best_train_candidate
            latest_snapshot_round = (
                out.heldout_snapshots[-1].round_idx if out.heldout_snapshots else None
            )
            if latest_snapshot_round != rounds_completed:
                heldout_best, _heldout_details, _heldout_latency = await _evaluate_candidate_once(
                    evaluate=evaluate_heldout,
                    candidate=best_train_candidate,
                    semaphore=semaphore,
                )
                baseline_score = out.heldout_baseline_score
                lift = (
                    heldout_best - baseline_score
                    if baseline_score is not None
                    else None
                )
                snapshot = MiproHeldoutSnapshot(
                    round_idx=rounds_completed,
                    best_candidate_id=str(best_train_candidate.candidate_id),
                    best_score=heldout_best,
                    baseline_score=baseline_score,
                    lift=lift,
                )
                out.heldout_snapshots.append(snapshot)
                if ledger is not None:
                    ledger.append_heldout_snapshot(seq=heldout_seq, snapshot=snapshot)
                    ledger.append_event(
                        event_type="heldout_snapshot",
                        payload={
                            "best_score": snapshot.best_score,
                            "baseline_score": snapshot.baseline_score,
                            "lift": snapshot.lift,
                        },
                        round_idx=snapshot.round_idx,
                        candidate_id=snapshot.best_candidate_id,
                    )
                    heldout_seq += 1

        if out.heldout_snapshots:
            final_snapshot = out.heldout_snapshots[-1]
            out.heldout_best_score = final_snapshot.best_score
            out.heldout_lift = final_snapshot.lift

        if ledger is not None:
            if out.heldout_baseline_score is not None:
                ledger.upsert_state(
                    key="heldout_baseline_score", value=out.heldout_baseline_score
                )
            ledger.upsert_state(key="heldout_best_score", value=out.heldout_best_score)
            ledger.upsert_state(key="heldout_lift", value=out.heldout_lift)
            ledger.upsert_state(
                key="heldout_snapshots_count", value=len(out.heldout_snapshots)
            )
            ledger.append_event(
                event_type="run_completed",
                payload={
                    "best_train_score": out.best_train_score,
                    "heldout_best_score": out.heldout_best_score,
                    "heldout_lift": out.heldout_lift,
                },
                round_idx=rounds_completed,
                candidate_id=(
                    out.best_train_candidate.candidate_id
                    if out.best_train_candidate is not None
                    else None
                ),
            )
            ledger.set_status(status="completed")

        return out
    except Exception:
        if ledger is not None:
            ledger.append_event(
                event_type="run_failed",
                payload={"error": "phase2_loop_exception"},
            )
            ledger.set_status(status="failed")
        raise
    finally:
        if ledger is not None:
            ledger.close()


__all__ = [
    "EvaluateCandidateFn",
    "EvaluateCandidateOutcome",
    "EvaluateCandidateWithDetails",
    "MiproHeldoutSnapshot",
    "MiproPhase2Config",
    "MiproPhase2Outcome",
    "build_baseline_config",
    "run_train_loop",
]
