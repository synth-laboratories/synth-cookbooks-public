"""Normalized benchmark summaries, comparison reports, and rebuild helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from synth_optimizers.miprov2.artifacts import (
    MiproArtifactPaths,
    write_miprov2_artifacts,
)
from synth_optimizers.miprov2.core.run_ledger import SQLiteMiproRunLedger


@dataclass(frozen=True, slots=True)
class MiproBenchmarkSummary:
    dataset: str
    task: str
    mode: str
    optimizer_family: str
    optimizer_impl: str
    task_model: str
    reflection_model: str | None
    seed: int
    train_n: int
    heldout_n: int
    baseline_score: float | None
    best_score: float | None
    lift: float | None
    run_id: str | None
    ledger_path: str | None
    artifact_manifest_path: str | None
    total_metric_calls: int
    completed: bool
    resumed: bool
    timed_out: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MiproBenchmarkSummary":
        return cls(
            dataset=str(payload.get("dataset") or ""),
            task=str(payload.get("task") or ""),
            mode=str(payload.get("mode") or ""),
            optimizer_family=str(payload.get("optimizer_family") or ""),
            optimizer_impl=str(payload.get("optimizer_impl") or ""),
            task_model=str(payload.get("task_model") or ""),
            reflection_model=(
                str(payload["reflection_model"])
                if payload.get("reflection_model") is not None
                else None
            ),
            seed=int(payload.get("seed") or 0),
            train_n=int(payload.get("train_n") or 0),
            heldout_n=int(payload.get("heldout_n") or 0),
            baseline_score=(
                float(payload["baseline_score"])
                if payload.get("baseline_score") is not None
                else None
            ),
            best_score=(
                float(payload["best_score"])
                if payload.get("best_score") is not None
                else None
            ),
            lift=float(payload["lift"]) if payload.get("lift") is not None else None,
            run_id=(
                str(payload["run_id"]) if payload.get("run_id") is not None else None
            ),
            ledger_path=(
                str(payload["ledger_path"])
                if payload.get("ledger_path") is not None
                else None
            ),
            artifact_manifest_path=(
                str(payload["artifact_manifest_path"])
                if payload.get("artifact_manifest_path") is not None
                else None
            ),
            total_metric_calls=int(payload.get("total_metric_calls") or 0),
            completed=bool(payload.get("completed")),
            resumed=bool(payload.get("resumed")),
            timed_out=bool(payload.get("timed_out")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class MiproComparisonReport:
    dataset: str
    task: str
    runs: list[dict[str, Any]]
    deltas: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_dump(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return str(path)


def build_comparison_report(
    *,
    dataset: str,
    task: str,
    summaries: list[MiproBenchmarkSummary],
    metadata: dict[str, Any] | None = None,
) -> MiproComparisonReport:
    ordered = sorted(
        summaries,
        key=lambda item: (
            item.mode,
            item.optimizer_family,
            item.optimizer_impl,
            item.run_id or "",
        ),
    )
    runs = [item.to_dict() for item in ordered]
    deltas: list[dict[str, Any]] = []
    for left_idx, left in enumerate(ordered):
        for right in ordered[left_idx + 1 :]:
            left_score = left.best_score
            right_score = right.best_score
            delta = (
                float(left_score - right_score)
                if left_score is not None and right_score is not None
                else None
            )
            deltas.append(
                {
                    "left_mode": left.mode,
                    "right_mode": right.mode,
                    "left_run_id": left.run_id,
                    "right_run_id": right.run_id,
                    "left_best_score": left_score,
                    "right_best_score": right_score,
                    "absolute_delta": delta,
                }
            )
    return MiproComparisonReport(
        dataset=dataset,
        task=task,
        runs=runs,
        deltas=deltas,
        metadata=dict(metadata or {}),
    )


def load_benchmark_summary(path: str | Path) -> MiproBenchmarkSummary:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    return MiproBenchmarkSummary.from_dict(payload)


def write_benchmark_bundle(
    *,
    output_dir: str,
    summary: MiproBenchmarkSummary,
    best_candidate: dict[str, Any],
    heldout_eval: dict[str, Any],
    result_manifest: dict[str, Any],
    run_read_model: dict[str, Any] | None = None,
    event_stream: list[dict[str, Any]] | None = None,
    comparison_report: dict[str, Any] | None = None,
    model_transforms: list[dict[str, Any]] | None = None,
    transform_failures: list[dict[str, Any]] | None = None,
    artifacts: MiproArtifactPaths | None = None,
) -> dict[str, str]:
    created = write_miprov2_artifacts(
        output_dir=output_dir,
        run_summary=summary.to_dict(),
        best_candidate=best_candidate,
        heldout_eval=heldout_eval,
        result_manifest=result_manifest,
        run_read_model=run_read_model,
        event_stream=event_stream,
        comparison_report=comparison_report,
        model_transforms=model_transforms,
        transform_failures=transform_failures,
        artifacts=artifacts,
    )
    return created


def rebuild_summary_from_ledger(
    *,
    dataset: str,
    task: str,
    mode: str,
    optimizer_family: str,
    optimizer_impl: str,
    task_model: str,
    reflection_model: str | None,
    seed: int,
    train_n: int,
    heldout_n: int,
    ledger_path: str,
) -> MiproBenchmarkSummary:
    resolved_ledger_path = str(Path(ledger_path).expanduser().resolve())
    run_id = Path(resolved_ledger_path).stem
    ledger = SQLiteMiproRunLedger(
        run_id=run_id,
        ledger_path=resolved_ledger_path,
        program_id="mipro_benchmark_rebuild",
        mode=mode,
        resume=True,
    )
    try:
        run_status = ledger.get_run_status()
        state = ledger.query_run_state()
        resumed = any(
            event.get("event_type") == "run_resumed"
            for event in ledger.query_events(limit=1000)
        )
        artifact_manifest_path = state.get("artifact_manifest_path")
        baseline_score = (
            float(state["heldout_baseline_score"])
            if state.get("heldout_baseline_score") is not None
            else None
        )
        best_score = (
            float(state["heldout_best_score"])
            if state.get("heldout_best_score") is not None
            else None
        )
        lift = float(state["heldout_lift"]) if state.get("heldout_lift") is not None else None
        total_metric_calls = int(
            state.get("total_metric_calls")
            or len(ledger.load_resume_state().observations)
        )
        return MiproBenchmarkSummary(
            dataset=dataset,
            task=task,
            mode=mode,
            optimizer_family=optimizer_family,
            optimizer_impl=optimizer_impl,
            task_model=task_model,
            reflection_model=reflection_model,
            seed=int(seed),
            train_n=int(train_n),
            heldout_n=int(heldout_n),
            baseline_score=baseline_score,
            best_score=best_score,
            lift=lift,
            run_id=run_status.get("run_id"),
            ledger_path=resolved_ledger_path,
            artifact_manifest_path=str(artifact_manifest_path)
            if artifact_manifest_path is not None
            else None,
            total_metric_calls=total_metric_calls,
            completed=(run_status.get("status") == "completed"),
            resumed=resumed,
            timed_out=bool(state.get("timed_out")),
            metadata={
                "recovered_from_ledger": True,
                "workspace_root": run_status.get("workspace_root"),
                "run_state_keys": sorted(state.keys()),
            },
        )
    finally:
        ledger.close()


def write_comparison_report(
    *,
    output_dir: str,
    report: MiproComparisonReport,
    artifacts: MiproArtifactPaths | None = None,
) -> str:
    paths = artifacts or MiproArtifactPaths()
    root = Path(output_dir).expanduser().resolve()
    return _json_dump(root / paths.comparison_report, report.to_dict())


__all__ = [
    "MiproBenchmarkSummary",
    "MiproComparisonReport",
    "build_comparison_report",
    "load_benchmark_summary",
    "rebuild_summary_from_ledger",
    "write_benchmark_bundle",
    "write_comparison_report",
]
