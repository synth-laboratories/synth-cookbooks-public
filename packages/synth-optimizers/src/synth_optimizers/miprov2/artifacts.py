"""Stable MIPROv2 artifact contract and file helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MiproArtifactPaths:
    best_candidate: str = "artifacts/best_candidate.json"
    heldout_eval: str = "artifacts/heldout_eval.json"
    run_summary: str = "artifacts/miprov2_run_summary.json"
    result_manifest: str = "artifacts/result_manifest.json"
    comparison_report: str = "artifacts/comparison_report.json"
    reportbench_output: str = "artifacts/reportbench_output.json"
    proposer_traces: str = "artifacts/proposer_traces"
    run_read_model: str = "artifacts/run_read_model.json"
    event_stream: str = "artifacts/run_events.jsonl"
    model_transforms: str = "artifacts/model_transforms.json"
    transform_failures: str = "artifacts/transform_failures.json"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MiproRunArtifactSummary:
    run_id: str
    task_id: str
    best_candidate_id: str | None = None
    best_score: float | None = None
    baseline_score: float | None = None
    heldout_score: float | None = None
    artifacts: MiproArtifactPaths = field(default_factory=MiproArtifactPaths)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = self.artifacts.as_dict()
        return payload


@dataclass(frozen=True, slots=True)
class MiproArtifactManifest:
    run_id: str
    task_id: str
    mode: str
    ledger_path: str | None = None
    workspace_root: str | None = None
    artifacts: MiproArtifactPaths = field(default_factory=MiproArtifactPaths)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = self.artifacts.as_dict()
        return payload


@dataclass(frozen=True, slots=True)
class MiproReportBenchResult:
    task_id: str
    answer: str
    score: float | None = None
    verifier: dict[str, Any] = field(default_factory=dict)
    artifacts: MiproArtifactPaths = field(default_factory=MiproArtifactPaths)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = self.artifacts.as_dict()
        return payload


def default_miprov2_artifact_paths() -> dict[str, str]:
    return MiproArtifactPaths().as_dict()


def write_miprov2_artifacts(
    *,
    output_dir: str,
    run_summary: dict[str, Any],
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
    paths = artifacts or MiproArtifactPaths()
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    created: dict[str, str] = {}

    def write_json(relative_path: str, payload: Any) -> None:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        created[relative_path] = str(path)

    write_json(paths.run_summary, run_summary)
    write_json(paths.best_candidate, best_candidate)
    write_json(paths.heldout_eval, heldout_eval)
    write_json(paths.result_manifest, result_manifest)
    if run_read_model is not None:
        write_json(paths.run_read_model, run_read_model)
    if comparison_report is not None:
        write_json(paths.comparison_report, comparison_report)
    if model_transforms is not None:
        write_json(paths.model_transforms, model_transforms)
    if transform_failures is not None:
        write_json(paths.transform_failures, transform_failures)
    if event_stream is not None:
        event_path = root / paths.event_stream
        event_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(item, sort_keys=True, ensure_ascii=True) for item in event_stream
        ]
        event_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        created[paths.event_stream] = str(event_path)
    return created


__all__ = [
    "MiproArtifactManifest",
    "MiproArtifactPaths",
    "MiproReportBenchResult",
    "MiproRunArtifactSummary",
    "default_miprov2_artifact_paths",
    "write_miprov2_artifacts",
]
