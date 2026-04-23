"""Shared typed contracts for public-facing MIPROv2 compatibility surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from math import ceil
from typing import Any

from synth_containers.capabilities import RuntimeCapabilitySurface
from synth_containers.contracts import ContainerExecutionContract
from synth_containers.ontology import RuntimeFamily as ContainerRuntimeFamily
from synth_containers.proxying import (
    InferenceTarget,
    ProxyMode,
    ProxyResolution,
    ToolCallStyle,
    TraceIdentity,
)


class MiproCandidateExecutionMode(StrEnum):
    PROMPT_ONLY = "prompt_only"
    TINKER_SELF_DISTILL_SFT = "tinker_self_distill_sft"


@dataclass(frozen=True, slots=True)
class MiproSftConfig:
    training_backend: str = "tinker"
    transform_type: str = "tinker_self_distill_sft"
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    api_key_env: str = "TINKER_API_KEY"
    rank: int = 16
    epochs: int = 2
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_parallel_eval: int = 16
    max_tokens: int = 16
    keep_strategy: str = "correct_only"
    min_kept_rollouts: int = 1
    extra_packages: tuple[str, ...] = ("tinker",)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_backend": self.training_backend,
            "transform_type": self.transform_type,
            "base_model": self.base_model,
            "api_key_env": self.api_key_env,
            "rank": int(self.rank),
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "max_parallel_eval": int(self.max_parallel_eval),
            "max_tokens": int(self.max_tokens),
            "keep_strategy": self.keep_strategy,
            "min_kept_rollouts": int(self.min_kept_rollouts),
            "extra_packages": [str(item) for item in self.extra_packages],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MiproModelTransformRecord:
    transform_id: str
    transform_type: str
    training_backend: str
    parent_candidate_id: str
    child_candidate_id: str | None = None
    finetune_ref: str | None = None
    status: str = ""
    training_summary_path: str | None = None
    holdout_model_compare_path: str | None = None
    contract_path: str | None = None
    failure_path: str | None = None
    baseline_holdout_score: float | None = None
    finetuned_holdout_score: float | None = None
    holdout_delta: float | None = None
    num_train_samples: int | None = None
    optimizer_steps: int | None = None
    transform_stage: str | None = None
    cost_proxy_train: float | None = None
    cost_proxy_heldout: float | None = None
    cost_proxy_total: float | None = None
    estimated_cost_usd: float | None = None
    estimate_source: str | None = None
    inference_target: InferenceTarget | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "transform_type": self.transform_type,
            "training_backend": self.training_backend,
            "parent_candidate_id": self.parent_candidate_id,
            "child_candidate_id": self.child_candidate_id,
            "finetune_ref": self.finetune_ref,
            "status": self.status,
            "training_summary_path": self.training_summary_path,
            "holdout_model_compare_path": self.holdout_model_compare_path,
            "contract_path": self.contract_path,
            "failure_path": self.failure_path,
            "baseline_holdout_score": self.baseline_holdout_score,
            "finetuned_holdout_score": self.finetuned_holdout_score,
            "holdout_delta": self.holdout_delta,
            "num_train_samples": self.num_train_samples,
            "optimizer_steps": self.optimizer_steps,
            "transform_stage": self.transform_stage,
            "cost_proxy_train": self.cost_proxy_train,
            "cost_proxy_heldout": self.cost_proxy_heldout,
            "cost_proxy_total": self.cost_proxy_total,
            "estimated_cost_usd": self.estimated_cost_usd,
            "estimate_source": self.estimate_source,
            "inference_target": asdict(self.inference_target)
            if self.inference_target is not None
            else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MiproComponentSpec:
    component_id: str
    description: str = ""
    kind: str = "instruction"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MiproRuntimeBinding:
    runtime_family: ContainerRuntimeFamily | str = ContainerRuntimeFamily.REQUEST_RESPONSE
    inference_target: InferenceTarget | None = None
    proxy_mode: ProxyMode | str = ProxyMode.ALLOW_DIRECT
    proxy_resolution: ProxyResolution | None = None
    trace_identity: TraceIdentity | None = None
    tool_call_style: ToolCallStyle | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        runtime_family = (
            self.runtime_family.value
            if isinstance(self.runtime_family, ContainerRuntimeFamily)
            else str(self.runtime_family)
        )
        proxy_mode = (
            self.proxy_mode.value
            if isinstance(self.proxy_mode, ProxyMode)
            else str(self.proxy_mode)
        )
        tool_call_style = (
            self.tool_call_style.value
            if isinstance(self.tool_call_style, ToolCallStyle)
            else self.tool_call_style
        )
        return {
            "runtime_family": runtime_family,
            "inference_target": asdict(self.inference_target)
            if self.inference_target is not None
            else None,
            "proxy_mode": proxy_mode,
            "proxy_resolution": asdict(self.proxy_resolution)
            if self.proxy_resolution is not None
            else None,
            "trace_identity": asdict(self.trace_identity)
            if self.trace_identity is not None
            else None,
            "tool_call_style": tool_call_style,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MiproExecutionContract:
    dataset: str
    task: str = ""
    runtime_binding: MiproRuntimeBinding = field(default_factory=MiproRuntimeBinding)
    container_contract: ContainerExecutionContract | None = None
    reward_source: str = "adapter_metric"
    verifier_contract: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "runtime_binding": self.runtime_binding.to_dict(),
            "container_contract": self.container_contract.to_dict()
            if self.container_contract is not None
            else None,
            "reward_source": self.reward_source,
            "verifier_contract": dict(self.verifier_contract),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MiproCandidateRecord:
    candidate_id: str
    component_values: dict[str, str]
    parent_candidate_id: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "component_values": dict(self.component_values),
            "parent_candidate_id": self.parent_candidate_id,
            "score": self.score,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MiproEvaluationBatch:
    outputs: list[Any]
    scores: list[float]
    traces: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.traces and len(self.traces) != len(self.outputs):
            raise ValueError("MiproEvaluationBatch.traces must match outputs length")
        if len(self.outputs) != len(self.scores):
            raise ValueError("MiproEvaluationBatch.outputs and scores must align")

    def aggregate_score(self) -> float:
        if not self.scores:
            return 0.0
        return float(sum(float(score) for score in self.scores) / len(self.scores))

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs": list(self.outputs),
            "scores": [float(score) for score in self.scores],
            "traces": list(self.traces),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_value(cls, value: Any) -> MiproEvaluationBatch:
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("evaluation batch must be a MiproEvaluationBatch or mapping")
        outputs = list(value.get("outputs") or [])
        scores = [float(score) for score in list(value.get("scores") or [])]
        traces = list(value.get("traces") or [])
        metadata = dict(value.get("metadata") or {})
        return cls(outputs=outputs, scores=scores, traces=traces, metadata=metadata)


@dataclass(frozen=True, slots=True)
class MiproCompatRunConfig:
    dataset: str
    task: str = ""
    train_n: int | None = None
    heldout_n: int | None = None
    seed: int = 42
    task_model: str = ""
    proposer_model: str = ""
    optimizer_budget: int = 8
    max_concurrency: int = 4
    use_proposer: bool = False
    resume: bool = False
    run_id: str | None = None
    output_dir: str | None = None
    ledger_path: str | None = None
    runtime_binding: MiproRuntimeBinding = field(default_factory=MiproRuntimeBinding)
    runtime_capabilities: RuntimeCapabilitySurface | None = None
    container_contract: ContainerExecutionContract | None = None
    execution_mode: MiproCandidateExecutionMode = MiproCandidateExecutionMode.PROMPT_ONLY
    sft_config: MiproSftConfig | None = None
    component_candidates: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.optimizer_budget) <= 0:
            raise ValueError("optimizer_budget must be > 0")
        if int(self.max_concurrency) <= 0:
            raise ValueError("max_concurrency must be > 0")

    def phase2_rounds(self, *, top_k: int) -> int:
        batch_width = max(1, int(top_k))
        # Budget counts total candidate evaluations; reserve one for the seeded baseline.
        return max(1, ceil(max(1, int(self.optimizer_budget) - 1) / batch_width))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["runtime_binding"] = self.runtime_binding.to_dict()
        payload["runtime_capabilities"] = (
            self.runtime_capabilities.to_dict()
            if self.runtime_capabilities is not None
            else None
        )
        payload["container_contract"] = (
            self.container_contract.to_dict()
            if self.container_contract is not None
            else None
        )
        payload["execution_mode"] = self.execution_mode.value
        payload["sft_config"] = (
            self.sft_config.to_dict() if self.sft_config is not None else None
        )
        return payload


@dataclass(frozen=True, slots=True)
class MiproCompatResult:
    run_id: str | None
    ledger_path: str | None
    best_idx: int
    best_candidate: dict[str, str]
    candidates: list[dict[str, str]]
    candidate_records: list[MiproCandidateRecord]
    parents: list[str | None]
    val_aggregate_scores: list[float]
    val_subscores: list[list[float]]
    total_metric_calls: int
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "ledger_path": self.ledger_path,
            "best_idx": int(self.best_idx),
            "best_candidate": dict(self.best_candidate),
            "candidates": [dict(candidate) for candidate in self.candidates],
            "candidate_records": [record.to_dict() for record in self.candidate_records],
            "parents": list(self.parents),
            "val_aggregate_scores": [float(score) for score in self.val_aggregate_scores],
            "val_subscores": [
                [float(score) for score in scores] for scores in self.val_subscores
            ],
            "total_metric_calls": int(self.total_metric_calls),
            "artifacts": dict(self.artifacts),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MiproCompatResult:
        return cls(
            run_id=str(payload["run_id"]) if payload.get("run_id") is not None else None,
            ledger_path=(
                str(payload["ledger_path"])
                if payload.get("ledger_path") is not None
                else None
            ),
            best_idx=int(payload.get("best_idx") or 0),
            best_candidate=dict(payload.get("best_candidate") or {}),
            candidates=[
                dict(candidate) for candidate in list(payload.get("candidates") or [])
            ],
            candidate_records=[
                MiproCandidateRecord(
                    candidate_id=str(item.get("candidate_id") or ""),
                    component_values=dict(item.get("component_values") or {}),
                    parent_candidate_id=(
                        str(item["parent_candidate_id"])
                        if item.get("parent_candidate_id") is not None
                        else None
                    ),
                    score=float(item["score"]) if item.get("score") is not None else None,
                    metadata=dict(item.get("metadata") or {}),
                )
                for item in list(payload.get("candidate_records") or [])
                if isinstance(item, dict)
            ],
            parents=[
                str(parent) if parent is not None else None
                for parent in list(payload.get("parents") or [])
            ],
            val_aggregate_scores=[
                float(score)
                for score in list(payload.get("val_aggregate_scores") or [])
            ],
            val_subscores=[
                [float(score) for score in list(scores or [])]
                for scores in list(payload.get("val_subscores") or [])
            ],
            total_metric_calls=int(payload.get("total_metric_calls") or 0),
            artifacts=dict(payload.get("artifacts") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class MiproRunEvent:
    seq: int
    event_type: str
    payload: dict[str, Any]
    round_idx: int | None = None
    candidate_id: str | None = None
    created_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": int(self.seq),
            "event_type": self.event_type,
            "payload": dict(self.payload),
            "round_idx": self.round_idx,
            "candidate_id": self.candidate_id,
            "created_at": self.created_at,
        }


__all__ = [
    "InferenceTarget",
    "ProxyMode",
    "ProxyResolution",
    "RuntimeCapabilitySurface",
    "ToolCallStyle",
    "TraceIdentity",
    "MiproCandidateExecutionMode",
    "MiproCandidateRecord",
    "MiproCompatResult",
    "MiproCompatRunConfig",
    "MiproComponentSpec",
    "MiproEvaluationBatch",
    "MiproExecutionContract",
    "MiproModelTransformRecord",
    "MiproRunEvent",
    "MiproRuntimeBinding",
    "MiproSftConfig",
]
