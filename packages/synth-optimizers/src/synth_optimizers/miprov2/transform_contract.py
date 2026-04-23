"""Transform execution records used by the public MIPROv2 ledger."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from synth_containers.proxying import InferenceTarget, TraceIdentity


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TransformModelMutationContract:
    provider: str = ""
    base_model: str = ""
    request_model: str = ""
    student_model: str = ""
    teacher_model: str = ""
    generation_model: str = ""
    scoring_model: str = ""
    api_key_env: str = ""
    inference_target: InferenceTarget | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class TransformExecutionContract:
    transform_id: str
    transform_type: str
    variant_name: str = ""
    group_type: str = ""
    parent_candidate_id: str = ""
    idempotency_key: str = ""
    training_mode: str = ""
    adaptation_mode: str = ""
    output_dir: str = ""
    trace_identity: TraceIdentity | None = None
    training_source: dict[str, Any] = field(default_factory=dict)
    selected_seeds: list[int] = field(default_factory=list)
    selected_checkpoint_ids: list[str] = field(default_factory=list)
    selected_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    model_runtime_mutation: TransformModelMutationContract = field(
        default_factory=TransformModelMutationContract
    )
    expected_artifacts: list[str] = field(default_factory=list)
    request_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def output_root(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class TransformFailure:
    failure_kind: str
    message: str
    phase: str = ""
    retriable: bool = False
    request_path: str = ""
    result_path: str = ""
    stdout_path: str = ""
    stderr_path: str = ""
    traceback_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class TransformExecutionSummary:
    transform_id: str
    transform_type: str = ""
    training_mode: str = ""
    adaptation_mode: str = ""
    variant_name: str = ""
    parent_candidate_id: str = ""
    output_dir: str = ""
    status: str = ""
    failure_kind: str = ""
    baseline_holdout_mean_reward: float | None = None
    transformed_holdout_mean_reward: float | None = None
    holdout_mean_reward_delta: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


__all__ = [
    "TransformExecutionContract",
    "TransformExecutionSummary",
    "TransformFailure",
    "TransformModelMutationContract",
]
