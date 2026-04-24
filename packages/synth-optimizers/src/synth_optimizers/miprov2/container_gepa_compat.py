"""Container-backed GEPA compatibility adapter for MIPROv2.

The adapter treats a Synth container as the evaluation oracle: GEPA/MIPRO passes
a candidate prompt map and a batch of task rows, and the adapter evaluates each
row through the container `/rollout` contract.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from synth_containers.http_client import HTTPContainerClient

from synth_optimizers.miprov2.core import MiproEvaluationBatch


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    for method_name in ("to_dict", "model_dump", "dict", "as_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            raw = method()
            if isinstance(raw, Mapping):
                return {str(key): item for key, item in raw.items()}
    if hasattr(value, "__dict__"):
        return {str(key): item for key, item in vars(value).items()}
    return {"value": value}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(item) for item in value]
    return str(value)


def _first_text(source: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = source.get(key)
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _score_from_rollout(payload: Mapping[str, Any]) -> float:
    reward_info = _as_dict(payload.get("reward_info") or {})
    if reward_info.get("outcome_reward") is not None:
        return float(reward_info["outcome_reward"])
    summary = _as_dict(payload.get("summary") or {})
    for key in ("outcome_reward", "reward", "score", "total_reward"):
        if summary.get(key) is not None:
            return float(summary[key])
    metadata = _as_dict(payload.get("metadata") or {})
    for key in ("outcome_reward", "reward", "score", "total_reward"):
        if metadata.get(key) is not None:
            return float(metadata[key])
    return 0.0


def _empty_usage_totals() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _cached_prompt_tokens_from_usage(usage: Mapping[str, Any]) -> int:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        return int(details.get("cached_tokens") or 0)
    details = usage.get("input_tokens_details")
    if isinstance(details, Mapping):
        return int(details.get("cached_tokens") or 0)
    return int(usage.get("cached_prompt_tokens") or usage.get("cached_input_tokens") or 0)


def _add_usage_totals(left: dict[str, int], right: Mapping[str, Any]) -> dict[str, int]:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        left[key] = int(left.get(key) or 0) + int(right.get(key) or 0)
    left["cached_prompt_tokens"] = int(left.get("cached_prompt_tokens") or 0) + _cached_prompt_tokens_from_usage(right)
    return left


def _output_from_rollout(payload: Mapping[str, Any]) -> Any:
    summary = _as_dict(payload.get("summary") or {})
    for key in ("output", "prediction", "answer", "final_answer", "completion"):
        if summary.get(key) is not None:
            return _json_safe(summary[key])
    trace = _as_dict(payload.get("trace") or {})
    for key in ("output", "prediction", "answer", "final_answer"):
        if trace.get(key) is not None:
            return _json_safe(trace[key])
    return {
        "rollout_id": payload.get("rollout_id"),
        "status": payload.get("status"),
        "summary": _json_safe(summary),
    }


def _candidate_patch_examples(
    *,
    candidate: Mapping[str, str],
    traces: Sequence[Mapping[str, Any]],
    components_to_update: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {str(key): [] for key in components_to_update}
    for trace in traces:
        summary = _as_dict(trace.get("summary") or {})
        reward_info = _as_dict(trace.get("reward_info") or {})
        score = reward_info.get("outcome_reward")
        if score is None:
            score = summary.get("outcome_reward")
        for component_id in examples:
            examples[component_id].append(
                {
                    "instruction": str(candidate.get(component_id) or ""),
                    "source": "container_rollout",
                    "score": score,
                    "rollout_id": trace.get("rollout_id"),
                    "status": trace.get("status"),
                    "summary": _json_safe(summary),
                }
            )
    return examples


@dataclass(frozen=True, slots=True)
class ContainerGepaRolloutBinding:
    """Mapping rules from benchmark rows to Synth container rollout payloads."""

    task_id: str = ""
    candidate_field: str = "candidate"
    example_field: str = "example"
    capture_traces_field: str = "capture_traces"
    seed_keys: tuple[str, ...] = ("seed",)
    split_keys: tuple[str, ...] = ("split", "dataset_split")
    task_instance_id_keys: tuple[str, ...] = ("task_instance_id", "instance_id")
    extra_request: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContainerGepaAdapter:
    """GEPA adapter that evaluates candidates through a Synth container runtime."""

    client: HTTPContainerClient
    binding: ContainerGepaRolloutBinding = field(default_factory=ContainerGepaRolloutBinding)
    component_candidates: dict[str, list[str]] = field(default_factory=dict)
    max_concurrency: int = 8
    metric_call_count: int = 0
    usage_totals: dict[str, int] = field(default_factory=_empty_usage_totals)

    async def _evaluate_one(
        self,
        row: Any,
        candidate: Mapping[str, str],
        *,
        capture_traces: bool,
        index: int,
    ) -> dict[str, Any]:
        example = _as_dict(row)
        request = dict(self.binding.extra_request)
        if self.binding.task_id:
            request.setdefault("task_id", self.binding.task_id)
        raw_task_payload = request.get("task_payload")
        task_payload: dict[str, Any] = dict(raw_task_payload) if isinstance(raw_task_payload, dict) else {}
        task_payload[self.binding.candidate_field] = dict(candidate)
        task_payload[self.binding.example_field] = example
        task_payload[self.binding.capture_traces_field] = bool(capture_traces)
        request["task_payload"] = task_payload
        request.setdefault("trace_correlation_id", f"container_gepa_{index:06d}")
        seed = _first_text(example, self.binding.seed_keys)
        if seed:
            raw_env = request.get("env")
            env: dict[str, Any] = dict(raw_env) if isinstance(raw_env, dict) else {}
            env.setdefault("seed", int(seed) if seed.isdigit() else seed)
            request["env"] = env
        split = _first_text(example, self.binding.split_keys)
        if split:
            raw_env = request.get("env")
            env: dict[str, Any] = dict(raw_env) if isinstance(raw_env, dict) else {}
            raw_config = env.get("config")
            config: dict[str, Any] = dict(raw_config) if isinstance(raw_config, dict) else {}
            config.setdefault("split", split)
            env["config"] = config
            request["env"] = env
        task_instance_id = _first_text(example, self.binding.task_instance_id_keys)
        if task_instance_id:
            request.setdefault("task_instance_id", task_instance_id)
        return await self.client.rollout(request)

    async def evaluate(
        self,
        batch: Sequence[Any],
        candidate: Mapping[str, str],
        capture_traces: bool = False,
    ) -> MiproEvaluationBatch:
        semaphore = asyncio.Semaphore(max(1, int(self.max_concurrency)))

        async def guarded(index: int, row: Any) -> dict[str, Any]:
            async with semaphore:
                return await self._evaluate_one(
                    row,
                    candidate,
                    capture_traces=capture_traces,
                    index=index,
                )

        rollouts = await asyncio.gather(
            *(guarded(index, row) for index, row in enumerate(list(batch)))
        )
        self.metric_call_count += len(rollouts)
        for rollout in rollouts:
            usage = rollout.get("usage")
            if isinstance(usage, Mapping):
                _add_usage_totals(self.usage_totals, usage)
        return MiproEvaluationBatch(
            outputs=[_output_from_rollout(item) for item in rollouts],
            scores=[_score_from_rollout(item) for item in rollouts],
            traces=[_json_safe(item) for item in rollouts] if capture_traces else [],
            metadata={
                "adapter": "container_gepa",
                "binding": self.binding.to_dict(),
                "capture_traces": bool(capture_traces),
            },
        )

    def make_reflective_dataset(
        self,
        candidate: Mapping[str, str],
        eval_batch: MiproEvaluationBatch,
        components_to_update: Sequence[str],
    ) -> dict[str, list[dict[str, Any]]]:
        out = _candidate_patch_examples(
            candidate=candidate,
            traces=[_as_dict(trace) for trace in eval_batch.traces],
            components_to_update=components_to_update,
        )
        for key, values in dict(self.component_candidates or {}).items():
            target = out.setdefault(str(key), [])
            for value in values:
                text = str(value or "").strip()
                if text:
                    target.append({"instruction": text, "source": "component_candidates"})
        return out


__all__ = [
    "ContainerGepaAdapter",
    "ContainerGepaRolloutBinding",
]
