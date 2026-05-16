"""Container-backed MIPROv2 adapter.

The adapter treats a Synth container as the evaluation oracle: MIPRO passes a
candidate prompt map and a batch of task rows, and the adapter evaluates each row
through the container `/rollout` contract.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

from synth_containers.http_client import HTTPContainerClient

from synth_optimizers.miprov2.core import (
    MiproEvaluationBatch,
    MiproModuleTemplate,
    MiproProgramTemplate,
    MiproStageTemplate,
)
from synth_optimizers.miprov2.local_interceptor import (
    InMemoryInterceptorTrialRegistry,
    InterceptorTrialRecord,
    create_local_interceptor_app,
)


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


def _safe_segment(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _stable_candidate_id(candidate: Mapping[str, Any]) -> str:
    explicit = str(candidate.get("candidate_id") or candidate.get("__candidate_id") or "").strip()
    if explicit:
        return _safe_segment(explicit)
    digest = hashlib.sha256(
        json.dumps(_json_safe(dict(candidate)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"candidate_{digest}"


def _selected_instruction_text(module: Any) -> str:
    candidates = getattr(module, "instruction_candidates", None)
    if isinstance(candidates, Sequence) and not isinstance(candidates, str | bytes | bytearray):
        return str(candidates[0] if candidates else "")
    if isinstance(candidates, Mapping):
        return str(candidates.get("i0") or next(iter(candidates.values()), ""))
    return ""


def build_stage_baseline_messages(
    program_template: Any,
    *,
    module_suffixes: Sequence[str] = ("_system",),
) -> dict[str, list[dict[str, str]]]:
    """Build interceptor baseline messages keyed by MIPRO stage id."""

    messages: dict[str, list[dict[str, str]]] = {}
    for stage in getattr(program_template, "stages", ()) or ():
        stage_id = str(getattr(stage, "stage_id", "") or "")
        if not stage_id:
            continue
        stage_messages: list[dict[str, str]] = []
        for module in getattr(stage, "modules", ()) or ():
            module_id = str(getattr(module, "module_id", "") or "")
            if not any(module_id.endswith(suffix) for suffix in module_suffixes):
                continue
            role = "user" if module_id.endswith("_user") else "system"
            stage_messages.append({"role": role, "content": _selected_instruction_text(module)})
        if stage_messages:
            messages[stage_id] = stage_messages
    return messages


def _stage_module_ids(program_template: Any) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for stage in getattr(program_template, "stages", ()) or ():
        stage_id = str(getattr(stage, "stage_id", "") or "")
        if not stage_id:
            continue
        for module in getattr(stage, "modules", ()) or ():
            module_id = str(getattr(module, "module_id", "") or "")
            if module_id.endswith("_system"):
                out.setdefault(stage_id, {})["system"] = module_id
            elif module_id.endswith("_user"):
                out.setdefault(stage_id, {})["user"] = module_id
    return out


def build_stage_prompt_deltas(
    *,
    program_template: Any,
    candidate: Mapping[str, str],
    baseline_messages: Mapping[str, Sequence[Mapping[str, Any]]],
    roles: Sequence[str] = ("system",),
) -> dict[str, dict[str, Any]]:
    """Build Rust-interceptor-compatible text replacement deltas."""

    module_ids_by_stage = _stage_module_ids(program_template)
    enabled_roles = {str(role) for role in roles}
    deltas: dict[str, dict[str, Any]] = {}
    for stage_id, stage_messages in baseline_messages.items():
        replacements: list[dict[str, Any]] = []
        order_by_role: dict[str, int] = {}
        stage_module_ids = module_ids_by_stage.get(str(stage_id), {})
        for message in stage_messages:
            role = str(message.get("role") or "")
            if role not in enabled_roles:
                continue
            order = order_by_role.get(role, 0)
            order_by_role[role] = order + 1
            module_id = stage_module_ids.get(role)
            baseline = str(message.get("content") or "")
            replacement = str(candidate.get(module_id or "") or baseline)
            replacements.append(
                {
                    "old_text": baseline,
                    "new_text": replacement,
                    "apply_to_role": role,
                    "apply_to_order": order,
                }
            )
        deltas[str(stage_id)] = {
            "transformation": {
                "text_replacements": replacements,
                "example_injections": [],
            }
        }
    return deltas


def program_template_from_prompt_contract(contract: Mapping[str, Any]) -> MiproProgramTemplate:
    """Build a MIPRO program template from a container prompt contract."""

    stages: list[MiproStageTemplate] = []
    search_space = contract.get("search_space")
    initial_candidates = (
        search_space.get("initial_candidates")
        if isinstance(search_space, Mapping) and isinstance(search_space.get("initial_candidates"), Mapping)
        else {}
    )
    raw_stages = contract.get("stages")
    if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, str | bytes | bytearray):
        raise ValueError("Container prompt contract must include a stages list.")
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, Mapping):
            continue
        modules: list[MiproModuleTemplate] = []
        raw_messages = raw_stage.get("messages")
        if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, str | bytes | bytearray):
            continue
        sorted_messages = sorted(
            (item for item in raw_messages if isinstance(item, Mapping)),
            key=lambda item: (str(item.get("role") or ""), int(item.get("order") or 0)),
        )
        for message in sorted_messages:
            module_id = str(message.get("module_id") or message.get("candidate_field") or "").strip()
            content = str(message.get("content") or "").strip()
            if not module_id or not content:
                continue
            candidates = [content]
            raw_candidates = initial_candidates.get(module_id) if isinstance(initial_candidates, Mapping) else None
            if isinstance(raw_candidates, Sequence) and not isinstance(raw_candidates, str | bytes | bytearray):
                candidates.extend(str(item).strip() for item in raw_candidates if str(item).strip())
            elif isinstance(raw_candidates, str) and raw_candidates.strip():
                candidates.append(raw_candidates.strip())
            deduped_candidates = tuple(dict.fromkeys(candidates))
            modules.append(
                MiproModuleTemplate(module_id=module_id, instruction_candidates=deduped_candidates)
            )
        if modules:
            stages.append(
                MiproStageTemplate(
                    stage_id=str(raw_stage.get("stage_id") or f"stage_{len(stages) + 1}"),
                    stage_name=str(raw_stage.get("stage_name") or "") or None,
                    modules=tuple(modules),
                )
            )
    if not stages:
        raise ValueError("Container prompt contract did not contain any usable stages.")
    return MiproProgramTemplate(
        program_id=str(contract.get("program_id") or contract.get("pipeline_id") or "container_prompt_contract"),
        stages=tuple(stages),
    )


def resolve_direct_inference_url() -> str:
    return str(os.environ.get("MIPRO_DIRECT_INFERENCE_URL") or "").strip()


def resolve_interceptor_base_url(value: str | None = None) -> str:
    return str(
        value
        or os.environ.get("MIPRO_PROXY_BASE_URL")
        or os.environ.get("INTERCEPTOR_BASE_URL")
        or ""
    ).strip().rstrip("/")


def build_interceptor_url(interceptor_base_url: str, trial_id: str, trace_id: str) -> str:
    base = resolve_interceptor_base_url(interceptor_base_url)
    if not base:
        raise RuntimeError(
            "MIPRO interceptor base URL is required; set MIPRO_PROXY_BASE_URL, "
            "or set MIPRO_DIRECT_INFERENCE_URL to bypass registration."
        )
    return f"{base}/api/interceptor/v1/{_safe_segment(trial_id)}/{_safe_segment(trace_id)}"


async def register_interceptor_trial(
    *,
    run_id: str,
    candidate_id: str,
    seed: int,
    baseline_messages: Mapping[str, Any],
    deltas: Mapping[str, Any],
    registry: InMemoryInterceptorTrialRegistry,
    forward_config: Mapping[str, Any] | None = None,
    ttl_seconds: int = 86_400,
    pipeline_id: str = "banking77",
    stage_id: str = "stage_default",
) -> str:
    trial_id = f"mipro_{_safe_segment(run_id)}_{_safe_segment(candidate_id)}_{int(seed)}"
    record = InterceptorTrialRecord(
        trial_id=trial_id,
        job_id=str(run_id),
        seed=int(seed),
        stage_key={"pipeline_id": str(pipeline_id), "stage_id": str(stage_id)},
        baseline_messages=_json_safe(dict(baseline_messages)),
        deltas=_json_safe(dict(deltas)),
        forward_config=_json_safe(dict(forward_config or {})),
        ttl_seconds=int(ttl_seconds),
    )
    await registry.set(record)
    return trial_id


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
class ContainerMiproRolloutBinding:
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
class ContainerMiproAdapter:
    """MIPRO adapter that evaluates candidates through a Synth container runtime."""

    client: HTTPContainerClient
    binding: ContainerMiproRolloutBinding = field(default_factory=ContainerMiproRolloutBinding)
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
        request.setdefault("trace_correlation_id", f"container_mipro_{index:06d}")
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
        batch_usage = _empty_usage_totals()
        for rollout in rollouts:
            usage = rollout.get("usage")
            if isinstance(usage, Mapping):
                _add_usage_totals(self.usage_totals, usage)
                _add_usage_totals(batch_usage, usage)
        return MiproEvaluationBatch(
            outputs=[_output_from_rollout(item) for item in rollouts],
            scores=[_score_from_rollout(item) for item in rollouts],
            traces=[_json_safe(item) for item in rollouts] if capture_traces else [],
            metadata={
                "adapter": "container_mipro",
                "binding": self.binding.to_dict(),
                "capture_traces": bool(capture_traces),
                "usage": batch_usage,
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


@dataclass(slots=True)
class ContainerMiproInterceptorAdapter(ContainerMiproAdapter):
    """Container adapter that routes candidate instructions through the MIPRO interceptor."""

    run_id: str = "mipro"
    program_template: Any | None = None
    interceptor_registry: InMemoryInterceptorTrialRegistry | None = None
    redis_client: Any | None = None
    redis_url: str | None = None
    interceptor_base_url: str | None = None
    direct_inference_url: str | None = None
    local_interceptor_host: str = "mipro-local-interceptor"
    ttl_seconds: int = 86_400
    pipeline_id: str = "banking77"
    stage_id: str = "stage_default"
    interceptor_roles: tuple[str, ...] = ("system",)
    policy_candidate_fields: tuple[str, ...] = ()
    _local_interceptor_app: Any | None = field(default=None, init=False, repr=False)

    async def aclose(self) -> None:
        await self.client.aclose()

    def _ensure_interceptor_registry(self) -> InMemoryInterceptorTrialRegistry:
        if self.interceptor_registry is None:
            self.interceptor_registry = InMemoryInterceptorTrialRegistry()
        return self.interceptor_registry

    def _register_local_interceptor_app(self, app: Any) -> None:
        host = str(self.local_interceptor_host or "").strip()
        if not host:
            raise RuntimeError("local_interceptor_host is required for Python MIPRO interception.")
        hooks: list[Any] = []
        for module_name in ("synth_service_app", "banking77_container.synth_service_app"):
            module = sys.modules.get(module_name)
            hook = getattr(module, "register_local_inference_app", None) if module is not None else None
            if callable(hook):
                hooks.append(hook)
        if not hooks:
            raise RuntimeError(
                "Local MIPRO interceptor app could not be registered with the container. "
                "Import the Banking77 synth_service_app module before constructing "
                "ContainerInterceptorAdapter, or set MIPRO_DIRECT_INFERENCE_URL for debugging."
            )
        for hook in hooks:
            hook(host, app)

    def _ensure_local_interceptor_base_url(self) -> str:
        registry = self._ensure_interceptor_registry()
        if self._local_interceptor_app is None:
            self._local_interceptor_app = create_local_interceptor_app(registry)
            self._register_local_interceptor_app(self._local_interceptor_app)
        return f"http://{self.local_interceptor_host}"

    async def _prepare_interceptor_rollout(
        self,
        *,
        row: Any,
        candidate: Mapping[str, str],
        capture_traces: bool,
        index: int,
        candidate_id: str,
    ) -> dict[str, Any]:
        example = _as_dict(row)
        request = dict(self.binding.extra_request)
        if self.binding.task_id:
            request.setdefault("task_id", self.binding.task_id)
        raw_task_payload = request.get("task_payload")
        task_payload: dict[str, Any] = dict(raw_task_payload) if isinstance(raw_task_payload, dict) else {}
        task_payload[self.binding.example_field] = example
        task_payload[self.binding.capture_traces_field] = bool(capture_traces)
        task_payload.pop(self.binding.candidate_field, None)
        request["task_payload"] = task_payload

        seed_text = _first_text(example, self.binding.seed_keys)
        seed = int(seed_text) if seed_text and seed_text.isdigit() else index
        trace_id = _safe_segment(
            request.get("trace_correlation_id")
            or f"trace_rollout_{_safe_segment(self.run_id)}_{_safe_segment(candidate_id)}_{seed}_{index}"
        )

        raw_policy = request.get("policy")
        policy: dict[str, Any] = dict(raw_policy) if isinstance(raw_policy, dict) else {}
        raw_policy_config = policy.get("config")
        policy_config: dict[str, Any] = dict(raw_policy_config) if isinstance(raw_policy_config, dict) else {}
        forward_config = dict(policy_config)

        direct_url = str(self.direct_inference_url or resolve_direct_inference_url()).strip()
        if direct_url:
            inference_url = direct_url
            trial_id = None
        else:
            if self.program_template is None:
                raise RuntimeError("program_template is required for interceptor-backed container rollouts.")
            baseline_messages = build_stage_baseline_messages(self.program_template)
            deltas = build_stage_prompt_deltas(
                program_template=self.program_template,
                candidate=candidate,
                baseline_messages=baseline_messages,
                roles=self.interceptor_roles,
            )
            trial_id = await register_interceptor_trial(
                run_id=self.run_id,
                candidate_id=candidate_id,
                seed=seed,
                baseline_messages=baseline_messages,
                deltas=deltas,
                registry=self._ensure_interceptor_registry(),
                forward_config=forward_config,
                ttl_seconds=self.ttl_seconds,
                pipeline_id=self.pipeline_id,
                stage_id=self.stage_id,
            )
            requested_base_url = resolve_interceptor_base_url(self.interceptor_base_url)
            if requested_base_url:
                parsed_host = str(urlparse(requested_base_url).hostname or "").lower()
                local_host = str(self.local_interceptor_host or "").lower()
                if parsed_host != local_host:
                    raise RuntimeError(
                        "External MIPRO interceptor URLs are not supported by the Python-local "
                        "cookbook path. Unset mipro_proxy_base_url/interceptor_base_url or set "
                        "MIPRO_DIRECT_INFERENCE_URL for direct container debugging."
                    )
                interceptor_base_url = requested_base_url
                self._ensure_local_interceptor_base_url()
            else:
                interceptor_base_url = self._ensure_local_interceptor_base_url()
            inference_url = build_interceptor_url(interceptor_base_url, trial_id, trace_id)

        policy_field_names = set(self.policy_candidate_fields)
        if direct_url:
            policy_field_names.update(
                str(key)
                for key in candidate
                if str(key).endswith("_system") or str(key).endswith("_user")
            )
        for field_name in sorted(policy_field_names):
            if candidate.get(field_name) is not None:
                policy_config[field_name] = str(candidate[field_name])
        policy_config["inference_url"] = inference_url
        policy["config"] = policy_config
        request["policy"] = policy
        request.setdefault("trace_correlation_id", trace_id)
        if trial_id:
            request.setdefault("trial_id", trial_id)
        request.setdefault("run_id", self.run_id)

        split = _first_text(example, self.binding.split_keys)
        if split:
            raw_env = request.get("env")
            env: dict[str, Any] = dict(raw_env) if isinstance(raw_env, dict) else {}
            raw_config = env.get("config")
            config: dict[str, Any] = dict(raw_config) if isinstance(raw_config, dict) else {}
            config.setdefault("split", split)
            env["config"] = config
            request["env"] = env
        raw_env = request.get("env")
        env = dict(raw_env) if isinstance(raw_env, dict) else {}
        env.setdefault("seed", seed)
        request["env"] = env
        return request

    async def evaluate(
        self,
        batch: Sequence[Any],
        candidate: Mapping[str, str],
        capture_traces: bool = False,
        *,
        candidate_id: str | None = None,
    ) -> MiproEvaluationBatch:
        candidate_map = {str(key): str(value) for key, value in dict(candidate).items()}
        resolved_candidate_id = _stable_candidate_id({**candidate_map, "candidate_id": candidate_id or ""})
        rows = list(batch)
        prepared = await asyncio.gather(
            *(
                self._prepare_interceptor_rollout(
                    row=row,
                    candidate=candidate_map,
                    capture_traces=capture_traces,
                    index=index,
                    candidate_id=resolved_candidate_id,
                )
                for index, row in enumerate(rows)
            )
        )
        semaphore = asyncio.Semaphore(max(1, int(self.max_concurrency)))

        async def guarded(request: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await self.client.rollout(request)

        rollouts = await asyncio.gather(*(guarded(request) for request in prepared))
        self.metric_call_count += len(rollouts)
        batch_usage = _empty_usage_totals()
        for rollout in rollouts:
            usage = rollout.get("usage")
            if isinstance(usage, Mapping):
                _add_usage_totals(self.usage_totals, usage)
                _add_usage_totals(batch_usage, usage)
        return MiproEvaluationBatch(
            outputs=[_output_from_rollout(item) for item in rollouts],
            scores=[_score_from_rollout(item) for item in rollouts],
            traces=[_json_safe(item) for item in rollouts] if capture_traces else [],
            metadata={
                "adapter": "container_interceptor",
                "binding": self.binding.to_dict(),
                "capture_traces": bool(capture_traces),
                "candidate_id": resolved_candidate_id,
                "interceptor_roles": list(self.interceptor_roles),
                "policy_candidate_fields": list(self.policy_candidate_fields),
                "direct_inference_url": bool(str(self.direct_inference_url or resolve_direct_inference_url()).strip()),
                "local_interceptor_host": self.local_interceptor_host,
                "usage": batch_usage,
            },
        )


ContainerInterceptorAdapter = ContainerMiproInterceptorAdapter

__all__ = [
    "ContainerInterceptorAdapter",
    "ContainerMiproAdapter",
    "ContainerMiproInterceptorAdapter",
    "ContainerMiproRolloutBinding",
    "build_interceptor_url",
    "build_stage_baseline_messages",
    "build_stage_prompt_deltas",
    "program_template_from_prompt_contract",
    "register_interceptor_trial",
    "resolve_direct_inference_url",
    "resolve_interceptor_base_url",
]
