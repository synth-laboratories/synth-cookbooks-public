from __future__ import annotations

import asyncio
import json
import os
import random
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlparse, urlunparse

import httpx
import uvicorn
from datasets import load_dataset

from synth_containers import (
    CapabilityLevel,
    DatasetDescriptor,
    ExecutionProfile,
    PrimitiveProtocol,
    RolloutMode,
    RuntimeCapabilitySurface,
    RuntimeKind,
    RuntimeMetadata,
    StatefulnessTier,
    TaskCatalog,
    TaskDefinition,
    TaskInfo,
    TaskInstance,
    create_reference_app,
)
from synth_containers.nouns import Actor, ExecutionRecord, Observation, Outcome, ToolCallRecord, TraceEvent, Trajectory, TurnRecord, VerifierResult
from synth_containers.ontology import OutcomeKind


DATASET_NAME = "banking77"
TASK_ID = "banking77.intent_classification"
DEFAULT_SPLIT = "test"

# 2-stage pipeline: Stage 1 predicts one of 7 coarse categories, Stage 2 predicts intent within it.
CATEGORY_GROUPS: dict[str, list[int]] = {
    "card_payment": [15, 16, 45],
    "declined_transaction": [25, 26],
    "cash_withdrawal": [19, 20, 46, 75],
    "exchange_rate": [17, 31, 76],
    "top_up": [47, 57, 61],
    "refund_and_reversal": [51, 53, 63],
    "charges_and_debits": [28, 34],
}
LABEL_ID_TO_CATEGORY: dict[int, str] = {
    label_id: cat for cat, ids in CATEGORY_GROUPS.items() for label_id in ids
}
STAGE1_TOOL_NAME = "predict_category"
STAGE2_TOOL_NAME = "banking77_classify"
TOOL_NAME = STAGE2_TOOL_NAME  # backward compat

DEFAULT_STAGE1_SYSTEM_PROMPT = (
    "Classify the customer banking query into exactly one category group. "
    "Return the answer only by calling predict_category with the best matching category."
)
DEFAULT_STAGE1_USER_PROMPT = (
    "Customer query:\n{query}\n\n"
    "Available categories:\n{available_categories}\n\n"
    "Call predict_category with the best matching category."
)
DEFAULT_STAGE2_SYSTEM_PROMPT = (
    "Classify the customer banking query into exactly one Banking77 intent. "
    "Return the answer only by calling banking77_classify with the best single label."
)
DEFAULT_STAGE2_USER_PROMPT = (
    "Customer query:\n{query}\n\n"
    "Available intents:\n{available_intents}\n\n"
    "Call banking77_classify with the best matching intent."
)
# Legacy single-stage aliases (kept for GEPA path backward compat)
DEFAULT_SYSTEM_PROMPT = DEFAULT_STAGE2_SYSTEM_PROMPT
DEFAULT_USER_PROMPT = DEFAULT_STAGE2_USER_PROMPT

_LOCAL_INFERENCE_APPS: dict[str, Any] = {}


def register_local_inference_app(host: str, app: Any) -> None:
    """Register an in-process inference app for container-first cookbook rollouts."""

    normalized = str(host or "").strip().lower()
    if not normalized:
        raise ValueError("Local inference app host is required.")
    _LOCAL_INFERENCE_APPS[normalized] = app


def prompt_candidate_variants() -> dict[str, list[str]]:
    """Initial instruction options owned by the container program contract."""

    return {
        "stage1_system": [
            DEFAULT_STAGE1_SYSTEM_PROMPT,
            (
                "Group the customer banking query into one category. "
                "Call predict_category with the best matching group name."
            ),
            (
                "Identify the broad category of this banking issue and call predict_category with it. "
                "Choose the single most relevant category."
            ),
        ],
        "stage1_user": [
            DEFAULT_STAGE1_USER_PROMPT,
            (
                "Customer query:\n{query}\n\n"
                "Category groups:\n{available_categories}\n\n"
                "Call predict_category with the best matching category."
            ),
            (
                "Query:\n{query}\n\n"
                "Valid categories:\n{available_categories}\n\n"
                "Return the best matching category with predict_category."
            ),
        ],
        "stage2_system": [
            DEFAULT_STAGE2_SYSTEM_PROMPT,
            (
                "Classify the customer banking query into exactly one Banking77 intent. "
                "Prefer the most specific available label and respond only with a banking77_classify tool call."
            ),
            (
                "You are a precise Banking77 intent classifier. Distinguish close banking-support intents carefully, "
                "then call banking77_classify with exactly one label from the provided intent list."
            ),
        ],
        "stage2_user": [
            DEFAULT_STAGE2_USER_PROMPT,
            (
                "Customer query:\n{query}\n\n"
                "Choose exactly one intent from this list:\n{available_intents}\n\n"
                "Call banking77_classify with the exact intent string."
            ),
            (
                "Query:\n{query}\n\n"
                "Valid Banking77 intents:\n{available_intents}\n\n"
                "Return the best matching intent with banking77_classify."
            ),
        ],
    }


def mipro_optimizer_contract() -> dict[str, str]:
    return {
        "version": "miprov2.container_contract.v1",
        "program_route": "/program",
        "dataset_route": "/dataset",
        "dataset_rows_route": "/dataset/rows",
        "rollout_route": "/rollout",
    }


def policy_prompt_contract() -> dict[str, Any]:
    """Describe the prompt stages this container exposes to MIPRO-style interceptors."""

    return {
        "version": "prompt_program.v1",
        "program_id": "banking77_2stage_miprov2",
        "pipeline_id": "banking77",
        "search_space": {
            "initial_candidates": prompt_candidate_variants(),
        },
        "stages": [
            {
                "stage_id": "stage_1_coarse",
                "stage_name": "Coarse category classification",
                "tool_name": STAGE1_TOOL_NAME,
                "messages": [
                    {
                        "module_id": "stage1_system",
                        "role": "system",
                        "order": 0,
                        "content": DEFAULT_STAGE1_SYSTEM_PROMPT,
                        "candidate_field": "stage1_system",
                        "interceptor_apply": True,
                        "runtime_templated": False,
                    },
                    {
                        "module_id": "stage1_user",
                        "role": "user",
                        "order": 0,
                        "content": DEFAULT_STAGE1_USER_PROMPT,
                        "candidate_field": "stage1_user",
                        "interceptor_apply": False,
                        "runtime_templated": True,
                        "template_variables": ["query", "available_categories"],
                    },
                ],
            },
            {
                "stage_id": "stage_2_fine",
                "stage_name": "Fine-grained intent classification",
                "tool_name": STAGE2_TOOL_NAME,
                "messages": [
                    {
                        "module_id": "stage2_system",
                        "role": "system",
                        "order": 0,
                        "content": DEFAULT_STAGE2_SYSTEM_PROMPT,
                        "candidate_field": "stage2_system",
                        "interceptor_apply": True,
                        "runtime_templated": False,
                    },
                    {
                        "module_id": "stage2_user",
                        "role": "user",
                        "order": 0,
                        "content": DEFAULT_STAGE2_USER_PROMPT,
                        "candidate_field": "stage2_user",
                        "interceptor_apply": False,
                        "runtime_templated": True,
                        "template_variables": ["query", "available_intents"],
                    },
                ],
            },
        ],
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_label(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def normalize_chat_completion_url(url: str) -> str:
    route = (url or "").rstrip("/")
    if not route:
        raise RuntimeError("Missing inference_url/api_base/base_url in policy config.")
    parsed = urlparse(route)
    path = parsed.path.rstrip("/")
    if "/api/interceptor/v1/" in path:
        return route
    if path.endswith("/v1/chat/completions") or path.endswith("/chat/completions"):
        return route
    if path.endswith("/v1"):
        new_path = f"{path}/chat/completions"
    else:
        new_path = f"{path}/v1/chat/completions" if path else "/v1/chat/completions"
    return urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, parsed.query, parsed.fragment))


def _provider_api_key(policy_cfg: Mapping[str, Any], endpoint: str) -> str:
    override = str(policy_cfg.get("api_key") or "").strip()
    if override:
        return override
    provider = str(policy_cfg.get("provider") or "").strip().lower()
    if "/api/interceptor/v1/" in urlparse(endpoint).path:
        return ""
    env_name = "SYNTH_API_KEY"
    if provider == "openai" or "api.openai.com" in endpoint.lower():
        env_name = "OPENAI_API_KEY"
    elif provider == "groq" or "api.groq.com" in endpoint.lower():
        env_name = "GROQ_API_KEY"
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    raise RuntimeError(f"{env_name} is required for Banking77 inference.")


async def call_chat_completion(
    *,
    policy_cfg: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
    tool_name: str = STAGE2_TOOL_NAME,
    tool_description: str = "Return the predicted Banking77 intent label.",
    tool_result_field: str = "intent",
    tool_enum: list[str] | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    model = str(policy_cfg.get("model") or "gpt-4.1-nano").strip()
    endpoint = normalize_chat_completion_url(
        str(
            policy_cfg.get("inference_url")
            or policy_cfg.get("api_base")
            or policy_cfg.get("base_url")
            or "https://api.openai.com/v1/chat/completions"
        )
    )
    api_key = _provider_api_key(policy_cfg, endpoint)
    _field_schema: dict[str, Any] = {"type": "string"}
    if tool_enum:
        _field_schema["enum"] = list(tool_enum)
    tool_schema = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": {tool_result_field: _field_schema},
                "required": [tool_result_field],
            },
        },
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(policy_cfg.get("temperature", 0.0)),
        "max_completion_tokens": int(policy_cfg.get("max_completion_tokens") or 120),
        "tools": [tool_schema],
        "tool_choice": "required",
    }
    _provider = str(policy_cfg.get("provider") or "").strip().lower()
    if _provider == "groq" or "groq.com" in endpoint.lower():
        payload["max_tokens"] = payload.pop("max_completion_tokens")
        payload["temperature"] = 1.0
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
        _json_hint_sys = f'\n\nDo NOT use tool calls. Return ONLY a JSON object: {{"{tool_result_field}": "<value>"}}'
        _json_hint_usr = f'\n\nIMPORTANT: Do not call any tool. Respond with JSON only: {{"{tool_result_field}": "<value>"}}'
        payload["messages"] = [
            {**msg, "content": str(msg.get("content") or "") + (
                _json_hint_sys if msg.get("role") == "system" else
                _json_hint_usr if msg.get("role") == "user" else ""
            )}
            for msg in payload.get("messages", [])
        ]
    parsed_endpoint = urlparse(endpoint)
    local_app = _LOCAL_INFERENCE_APPS.get((parsed_endpoint.hostname or "").lower())
    if local_app is not None:
        local_path = urlunparse(("", "", parsed_endpoint.path, parsed_endpoint.params, parsed_endpoint.query, ""))
        transport = httpx.ASGITransport(app=local_app)
        client_context = httpx.AsyncClient(
            transport=transport,
            base_url=f"{parsed_endpoint.scheme}://{parsed_endpoint.netloc}",
            timeout=60.0,
        )
        request_url = local_path
    else:
        client_context = httpx.AsyncClient(timeout=60.0)
        request_url = endpoint
    async with client_context as client:
        for _attempt in range(12):
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            response = await client.post(
                request_url,
                json=payload,
                headers=headers,
            )
            if response.status_code == 429:
                # Honor the provider's retry-after time, plus jitter to stagger concurrent callers.
                # Fall back to short exponential backoff if the time is not parseable.
                _retry_secs = 0.1 * (2.0 ** min(_attempt, 6))
                _body_text = response.text
                _match = re.search(r"try again in ([\d.]+)(ms|s)", _body_text)
                if _match:
                    _val = float(_match.group(1))
                    _from_header = _val / 1000.0 if _match.group(2) == "ms" else _val
                    _retry_secs = max(_retry_secs, _from_header)
                _retry_secs += random.uniform(0.05, 0.5)
                await asyncio.sleep(min(_retry_secs, 60.0))
                continue
            if response.status_code >= 500:
                _retry_secs = 0.5 * (2.0 ** min(_attempt, 5)) + random.uniform(0.05, 0.3)
                await asyncio.sleep(min(_retry_secs, 30.0))
                continue
            if response.is_error:
                raise RuntimeError(
                    f"Inference request failed {response.status_code}: {response.text}"
                )
            break
        else:
            raise RuntimeError(
                f"Inference request failed after retries {response.status_code}: {response.text}"
            )
        body = response.json()
    choices = body.get("choices") or []
    first_message = (choices[0] or {}).get("message") if choices else {}
    raw_text = str((first_message or {}).get("content") or "")
    tool_calls = list((first_message or {}).get("tool_calls") or [])
    return raw_text, body, tool_calls


def extract_prediction(
    *,
    raw_text: str,
    tool_calls: list[dict[str, Any]],
    label_names: list[str],
    tool_name: str = STAGE2_TOOL_NAME,
    result_field: str = "intent",
) -> str:
    for call in tool_calls:
        fn = call.get("function") if isinstance(call, dict) else {}
        if not isinstance(fn, dict) or fn.get("name") != tool_name:
            continue
        try:
            payload = json.loads(str(fn.get("arguments") or "{}"))
        except Exception:
            payload = {}
        value = str(payload.get(result_field) or "").strip()
        if value:
            return value
    normalized_lookup = {_normalize_label(label): label for label in label_names}
    raw = str(raw_text or "").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        value = str(parsed.get(result_field) or "").strip()
        if value:
            return value
    normalized_raw = _normalize_label(raw)
    return normalized_lookup.get(normalized_raw, raw.splitlines()[0].strip() if raw else "")


def _cached_prompt_tokens_from_usage(usage: Mapping[str, Any]) -> int:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        return int(details.get("cached_tokens") or 0)
    details = usage.get("input_tokens_details")
    if isinstance(details, Mapping):
        return int(details.get("cached_tokens") or 0)
    return int(usage.get("cached_prompt_tokens") or usage.get("cached_input_tokens") or 0)


def usage_from_response(response_json: Mapping[str, Any]) -> dict[str, int]:
    usage = response_json.get("usage") if isinstance(response_json, Mapping) else None
    if not isinstance(usage, Mapping):
        return {
            "prompt_tokens": 0,
            "cached_prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": _cached_prompt_tokens_from_usage(usage),
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage.get("total_tokens") or (prompt_tokens + completion_tokens)),
    }


class Banking77Dataset:
    def __init__(self) -> None:
        self._train: Any | None = None
        self._test: Any | None = None
        self._label_names: list[str] | None = None

    def _load(self) -> None:
        if self._train is None:
            self._train = load_dataset(DATASET_NAME, split="train")
        if self._test is None:
            self._test = load_dataset(DATASET_NAME, split="test")
        if self._label_names is None:
            features = getattr(self._train, "features", {}) or {}
            self._label_names = list(getattr(features.get("label"), "names", None) or [])

    @property
    def label_names(self) -> list[str]:
        self._load()
        return list(self._label_names or [])

    def size(self, split: str) -> int:
        self._load()
        return len(self._train if split == "train" else self._test)

    def sample(self, *, split: str, seed: int) -> dict[str, Any]:
        self._load()
        ds = self._train if split == "train" else self._test
        index = int(seed) % len(ds)
        row = ds[index]
        label_idx = int(row.get("label", 0))
        return {
            "index": index,
            "split": split,
            "text": str(row.get("text", "")),
            "label": self.label_names[label_idx],
            "label_idx": label_idx,
        }

    def rows_for_seeds(
        self,
        *,
        split: str,
        seeds: list[int],
        label_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        self._load()
        ds = self._train if split == "train" else self._test
        labels = self.label_names
        allowed_labels = [int(label_id) for label_id in (label_ids or [])]
        allowed_label_set = set(allowed_labels)
        filtered_indices: list[int] = []
        indices_by_label: dict[int, list[int]] = {label_id: [] for label_id in allowed_labels}
        for index, row in enumerate(ds):
            label_idx = int(row.get("label", 0))
            if not allowed_label_set or label_idx in allowed_label_set:
                filtered_indices.append(index)
            if label_idx in indices_by_label:
                indices_by_label[label_idx].append(index)

        rows: list[dict[str, Any]] = []
        for offset, seed in enumerate(seeds):
            if allowed_labels:
                target_label = allowed_labels[offset % len(allowed_labels)]
                label_indices = indices_by_label.get(target_label) or []
                if not label_indices:
                    continue
                index = label_indices[int(seed) % len(label_indices)]
            else:
                if not filtered_indices:
                    continue
                index = filtered_indices[int(seed) % len(filtered_indices)]
            row = ds[index]
            label_idx = int(row.get("label", 0))
            label = labels[label_idx] if 0 <= label_idx < len(labels) else f"label_{label_idx}"
            rows.append(
                {
                    "seed": int(seed),
                    "index": int(index),
                    "split": split,
                    "text": str(row.get("text", "")),
                    "label": label,
                    "label_idx": label_idx,
                }
            )
        return rows


class Banking77Runtime:
    def __init__(self) -> None:
        self.dataset = Banking77Dataset()
        self._executions: dict[str, ExecutionRecord] = {}

    def _capabilities(self) -> RuntimeCapabilitySurface:
        return RuntimeCapabilitySurface(
            runtime_kind=RuntimeKind.ENVIRONMENT,
            profiles=[ExecutionProfile.STATELESS_EVALUATOR],
            rollout_modes=[RolloutMode.BLOCKING],
            statefulness_tier=StatefulnessTier.EPISODIC,
            protocol_fidelity={
                PrimitiveProtocol.CATALOG_BACKED: CapabilityLevel.NATIVE,
                PrimitiveProtocol.ROLLOUT_RUNNABLE: CapabilityLevel.NATIVE,
                PrimitiveProtocol.TRACE_EMITTING: CapabilityLevel.DERIVED,
                PrimitiveProtocol.REWARD_EMITTING: CapabilityLevel.NATIVE,
            },
            trace_support=True,
            reward_support=True,
            verifier_support=True,
            metadata={
                "task_family": "banking77",
                "policy_prompt_contract": policy_prompt_contract(),
                "optimizer_contracts": {"miprov2": mipro_optimizer_contract()},
            },
        )

    def metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(
            runtime_id="banking77.synth_containers",
            name="Banking77 synth-containers runtime",
            description="Banking77 exact-label intent classification runtime.",
            capabilities=self._capabilities(),
            metadata={
                "dataset": DATASET_NAME,
                "policy_prompt_contract": policy_prompt_contract(),
                "optimizer_contracts": {"miprov2": mipro_optimizer_contract()},
            },
        )

    def task_info(self) -> TaskInfo:
        return TaskInfo(
            task=self._task_definition(),
            dataset=DatasetDescriptor(
                dataset_id=DATASET_NAME,
                split=DEFAULT_SPLIT,
                visible_splits=["train", "test"],
                default_split=DEFAULT_SPLIT,
                row_count=self.dataset.size(DEFAULT_SPLIT),
                source="huggingface",
                metadata={"label_count": len(self.dataset.label_names)},
            ),
            capabilities=self._capabilities(),
            inference={"default_model": "gpt-4.1-nano", "api_key_env": "OPENAI_API_KEY"},
            task_metadata={"available_intents": self.dataset.label_names},
            environment="banking77",
            metadata={
                "policy_prompt_contract": policy_prompt_contract(),
                "optimizer_contracts": {"miprov2": mipro_optimizer_contract()},
            },
        )

    def program(self) -> dict[str, Any]:
        return policy_prompt_contract()

    def dataset_info(self) -> dict[str, Any]:
        return {
            "version": "dataset_contract.v1",
            "dataset_id": DATASET_NAME,
            "task_id": TASK_ID,
            "splits": {
                "train": {"row_count": self.dataset.size("train")},
                "test": {"row_count": self.dataset.size("test")},
            },
            "default_split": DEFAULT_SPLIT,
            "label_schema": {
                "type": "classification",
                "names": self.dataset.label_names,
                "category_groups": CATEGORY_GROUPS,
            },
            "row_request": {
                "supported_splits": ["train", "test"],
                "seed_field": "seeds",
                "filters": {"label_ids": "list[int]"},
            },
        }

    def dataset_rows(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request)
        split = str(payload.get("split") or DEFAULT_SPLIT).strip().lower()
        if split not in {"train", "test"}:
            raise ValueError(f"unsupported_banking77_split:{split}")
        raw_seeds = payload.get("seeds")
        if isinstance(raw_seeds, list):
            seeds = [int(seed) for seed in raw_seeds]
        else:
            limit = int(payload.get("limit") or 0)
            seeds = list(range(max(0, limit)))
        filters = payload.get("filters") if isinstance(payload.get("filters"), Mapping) else {}
        label_ids = [int(label_id) for label_id in (filters.get("label_ids") or payload.get("label_ids") or [])]
        rows = self.dataset.rows_for_seeds(split=split, seeds=seeds, label_ids=label_ids)
        return {
            "version": "dataset_rows.v1",
            "dataset_id": DATASET_NAME,
            "task_id": TASK_ID,
            "split": split,
            "rows": [
                {
                    "seed": row["seed"],
                    "task_instance_id": f"banking77:{split}:{row['index']}",
                    "example": dict(row),
                }
                for row in rows
            ],
        }

    def _task_definition(self) -> TaskDefinition:
        return TaskDefinition(
            task_id=TASK_ID,
            task_name="Banking77 Intent Classification",
            task_family="banking77",
            description="Classify one banking-support utterance into exactly one Banking77 intent.",
            version="v1",
            benchmark="banking77",
        )

    def task_catalog(self) -> TaskCatalog:
        instances = [
            TaskInstance(
                task_instance_id=f"banking77:test:{seed}",
                task_id=TASK_ID,
                split="test",
                seed=seed,
            )
            for seed in range(10)
        ]
        return TaskCatalog(catalog_id="banking77:catalog", tasks=[self._task_definition()], instances=instances)

    async def submit_rollout(self, request: Mapping[str, Any]) -> ExecutionRecord:
        payload = dict(request)
        task_payload = payload.get("task_payload") if isinstance(payload.get("task_payload"), dict) else {}
        env = payload.get("env") if isinstance(payload.get("env"), dict) else {}
        env_config = env.get("config") if isinstance(env.get("config"), dict) else {}
        example = task_payload.get("example") if isinstance(task_payload.get("example"), dict) else None
        if example is None:
            example = payload.get("example") if isinstance(payload.get("example"), dict) else None
        split = str((example or {}).get("split") or env_config.get("split") or payload.get("split") or DEFAULT_SPLIT)
        seed = int((example or {}).get("seed") or env.get("seed") or payload.get("seed") or 0)
        if example is not None and example.get("text") is not None and example.get("label") is not None:
            sample = {
                "index": int(example.get("index") or seed),
                "split": split,
                "text": str(example.get("text") or ""),
                "label": str(example.get("label") or ""),
                "label_idx": int(example.get("label_idx") or 0),
            }
        else:
            sample = self.dataset.sample(split=split, seed=seed)
        policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
        policy_cfg = policy.get("config") if isinstance(policy.get("config"), dict) else {}
        candidate = task_payload.get("candidate") if isinstance(task_payload.get("candidate"), dict) else {}
        if not candidate:
            candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
        stage1_system = str(candidate.get("stage1_system") or policy_cfg.get("stage1_system") or DEFAULT_STAGE1_SYSTEM_PROMPT)
        stage1_user_tmpl = str(candidate.get("stage1_user") or policy_cfg.get("stage1_user") or DEFAULT_STAGE1_USER_PROMPT)
        stage2_system = str(candidate.get("stage2_system") or policy_cfg.get("stage2_system") or DEFAULT_STAGE2_SYSTEM_PROMPT)
        stage2_user_tmpl = str(candidate.get("stage2_user") or policy_cfg.get("stage2_user") or DEFAULT_STAGE2_USER_PROMPT)

        category_names = list(CATEGORY_GROUPS.keys())
        available_categories = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(category_names))
        stage1_user = stage1_user_tmpl.format(query=sample["text"], available_categories=available_categories)
        raw1, rj1, tc1 = await call_chat_completion(
            policy_cfg=policy_cfg,
            system_prompt=stage1_system,
            user_prompt=stage1_user,
            tool_name=STAGE1_TOOL_NAME,
            tool_description="Predict the broad category group for this banking query.",
            tool_result_field="category",
            tool_enum=category_names,
        )
        predicted_category = extract_prediction(
            raw_text=raw1, tool_calls=tc1, label_names=category_names,
            tool_name=STAGE1_TOOL_NAME, result_field="category",
        )
        cat_norm = _normalize_label(predicted_category)
        cat_label_ids = next(
            (ids for cat, ids in CATEGORY_GROUPS.items() if _normalize_label(cat) == cat_norm), None
        )
        cat_intents = (
            [self.dataset.label_names[lid] for lid in cat_label_ids if lid < len(self.dataset.label_names)]
            if cat_label_ids else self.dataset.label_names
        )

        available_intents = "\n".join(f"{i + 1}. {intent}" for i, intent in enumerate(cat_intents))
        stage2_user = stage2_user_tmpl.format(query=sample["text"], available_intents=available_intents)
        raw2, rj2, tc2 = await call_chat_completion(
            policy_cfg=policy_cfg,
            system_prompt=stage2_system,
            user_prompt=stage2_user,
            tool_name=STAGE2_TOOL_NAME,
            tool_description="Return the predicted Banking77 intent label.",
            tool_result_field="intent",
        )
        usage1 = usage_from_response(rj1)
        usage2 = usage_from_response(rj2)
        usage = {k: usage1[k] + usage2[k] for k in usage1}
        predicted = extract_prediction(raw_text=raw2, tool_calls=tc2, label_names=self.dataset.label_names)
        expected = str(sample["label"])
        reward = 1.0 if _normalize_label(predicted) == _normalize_label(expected) else 0.0
        expected_category = LABEL_ID_TO_CATEGORY.get(int(sample.get("label_idx") or 0), "")
        s1_correct = bool(expected_category and _normalize_label(predicted_category) == _normalize_label(expected_category))
        trace_id = str(payload.get("trace_correlation_id") or f"banking77_{uuid.uuid4().hex[:10]}")
        now = _utc_now_iso()
        task = self._task_definition()
        turn = TurnRecord(
            turn_index=1,
            actor_id="policy",
            prompt_messages=[
                {"role": "system", "content": stage1_system},
                {"role": "user", "content": stage1_user},
                {"role": "assistant", "content": raw1 or ""},
                {"role": "system", "content": stage2_system},
                {"role": "user", "content": stage2_user},
            ],
            assistant_text=raw2,
            actions=[
                {"tool": STAGE1_TOOL_NAME, "category": predicted_category},
                {"tool": STAGE2_TOOL_NAME, "intent": predicted},
            ],
            executed_actions=[
                {"tool": STAGE1_TOOL_NAME, "category": predicted_category},
                {"tool": STAGE2_TOOL_NAME, "intent": predicted},
            ],
            observation=Observation(
                content={
                    "query": sample["text"],
                    "predicted_category": predicted_category,
                    "predicted_intent": predicted,
                    "expected_intent": expected,
                },
                channels={"reward": reward, "correct": reward >= 1.0},
                actor_id="verifier",
                created_at=now,
            ),
            event_rewards=[reward],
            outcome_reward=reward,
            tool_calls=[
                ToolCallRecord(tool_name=STAGE1_TOOL_NAME, arguments={"category": predicted_category}, success=True),
                ToolCallRecord(tool_name=STAGE2_TOOL_NAME, arguments={"intent": predicted}, success=True),
            ],
            metadata={"split": split, "seed": seed, "index": sample["index"]},
        )
        execution = ExecutionRecord(
            execution_id=trace_id,
            trace_correlation_id=trace_id,
            status="completed",
            success_status="success",
            created_at=now,
            updated_at=now,
            task=task,
            task_instance=TaskInstance(
                task_instance_id=f"banking77:{split}:{sample['index']}",
                task_id=TASK_ID,
                split=split,
                seed=seed,
                input_payload={"query": sample["text"]},
                metadata={"expected_intent": expected},
            ),
            actors=[Actor(actor_id="policy", role="agent", display_name="Policy")],
            trajectory=Trajectory(
                turns=[turn],
                events=[
                    TraceEvent(
                        event_type="banking77_classification",
                        at=now,
                        step_index=1,
                        actor_id="policy",
                        payload={"predicted_category": predicted_category, "predicted_intent": predicted, "expected_intent": expected, "reward": reward},
                    )
                ],
                metadata={"dataset": DATASET_NAME, "split": split, "seed": seed},
            ),
            outcome=Outcome(
                kind=OutcomeKind.REWARD,
                reward=reward,
                passed=reward >= 1.0,
                verifier=VerifierResult(verdict="correct" if reward >= 1.0 else "incorrect", score=reward, passed=reward >= 1.0),
                details={
                    "correct": reward >= 1.0,
                    "predicted_category": predicted_category,
                    "predicted_intent": predicted,
                    "expected_intent": expected,
                    "query": sample["text"],
                },
            ),
            summary={
                "outcome_reward": reward,
                "correct": reward >= 1.0,
                "output": predicted,
                "prediction": predicted,
                "predicted_intent": predicted,
                "expected_intent": expected,
                "split": split,
                "seed": seed,
            },
            usage=usage,
            metadata={
                "status_detail": "completed",
                "reward_source": "exact_label_match",
                "predicted_intent": predicted,
                "expected_intent": expected,
                "correct": reward >= 1.0,
                "achievements": {"s1_correct": s1_correct},
            },
        )
        self._executions[trace_id] = execution
        return execution

    async def get_execution(self, rollout_id: str) -> ExecutionRecord | None:
        return self._executions.get(str(rollout_id))

    async def get_execution_state(self, rollout_id: str) -> ExecutionRecord | None:
        return await self.get_execution(rollout_id)

    async def pause_execution(self, rollout_id: str, request: Mapping[str, Any]) -> ExecutionRecord | None:
        return await self.get_execution(rollout_id)

    async def terminate_execution(self, rollout_id: str, request: Mapping[str, Any]) -> ExecutionRecord | None:
        return await self.get_execution(rollout_id)

    async def create_checkpoint(self, rollout_id: str, request: Mapping[str, Any]) -> None:
        return None

    async def get_checkpoint(self, checkpoint_id: str) -> None:
        return None

    async def list_checkpoints(self, rollout_id: str | None = None) -> list[Any]:
        return []

    async def get_rollout_checkpoint(self, rollout_id: str, checkpoint_id: str) -> None:
        return None

    async def update_checkpoint_labels(self, checkpoint_id: str, request: Mapping[str, Any]) -> None:
        return None

    async def resume_execution(self, rollout_id: str, request: Mapping[str, Any]) -> None:
        return None


app = create_reference_app(Banking77Runtime(), title="banking77-synth-container")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8942"))
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)
