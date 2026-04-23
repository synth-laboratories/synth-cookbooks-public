from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
import uvicorn

try:
    from synth_ai.data.enums import SuccessStatus
except ModuleNotFoundError:
    class SuccessStatus:
        SUCCESS = "success"

from synth_ai.sdk.container.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)
from synth_ai.sdk.container.server import ContainerConfig, create_container


DEFAULT_SYSTEM_PROMPT = (
    "Classify the customer banking query into exactly one Banking77 intent. "
    "Return the answer only by calling banking77_classify with the best single label."
)
DEFAULT_USER_PROMPT = (
    "Customer query:\n{query}\n\n"
    "Available intents:\n{available_intents}\n\n"
    "Call banking77_classify with the best matching intent."
)
TOOL_NAME = "banking77_classify"
DEFAULT_MAX_COMPLETION_TOKENS = 16192
DATASET_NAME = "banking77"
DEFAULT_SPLIT = "train"
AVAILABLE_SPLITS: tuple[str, ...] = ("train", "test")
DIRECT_PROVIDER_HOSTS = (
    "api.openai.com",
    "api.groq.com",
    "openrouter.ai",
    "api.openrouter.ai",
)


def normalize_chat_completion_url(url: str) -> str:
    route = (url or "").rstrip("/")
    if not route:
        raise RuntimeError("Missing inference_url/api_base/base_url in policy config.")

    parsed = urlparse(route)
    path = parsed.path.rstrip("/")

    if path.endswith("/v1/chat/completions") or path.endswith("/chat/completions"):
        return route
    if "/v1/" in path and not path.endswith("/v1"):
        new_path = f"{path}/chat/completions"
    elif path.endswith("/v1"):
        new_path = f"{path}/chat/completions"
    elif "/policy/" in path or "/proposer/" in path:
        new_path = f"{path}/chat/completions"
    elif path.endswith("/completions"):
        new_path = path.rsplit("/", 1)[0] + "/chat/completions"
    else:
        new_path = f"{path}/v1/chat/completions" if path else "/v1/chat/completions"

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            new_path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def is_direct_provider_host(endpoint: str) -> bool:
    lowered = endpoint.lower()
    return any(host in lowered for host in DIRECT_PROVIDER_HOSTS)


def _provider_api_key(provider: str, endpoint: str, override: str | None = None) -> str:
    if override:
        return override
    provider_lower = str(provider or "").strip().lower()
    env_name = "SYNTH_API_KEY"
    if provider_lower == "openai" or "api.openai.com" in endpoint.lower():
        env_name = "OPENAI_API_KEY"
    elif provider_lower == "groq" or "api.groq.com" in endpoint.lower():
        env_name = "GROQ_API_KEY"
    elif provider_lower == "openrouter" or "openrouter.ai" in endpoint.lower():
        env_name = "OPENROUTER_API_KEY"
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    synth_env = Path(__file__).resolve().parents[2].parent / "synth-ai" / ".env"
    if synth_env.exists():
        for raw_line in synth_env.read_text(encoding="utf-8").splitlines():
            if raw_line.startswith(f"{env_name}="):
                return raw_line.partition("=")[2].strip().strip("'").strip('"')
    raise RuntimeError(
        f"{env_name} is required for {provider_lower or 'provider'} inference."
    )


async def call_chat_completion(
    *,
    policy_cfg: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    http_client: httpx.AsyncClient | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    provider = str(policy_cfg.get("provider") or "openai").strip().lower()
    model = str(policy_cfg.get("model") or "").strip()
    if not model:
        raise RuntimeError("Missing model in policy config.")
    route = str(
        policy_cfg.get("inference_url")
        or policy_cfg.get("api_base")
        or policy_cfg.get("base_url")
        or ""
    ).strip()
    endpoint = normalize_chat_completion_url(route)
    api_key = _provider_api_key(
        provider,
        endpoint,
        override=str(policy_cfg.get("api_key") or "").strip() or None,
    )

    headers = {"Content-Type": "application/json"}
    if is_direct_provider_host(endpoint):
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["X-API-Key"] = api_key

    tool_schema = {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Return the predicted Banking77 intent label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                },
                "required": ["intent"],
            },
        },
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(policy_cfg.get("temperature", 1)),
        "max_completion_tokens": int(
            policy_cfg.get("max_completion_tokens") or DEFAULT_MAX_COMPLETION_TOKENS
        ),
        "tools": [tool_schema],
        "tool_choice": "required",
    }

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=60.0)
    try:
        response = await client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
    finally:
        if owns_client:
            await client.aclose()

    choices = body.get("choices") or []
    first_message = (choices[0] or {}).get("message") or {}
    raw_text = str(first_message.get("content") or "")
    tool_calls = list(first_message.get("tool_calls") or [])
    return raw_text, body, tool_calls


class Banking77Dataset:
    def __init__(self, train_ds: Any | None = None, test_ds: Any | None = None) -> None:
        self.train_ds = train_ds
        self.test_ds = test_ds
        self._label_names: list[str] | None = None

    def _load(self) -> None:
        if self.train_ds is None or self.test_ds is None:
            from datasets import load_dataset

            if self.train_ds is None:
                self.train_ds = load_dataset(DATASET_NAME, split="train")
            if self.test_ds is None:
                self.test_ds = load_dataset(DATASET_NAME, split="test")
        if self._label_names is None:
            ds = self.train_ds if self.train_ds is not None else self.test_ds
            features = getattr(ds, "features", {}) or {}
            label_feature = features.get("label")
            names = getattr(label_feature, "names", None)
            self._label_names = list(names or [])

    def size(self, split: str) -> int:
        self._load()
        ds = self.train_ds if split == "train" else self.test_ds
        return len(ds)

    def sample(self, *, split: str, index: int) -> dict[str, Any]:
        self._load()
        ds = self.train_ds if split == "train" else self.test_ds
        row = ds[int(index) % len(ds)]
        label_idx = int(row.get("label", 0))
        return {
            "index": int(index) % len(ds),
            "split": split,
            "text": str(row.get("text", "")),
            "label_idx": label_idx,
            "label": self.get_label_name(label_idx),
        }

    def get_label_name(self, label_idx: int) -> str:
        self._load()
        if self._label_names and 0 <= label_idx < len(self._label_names):
            return self._label_names[label_idx]
        return f"label_{label_idx}"

    @property
    def label_names(self) -> list[str]:
        self._load()
        return list(self._label_names or [])


def _normalize_label(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def extract_prediction(
    *,
    raw_text: str,
    tool_calls: list[dict[str, Any]],
    label_names: list[str],
) -> str:
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        if function.get("name") != TOOL_NAME:
            continue
        arguments_raw = function.get("arguments") or "{}"
        try:
            payload = json.loads(arguments_raw)
        except Exception:
            continue
        intent = str(payload.get("intent") or "").strip()
        if intent:
            return intent

    normalized_lookup = {_normalize_label(label): label for label in label_names}
    raw = str(raw_text or "").strip()
    if not raw:
        return ""

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        intent = str(parsed.get("intent") or parsed.get("label") or "").strip()
        if intent:
            return intent

    candidate = raw.splitlines()[0].strip().split()[0] if raw.strip() else ""
    normalized_candidate = _normalize_label(candidate)
    if normalized_candidate in normalized_lookup:
        return normalized_lookup[normalized_candidate]

    normalized_raw = _normalize_label(raw)
    for normalized_label, label in normalized_lookup.items():
        if normalized_label in normalized_raw:
            return label
    return candidate


async def rollout_executor(
    request: RolloutRequest,
    fastapi_request: Any,
) -> RolloutResponse:
    dataset: Banking77Dataset = fastapi_request.app.state.banking77_dataset
    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = int(request.env.seed or 0)
    sample = dataset.sample(split=split, index=seed)

    policy_cfg = dict(request.policy.config or {})
    system_prompt = str(policy_cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
    user_template = str(policy_cfg.get("user_prompt") or DEFAULT_USER_PROMPT)
    available_intents = "\n".join(
        f"{idx + 1}. {label}" for idx, label in enumerate(dataset.label_names)
    )
    user_prompt = user_template.format(
        query=sample["text"],
        available_intents=available_intents,
    )

    http_client = getattr(fastapi_request.app.state, "http_client", None)
    raw_response, response_json, tool_calls = await call_chat_completion(
        policy_cfg=policy_cfg,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        http_client=http_client,
    )
    predicted_intent = extract_prediction(
        raw_text=raw_response,
        tool_calls=tool_calls,
        label_names=dataset.label_names,
    )
    expected_intent = str(sample["label"])
    reward = (
        1.0
        if _normalize_label(predicted_intent) == _normalize_label(expected_intent)
        else 0.0
    )

    return RolloutResponse(
        trace_correlation_id=request.trace_correlation_id,
        reward_info=RolloutMetrics(
            outcome_reward=reward,
            outcome_objectives={"reward": reward},
            details={
                "correct": reward >= 1.0,
                "predicted_intent": predicted_intent,
                "expected_intent": expected_intent,
                "prediction": predicted_intent,
                "label": predicted_intent,
                "query": sample["text"],
                "raw_response": raw_response,
                "response_json": response_json,
                "seed": seed,
                "split": split,
            },
        ),
        trace=None,
        inference_url=str(policy_cfg.get("inference_url") or ""),
        success_status=SuccessStatus.SUCCESS,
    )


def build_banking77_app(
    *,
    dataset: Banking77Dataset | None = None,
    require_api_key: bool = False,
) -> Any:
    banking77_dataset = dataset or Banking77Dataset()

    async def startup_http_client() -> None:
        app.state.http_client = httpx.AsyncClient(timeout=60.0)

    async def shutdown_http_client() -> None:
        client = getattr(app.state, "http_client", None)
        if client is not None:
            await client.aclose()

    def provide_taskset_description() -> dict[str, Any]:
        return {
            "task": "banking77",
            "sizes": {
                split: banking77_dataset.size(split) for split in AVAILABLE_SPLITS
            },
            "num_labels": len(banking77_dataset.label_names),
        }

    def provide_task_instances(seeds: list[int]) -> list[TaskInfo]:
        rows: list[TaskInfo] = []
        for seed in seeds:
            sample = banking77_dataset.sample(split=DEFAULT_SPLIT, index=int(seed))
            rows.append(
                TaskInfo(
                    task={"id": "banking77", "name": "Banking77"},
                    dataset={
                        "id": DATASET_NAME,
                        "split": DEFAULT_SPLIT,
                        "index": int(seed),
                    },
                    inference={"model": "arcee-ai/trinity-mini:free"},
                    limits={"max_turns": 1},
                    task_metadata={
                        "query": sample["text"],
                        "expected_intent": sample["label"],
                        "available_intents": banking77_dataset.label_names,
                    },
                    environment="banking77",
                )
            )
        return rows

    config = ContainerConfig(
        app_id="banking77",
        name="Banking77",
        description="Banking77 intent classification container for nanoprogram.",
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=rollout_executor,
        app_state={"banking77_dataset": banking77_dataset},
        require_api_key=require_api_key,
        cors_origins=["*"],
        startup_hooks=[startup_http_client],
        shutdown_hooks=[shutdown_http_client],
    )
    app = create_container(config)
    return app


def fastapi_app() -> Any:
    return build_banking77_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Banking77 container")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--require-api-key", action="store_true")
    args = parser.parse_args()
    app = build_banking77_app(require_api_key=bool(args.require_api_key))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
