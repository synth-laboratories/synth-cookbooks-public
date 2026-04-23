from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
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
    env_name = "SYNTH_API_KEY"
    if provider == "openai" or "api.openai.com" in endpoint.lower():
        env_name = "OPENAI_API_KEY"
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    raise RuntimeError(f"{env_name} is required for Banking77 inference.")


async def call_chat_completion(
    *,
    policy_cfg: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
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
    tool_schema = {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Return the predicted Banking77 intent label.",
            "parameters": {
                "type": "object",
                "properties": {"intent": {"type": "string"}},
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
        "temperature": float(policy_cfg.get("temperature", 0.0)),
        "max_completion_tokens": int(policy_cfg.get("max_completion_tokens") or 120),
        "tools": [tool_schema],
        "tool_choice": "required",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        response.raise_for_status()
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
) -> str:
    for call in tool_calls:
        fn = call.get("function") if isinstance(call, dict) else {}
        if not isinstance(fn, dict) or fn.get("name") != TOOL_NAME:
            continue
        try:
            payload = json.loads(str(fn.get("arguments") or "{}"))
        except Exception:
            payload = {}
        intent = str(payload.get("intent") or "").strip()
        if intent:
            return intent
    normalized_lookup = {_normalize_label(label): label for label in label_names}
    raw = str(raw_text or "").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        intent = str(parsed.get("intent") or parsed.get("label") or "").strip()
        if intent:
            return intent
    normalized_raw = _normalize_label(raw)
    return normalized_lookup.get(normalized_raw, raw.splitlines()[0].strip() if raw else "")


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
            metadata={"task_family": "banking77"},
        )

    def metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(
            runtime_id="banking77.synth_containers",
            name="Banking77 synth-containers runtime",
            description="Banking77 exact-label intent classification runtime.",
            capabilities=self._capabilities(),
            metadata={"dataset": DATASET_NAME},
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
        )

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
        env = payload.get("env") if isinstance(payload.get("env"), dict) else {}
        env_config = env.get("config") if isinstance(env.get("config"), dict) else {}
        split = str(env_config.get("split") or DEFAULT_SPLIT)
        seed = int(env.get("seed") or payload.get("seed") or 0)
        sample = self.dataset.sample(split=split, seed=seed)
        policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
        policy_cfg = policy.get("config") if isinstance(policy.get("config"), dict) else {}
        system_prompt = str(policy_cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
        user_template = str(policy_cfg.get("user_prompt") or DEFAULT_USER_PROMPT)
        available_intents = "\n".join(f"{idx + 1}. {label}" for idx, label in enumerate(self.dataset.label_names))
        user_prompt = user_template.format(query=sample["text"], available_intents=available_intents)
        raw_response, response_json, tool_calls = await call_chat_completion(
            policy_cfg=policy_cfg,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        predicted = extract_prediction(raw_text=raw_response, tool_calls=tool_calls, label_names=self.dataset.label_names)
        expected = str(sample["label"])
        reward = 1.0 if _normalize_label(predicted) == _normalize_label(expected) else 0.0
        trace_id = str(payload.get("trace_correlation_id") or f"banking77_{uuid.uuid4().hex[:10]}")
        now = _utc_now_iso()
        task = self._task_definition()
        turn = TurnRecord(
            turn_index=1,
            actor_id="policy",
            prompt_messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            assistant_text=raw_response,
            actions=[{"tool": TOOL_NAME, "intent": predicted}],
            executed_actions=[{"tool": TOOL_NAME, "intent": predicted}],
            observation=Observation(
                content={"query": sample["text"], "predicted_intent": predicted, "expected_intent": expected},
                channels={"reward": reward, "correct": reward >= 1.0},
                actor_id="verifier",
                created_at=now,
            ),
            event_rewards=[reward],
            outcome_reward=reward,
            tool_calls=[ToolCallRecord(tool_name=TOOL_NAME, arguments={"intent": predicted}, success=True)],
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
                        payload={"predicted_intent": predicted, "expected_intent": expected, "reward": reward},
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
                    "predicted_intent": predicted,
                    "expected_intent": expected,
                    "query": sample["text"],
                    "response_json": response_json,
                },
            ),
            summary={
                "outcome_reward": reward,
                "correct": reward >= 1.0,
                "predicted_intent": predicted,
                "expected_intent": expected,
                "split": split,
                "seed": seed,
            },
            metadata={
                "status_detail": "completed",
                "reward_source": "exact_label_match",
                "predicted_intent": predicted,
                "expected_intent": expected,
                "correct": reward >= 1.0,
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
