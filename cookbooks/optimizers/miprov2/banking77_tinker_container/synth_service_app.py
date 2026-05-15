from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

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
from synth_containers.nouns import (
    Actor,
    ExecutionRecord,
    Observation,
    Outcome,
    ToolCallRecord,
    TraceEvent,
    Trajectory,
    TurnRecord,
    VerifierResult,
)
from synth_containers.ontology import OutcomeKind
from synth_containers.rollout_tracing.v4 import (
    CanonicalChoice,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalUsage,
    ReasoningPart,
    RolloutTraceSpanV4,
    RolloutTraceV4,
    TextPart,
    ToolCallPart,
)


DATASET_NAME = "banking77"
TASK_ID = "banking77.tinker_intent_classification"
DEFAULT_SPLIT = "test"
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B"
LEGACY_4B_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_MAX_TOKENS = 256
DEFAULT_SYSTEM_PROMPT = (
    "You classify customer banking queries into exactly one intent.\n"
    "Choose the best intent from the available labels. You may reason briefly, "
    "then call banking77_classify with the selected snake_case intent."
)
DEFAULT_USER_PROMPT = "Customer query:\n{query}\n\nClassify this query using the banking77_classify function."
DEFAULT_LABELS = [
    "pending_card_payment",
    "pending_transfer",
    "pending_top_up",
    "pending_cash_withdrawal",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "declined_cash_withdrawal",
]
LABEL_MODE_CONFUSABLE7 = "confusable7"
LABEL_MODE_FULL77 = "full77"
TOOL_NAME = "banking77_classify"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_label(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _coerce_label_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _build_messages(*, query: str, labels: list[str], system_prompt: str, user_template: str) -> list[dict[str, str]]:
    label_block = "\n".join(f"- {label}" for label in labels)
    system = f"{system_prompt}\nValid intents:\n{label_block}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_template.format(query=query, available_intents=label_block)},
    ]


def _tool_schema(labels: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Submit the predicted Banking77 intent label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": labels,
                        "description": "The exact Banking77 intent label.",
                    }
                },
                "required": ["intent"],
            },
        },
    }


def _render_prompt_tokens(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    labels: list[str],
    native_tool_calling: bool,
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=[_tool_schema(labels)] if native_tool_calling else None,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(rendered, "get") and not isinstance(rendered, (list, tuple)):
        rendered = rendered.get("input_ids")
    if rendered is None:
        raise RuntimeError("apply_chat_template returned no input_ids")
    if rendered and isinstance(rendered[0], (list, tuple)):
        rendered = rendered[0]
    return [int(token) for token in rendered]


def _parse_prediction(text: str, labels: list[str]) -> str:
    cleaned = str(text or "").strip().lower()
    tool_match = re.search(
        rf"<function={re.escape(TOOL_NAME)}>\s*.*?<parameter=intent>\s*(.*?)\s*</parameter>",
        str(text or ""),
        flags=re.DOTALL | re.IGNORECASE,
    )
    if tool_match:
        cleaned = tool_match.group(1).strip().lower()
    else:
        json_match = re.search(r"\{.*?\"intent\"\s*:\s*\"([^\"]+)\".*?\}", str(text or ""), flags=re.DOTALL)
        if json_match:
            cleaned = json_match.group(1).strip().lower()
    cleaned = re.split(r"[\n\r]", cleaned, maxsplit=1)[0].strip()
    cleaned = cleaned.strip("`'\" .:;,")
    by_normalized = {_normalize_label(label): label for label in labels}
    if cleaned in by_normalized:
        return by_normalized[cleaned]
    for label in labels:
        if _normalize_label(label) in cleaned:
            return label
    return cleaned


def _split_reasoning_and_tool_text(text: str) -> tuple[str, str]:
    raw = str(text or "")
    tool_start = raw.find("<tool_call>")
    if tool_start < 0:
        return raw.strip(), ""
    reasoning = raw[:tool_start].strip()
    if reasoning.endswith("</think>"):
        reasoning = reasoning[: -len("</think>")].strip()
    return reasoning, raw[tool_start:].strip()


def _tool_call_from_output(text: str, predicted: str) -> dict[str, Any]:
    call_id = f"call_{uuid.uuid4().hex[:10]}"
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "arguments": json.dumps({"intent": predicted}, sort_keys=True),
        },
    }


def _assistant_message_from_output(text: str, predicted: str) -> dict[str, Any]:
    reasoning, tool_text = _split_reasoning_and_tool_text(text)
    tool_call = _tool_call_from_output(text, predicted)
    return {
        "role": "assistant",
        "content": tool_text,
        "reasoning_content": reasoning,
        "tool_calls": [tool_call],
    }


def _span_from_call(
    *,
    span_id: str,
    trace_id: str,
    request_messages: list[dict[str, str]],
    response_message: dict[str, Any],
    model: str,
    labels: list[str],
    max_tokens: int,
    temperature: float,
    native_tool_calling: bool,
    usage: dict[str, int],
    sample: Mapping[str, Any],
    expected: str,
    predicted: str,
    reward: float,
) -> RolloutTraceSpanV4:
    raw_request = {
        "messages": request_messages,
        "tools": [_tool_schema(labels)] if native_tool_calling else [],
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "native_tool_calling": native_tool_calling,
    }
    raw_response = {
        "message": response_message,
        "prediction": predicted,
        "expected": expected,
        "reward": reward,
    }
    request = CanonicalRequest(
        messages=tuple(CanonicalMessage.text(str(message["role"]), str(message["content"])) for message in request_messages),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tuple([_tool_schema(labels)]) if native_tool_calling else None,
        tool_choice={"type": "function", "function": {"name": TOOL_NAME}} if native_tool_calling else None,
        provider_hint="tinker",
    )
    tool_call = response_message["tool_calls"][0]
    response = CanonicalResponse(
        choices=(
            CanonicalChoice(
                index=0,
                message=CanonicalMessage(
                    role="assistant",
                    parts=(
                        ReasoningPart(content=str(response_message.get("reasoning_content") or ""), kind="model_reasoning"),
                        TextPart(text=str(response_message.get("content") or "")),
                        ToolCallPart(
                            id=str(tool_call.get("id") or ""),
                            name=str((tool_call.get("function") or {}).get("name") or TOOL_NAME),
                            arguments_json=str((tool_call.get("function") or {}).get("arguments") or "{}"),
                        ),
                    ),
                ),
                finish_reason="tool_call",
            ),
        ),
        usage=CanonicalUsage(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        ),
        model=model,
        provider_hint="tinker",
    )
    return RolloutTraceSpanV4(
        span_id=span_id,
        call_index=0,
        request=request,
        response=response,
        run_id=trace_id,
        api_format="tinker_sampling_chat_template_tools",
        raw_request=raw_request,
        raw_response=raw_response,
        metrics={"reward": reward, "correct": reward >= 1.0},
        metadata={
            "dataset": DATASET_NAME,
            "split": sample.get("split"),
            "index": sample.get("index"),
            "expected_intent": expected,
            "predicted_intent": predicted,
        },
    )


class TinkerBanking77Policy:
    def __init__(self, *, api_key: str, base_model: str, sampler_path: str | None = None) -> None:
        import tinker  # type: ignore

        self._tinker = tinker
        self._base_model = base_model
        self._sampler_path = sampler_path
        self._service_client = tinker.ServiceClient(api_key=api_key)
        if sampler_path:
            self._sampling_client = self._service_client.create_sampling_client(model_path=sampler_path)
        else:
            self._sampling_client = self._service_client.create_sampling_client(base_model=base_model)
        self._tokenizer = self._sampling_client.get_tokenizer()

    @property
    def model_ref(self) -> str:
        return self._sampler_path or self._base_model

    def classify(
        self,
        *,
        query: str,
        labels: list[str],
        system_prompt: str,
        user_template: str,
        max_tokens: int,
        temperature: float,
        seed: int,
        native_tool_calling: bool,
    ) -> tuple[str, str, dict[str, Any]]:
        messages = _build_messages(
            query=query,
            labels=labels,
            system_prompt=system_prompt,
            user_template=user_template,
        )
        prompt_ids = _render_prompt_tokens(
            self._tokenizer,
            messages,
            labels=labels,
            native_tool_calling=native_tool_calling,
        )
        sampling_params = self._tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            stop=None if native_tool_calling else ["\n"],
        )
        response = self._sampling_client.sample(
            prompt=self._tinker.ModelInput.from_ints(tokens=prompt_ids),
            num_samples=1,
            sampling_params=sampling_params,
        ).result()
        sequence = response.sequences[0]
        raw_text = self._tokenizer.decode(sequence.tokens, skip_special_tokens=True)
        prediction = _parse_prediction(raw_text, labels)
        metadata = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(sequence.tokens),
            "model_ref": self.model_ref,
            "native_tool_calling": native_tool_calling,
        }
        return prediction, raw_text, metadata


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

    def labels_for_mode(self, mode: str | None) -> list[str]:
        normalized = str(mode or LABEL_MODE_CONFUSABLE7).strip().lower()
        if normalized in {"full", "full77", "all", "77"}:
            return self.label_names
        if normalized in {"confusable", "confusable7", "seven", "7"}:
            return list(DEFAULT_LABELS)
        raise ValueError(f"unsupported Banking77 label mode: {mode!r}")

    def size(self, split: str) -> int:
        self._load()
        return len(self._train if split == "train" else self._test)

    def sample(self, *, split: str, seed: int, labels: list[str] | None = None) -> dict[str, Any]:
        self._load()
        ds = self._train if split == "train" else self._test
        allowed = set(labels or [])
        if allowed:
            indexes = [
                idx
                for idx, row in enumerate(ds)
                if self.label_names[int(row.get("label", 0))] in allowed
            ]
        else:
            indexes = list(range(len(ds)))
        if not indexes:
            raise RuntimeError(f"No Banking77 rows found for requested labels: {sorted(allowed)}")
        index = indexes[int(seed) % len(indexes)]
        row = ds[index]
        label_idx = int(row.get("label", 0))
        return {
            "index": index,
            "split": split,
            "text": str(row.get("text", "")),
            "label": self.label_names[label_idx],
            "label_idx": label_idx,
        }


class Banking77TinkerRuntime:
    def __init__(self) -> None:
        self.dataset = Banking77Dataset()
        self._executions: dict[str, ExecutionRecord] = {}
        self._policy_cache: dict[tuple[str, str | None], TinkerBanking77Policy] = {}

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
            metadata={"task_family": "banking77", "inference_provider": "tinker"},
        )

    def metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(
            runtime_id="banking77.tinker_synth_containers",
            name="Banking77 Tinker synth-containers runtime",
            description="Banking77 exact-label intent classification runtime using Tinker SamplingClient inference.",
            capabilities=self._capabilities(),
            metadata={
                "dataset": DATASET_NAME,
                "default_base_model": DEFAULT_BASE_MODEL,
                "legacy_4b_model": LEGACY_4B_MODEL,
                "default_label_mode": LABEL_MODE_CONFUSABLE7,
                "default_labels": DEFAULT_LABELS,
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
            inference={
                "provider": "tinker",
                "default_model": DEFAULT_BASE_MODEL,
                "api_key_env": "TINKER_API_KEY",
                "supports_sampler_path": True,
            },
            task_metadata={"available_intents": DEFAULT_LABELS, "all_intents": self.dataset.label_names},
            environment="banking77_tinker",
        )

    def _task_definition(self) -> TaskDefinition:
        return TaskDefinition(
            task_id=TASK_ID,
            task_name="Banking77 Tinker Intent Classification",
            task_family="banking77",
            description="Classify one banking-support utterance into exactly one Banking77 intent using Tinker inference.",
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
        return TaskCatalog(catalog_id="banking77:tinker_catalog", tasks=[self._task_definition()], instances=instances)

    def _policy_for(self, policy_cfg: Mapping[str, Any]) -> TinkerBanking77Policy:
        api_key = str(policy_cfg.get("api_key") or os.environ.get("TINKER_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("TINKER_API_KEY is required for Banking77 Tinker inference.")
        base_model = str(policy_cfg.get("model") or policy_cfg.get("base_model") or DEFAULT_BASE_MODEL).strip()
        sampler_path = str(policy_cfg.get("sampler_path") or policy_cfg.get("model_path") or "").strip() or None
        cache_key = (base_model, sampler_path)
        cached = self._policy_cache.get(cache_key)
        if cached is None:
            cached = TinkerBanking77Policy(api_key=api_key, base_model=base_model, sampler_path=sampler_path)
            self._policy_cache[cache_key] = cached
        return cached

    async def submit_rollout(self, request: Mapping[str, Any]) -> ExecutionRecord:
        payload = dict(request)
        task_payload = payload.get("task_payload") if isinstance(payload.get("task_payload"), dict) else {}
        env = payload.get("env") if isinstance(payload.get("env"), dict) else {}
        env_config = env.get("config") if isinstance(env.get("config"), dict) else {}
        policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
        policy_cfg = policy.get("config") if isinstance(policy.get("config"), dict) else {}
        candidate = task_payload.get("candidate") if isinstance(task_payload.get("candidate"), dict) else {}
        if not candidate:
            candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
        labels = (
            _coerce_label_list(candidate.get("labels"))
            or _coerce_label_list(policy_cfg.get("labels"))
            or self.dataset.labels_for_mode(
                str(
                    candidate.get("label_mode")
                    or policy_cfg.get("label_mode")
                    or env_config.get("label_mode")
                    or LABEL_MODE_CONFUSABLE7
                )
            )
        )
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
            sample = self.dataset.sample(split=split, seed=seed, labels=labels)
        system_prompt = str(candidate.get("system_prompt") or policy_cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
        user_template = str(candidate.get("user_prompt") or policy_cfg.get("user_prompt") or DEFAULT_USER_PROMPT)
        max_tokens = int(policy_cfg.get("max_tokens") or policy_cfg.get("max_completion_tokens") or DEFAULT_MAX_TOKENS)
        temperature = float(policy_cfg.get("temperature", 0.0))
        native_tool_calling = bool(policy_cfg.get("native_tool_calling", True))
        policy_runtime = self._policy_for(policy_cfg)
        predicted, raw_response, model_metadata = await asyncio.to_thread(
            policy_runtime.classify,
            query=sample["text"],
            labels=labels,
            system_prompt=system_prompt,
            user_template=user_template,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            native_tool_calling=native_tool_calling,
        )
        expected = str(sample["label"])
        reward = 1.0 if _normalize_label(predicted) == _normalize_label(expected) else 0.0
        usage = {
            "prompt_tokens": int(model_metadata.get("prompt_tokens") or 0),
            "cached_prompt_tokens": 0,
            "completion_tokens": int(model_metadata.get("completion_tokens") or 0),
            "total_tokens": int(model_metadata.get("prompt_tokens") or 0) + int(model_metadata.get("completion_tokens") or 0),
        }
        trace_id = str(payload.get("trace_correlation_id") or f"banking77_tinker_{uuid.uuid4().hex[:10]}")
        now = _utc_now_iso()
        task = self._task_definition()
        prompt_messages = _build_messages(
            query=sample["text"],
            labels=labels,
            system_prompt=system_prompt,
            user_template=user_template,
        )
        response_message = _assistant_message_from_output(raw_response, predicted)
        span = _span_from_call(
            span_id=f"{trace_id}:lm:0",
            trace_id=trace_id,
            request_messages=prompt_messages,
            response_message=response_message,
            model=str(model_metadata.get("model_ref") or DEFAULT_BASE_MODEL),
            labels=labels,
            max_tokens=max_tokens,
            temperature=temperature,
            native_tool_calling=native_tool_calling,
            usage=usage,
            sample=sample,
            expected=expected,
            predicted=predicted,
            reward=reward,
        )
        v4_trace = RolloutTraceV4(
            rollout_id=trace_id,
            trace_correlation_id=trace_id,
            status="completed",
            spans=(span,),
            summary={
                "reward": reward,
                "correct": reward >= 1.0,
                "predicted_intent": predicted,
                "expected_intent": expected,
            },
            metadata={"dataset": DATASET_NAME, "split": split, "seed": seed},
        ).to_dict()
        turn = TurnRecord(
            turn_index=1,
            actor_id="policy",
            prompt_messages=prompt_messages,
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
            actors=[Actor(actor_id="policy", role="agent", display_name="Tinker policy")],
            trajectory=Trajectory(
                turns=[turn],
                events=[
                    TraceEvent(
                        event_type="banking77_tinker_classification",
                        at=now,
                        step_index=1,
                        actor_id="policy",
                            payload={
                                "predicted_intent": predicted,
                                "expected_intent": expected,
                                "reward": reward,
                                "model_ref": model_metadata.get("model_ref"),
                                "v4_trace_span_id": span.span_id,
                            },
                        )
                ],
                metadata={"dataset": DATASET_NAME, "split": split, "seed": seed, "provider": "tinker"},
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
                        "raw_response": raw_response,
                        "model": model_metadata,
                        "v4_trace": v4_trace,
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
                "model_ref": model_metadata.get("model_ref"),
                "raw_response": raw_response,
            },
            usage=usage,
            metadata={
                "status_detail": "completed",
                "reward_source": "exact_label_match",
                "provider": "tinker",
                "model_ref": model_metadata.get("model_ref"),
                "predicted_intent": predicted,
                "expected_intent": expected,
                    "correct": reward >= 1.0,
                    "v4_trace": v4_trace,
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


app = create_reference_app(Banking77TinkerRuntime(), title="banking77-tinker-synth-container")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8943"))
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)
