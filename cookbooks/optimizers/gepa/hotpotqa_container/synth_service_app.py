from __future__ import annotations

import argparse
import asyncio
import os
import random
import re
import string
import time
import uuid
from collections import Counter
from typing import Any

import uvicorn
from datasets import load_dataset
from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI

try:
    from synth_containers import GEPA_OPTIMIZER_CONTRACT_VERSION
except Exception:
    GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"


DATASET_NAME = "hotpot_qa"
DATASET_CONFIG = "distractor"
TASK_ID = "hotpotqa.multihop_qa"
DEFAULT_STAGE1_SYSTEM = (
    "You are a HotpotQA answer extractor. Read the question first to determine the "
    "requested answer type, then use the passages to find the entity, date, number, "
    "title, yes/no answer, or short phrase directly supported by the evidence. Connect "
    "facts across passages when needed. Return only the shortest answer string that "
    "fully answers the question, preserving the passage surface form for names, titles, "
    "dates, numbers, and quoted phrases. Do not return a related entity, category, "
    "alias, explanation, citation, or reasoning."
)
DEFAULT_STAGE1_USER = (
    "Question:\n{question}\n\n"
    "Passages:\n{context}\n\n"
    "Return only the short answer string."
)

POLICY_TIMEOUT_SECONDS = float(os.environ.get("HOTPOTQA_POLICY_TIMEOUT_SECONDS", "20"))
POLICY_RETRIES = int(os.environ.get("HOTPOTQA_POLICY_RETRIES", "1"))
POLICY_CONCURRENCY = int(os.environ.get("HOTPOTQA_POLICY_CONCURRENCY", "120"))
ROLLOUT_TIMEOUT_SECONDS = float(os.environ.get("HOTPOTQA_ROLLOUT_TIMEOUT_SECONDS", "25"))
DEFAULT_POLICY_MAX_TOKENS = 32

HOTPOTQA_PROPOSER_HINTS = {
    "task_output_space": "open_short_answer",
    "literal_training_targets": "forbid",
    "proposal_goal": (
        "Infer reusable answer-extraction and evidence-selection procedures from rollout "
        "traces. Do not turn observed train questions, gold answers, or predictions into "
        "candidate prompt mappings."
    ),
    "trace_review": [
        "Inspect question text, passage context, expected answer, prediction, score, and rationale.",
        "Identify answer-type mistakes, wrong-hop mistakes, related-entity mistakes, and output-format mistakes.",
        "Convert repeated trace patterns into general extraction rules, not memorized answer lists.",
    ],
    "valid_examples": "Use abstract or synthetic examples only; do not quote train question-answer pairs.",
}

_clients: dict[tuple[str, str, str], AsyncOpenAI] = {}
_policy_semaphore: asyncio.Semaphore | None = None
_dataset_lock = asyncio.Lock()
_train_dataset: Any | None = None
_validation_dataset: Any | None = None
_async_rollouts: dict[str, dict[str, Any]] = {}
_async_rollouts_lock = asyncio.Lock()
_terminal_statuses = {"completed", "failed", "cancelled", "terminated"}
_RAW_CREDENTIAL_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "openai_api_key",
    "openrouter_api_key",
    "secret_key",
}


def _find_raw_credential_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for raw_key, raw_value in value.items():
            normalized = str(raw_key).strip().lower().replace("-", "_")
            if normalized in _RAW_CREDENTIAL_KEYS or normalized.endswith("_api_key"):
                return str(raw_key)
            nested = _find_raw_credential_key(raw_value)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _find_raw_credential_key(item)
            if nested is not None:
                return nested
    return None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_policy_enum(value: Any, default: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return text or default


def _strip_openai_endpoint_suffix(url: str) -> str:
    normalized = url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _require_policy(payload: dict[str, Any]) -> dict[str, Any]:
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        raise HTTPException(
            status_code=422,
            detail="rollout.policy is required for GEPA optimizer contract v2.",
        )
    raw_key = _find_raw_credential_key(policy.get("config", {}))
    if raw_key is not None:
        raise HTTPException(
            status_code=422,
            detail=f"rollout.policy.config must not carry raw credential field {raw_key!r}.",
        )
    provider = str(policy.get("provider") or "").strip()
    model = str(policy.get("model") or "").strip()
    if not provider or not model:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.provider and rollout.policy.model are required.",
        )
    api_family = _normalize_policy_enum(policy.get("api_family"), "chat_completions")
    if api_family != "chat_completions":
        raise HTTPException(
            status_code=422,
            detail=f"{TASK_ID} supports rollout.policy.api_family='chat_completions'; got {api_family!r}.",
        )
    credential_mode = _normalize_policy_enum(policy.get("credential_mode"), "byok")
    if credential_mode not in {"byok", "proxy"}:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported rollout.policy.credential_mode: {credential_mode!r}",
        )
    raw_base_url = (
        str(policy.get("inference_url") or "").strip()
        if credential_mode == "proxy"
        else str(policy.get("base_url") or "").strip()
    )
    if credential_mode == "proxy" and not raw_base_url:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.inference_url is required when credential_mode=proxy.",
        )
    if provider.lower() == "openrouter" and credential_mode == "byok" and not raw_base_url:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.base_url is required for provider=openrouter.",
        )
    max_tokens = policy.get("max_tokens", DEFAULT_POLICY_MAX_TOKENS)
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.max_tokens must be an integer when set.",
        ) from exc
    if max_tokens <= 0:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.max_tokens must be positive.",
        )
    return {
        "provider": provider,
        "model": model,
        "base_url": _strip_openai_endpoint_suffix(raw_base_url) if raw_base_url else None,
        "credential_mode": credential_mode,
        "max_tokens": max_tokens,
    }


def _api_key(policy: dict[str, Any]) -> str:
    if policy["credential_mode"] == "proxy":
        return "proxy"
    env_name = "OPENROUTER_API_KEY" if policy["provider"].lower() == "openrouter" else "OPENAI_API_KEY"
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    raise HTTPException(
        status_code=503,
        detail=f"{env_name} is not set; rollout.policy credential_mode=byok requires a container env credential.",
    )


def _get_client(policy: dict[str, Any]) -> AsyncOpenAI:
    base_url = policy.get("base_url")
    key = (policy["provider"].lower(), policy["credential_mode"], str(base_url or ""))
    client = _clients.get(key)
    if client is None:
        client_kwargs: dict[str, Any] = {
            "api_key": _api_key(policy),
            "timeout": POLICY_TIMEOUT_SECONDS,
            "max_retries": 0,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        client = AsyncOpenAI(**client_kwargs)
        _clients[key] = client
    return client


def _get_policy_semaphore() -> asyncio.Semaphore:
    global _policy_semaphore
    if _policy_semaphore is None:
        _policy_semaphore = asyncio.Semaphore(max(1, POLICY_CONCURRENCY))
    return _policy_semaphore


async def _ensure_dataset_loaded() -> None:
    global _train_dataset, _validation_dataset
    if _train_dataset is not None and _validation_dataset is not None:
        return
    async with _dataset_lock:
        if _train_dataset is None:
            _train_dataset = await asyncio.to_thread(
                load_dataset,
                DATASET_NAME,
                DATASET_CONFIG,
                split="train",
            )
        if _validation_dataset is None:
            _validation_dataset = await asyncio.to_thread(
                load_dataset,
                DATASET_NAME,
                DATASET_CONFIG,
                split="validation",
            )


def _canonical_split(split: str) -> str:
    normalized = str(split or "train").strip().lower()
    if normalized in {"validation", "test", "heldout", "dev"}:
        return "validation"
    return "train"


def _dataset_for_split(split: str) -> Any:
    canonical = _canonical_split(split)
    return _validation_dataset if canonical == "validation" else _train_dataset


def _format_context(context: dict[str, Any]) -> str:
    blocks: list[str] = []
    titles = context.get("title") or []
    sentences = context.get("sentences") or []
    for title, sentence_list in zip(titles, sentences):
        text = " ".join(str(sentence) for sentence in sentence_list)
        blocks.append(f"[{title}]\n{text}")
    return "\n\n".join(blocks)


async def _row_for_seed(*, split: str, seed: int) -> dict[str, Any]:
    await _ensure_dataset_loaded()
    canonical = _canonical_split(split)
    dataset = _dataset_for_split(canonical)
    if dataset is None:
        raise HTTPException(status_code=503, detail="HotpotQA dataset did not load.")
    index = int(seed) % len(dataset)
    row = dataset[index]
    return {
        "seed": int(seed),
        "index": index,
        "split": canonical,
        "task_instance_id": f"hotpotqa:{canonical}:{index}",
        "question": str(row.get("question") or ""),
        "answer": str(row.get("answer") or ""),
        "context": _format_context(dict(row.get("context") or {})),
    }


def _normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    return " ".join(text.split())


def _token_f1(prediction: str, expected: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(expected).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap <= 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _extract_answer(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    json_match = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
    if json_match:
        return json_match.group(1).strip()
    for prefix in ("answer:", "final answer:"):
        if text.lower().startswith(prefix):
            return text[len(prefix) :].strip()
    return text.splitlines()[0].strip().strip('"')


async def _predict_answer(
    policy: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, dict[str, Any]]:
    client = _get_client(policy)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    last_error: Exception | None = None
    async with _get_policy_semaphore():
        for attempt in range(POLICY_RETRIES + 1):
            try:
                response = await client.chat.completions.create(
                    model=policy["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=policy["max_tokens"],
                )
                if response.usage is not None:
                    usage = {
                        "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(
                            getattr(response.usage, "completion_tokens", 0) or 0
                        ),
                        "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
                    }
                content = response.choices[0].message.content if response.choices else ""
                return _extract_answer(content or ""), usage
            except Exception as exc:
                last_error = exc
                if attempt >= POLICY_RETRIES:
                    break
                await asyncio.sleep(0.25 + random.random() * 0.25)
    raise HTTPException(
        status_code=502,
        detail=f"Policy model {policy['model']!r} failed after {POLICY_RETRIES + 1} attempt(s): {last_error}",
    )


app = FastAPI(title="hotpotqa-gepa-container")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metadata")
@app.get("/info")
async def metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": "hotpotqa_gepa_live",
            "name": "HotpotQA GEPA (live OpenAI-compatible policy)",
            "description": "Public prompt-optimizer cookbook for HotpotQA multi-hop QA.",
        },
        "capabilities": {
            "contract_version": "container_contract.v1",
            "rollout_modes": ["blocking"],
            "metadata": {},
        },
        "metadata": {
            "optimizer_contracts": {
                "gepa": {
                    "version": GEPA_OPTIMIZER_CONTRACT_VERSION,
                    "program_route": "/program",
                    "dataset_route": "/dataset",
                    "dataset_rows_route": "/dataset/rows",
                    "rollout_route": "/rollout",
                }
            }
        },
    }


@app.get("/task_info")
async def task_info() -> dict[str, Any]:
    await _ensure_dataset_loaded()
    return {
        "task": {
            "task_id": TASK_ID,
            "name": "HotpotQA multi-hop QA",
            "description": "Answer a multi-hop question using the supplied distractor passages.",
        },
        "output_space": {
            "kind": "open_short_answer",
            "contract": "Return only the short final answer string.",
            "valid_answer_types": [
                "entity",
                "person",
                "organization",
                "place",
                "title",
                "date",
                "number",
                "yes/no",
                "short phrase",
            ],
        },
        "dataset": {
            "dataset_id": "hotpot_qa:distractor",
            "visible_splits": ["train", "validation"],
            "default_split": "train",
            "row_count": len(_train_dataset or []),
            "heldout_row_count": len(_validation_dataset or []),
        },
        "proposer_hints": HOTPOTQA_PROPOSER_HINTS,
        "metadata": {
            "primary_metric": "token_f1",
            "policy_model_source": "rollout.policy.model",
            "answer_contract": "Return only the short final answer string.",
            "proposer_hints": HOTPOTQA_PROPOSER_HINTS,
        },
    }


@app.get("/program")
async def program() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "hotpotqa_single_stage_gepa",
        "modules": [
            {
                "module_id": "stage1_system",
                "role": "system",
                "content": DEFAULT_STAGE1_SYSTEM,
                "mutable": True,
                "candidate_field": "stage1_system",
                "template_variables": [],
            },
            {
                "module_id": "stage1_user",
                "role": "user",
                "content": DEFAULT_STAGE1_USER,
                "mutable": False,
                "candidate_field": "stage1_user",
                "template_variables": ["question", "context"],
            },
        ],
        "target_modules": [
            {
                "module_id": "stage1_system",
                "candidate_field": "stage1_system",
                "objective": "token_f1",
            }
        ],
        "seed_candidate": {"stage1_system": DEFAULT_STAGE1_SYSTEM},
        "rollout_overlay_schema": {"candidate_fields": ["stage1_system"]},
        "metadata": {
            "task_id": TASK_ID,
            "dataset_id": "hotpot_qa:distractor",
            "primary_metric": "token_f1",
            "answer_contract": "Return only the short final answer string.",
            "proposer_hints": HOTPOTQA_PROPOSER_HINTS,
        },
    }


@app.get("/dataset")
async def dataset() -> dict[str, Any]:
    await _ensure_dataset_loaded()
    return {
        "dataset_id": "hotpot_qa:distractor",
        "splits": {
            "train": len(_train_dataset or []),
            "validation": len(_validation_dataset or []),
        },
        "default_split": "train",
        "sampling": {"method": "seed_mod_dataset_index"},
    }


@app.post("/dataset/rows")
async def dataset_rows(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    seeds = [int(seed) for seed in payload.get("seeds") or []]
    rows = [await _row_for_seed(split=split, seed=seed) for seed in seeds]
    return {"rows": rows}


@app.post("/rollout")
@app.post("/rollouts")
async def rollout(request: Request) -> dict[str, Any]:
    payload = await request.json()
    submission_mode = str(payload.get("submission_mode") or "sync").strip().lower()
    if submission_mode == "sync":
        return await _execute_rollout_payload_with_timeout(payload)
    if submission_mode != "async":
        raise HTTPException(status_code=400, detail="submission_mode must be one of: sync, async")
    rollout_id = str(
        payload.get("rollout_id")
        or payload.get("trace_correlation_id")
        or f"rollout_{uuid.uuid4().hex[:12]}"
    )
    payload = {**payload, "rollout_id": rollout_id}
    now = _now()
    queued = {
        "rollout_id": rollout_id,
        "status": "queued",
        "success_status": "pending",
        "status_detail": "queued",
        "task_id": TASK_ID,
        "seed": int(payload.get("seed") or 0),
        "summary": {},
        "usage": {},
        "metadata": {"submission_mode": "async"},
        "created_at": now,
        "updated_at": now,
    }
    async with _async_rollouts_lock:
        _async_rollouts[rollout_id] = queued
    asyncio.create_task(_complete_async_rollout(rollout_id, payload))
    return queued


@app.get("/rollouts/{rollout_id}/state")
async def rollout_state(rollout_id: str) -> dict[str, Any]:
    return await _async_rollout_record(rollout_id)


@app.get("/rollouts/{rollout_id}")
async def rollout_record(rollout_id: str) -> dict[str, Any]:
    return await _async_rollout_record(rollout_id)


@app.post("/rollouts/{rollout_id}/terminate")
async def terminate_rollout(rollout_id: str, request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    reason = str(payload.get("reason") or "terminated")
    async with _async_rollouts_lock:
        current = _async_rollouts.get(rollout_id)
        if current is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        if str(current.get("status") or "") not in _terminal_statuses:
            now = _now()
            current = {
                **current,
                "status": "cancelled",
                "success_status": "cancelled",
                "status_detail": reason,
                "updated_at": now,
                "metadata": {
                    **dict(current.get("metadata") or {}),
                    "termination": {"reason": reason},
                },
            }
            _async_rollouts[rollout_id] = current
        return dict(current)


async def _execute_rollout_payload(payload: dict[str, Any]) -> dict[str, Any]:
    policy = _require_policy(payload)
    row = payload.get("dataset_row") if isinstance(payload.get("dataset_row"), dict) else None
    if row is None:
        row = await _row_for_seed(
            split=str(payload.get("split") or "train"),
            seed=int(payload.get("seed") or 0),
        )
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    system_prompt = str(candidate.get("stage1_system") or DEFAULT_STAGE1_SYSTEM)
    user_prompt = DEFAULT_STAGE1_USER.format(
        question=str(row.get("question") or ""),
        context=str(row.get("context") or ""),
    )
    prediction, usage = await _predict_answer(policy, system_prompt, user_prompt)
    expected = str(row.get("answer") or "")
    f1 = _token_f1(prediction, expected)
    exact_match = 1.0 if _normalize_text(prediction) == _normalize_text(expected) else 0.0
    rollout_id = str(payload.get("rollout_id") or f"rollout_{uuid.uuid4().hex[:12]}")
    now = _now()
    return {
        "rollout_id": rollout_id,
        "status": "completed",
        "success_status": "succeeded" if f1 > 0.0 else "failed",
        "task_id": TASK_ID,
        "seed": int(row.get("seed") or 0),
        "reward_info": {
            "outcome_reward": f1,
            "event_rewards": [f1],
            "details": {
                "prediction": prediction,
                "expected": expected,
                "exact_match": exact_match,
                "policy_model": policy["model"],
            },
        },
        "summary": {
            "outcome_reward": f1,
            "f1": f1,
            "exact_match": exact_match,
            "prediction": prediction,
            "expected": expected,
        },
        "usage": {**usage, "cost_usd": 0.0, "model": policy["model"]},
        "trace": {
            "event_history": [
                {"type": "question", "text": row.get("question")},
                {"type": "prediction", "answer": prediction},
            ],
            "metadata": {
                "expected": expected,
                "split": row.get("split"),
                "index": row.get("index"),
            },
        },
        "metadata": {
            "candidate": candidate,
            "question": row.get("question"),
            "expected_answer": expected,
            "predicted_answer": prediction,
        },
        "created_at": now,
        "updated_at": now,
    }


async def _execute_rollout_payload_with_timeout(payload: dict[str, Any]) -> dict[str, Any]:
    policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
    policy_model = str(policy.get("model") or "unknown")
    try:
        return await asyncio.wait_for(
            _execute_rollout_payload(payload),
            timeout=ROLLOUT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                f"rollout request timed out after {ROLLOUT_TIMEOUT_SECONDS:.1f}s; "
                f"policy_model={policy_model} policy_timeout={POLICY_TIMEOUT_SECONDS:.1f}s "
                f"policy_retries={POLICY_RETRIES} policy_concurrency={POLICY_CONCURRENCY}"
            ),
        ) from exc


async def _async_rollout_record(rollout_id: str) -> dict[str, Any]:
    async with _async_rollouts_lock:
        current = _async_rollouts.get(rollout_id)
    if current is None:
        raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
    return dict(current)


async def _complete_async_rollout(rollout_id: str, payload: dict[str, Any]) -> None:
    try:
        result = await _execute_rollout_payload_with_timeout(payload)
    except Exception as exc:
        now = _now()
        result = {
            "rollout_id": rollout_id,
            "status": "failed",
            "success_status": "failed",
            "status_detail": str(exc),
            "task_id": TASK_ID,
            "summary": {},
            "usage": {},
            "metadata": {"error": str(exc)},
            "created_at": now,
            "updated_at": now,
        }
    async with _async_rollouts_lock:
        current = _async_rollouts.get(rollout_id, {})
        if str(current.get("status") or "") not in _terminal_statuses:
            _async_rollouts[rollout_id] = result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8769")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
