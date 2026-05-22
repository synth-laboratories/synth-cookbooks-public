from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request

# Live OpenAI-compatible policy. OpenRouter is supported through
# BANKING77_POLICY_BASE_URL plus OPENROUTER_API_KEY.
try:
    from openai import AsyncOpenAI
except Exception as _openai_err:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = _openai_err
else:
    _OPENAI_IMPORT_ERROR = None

POLICY_MODEL = os.environ.get("BANKING77_POLICY_MODEL", "qwen/qwen-2.5-7b-instruct")
POLICY_BASE_URL = os.environ.get("BANKING77_POLICY_BASE_URL") or os.environ.get(
    "OPENAI_BASE_URL"
)

# Fixed cap for OpenAI-compatible calls inside this service. Adaptive fan-out is
# owned by the Rust GEPA scheduler so run behavior is checkpointed with the run.
POLICY_CONCURRENCY = int(os.environ.get("BANKING77_POLICY_CONCURRENCY", "30"))
POLICY_TIMEOUT_SECONDS = float(os.environ.get("BANKING77_POLICY_TIMEOUT_SECONDS", "20"))
POLICY_RETRIES = int(os.environ.get("BANKING77_POLICY_RETRIES", "1"))
POLICY_RETRY_BACKOFF_SECONDS = float(
    os.environ.get("BANKING77_POLICY_RETRY_BACKOFF_SECONDS", "1.5")
)
ROLLOUT_TIMEOUT_SECONDS = float(
    os.environ.get("BANKING77_ROLLOUT_TIMEOUT_SECONDS", str(POLICY_TIMEOUT_SECONDS + 5))
)
POLICY_API_MODE = os.environ.get("BANKING77_POLICY_API_MODE", "auto").strip().lower()
POLICY_MAX_TOKENS = int(os.environ.get("BANKING77_POLICY_MAX_TOKENS", "16"))
POLICY_DISABLE_REASONING = (
    os.environ.get("BANKING77_POLICY_DISABLE_REASONING", "auto").strip().lower()
)
_openai_client: Any = None
_openai_semaphore: asyncio.Semaphore | None = None


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    if AsyncOpenAI is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "openai package not installed; install with `pip install openai>=1.0`. "
                f"Original import error: {_OPENAI_IMPORT_ERROR!r}"
            ),
        )
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY or OPENAI_API_KEY not set in container env; cannot serve live rollouts.",
        )
    client_kwargs = {"api_key": api_key, "timeout": POLICY_TIMEOUT_SECONDS}
    if POLICY_BASE_URL:
        client_kwargs["base_url"] = POLICY_BASE_URL
    _openai_client = AsyncOpenAI(**client_kwargs)
    return _openai_client


def _get_openai_semaphore() -> asyncio.Semaphore:
    """Lazy semaphore creation so it binds to the running event loop."""
    global _openai_semaphore
    if _openai_semaphore is None:
        _openai_semaphore = asyncio.Semaphore(max(1, POLICY_CONCURRENCY))
    return _openai_semaphore


def _is_policy_timeout(error: Exception) -> bool:
    if isinstance(error, TimeoutError):
        return True
    name = type(error).__name__.lower()
    return "timeout" in name or "timedout" in name


def _policy_prefers_chat() -> bool:
    if POLICY_API_MODE in {"chat", "chat_completions", "completions"}:
        return True
    if POLICY_API_MODE in {"responses", "response"}:
        return False
    return "openrouter.ai" in str(POLICY_BASE_URL or "").lower()


def _policy_retry_delay(attempt: int) -> float:
    return min(POLICY_RETRY_BACKOFF_SECONDS * (2 ** max(0, attempt - 1)), 8.0)


def _policy_chat_extra_body() -> dict[str, Any] | None:
    disable_reasoning = POLICY_DISABLE_REASONING in {"1", "true", "yes", "on"}
    if POLICY_DISABLE_REASONING == "auto":
        disable_reasoning = "openrouter.ai" in str(POLICY_BASE_URL or "").lower()
    if not disable_reasoning:
        return None
    return {
        "reasoning": {"effort": "none", "exclude": True},
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

try:
    from synth_containers import GEPA_OPTIMIZER_CONTRACT_VERSION
except Exception:
    GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v1"


TASK_ID = "banking77.intent_classification"

# Sample sizes: pick a balanced random subset from each split, then expose
# stable 0..N-1 seed indices for the optimizer inside that process.
TRAIN_SAMPLE = int(os.environ.get("BANKING77_TRAIN_SAMPLE", "24"))
TEST_SAMPLE = int(os.environ.get("BANKING77_TEST_SAMPLE", "200"))
TRAIN_SHUFFLE_SEED = int(os.environ.get("BANKING77_TRAIN_SHUFFLE_SEED", "1009"))
TEST_SHUFFLE_SEED = int(os.environ.get("BANKING77_TEST_SHUFFLE_SEED", "2003"))


def _load_banking77_rows() -> tuple[list[str], list[dict[str, Any]]]:
    """Load deterministic mixed PolyAI/banking77 train+test slices."""
    from datasets import load_dataset
    ds = load_dataset("PolyAI/banking77", trust_remote_code=True)
    label_names: list[str] = list(ds["train"].features["label"].names)

    def mixed_rows(split_name: str, sample_size: int, shuffle_seed: int) -> list[dict[str, Any]]:
        split = ds[split_name]
        grouped: dict[int, list[int]] = {idx: [] for idx in range(len(label_names))}
        for source_index, ex in enumerate(split):
            grouped[int(ex["label"])].append(source_index)
        rng = random.Random(shuffle_seed)
        for indices in grouped.values():
            rng.shuffle(indices)
        label_order = list(grouped)
        rng.shuffle(label_order)
        selected: list[int] = []
        while len(selected) < min(sample_size, len(split)):
            progressed = False
            for label_idx in label_order:
                if grouped[label_idx]:
                    selected.append(grouped[label_idx].pop())
                    progressed = True
                    if len(selected) >= sample_size:
                        break
            if not progressed:
                break
        rng.shuffle(selected)
        rows: list[dict[str, Any]] = []
        for seed, source_index in enumerate(selected):
            ex = split[source_index]
            rows.append({
                "seed": seed,
                "source_index": source_index,
                "split": split_name,
                "text": str(ex["text"]),
                "label": label_names[int(ex["label"])],
            })
        return rows

    rows = []
    rows.extend(mixed_rows("train", TRAIN_SAMPLE, TRAIN_SHUFFLE_SEED))
    rows.extend(mixed_rows("test", TEST_SAMPLE, TEST_SHUFFLE_SEED))
    return label_names, rows


LABELS, ROWS = _load_banking77_rows()
_LABEL_BY_LOWER = {label.lower(): label for label in LABELS}
_LABEL_BY_SIMPLIFIED = {
    "".join(ch for ch in label.lower() if ch.isalnum() or ch == "_"): label
    for label in LABELS
}

DEFAULT_STAGE2_SYSTEM = (
    "Classify the customer banking query into exactly one Banking77 intent. "
    "Return exactly one label from the allowed label list, preserving the label's spelling, "
    "underscores, capitalization, and punctuation. Use the full query, not one keyword. "
    "Prefer the label for the user's concrete banking action, status, or problem: separate "
    "physical-card ordering from delivery timing, virtual-card creation from virtual-card "
    "problems, card payments from cash withdrawals, top-ups from incoming transfers, "
    "pending from failed/declined/reverted, passcodes from card PINs, and phone loss from "
    "card compromise. Return only the label."
)

BANKING77_LABEL_GUIDANCE = {
    "output_contract": [
        "Return one canonical label exactly as it appears in the allowed list.",
        "Some dataset labels intentionally contain capitalization or punctuation; preserve them.",
        "Do not rewrite labels into normalized lowercase if the allowed label differs.",
    ],
    "high_value_boundaries": [
        {
            "boundary": "card_delivery_estimate vs order_physical_card vs get_physical_card",
            "rule": "Delivery timing or choosing a delivery day is card_delivery_estimate; asking to order/request a physical card is order_physical_card; asking how to obtain one generally is get_physical_card.",
        },
        {
            "boundary": "getting_virtual_card vs get_disposable_virtual_card vs virtual_card_not_working",
            "rule": "Getting a normal virtual card is getting_virtual_card; one-time disposable virtual cards are get_disposable_virtual_card; an existing virtual card failing is virtual_card_not_working.",
        },
        {
            "boundary": "pending_* vs failed_* vs declined_* vs reverted_*",
            "rule": "Pending means delayed or not completed yet; failed means the attempted operation did not complete; declined is an explicit refusal; reverted means a completed-looking top-up/payment was reversed.",
        },
        {
            "boundary": "card payment vs cash withdrawal",
            "rule": "Purchases/payments with a card use card_payment labels; ATM/cash-machine withdrawal issues use cash_withdrawal labels.",
        },
        {
            "boundary": "passcode_forgotten vs change_pin vs pin_blocked",
            "rule": "App passcode reset is passcode_forgotten; changing a card PIN is change_pin; too many wrong PIN attempts is pin_blocked.",
        },
        {
            "boundary": "lost_or_stolen_phone vs compromised_card vs lost_or_stolen_card",
            "rule": "Lost phone or app access on a new device is lost_or_stolen_phone; stolen physical card is lost_or_stolen_card; unauthorized card/account use is compromised_card.",
        },
        {
            "boundary": "receiving_money vs transfer_into_account vs topping_up_by_card",
            "rule": "Receiving external money or salary is receiving_money; bank transfer into the account is transfer_into_account; adding money by debit/credit card is topping_up_by_card.",
        },
        {
            "boundary": "supported_cards_and_currencies vs fiat_currency_support",
            "rule": "Supported cards/currencies for adding money is supported_cards_and_currencies; account or holding support for fiat currencies is fiat_currency_support.",
        },
    ],
}

BANKING77_PROPOSER_HINTS = {
    "task_output_space": "finite_intent_label",
    "literal_training_targets": "allow",
    "proposal_goal": (
        "Infer reusable Banking77 label-boundary rules from rollout traces, mistakes, and guard "
        "wins. Concrete query-to-label examples are valid when they teach a reusable distinction."
    ),
    "trace_review": [
        "Compare expected labels, predicted labels, rationales, and trace summaries for losses.",
        "Look for repeated boundary confusions rather than only isolated one-off examples.",
        "Preserve the exact allowed-label output contract.",
    ],
}


app = FastAPI(title="banking77-gepa-container")
_ASYNC_ROLLOUTS: dict[str, dict[str, Any]] = {}
_ASYNC_ROLLOUT_LOCK = asyncio.Lock()
_TERMINAL_ROLLOUT_STATUSES = {"completed", "failed", "cancelled"}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metadata")
@app.get("/info")
async def metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": "banking77_gepa_live",
            "name": "Banking77 GEPA (live OpenAI-compatible policy)",
            "description": "Public prompt-optimizer cookbook for Banking77 with a live OpenAI-compatible policy model.",
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
    return {
        "task": {
            "task_id": TASK_ID,
            "name": "Banking77 intent classification",
            "description": "Classify a customer banking question into one Banking77 label.",
        },
        "output_space": {
            "kind": "finite_intent_label",
            "label_count": len(LABELS),
            "labels": LABELS,
            "contract": "Return exactly one canonical label from the allowed list.",
        },
        "dataset": {
            "dataset_id": "banking77_public_rows",
            "visible_splits": ["train", "test"],
            "default_split": "train",
            "row_count": len(ROWS),
            "sampling": {
                "train_sample": TRAIN_SAMPLE,
                "test_sample": TEST_SAMPLE,
                "train_shuffle_seed": TRAIN_SHUFFLE_SEED,
                "test_shuffle_seed": TEST_SHUFFLE_SEED,
                "method": "balanced_random_per_label",
            },
        },
        "proposer_hints": BANKING77_PROPOSER_HINTS,
        "metadata": {
            "primary_metric": "classification_accuracy",
            "labels": LABELS,
            "label_guidance": BANKING77_LABEL_GUIDANCE,
            "proposer_hints": BANKING77_PROPOSER_HINTS,
        },
    }


@app.get("/program")
async def program() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "banking77_single_stage_gepa",
        "modules": [
            {
                "module_id": "stage2_system",
                "role": "system",
                "content": DEFAULT_STAGE2_SYSTEM,
                "mutable": True,
                "candidate_field": "stage2_system",
                "template_variables": [],
            }
        ],
        "target_modules": [
            {
                "module_id": "stage2_system",
                "candidate_field": "stage2_system",
                "objective": "classification_accuracy",
            }
        ],
        "seed_candidate": {"stage2_system": DEFAULT_STAGE2_SYSTEM},
        "rollout_overlay_schema": {"candidate_fields": ["stage2_system"]},
        "metadata": {
            "task_id": TASK_ID,
            "dataset_id": "banking77_public_rows",
            "labels": LABELS,
            "label_guidance": BANKING77_LABEL_GUIDANCE,
            "proposer_hints": BANKING77_PROPOSER_HINTS,
        },
    }


@app.get("/dataset")
async def dataset() -> dict[str, Any]:
    return {
        "dataset_id": "banking77_public_rows",
        "splits": {
            "train": sum(1 for row in ROWS if row["split"] == "train"),
            "test": sum(1 for row in ROWS if row["split"] == "test"),
        },
        "sampling": {
            "train_sample": TRAIN_SAMPLE,
            "test_sample": TEST_SAMPLE,
            "train_shuffle_seed": TRAIN_SHUFFLE_SEED,
            "test_shuffle_seed": TEST_SHUFFLE_SEED,
            "method": "balanced_random_per_label",
        },
        "labels": LABELS,
        "label_guidance": BANKING77_LABEL_GUIDANCE,
    }


@app.post("/dataset/rows")
async def dataset_rows(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    seeds = [int(seed) for seed in payload.get("seeds") or []]
    selected = []
    for seed in seeds:
        selected.append(_row_for_seed(split=split, seed=seed))
    return {"rows": selected}


@app.post("/rollout")
@app.post("/rollouts")
async def rollout(request: Request) -> dict[str, Any]:
    payload = await request.json()
    submission_mode = str(payload.get("submission_mode") or "sync").strip().lower()
    if submission_mode == "sync":
        return await _execute_rollout_payload_with_timeout(payload)
    if submission_mode != "async":
        raise HTTPException(
            status_code=400,
            detail="submission_mode must be one of: sync, async",
        )
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
    async with _ASYNC_ROLLOUT_LOCK:
        _ASYNC_ROLLOUTS[rollout_id] = queued
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
    async with _ASYNC_ROLLOUT_LOCK:
        current = _ASYNC_ROLLOUTS.get(rollout_id)
        if current is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        if str(current.get("status") or "") not in _TERMINAL_ROLLOUT_STATUSES:
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
            _ASYNC_ROLLOUTS[rollout_id] = current
        return dict(current)


async def _execute_rollout_payload(payload: dict[str, Any]) -> dict[str, Any]:
    row = payload.get("dataset_row") if isinstance(payload.get("dataset_row"), dict) else None
    if not row:
        row = _row_for_seed(
            split=str(payload.get("split") or "train"),
            seed=int(payload.get("seed") or 0),
        )
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    system_prompt = str(candidate.get("stage2_system") or DEFAULT_STAGE2_SYSTEM)
    # Direct await on AsyncOpenAI; concurrency capped inside _predict_label
    # via a module-level asyncio.Semaphore(POLICY_CONCURRENCY).
    prediction, usage = await _predict_label(
        str(row.get("text") or ""),
        system_prompt=system_prompt,
    )
    expected = str(row.get("label") or "")
    reward = 1.0 if prediction == expected else 0.0
    rollout_id = str(payload.get("rollout_id") or f"rollout_{uuid.uuid4().hex[:12]}")
    now = _now()
    return {
        "rollout_id": rollout_id,
        "status": "completed",
        "success_status": "succeeded" if reward > 0 else "failed",
        "task_id": TASK_ID,
        "seed": int(row.get("seed") or 0),
        "reward_info": {
            "outcome_reward": reward,
            "event_rewards": [reward],
            "details": {
                "prediction": prediction,
                "expected": expected,
                "system_prompt_len": len(system_prompt),
                "policy_model": POLICY_MODEL,
            },
        },
        "summary": {
            "outcome_reward": reward,
            "prediction": prediction,
            "expected": expected,
        },
        "usage": {**usage, "cost_usd": 0.0},
        "trace": {
            "event_history": [
                {"type": "input", "text": row.get("text")},
                {"type": "prediction", "label": prediction},
            ],
            "metadata": {"label": expected},
        },
        "metadata": {"candidate": candidate},
        "created_at": now,
        "updated_at": now,
    }


async def _execute_rollout_payload_with_timeout(payload: dict[str, Any]) -> dict[str, Any]:
    row = payload.get("dataset_row") if isinstance(payload.get("dataset_row"), dict) else None
    example_id = str((row or {}).get("example_id") or payload.get("trace_correlation_id") or "-")
    try:
        return await asyncio.wait_for(
            _execute_rollout_payload(payload),
            timeout=ROLLOUT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                f"rollout request timed out after {ROLLOUT_TIMEOUT_SECONDS:.1f}s "
                f"for example_id={example_id}; policy_model={POLICY_MODEL} "
                f"api_mode={POLICY_API_MODE} policy_timeout={POLICY_TIMEOUT_SECONDS:.1f}s "
                f"policy_retries={POLICY_RETRIES} policy_concurrency={POLICY_CONCURRENCY}"
            ),
        ) from exc


async def _async_rollout_record(rollout_id: str) -> dict[str, Any]:
    async with _ASYNC_ROLLOUT_LOCK:
        current = _ASYNC_ROLLOUTS.get(rollout_id)
    if current is None:
        raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
    return dict(current)


async def _complete_async_rollout(rollout_id: str, payload: dict[str, Any]) -> None:
    async with _ASYNC_ROLLOUT_LOCK:
        current = _ASYNC_ROLLOUTS.get(rollout_id)
        if current is None or str(current.get("status") or "") == "cancelled":
            return
        _ASYNC_ROLLOUTS[rollout_id] = {
            **current,
            "status": "running",
            "success_status": "running",
            "status_detail": "running",
            "updated_at": _now(),
        }
    try:
        completed = await _execute_rollout_payload_with_timeout(payload)
    except Exception as exc:
        completed = {
            "rollout_id": rollout_id,
            "status": "failed",
            "success_status": "failed",
            "status_detail": str(exc),
            "task_id": TASK_ID,
            "seed": int(payload.get("seed") or 0),
            "summary": {"status_detail": str(exc)},
            "usage": {},
            "metadata": {"submission_mode": "async"},
            "created_at": _now(),
            "updated_at": _now(),
        }
    async with _ASYNC_ROLLOUT_LOCK:
        current = _ASYNC_ROLLOUTS.get(rollout_id)
        if current is None or str(current.get("status") or "") == "cancelled":
            return
        _ASYNC_ROLLOUTS[rollout_id] = completed


def _row_for_seed(*, split: str, seed: int) -> dict[str, Any]:
    rows = [row for row in ROWS if row["split"] == split]
    if not rows:
        rows = list(ROWS)
    match = next((row for row in rows if int(row["seed"]) == int(seed)), None)
    row = match or rows[int(seed) % len(rows)]
    result = dict(row)
    result.setdefault("example_id", f"{result.get('split', split)}:{result.get('seed', seed)}")
    return result


async def _predict_label(text: str, *, system_prompt: str) -> tuple[str, dict[str, int]]:
    """Call the live policy model. Returns (predicted_label, token_usage).

    Uses AsyncOpenAI + a module-level Semaphore so the container only ever
    has `POLICY_CONCURRENCY` OpenAI calls in flight at once. Lets the
    container accept any number of concurrent /rollout requests without
    overrunning OpenAI's per-key connection pool.
    """
    client = _get_openai_client()
    semaphore = _get_openai_semaphore()
    user_content = (
        f"Customer query:\n{text}\n\n"
        f"Complete allowed Banking77 label list ({len(LABELS)} labels total). "
        "Return EXACTLY one label from this list as written, no other text:\n"
        + "\n".join(f"- {label}" for label in LABELS)
    )
    # Deterministic policy: temperature=0 so identical (seed, candidate)
    # pairs produce byte-identical predictions across both stacks.
    async with semaphore:
        if _policy_prefers_chat():
            last_error: Exception | None = None
            for attempt in range(1, POLICY_RETRIES + 1):
                try:
                    extra_body = _policy_chat_extra_body()
                    request_kwargs = {
                        "model": POLICY_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        "temperature": 0,
                        "max_tokens": POLICY_MAX_TOKENS,
                    }
                    if extra_body is not None:
                        request_kwargs["extra_body"] = extra_body
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(**request_kwargs),
                        timeout=POLICY_TIMEOUT_SECONDS,
                    )
                    break
                except Exception as chat_error:
                    last_error = chat_error
                    if attempt >= POLICY_RETRIES or not _is_policy_timeout(chat_error):
                        status_code = 504 if _is_policy_timeout(chat_error) else 502
                        raise HTTPException(
                            status_code=status_code,
                            detail=(
                                f"Policy model {POLICY_MODEL!r} failed through Chat Completions API "
                                f"after {attempt}/{POLICY_RETRIES} attempts, "
                                f"timeout={POLICY_TIMEOUT_SECONDS:.1f}s."
                            ),
                        ) from chat_error
                    await asyncio.sleep(_policy_retry_delay(attempt))
            else:
                raise HTTPException(
                    status_code=504,
                    detail=f"Policy model {POLICY_MODEL!r} failed: {last_error!r}",
                )
            raw = (resp.choices[0].message.content or "").strip()
            usage = {
                "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
            }
            return _normalize_policy_label(raw), usage
        try:
            resp = await asyncio.wait_for(
                client.responses.create(
                    model=POLICY_MODEL,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0,
                ),
                timeout=POLICY_TIMEOUT_SECONDS,
            )
            raw = (resp.output_text or "").strip()
            usage = {
                "prompt_tokens": int(getattr(resp.usage, "input_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp.usage, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
            }
        except Exception as responses_error:
            if _is_policy_timeout(responses_error):
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"Policy model {POLICY_MODEL!r} timed out after "
                        f"{POLICY_TIMEOUT_SECONDS:.1f}s through Responses API."
                    ),
                ) from responses_error
            # Fallback to Chat Completions for endpoints that don't support Responses API.
            try:
                extra_body = _policy_chat_extra_body()
                request_kwargs = {
                    "model": POLICY_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0,
                    "max_tokens": POLICY_MAX_TOKENS,
                }
                if extra_body is not None:
                    request_kwargs["extra_body"] = extra_body
                resp = await asyncio.wait_for(
                    client.chat.completions.create(**request_kwargs),
                    timeout=POLICY_TIMEOUT_SECONDS,
                )
            except Exception as chat_error:
                status_code = 504 if _is_policy_timeout(chat_error) else 502
                raise HTTPException(
                    status_code=status_code,
                    detail=(
                        f"Policy model {POLICY_MODEL!r} failed through OpenAI-compatible API. "
                        f"Responses error: {responses_error!r}; chat completions error: {chat_error!r}"
                    ),
                ) from chat_error
            raw = (resp.choices[0].message.content or "").strip()
            usage = {
                "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
            }
    return _normalize_policy_label(raw), usage


def _normalize_policy_label(raw: str) -> str:
    # Normalize: map common exact/case/punctuation variants back to the canonical
    # dataset label so the scorer honors labels like Refund_not_showing_up.
    candidate = raw.strip().strip("`'\"").splitlines()[0].strip()
    if candidate in LABELS:
        return candidate
    lowered = candidate.lower()
    if lowered in _LABEL_BY_LOWER:
        return _LABEL_BY_LOWER[lowered]
    simplified = "".join(ch for ch in lowered if ch.isalnum() or ch == "_")
    if simplified in _LABEL_BY_SIMPLIFIED:
        return _LABEL_BY_SIMPLIFIED[simplified]
    for label in LABELS:
        if label.lower() in lowered:
            return label
        label_simplified = "".join(ch for ch in label.lower() if ch.isalnum() or ch == "_")
        if label_simplified and label_simplified in simplified:
            return label
    # Last-resort: no recognized label in response — return the raw first-line so
    # the scorer marks it incorrect. Optimizer sees a real "wrong" signal.
    return candidate or "<no_label>"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
