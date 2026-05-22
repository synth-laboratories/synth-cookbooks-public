from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request

# Live OpenAI policy. The container will not start without OPENAI_API_KEY
# unless a non-default policy is wired in at runtime.
try:
    from openai import AsyncOpenAI
except Exception as _openai_err:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = _openai_err
else:
    _OPENAI_IMPORT_ERROR = None

POLICY_MODEL = os.environ.get("BANKING77_POLICY_MODEL", "gpt-4.1-nano")

# Cap concurrent OpenAI calls regardless of how many incoming /rollout
# requests the container accepts. Decouples client-side fan-out from the
# API connection pool so a burst of 32 client requests doesn't blow up
# OpenAI's per-key connection budget.
POLICY_CONCURRENCY = int(os.environ.get("BANKING77_POLICY_CONCURRENCY", "16"))
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
    if "OPENAI_API_KEY" not in os.environ:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY not set in container env; cannot serve live rollouts.",
        )
    _openai_client = AsyncOpenAI()
    return _openai_client


def _get_openai_semaphore() -> asyncio.Semaphore:
    """Lazy semaphore creation so it binds to the running event loop."""
    global _openai_semaphore
    if _openai_semaphore is None:
        _openai_semaphore = asyncio.Semaphore(POLICY_CONCURRENCY)
    return _openai_semaphore

try:
    from synth_containers import GEPA_OPTIMIZER_CONTRACT_VERSION
except Exception:
    GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v1"


TASK_ID = "banking77.intent_classification"

# Sample sizes: take first N from each split deterministically so seed indices
# are stable across container restarts. Override via env if you need more.
TRAIN_SAMPLE = int(os.environ.get("BANKING77_TRAIN_SAMPLE", "24"))
TEST_SAMPLE = int(os.environ.get("BANKING77_TEST_SAMPLE", "200"))


def _load_banking77_rows() -> tuple[list[str], list[dict[str, Any]]]:
    """Load real PolyAI/banking77 train+test slices.

    Returns (LABELS, ROWS) where ROWS is a flat list with `seed` indexing
    into the original split: train[i] => seed=i, split="train";
    test[i] => seed=i, split="test". This makes the toml configurable via
    `train_seeds = [0,1,2,...]` and `heldout_seeds = [0,1,2,...]`.
    """
    from datasets import load_dataset
    ds = load_dataset("PolyAI/banking77", trust_remote_code=True)
    label_names: list[str] = list(ds["train"].features["label"].names)
    rows: list[dict[str, Any]] = []
    for i in range(min(TRAIN_SAMPLE, len(ds["train"]))):
        ex = ds["train"][i]
        rows.append({
            "seed": i,
            "split": "train",
            "text": str(ex["text"]),
            "label": label_names[int(ex["label"])],
        })
    for i in range(min(TEST_SAMPLE, len(ds["test"]))):
        ex = ds["test"][i]
        rows.append({
            "seed": i,
            "split": "test",
            "text": str(ex["text"]),
            "label": label_names[int(ex["label"])],
        })
    return label_names, rows


LABELS, ROWS = _load_banking77_rows()

DEFAULT_STAGE2_SYSTEM = (
    "Classify the customer banking query into exactly one Banking77 intent. "
    "Return the answer only with the best single label."
)


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
            "name": "Banking77 GEPA (live OpenAI policy)",
            "description": "Public prompt-optimizer cookbook for Banking77 with a live OpenAI policy model.",
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
        "dataset": {
            "dataset_id": "banking77_public_rows",
            "visible_splits": ["train", "test"],
            "default_split": "train",
            "row_count": len(ROWS),
        },
        "metadata": {},
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
        "labels": LABELS,
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
        return await _execute_rollout_payload(payload)
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
        completed = await _execute_rollout_payload(payload)
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
        f"Allowed Banking77 labels (return EXACTLY one, lowercase snake_case, no other text):\n"
        + "\n".join(f"- {label}" for label in LABELS)
    )
    # Deterministic policy: temperature=0 so identical (seed, candidate)
    # pairs produce byte-identical predictions across both stacks.
    async with semaphore:
        try:
            resp = await client.responses.create(
                model=POLICY_MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
            )
            raw = (resp.output_text or "").strip()
            usage = {
                "prompt_tokens": int(getattr(resp.usage, "input_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp.usage, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
            }
        except Exception:
            # Fallback to Chat Completions for endpoints that don't support Responses API.
            resp = await client.chat.completions.create(
                model=POLICY_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            usage = {
                "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
            }
    # Normalize: pick the first token-like substring that matches a known label.
    candidate = raw.strip().strip("`'\"").splitlines()[0].strip()
    if candidate in LABELS:
        return candidate, usage
    lowered = candidate.lower()
    for label in LABELS:
        if label in lowered:
            return label, usage
    # Last-resort: no recognized label in response — return the raw first-line so
    # the scorer marks it incorrect. Optimizer sees a real "wrong" signal.
    return candidate or "<no_label>", usage


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
