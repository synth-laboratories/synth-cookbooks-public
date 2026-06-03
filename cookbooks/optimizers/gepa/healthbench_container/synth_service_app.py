"""
HealthBench Professional GEPA cookbook container (live OpenAI policy + rubric judge).

Speaks the public synth-optimizers GEPA contract:
  GET  /health
  GET  /metadata   (also /info)
  GET  /task_info
  GET  /program
  GET  /dataset
  POST /dataset/rows
  POST /rollout    (also /rollouts)

Each rollout builds a system prompt from the candidate's `stage1_system`, calls
the policy model with the clinical conversation context, then scores the response
against the per-row HealthBench rubric criteria using a lightweight judge.

Dataset: openai/healthbench-professional (HuggingFace).
If only a single HF split exists the container creates synthetic train/test
slices: first 1000 rows → train, remainder → test.

Required env:
  OPENAI_API_KEY                    — required for policy and judge calls.
  HEALTHBENCH_POLICY_MODEL          — default: gpt-4.1-nano
  HEALTHBENCH_POLICY_BASE_URL       — optional (OpenRouter, etc.)
  HEALTHBENCH_POLICY_API_KEY_ENV    — default: OPENAI_API_KEY
  HEALTHBENCH_JUDGE_MODEL           — default: same as policy model
  HEALTHBENCH_POLICY_TIMEOUT_SECONDS — default: 30
  HEALTHBENCH_POLICY_CONCURRENCY    — default: 30
  HEALTHBENCH_POLICY_MAX_TOKENS     — default: 512
  HEALTHBENCH_JUDGE_MAX_TOKENS      — default: 4
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request

try:
    from openai import AsyncOpenAI
except Exception as _openai_err:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment,misc]
    _OPENAI_IMPORT_ERROR = _openai_err
else:
    _OPENAI_IMPORT_ERROR = None

GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TASK_ID = "healthbench.medical_qa"
DATASET_ID = "openai:healthbench-professional"
HF_DATASET_NAME = "openai/healthbench-professional"

POLICY_MODEL = os.environ.get("HEALTHBENCH_POLICY_MODEL", "gpt-4.1-nano")
POLICY_BASE_URL = os.environ.get("HEALTHBENCH_POLICY_BASE_URL") or None
POLICY_API_KEY_ENV = os.environ.get("HEALTHBENCH_POLICY_API_KEY_ENV", "OPENAI_API_KEY")
JUDGE_MODEL = os.environ.get("HEALTHBENCH_JUDGE_MODEL") or POLICY_MODEL
POLICY_TIMEOUT_SECONDS = float(os.environ.get("HEALTHBENCH_POLICY_TIMEOUT_SECONDS", "30"))
POLICY_CONCURRENCY = int(os.environ.get("HEALTHBENCH_POLICY_CONCURRENCY", "30"))
POLICY_MAX_TOKENS = int(os.environ.get("HEALTHBENCH_POLICY_MAX_TOKENS", "512"))
JUDGE_MAX_TOKENS = int(os.environ.get("HEALTHBENCH_JUDGE_MAX_TOKENS", "4"))

ROLLOUT_TIMEOUT_SECONDS = POLICY_TIMEOUT_SECONDS + 60.0  # extra budget for judge calls

# Synthetic split size when the dataset has only one HF split.
TRAIN_CAP = int(os.environ.get("HEALTHBENCH_TRAIN_CAP", "100"))
DEFAULT_PORT = 8814

DEFAULT_STAGE1_SYSTEM = (
    "You are an expert physician. Answer the clinical question directly and accurately. "
    "Provide a clear, evidence-based response that addresses all aspects of the question. "
    "Include relevant considerations for diagnosis, treatment, or clinical management as "
    "appropriate. Be concise and precise."
)

HEALTHBENCH_PROPOSER_HINTS = {
    "task_output_space": "open_medical_qa",
    "literal_training_targets": "forbid",
    "proposal_goal": (
        "Infer reusable clinical reasoning principles from rollout traces and rubric scores. "
        "The rubric criteria cover accuracy, safety, completeness, and clinical reasoning. "
        "Do not memorize specific patient cases; generalize to clinical reasoning principles "
        "that improve rubric coverage across unseen questions."
    ),
    "trace_review": [
        "Inspect rubric_score, criteria met/missed, and the physician response for each rollout.",
        "Identify patterns in missed criteria: accuracy errors, missing safety caveats, incomplete differentials, poor structure.",
        "Propose system prompt edits that improve rubric coverage by teaching the model better clinical reasoning, not by memorizing examples.",
    ],
    "rubric_note": (
        "Each row has a different rubric. Criteria are scored by a judge LLM per criterion. "
        "Improving clarity, completeness, and safety coverage raises rubric_score across diverse questions."
    ),
    "constraints": [
        "Do not reference or echo specific patient details from train rollouts.",
        "Do not add fabricated citations or drug dosages not grounded in the response.",
        "Keep the system prompt clinically precise but model-agnostic.",
    ],
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_dataset_lock = asyncio.Lock()
_train_rows: list[dict[str, Any]] | None = None
_test_rows: list[dict[str, Any]] | None = None


def _extract_rows_from_split(
    split_data: Any,
    split_name: str,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    """Convert HF split rows to our internal format."""
    rows: list[dict[str, Any]] = []
    for i, example in enumerate(split_data):
        prompt_messages: list[dict[str, str]] = list(
            example.get("prompt_messages")
            or (example.get("conversation") or {}).get("messages")
            or []
        )
        criteria_raw: list[dict[str, Any]] = list(
            example.get("criteria_v0.1")
            or example.get("rubric_items")
            or example.get("rubrics")
            or []
        )
        criteria = [
            {
                "criterion": str(c.get("criterion") or c.get("criterion_text") or ""),
                "points": float(c.get("points") if c.get("points") is not None else 1.0),
            }
            for c in criteria_raw
            if c.get("criterion") or c.get("criterion_text")
        ]
        # Last user message is the primary question.
        question = ""
        for msg in reversed(prompt_messages):
            if str(msg.get("role") or "") == "user":
                question = str(msg.get("content") or "")
                break
        if not question and prompt_messages:
            question = str(prompt_messages[-1].get("content") or "")

        seed = seed_offset + i
        rows.append(
            {
                "seed": seed,
                "split": split_name,
                "task_instance_id": f"healthbench:{split_name}:{i}",
                "question": question,
                "prompt_messages": prompt_messages,
                "criteria": criteria,
            }
        )
    return rows


async def _ensure_dataset_loaded() -> None:
    global _train_rows, _test_rows
    if _train_rows is not None and _test_rows is not None:
        return
    async with _dataset_lock:
        if _train_rows is not None and _test_rows is not None:
            return

        def _load() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            from datasets import load_dataset  # type: ignore[import]

            ds = load_dataset(HF_DATASET_NAME)
            split_names = list(ds.keys())

            if "train" in split_names and "test" in split_names:
                train = _extract_rows_from_split(ds["train"], "train", seed_offset=0)
                test = _extract_rows_from_split(ds["test"], "test", seed_offset=0)
                return train, test

            if "train" in split_names and len(split_names) == 1:
                all_rows = _extract_rows_from_split(ds["train"], "train", seed_offset=0)
            else:
                # Use whatever the first split is.
                first_key = split_names[0]
                all_rows = _extract_rows_from_split(ds[first_key], "train", seed_offset=0)

            split_idx = min(TRAIN_CAP, len(all_rows))
            train_raw = all_rows[:split_idx]
            test_raw = all_rows[split_idx:]

            # Re-tag splits and re-index seeds.
            train = [{**r, "split": "train", "seed": i} for i, r in enumerate(train_raw)]
            test = [{**r, "split": "test", "seed": i} for i, r in enumerate(test_raw)]
            return train, test

        train, test = await asyncio.to_thread(_load)
        _train_rows = train
        _test_rows = test


def _rows_for_split(split: str) -> list[dict[str, Any]]:
    normalized = "test" if split.strip().lower() in {"test", "heldout", "validation", "val"} else "train"
    return _test_rows if normalized == "test" else (_train_rows or [])


def _row_for_seed(*, split: str, seed: int) -> dict[str, Any]:
    rows = _rows_for_split(split)
    if not rows:
        rows = list(_train_rows or [])
    match = next((r for r in rows if int(r["seed"]) == int(seed)), None)
    row = match or rows[int(seed) % len(rows)]
    result = dict(row)
    result.setdefault("example_id", f"{result.get('split', split)}:{result.get('seed', seed)}")
    return result


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

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
                f"Import error: {_OPENAI_IMPORT_ERROR!r}"
            ),
        )
    api_key = os.environ.get(POLICY_API_KEY_ENV, "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=f"{POLICY_API_KEY_ENV} is not set; cannot serve live rollouts.",
        )
    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": POLICY_TIMEOUT_SECONDS}
    if POLICY_BASE_URL:
        client_kwargs["base_url"] = POLICY_BASE_URL
    _openai_client = AsyncOpenAI(**client_kwargs)
    return _openai_client


def _get_semaphore() -> asyncio.Semaphore:
    global _openai_semaphore
    if _openai_semaphore is None:
        _openai_semaphore = asyncio.Semaphore(max(1, POLICY_CONCURRENCY))
    return _openai_semaphore


# ---------------------------------------------------------------------------
# Policy call
# ---------------------------------------------------------------------------

def _build_policy_messages(
    system_prompt: str,
    prompt_messages: list[dict[str, str]],
    question: str,
) -> list[dict[str, str]]:
    """Build the message list for the policy call.

    Includes conversation history (all messages except the final user turn which
    we already captured as `question`) then appends a fresh user message.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Include full conversation history if there are multiple turns.
    if len(prompt_messages) > 1:
        # Omit the last message — we will re-add it below as the question.
        for msg in prompt_messages[:-1]:
            role = str(msg.get("role") or "user")
            content = str(msg.get("content") or "")
            if role in {"user", "assistant", "system"} and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})
    return messages


async def _call_policy(
    system_prompt: str,
    prompt_messages: list[dict[str, str]],
    question: str,
) -> tuple[str, dict[str, int]]:
    """Call the policy model. Returns (response_text, token_usage)."""
    client = _get_openai_client()
    semaphore = _get_semaphore()
    messages = _build_policy_messages(system_prompt, prompt_messages, question)
    async with semaphore:
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=POLICY_MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=POLICY_MAX_TOKENS,
                ),
                timeout=POLICY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail=f"Policy model {POLICY_MODEL!r} timed out after {POLICY_TIMEOUT_SECONDS:.1f}s.",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Policy model {POLICY_MODEL!r} failed: {exc!r}",
            ) from exc
    content = (resp.choices[0].message.content or "").strip() if resp.choices else ""
    usage = {
        "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
    }
    return content, usage


# ---------------------------------------------------------------------------
# Rubric judge
# ---------------------------------------------------------------------------

async def _judge_criterion(
    client: Any,
    response_text: str,
    criterion: str,
) -> tuple[bool, dict[str, int]]:
    """Ask the judge whether the response satisfies a single criterion.

    Returns True if met, False otherwise. Uses a minimal yes/no prompt to
    keep judge token cost proportional to JUDGE_MAX_TOKENS.
    """
    system = (
        "You are a medical response evaluator. "
        "Judge whether the criterion is met, even when the criterion describes undesirable behavior. "
        "Answer only YES or NO — no other text."
    )
    user = (
        f"Physician response:\n{response_text}\n\n"
        f"Criterion: {criterion}\n\n"
        "Does the physician response satisfy this criterion? Answer YES or NO."
    )
    semaphore = _get_semaphore()
    async with semaphore:
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=JUDGE_MAX_TOKENS,
                ),
                timeout=POLICY_TIMEOUT_SECONDS,
            )
        except Exception:
            # Judge failure -> conservative miss (counts as 0 for this criterion).
            return False, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    raw = (resp.choices[0].message.content or "").strip().upper()
    usage = {
        "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
    }
    return raw.startswith("Y"), usage


async def _score_rubric(
    response_text: str,
    criteria: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]], dict[str, int]]:
    """Score the response against all rubric criteria.

    Returns (rubric_score, per_criterion_results, judge_token_usage).
    rubric_score follows HealthBench scoring: sum of signed points for met criteria
    divided by the maximum possible positive points.
    """
    if not criteria:
        return 0.5, [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    client = _get_openai_client()
    total_possible_points = sum(
        float(c.get("points") if c.get("points") is not None else 1.0)
        for c in criteria
        if float(c.get("points") if c.get("points") is not None else 1.0) > 0
    )
    if total_possible_points <= 0:
        return 0.5, [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Fan out judge calls concurrently.
    tasks = [
        _judge_criterion(client, response_text, str(c.get("criterion") or ""))
        for c in criteria
    ]
    results = await asyncio.gather(*tasks)

    earned = 0.0
    per_criterion: list[dict[str, Any]] = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for criterion_dict, (met, usage) in zip(criteria, results):
        usage_totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        usage_totals["completion_tokens"] += int(usage.get("completion_tokens", 0))
        usage_totals["total_tokens"] += int(usage.get("total_tokens", 0))
        pts = float(
            criterion_dict.get("points")
            if criterion_dict.get("points") is not None
            else 1.0
        )
        per_criterion.append(
            {
                "criterion": criterion_dict.get("criterion"),
                "points": pts,
                "met": bool(met),
                "earned": pts if met else 0.0,
            }
        )
        if met:
            earned += pts

    rubric_score = earned / total_possible_points
    return rubric_score, per_criterion, usage_totals


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="healthbench-gepa-container")
_ASYNC_ROLLOUTS: dict[str, dict[str, Any]] = {}
_ASYNC_ROLLOUT_LOCK = asyncio.Lock()
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metadata")
@app.get("/info")
async def metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": "healthbench_gepa_live",
            "name": "HealthBench Professional GEPA (live OpenAI policy + rubric judge)",
            "description": (
                "Public prompt-optimizer cookbook for HealthBench Professional medical QA. "
                "Each rollout calls a live physician-role policy model and scores the response "
                "against the per-row HealthBench rubric criteria using a lightweight judge LLM."
            ),
        },
        "capabilities": {
            "contract_version": "container_contract.v1",
            "rollout_modes": ["blocking", "async"],
            "metadata": {"policy_ready": True},
        },
        "metadata": {
            "optimizer_contracts": {
                "gepa": {
                    "version": GEPA_OPTIMIZER_CONTRACT_VERSION,
                    "program_route": "/program",
                    "taskset_route": "/taskset",
                    "taskset_tasks_route": "/taskset/tasks",
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
    train_count = len(_train_rows or [])
    test_count = len(_test_rows or [])
    return {
        "task": {
            "task_id": TASK_ID,
            "name": "HealthBench Professional medical QA",
            "description": (
                "Generate a physician-quality response to a clinical question, then score it "
                "against the per-row HealthBench rubric (accuracy, safety, completeness, "
                "clinical reasoning)."
            ),
        },
        "output_space": {
            "kind": "open_medical_qa",
            "contract": (
                "Produce a direct, evidence-based physician response to the clinical question. "
                "The response is scored by a rubric judge against per-row criteria."
            ),
            "primary_metric": "rubric_score",
            "metric_range": "HealthBench signed point fraction; aggregate score is clipped to [0.0, 1.0]",
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "hf_name": HF_DATASET_NAME,
            "visible_splits": ["train", "test"],
            "default_split": "train",
            "row_count": train_count,
            "heldout_row_count": test_count,
        },
        "proposer_hints": HEALTHBENCH_PROPOSER_HINTS,
        "metadata": {
            "primary_metric": "rubric_score",
            "policy_model": POLICY_MODEL,
            "judge_model": JUDGE_MODEL,
            "proposer_hints": HEALTHBENCH_PROPOSER_HINTS,
        },
    }


@app.get("/program")
async def program() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "healthbench_single_stage_gepa",
        "modules": [
            {
                "module_id": "stage1_system",
                "role": "system",
                "content": DEFAULT_STAGE1_SYSTEM,
                "mutable": True,
                "candidate_field": "stage1_system",
                "template_variables": [],
            }
        ],
        "target_modules": [
            {
                "module_id": "stage1_system",
                "candidate_field": "stage1_system",
                "objective": "rubric_score",
            }
        ],
        "seed_candidate": {"stage1_system": DEFAULT_STAGE1_SYSTEM},
        "rollout_overlay_schema": {"candidate_fields": ["stage1_system"]},
        "metadata": {
            "task_id": TASK_ID,
            "dataset_id": DATASET_ID,
            "primary_metric": "rubric_score",
            "proposer_hints": HEALTHBENCH_PROPOSER_HINTS,
        },
    }


@app.get("/dataset")
async def dataset() -> dict[str, Any]:
    await _ensure_dataset_loaded()
    return {
        "dataset_id": DATASET_ID,
        "splits": {
            "train": len(_train_rows or []),
            "test": len(_test_rows or []),
        },
        "hf_name": HF_DATASET_NAME,
        "split_note": (
            "If openai/healthbench-professional has no train/test split, "
            f"the first {TRAIN_CAP} rows become train and the remainder become test."
        ),
    }


@app.post("/dataset/rows")
async def dataset_rows(request: Request) -> dict[str, Any]:
    await _ensure_dataset_loaded()
    payload = await request.json()
    split = str(payload.get("split") or "train")
    seeds = [int(s) for s in (payload.get("seeds") or [])]
    rows = [_row_for_seed(split=split, seed=seed) for seed in seeds]
    return {"rows": rows}


@app.get("/taskset")
async def taskset() -> dict[str, Any]:
    await _ensure_dataset_loaded()
    return {
        "taskset_id": "healthbench_professional",
        "splits": {
            "train": {"num_tasks": len(_train_rows or [])},
            "test": {"num_tasks": len(_test_rows or [])},
        },
        "source": HF_DATASET_NAME,
        "metadata": {
            "primary_metric": "rubric_score",
            "metric_range": "HealthBench signed point fraction; aggregate score is clipped to [0.0, 1.0]",
            "policy_model": POLICY_MODEL,
            "judge_model": JUDGE_MODEL,
            "proposer_hints": HEALTHBENCH_PROPOSER_HINTS,
        },
    }


def _seed_from_task_id(task_id: Any) -> int:
    return int(str(task_id).rsplit(":", 1)[-1])


def _task_for_id(split: str, task_id: Any) -> dict[str, Any]:
    seed = _seed_from_task_id(task_id)
    row = _row_for_seed(split=split, seed=seed)
    return {**row, "task_id": str(task_id)}


@app.post("/taskset/tasks")
async def taskset_tasks(request: Request) -> dict[str, Any]:
    await _ensure_dataset_loaded()
    payload = await request.json()
    split = str(payload.get("split") or "train")
    task_ids = payload.get("task_ids") or []
    tasks = [_task_for_id(split, tid) for tid in task_ids]
    return {"tasks": tasks, "metadata": {"split": split, "count": len(tasks)}}


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
    queued: dict[str, Any] = {
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
async def rollout_record_route(rollout_id: str) -> dict[str, Any]:
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
        if str(current.get("status") or "") not in _TERMINAL_STATUSES:
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


# ---------------------------------------------------------------------------
# Rollout execution
# ---------------------------------------------------------------------------

async def _execute_rollout_payload(payload: dict[str, Any]) -> dict[str, Any]:
    await _ensure_dataset_loaded()
    row = None
    for key in ("task", "dataset_row"):
        value = payload.get(key)
        if isinstance(value, dict) and value.get("question") is not None:
            row = value
            break
    if row is None:
        task = payload.get("task") if isinstance(payload.get("task"), dict) else {}
        row = _row_for_seed(
            split=str(payload.get("split") or task.get("split") or "train"),
            seed=int(payload.get("seed") or task.get("seed") or 0),
        )
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    system_prompt = str(candidate.get("stage1_system") or DEFAULT_STAGE1_SYSTEM)

    prompt_messages: list[dict[str, str]] = list(row.get("prompt_messages") or [])
    question = str(row.get("question") or "")
    criteria: list[dict[str, Any]] = list(row.get("criteria") or [])

    # Step 1: call policy model.
    response_text, policy_usage = await _call_policy(system_prompt, prompt_messages, question)

    # Step 2: score against rubric.
    rubric_score, per_criterion, judge_usage = await _score_rubric(response_text, criteria)

    rollout_id = str(payload.get("rollout_id") or f"rollout_{uuid.uuid4().hex[:12]}")
    now = _now()
    combined_usage = {
        "prompt_tokens": policy_usage["prompt_tokens"] + judge_usage["prompt_tokens"],
        "completion_tokens": policy_usage["completion_tokens"] + judge_usage["completion_tokens"],
        "total_tokens": policy_usage["total_tokens"] + judge_usage["total_tokens"],
        "cost_usd": 0.0,
    }

    return {
        "rollout_id": rollout_id,
        "status": "completed",
        "success_status": "succeeded" if rubric_score > 0.0 else "failed",
        "task_id": TASK_ID,
        "seed": int(row.get("seed") or 0),
        "reward_info": {
            "outcome_reward": rubric_score,
            "event_rewards": [rubric_score],
            "details": {
                "rubric_score": rubric_score,
                "criteria_count": len(criteria),
                "criteria_met": sum(1 for c in per_criterion if c.get("met")),
                "policy_model": POLICY_MODEL,
                "judge_model": JUDGE_MODEL,
                "response_length": len(response_text),
            },
        },
        "summary": {
            "outcome_reward": rubric_score,
            "rubric_score": rubric_score,
            "criteria_count": len(criteria),
            "criteria_met": sum(1 for c in per_criterion if c.get("met")),
            "question_snippet": question[:120],
        },
        "usage": combined_usage,
        "trace": {
            "event_history": [
                {"type": "question", "text": question},
                {"type": "policy_response", "text": response_text},
                {"type": "rubric_score", "score": rubric_score, "criteria": per_criterion},
            ],
            "metadata": {
                "task_instance_id": row.get("task_instance_id"),
                "split": row.get("split"),
                "seed": row.get("seed"),
            },
        },
        "metadata": {
            "candidate": candidate,
            "question": question,
            "response": response_text,
            "per_criterion": per_criterion,
        },
        "created_at": now,
        "updated_at": now,
    }


async def _execute_rollout_payload_with_timeout(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(
            _execute_rollout_payload(payload),
            timeout=ROLLOUT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                f"rollout timed out after {ROLLOUT_TIMEOUT_SECONDS:.1f}s; "
                f"policy_model={POLICY_MODEL} judge_model={JUDGE_MODEL} "
                f"policy_timeout={POLICY_TIMEOUT_SECONDS:.1f}s"
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
            "metadata": {"submission_mode": "async", "error": str(exc)},
            "created_at": _now(),
            "updated_at": _now(),
        }
    async with _ASYNC_ROLLOUT_LOCK:
        current = _ASYNC_ROLLOUTS.get(rollout_id)
        if current is None or str(current.get("status") or "") == "cancelled":
            return
        _ASYNC_ROLLOUTS[rollout_id] = completed


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    parser = argparse.ArgumentParser(description="HealthBench Professional GEPA container")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", str(DEFAULT_PORT))))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
