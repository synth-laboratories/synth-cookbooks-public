"""
Harvey LAB (Tax) GEPA cookbook container — public, text-in/text-out.

Mirrors `crafter_container` / `dungeongrid_container` and speaks the public
synth-optimizers GEPA contract:
  GET  /metadata
  GET  /task_info
  GET  /program
  GET  /dataset
  POST /dataset/rows
  POST /rollout

Each rollout: the candidate's `system_prompt` (the legal-associate guidance GEPA
optimizes) drives an LLM that reads a Tax matter's instructions + document text
and produces a written work product. An LLM **rubric judge** then scores every
atomic PASS/FAIL criterion from Harvey LAB's rubric. Reward = fraction of
criteria passed (no string matching, no fixture).

This is a disclosed simplification of the full LAB agentic task: documents are
provided as text in context rather than navigated in a sandboxed file system,
and the work product is text (no .docx/pandoc). Dataset = Harvey AI's MIT
Legal Agent Benchmark; run `prepare_dataset.py` first.

Required env:
  OPENAI_API_KEY            — required.
  OPENAI_BASE_URL           — optional (route to a different OpenAI-compatible gateway).
  OPENROUTER_API_KEY        — optional fallback when OPENAI_BASE_URL routes to OpenRouter.
  HARVEY_LAB_POLICY_MODEL   — default: gpt-4.1-nano
  HARVEY_LAB_JUDGE_MODEL    — default: gpt-4.1-mini
  HARVEY_LAB_MAX_DOC_CHARS  — default: 9000  (per-document context budget)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request

try:
    from synth_containers import GEPA_OPTIMIZER_CONTRACT_VERSION
except Exception:
    GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"
else:
    GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"

try:
    from openai import OpenAI
except Exception as _openai_err:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = _openai_err
else:
    _OPENAI_IMPORT_ERROR = None


TASK_ID = "harvey_lab.tax_legal_agent"
DATASET_ID = "harvey_lab_tax"
DATA_PATH = Path(__file__).resolve().parent / "data" / "harvey_lab_tax_tasks.jsonl"

POLICY_MODEL = os.environ.get("HARVEY_LAB_POLICY_MODEL", "gpt-4.1-nano")
JUDGE_MODEL = os.environ.get("HARVEY_LAB_JUDGE_MODEL", "gpt-4.1-mini")
MAX_DOC_CHARS = int(os.environ.get("HARVEY_LAB_MAX_DOC_CHARS", "9000"))

DEFAULT_SYSTEM_PROMPT = (
    "You are a meticulous law-firm associate. You are given a client matter with a "
    "document data room and a partner's instructions. Read the relevant documents, "
    "do the legal analysis, and produce a precise, well-supported work product. "
    "Cite specific documents and facts; do not invent authorities."
)


# --- dataset (lazy) -----------------------------------------------------------

_rows_cache: list[dict[str, Any]] | None = None


def _load_rows() -> list[dict[str, Any]]:
    global _rows_cache
    if _rows_cache is not None:
        return _rows_cache
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"dataset bundle missing at {DATA_PATH}. Run prepare_dataset.py first "
                "(clones the MIT Harvey LAB benchmark and writes the Tax bundle)."
            ),
        )
    rows = [json.loads(line) for line in DATA_PATH.read_text().splitlines() if line.strip()]
    _rows_cache = rows
    return rows


def _split_rows(split: str) -> list[dict[str, Any]]:
    normalized = "heldout" if split in {"heldout", "test", "validation", "val"} else "train"
    rows = [r for r in _load_rows() if r.get("split") == normalized]
    return rows or _load_rows()


def _row_for_seed(*, split: str, seed: int) -> dict[str, Any]:
    rows = _split_rows(split)
    row = rows[int(seed) % len(rows)]
    return {
        "seed": int(seed),
        "split": "heldout" if split in {"heldout", "test", "validation", "val"} else "train",
        "example_id": row["task_id"],
        "task": row,
    }


# --- OpenAI client (lazy) -----------------------------------------------------

_openai_client: Any = None


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    if OpenAI is None:
        raise HTTPException(
            status_code=503,
            detail=f"openai package not installed. Import error: {_OPENAI_IMPORT_ERROR!r}",
        )
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if base_url and "openrouter.ai" in base_url and os.environ.get("OPENROUTER_API_KEY"):
        api_key = os.environ["OPENROUTER_API_KEY"]
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not set in container env.")
    _openai_client = OpenAI(api_key=api_key, base_url=base_url)
    return _openai_client


# --- agent + rubric judge -----------------------------------------------------


def _documents_block(task: dict[str, Any]) -> str:
    parts = []
    for doc in task.get("documents", []):
        text = str(doc.get("text", ""))[:MAX_DOC_CHARS]
        parts.append(f"=== DOCUMENT: {doc.get('name', '?')} ===\n{text}")
    return "\n\n".join(parts) if parts else "(no documents provided)"


def _produce_work_product(client: Any, system_prompt: str, task: dict[str, Any]) -> tuple[str, dict[str, int]]:
    user = (
        f"MATTER: {task.get('title', '')}\n\n"
        f"PARTNER INSTRUCTIONS:\n{task.get('instructions', '')}\n\n"
        f"DOCUMENT DATA ROOM:\n{_documents_block(task)}\n\n"
        "Produce the requested work product now."
    )
    resp = client.chat.completions.create(
        model=POLICY_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content or ""
    usage = {
        "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
    }
    return text, usage


def _judge_criteria(
    client: Any, task: dict[str, Any], work_product: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One judge call returns a pass/fail verdict for every rubric criterion."""
    criteria = task.get("criteria", [])
    rubric = "\n".join(
        f'{i + 1}. (id={c["id"]}) {c["match_criteria"]}' for i, c in enumerate(criteria)
    )
    judge_system = (
        "You are a strict legal work-product grader. For each numbered rubric "
        "criterion, decide whether the work product fully satisfies it. Be exacting: "
        "a criterion passes only if the work product clearly meets it. Respond with "
        'JSON: {"verdicts": [{"n": <number>, "verdict": "pass"|"fail"}]}.'
    )
    judge_user = (
        f"RUBRIC CRITERIA:\n{rubric}\n\n"
        f"WORK PRODUCT TO GRADE:\n{work_product[:20000]}\n\n"
        "Return the JSON verdicts now."
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": judge_system},
            {"role": "user", "content": judge_user},
        ],
        response_format={"type": "json_object"},
    )
    usage = {
        "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
    }
    verdict_by_n: dict[int, str] = {}
    try:
        parsed = json.loads(resp.choices[0].message.content or "{}")
        for v in parsed.get("verdicts", []):
            verdict_by_n[int(v["n"])] = str(v.get("verdict", "fail")).lower()
    except Exception:
        pass
    results = [
        {
            "id": c["id"],
            "verdict": "pass" if verdict_by_n.get(i + 1) == "pass" else "fail",
        }
        for i, c in enumerate(criteria)
    ]
    return results, usage


def _run_matter(seed_split: str, seed: int, system_prompt: str) -> dict[str, Any]:
    client = _get_openai_client()
    row = _row_for_seed(split=seed_split, seed=seed)
    task = row["task"]
    work_product, policy_usage = _produce_work_product(client, system_prompt, task)
    verdicts, judge_usage = _judge_criteria(client, task, work_product)
    n = len(verdicts)
    n_pass = sum(1 for v in verdicts if v["verdict"] == "pass")
    usage = {k: policy_usage.get(k, 0) + judge_usage.get(k, 0) for k in policy_usage}
    return {
        "example_id": row["example_id"],
        "reward": float(n_pass / n) if n else 0.0,
        "n_criteria": n,
        "n_passed": n_pass,
        "all_pass": n > 0 and n_pass == n,
        "usage": usage,
    }


# --- FastAPI app --------------------------------------------------------------

app = FastAPI(title="harvey-lab-tax-gepa-container")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metadata")
@app.get("/info")
async def metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": "harvey_lab_tax_gepa_live",
            "name": "Harvey LAB Tax GEPA (live legal-agent task, LLM policy + rubric judge)",
            "description": "Public ReAct/work-product cookbook running the MIT Harvey LAB Tax benchmark with an LLM associate and an LLM rubric judge.",
        },
        "capabilities": {
            "contract_version": "container_contract.v1",
            "rollout_modes": ["blocking"],
            "metadata": {
                "policy_ready": True,
                "trace_schema": "prompt_calls.llm_request.messages.v1",
            },
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
    return {
        "task": {
            "task_id": TASK_ID,
            "name": "Harvey LAB Tax legal associate",
            "description": (
                "Optimize a legal-associate system prompt for an LLM that reads a Tax "
                "matter's instructions + documents and writes a work product, scored "
                "against Harvey LAB's atomic PASS/FAIL rubric criteria."
            ),
            "objective": "Maximize the fraction of rubric criteria passed.",
            "domain": "long-horizon legal work-product generation (Tax), simplified to text-in/text-out",
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "visible_splits": ["train", "test"],
            "default_split": "train",
            "seed_semantics": "Seed indexes into the split's task list (Harvey LAB Tax tasks).",
        },
        "prompt_program": {
            "mutable_modules": ["system_prompt"],
            "candidate_field": "system_prompt",
            "output_contract": "Policy returns the written work product as text.",
        },
        "evaluation": {
            "primary_metric": "fraction_criteria_passed",
            "success_status": "succeeded when reward is positive",
            "rollout_trace_contains": ["work_product_complete", "n_criteria", "n_passed"],
        },
        "proposal_guidance": {
            "premises": [
                "The associate sees partner instructions and document text, and writes one work product.",
                "Each task is graded by many atomic PASS/FAIL rubric criteria; reward is the fraction passed.",
            ],
            "constraints": [
                "Keep the system prompt model-agnostic and concise enough to run every rollout.",
                "Do not encode task-specific answers; optimize general legal-reasoning guidance.",
            ],
            "high_leverage_heuristics": [
                "Instruct the model to ground every claim in named documents and quoted facts.",
                "Require it to address each part of the partner instructions explicitly.",
                "Discourage invented authorities; prefer precise, well-supported analysis.",
            ],
        },
        "metadata": {
            "policy_model": POLICY_MODEL,
            "judge_model": JUDGE_MODEL,
            "trace_schema": "prompt_calls.llm_request.messages.v1",
        },
    }


@app.get("/program")
async def program() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "harvey_lab_tax_prompt_gepa",
        "modules": [
            {
                "module_id": "system_prompt",
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
                "mutable": True,
                "candidate_field": "system_prompt",
                "template_variables": [],
                "metadata": {"surface": "system_prompt"},
            }
        ],
        "target_modules": [
            {
                "module_id": "system_prompt",
                "candidate_field": "system_prompt",
                "objective": "fraction_criteria_passed",
            }
        ],
        "seed_candidate": {"system_prompt": DEFAULT_SYSTEM_PROMPT},
        "rollout_overlay_schema": {"candidate_fields": ["system_prompt"]},
        "metadata": {"task_id": TASK_ID, "dataset_id": DATASET_ID},
    }


@app.get("/dataset")
async def dataset() -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "splits": {
            "train": len(_split_rows("train")),
            "test": len(_split_rows("test")),
        },
        "source": "harveyai/harvey-labs (MIT) — Tax practice area",
    }


@app.post("/dataset/rows")
async def dataset_rows(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    seeds = [int(seed) for seed in payload.get("seeds") or []]
    rows = []
    for seed in seeds:
        r = _row_for_seed(split=split, seed=seed)
        rows.append({"seed": r["seed"], "split": r["split"], "example_id": r["example_id"]})
    return {"rows": rows}


@app.get("/taskset")
async def taskset() -> dict[str, Any]:
    return {
        "taskset_id": DATASET_ID,
        "name": "Harvey LAB Tax",
        "splits": {
            "train": {"num_tasks": len(_split_rows("train"))},
            "test": {"num_tasks": len(_split_rows("test"))},
        },
    }


def _seed_from_task_id(task_id: Any) -> int:
    return int(str(task_id).rsplit(":", 1)[-1])


def _task_for_id(split: str, task_id: Any) -> dict[str, Any]:
    seed = _seed_from_task_id(task_id)
    row = _row_for_seed(split=split, seed=seed)
    return {
        "task_id": str(task_id),
        "seed": row["seed"],
        "split": row["split"],
        "example_id": row["example_id"],
    }


@app.post("/taskset/tasks")
async def taskset_tasks(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    task_ids = payload.get("task_ids") or []
    tasks = [_task_for_id(split, task_id) for task_id in task_ids]
    return {"tasks": tasks, "metadata": {"split": split, "count": len(tasks)}}


@app.post("/rollout")
@app.post("/rollouts")
def rollout(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    split = str(payload.get("split") or "train")
    seed = int(payload.get("seed") or 0)
    if isinstance(payload.get("task"), dict):
        task = payload["task"]
        split = str(task.get("split") or split)
        seed = int(task.get("seed") or seed)
    if isinstance(payload.get("dataset_row"), dict):
        dr = payload["dataset_row"]
        split = str(dr.get("split") or split)
        seed = int(dr.get("seed") or seed)
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    system_prompt = str(candidate.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)

    result = _run_matter(split, seed, system_prompt)
    reward = float(result["reward"])

    rollout_id = str(payload.get("rollout_id") or f"rollout_{uuid.uuid4().hex[:12]}")
    now = _now()
    return {
        "rollout_id": rollout_id,
        "status": "completed",
        "success_status": "succeeded" if reward > 0 else "failed",
        "task_id": TASK_ID,
        "seed": seed,
        "reward_info": {
            "outcome_reward": reward,
            "event_rewards": [reward],
            "details": {
                "example_id": result["example_id"],
                "n_criteria": result["n_criteria"],
                "n_passed": result["n_passed"],
                "all_pass": result["all_pass"],
                "policy_model": POLICY_MODEL,
                "judge_model": JUDGE_MODEL,
            },
        },
        "summary": {
            "outcome_reward": reward,
            "example_id": result["example_id"],
            "n_criteria": result["n_criteria"],
            "n_passed": result["n_passed"],
        },
        "usage": {**result["usage"], "cost_usd": 0.0},
        "trace": {
            "event_history": [
                {
                    "type": "work_product_complete",
                    "example_id": result["example_id"],
                    "reward": reward,
                    "n_criteria": result["n_criteria"],
                    "n_passed": result["n_passed"],
                }
            ],
            "metadata": {"example_id": result["example_id"], "call_site_id": TASK_ID},
        },
        "metadata": {"candidate": candidate},
        "created_at": now,
        "updated_at": now,
    }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
