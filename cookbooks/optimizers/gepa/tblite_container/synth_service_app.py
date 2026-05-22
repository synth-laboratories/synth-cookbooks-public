"""
Terminal-Bench-Lite GEPA cookbook container (real pytest verifier, OpenAI agent).

Speaks the public synth-optimizers GEPA contract:
  GET  /metadata
  GET  /task_info
  GET  /program
  GET  /dataset
  POST /dataset/rows
  POST /rollout

Each rollout:
  1. Picks a coding task by seed (function spec + hidden tests).
  2. Calls OpenAI with the candidate's `starting_prompt` as system, the
     task spec as user, asking for the function source.
  3. Writes the response + hidden tests to a temp dir.
  4. Runs `pytest` in a subprocess; reward = fraction of tests passing.

No fixture, no string matching. Reward comes from a real pytest verdict.

Required env:
  OPENAI_API_KEY               — required.
  TBLITE_POLICY_MODEL          — default: gpt-4.1-nano
  TBLITE_TEST_TIMEOUT_SECONDS  — default: 30 (pytest subprocess hard cap)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request

try:
    from synth_containers import GEPA_OPTIMIZER_CONTRACT_VERSION
except Exception:
    GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v1"

try:
    from openai import OpenAI
except Exception as _openai_err:
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = _openai_err
else:
    _OPENAI_IMPORT_ERROR = None


TASK_ID = "tblite.python_function_impl"
DATASET_ID = "tblite_public_pytest_tasks"

POLICY_MODEL = os.environ.get("TBLITE_POLICY_MODEL", "gpt-4.1-nano")
TEST_TIMEOUT_SECONDS = int(os.environ.get("TBLITE_TEST_TIMEOUT_SECONDS", "30"))

DEFAULT_STARTING_PROMPT = (
    "You are a Python coding agent. Implement the requested function so that "
    "all of its hidden tests pass. Output ONLY the function source code, no "
    "markdown fences, no surrounding prose, no example usage. Match the "
    "function signature exactly. Handle edge cases. Use the standard library."
)


# Each row is a real, self-contained Python task: function signature + spec +
# hidden pytest. Tasks are small and independent so this container can be a
# DIY benchmark, not just a smoke harness.
ROWS = [
    {
        "seed": 0,
        "split": "train",
        "example_id": "is_palindrome",
        "signature": "def is_palindrome(s: str) -> bool:",
        "spec": (
            "Return True iff `s` reads the same forwards and backwards, "
            "ignoring case and any non-alphanumeric characters."
        ),
        "tests": (
            "from solution import is_palindrome\n"
            "def test_basic():\n"
            "    assert is_palindrome('racecar')\n"
            "    assert not is_palindrome('hello')\n"
            "def test_case_and_spaces():\n"
            "    assert is_palindrome('A man, a plan, a canal: Panama')\n"
            "    assert is_palindrome('No lemon, no melon')\n"
            "def test_empty_and_single():\n"
            "    assert is_palindrome('')\n"
            "    assert is_palindrome('a')\n"
            "def test_mixed():\n"
            "    assert not is_palindrome('abca')\n"
            "    assert is_palindrome('Was it a car or a cat I saw?')\n"
        ),
    },
    {
        "seed": 1,
        "split": "train",
        "example_id": "two_sum_indices",
        "signature": "def two_sum_indices(nums: list[int], target: int) -> list[int]:",
        "spec": (
            "Given a list of integers `nums` and a `target`, return the "
            "indices [i, j] (i < j) of the two numbers that add up to "
            "target. Assume exactly one solution exists. Return [-1, -1] "
            "only if no such pair exists."
        ),
        "tests": (
            "from solution import two_sum_indices\n"
            "def test_simple():\n"
            "    assert two_sum_indices([2, 7, 11, 15], 9) == [0, 1]\n"
            "def test_skipped_pair():\n"
            "    assert two_sum_indices([3, 2, 4], 6) == [1, 2]\n"
            "def test_negatives():\n"
            "    assert two_sum_indices([-3, 4, 3, 90], 0) == [0, 2]\n"
            "def test_no_pair():\n"
            "    assert two_sum_indices([1, 2, 3], 100) == [-1, -1]\n"
        ),
    },
    {
        "seed": 2,
        "split": "train",
        "example_id": "rle_encode",
        "signature": "def rle_encode(s: str) -> str:",
        "spec": (
            "Run-length encode `s`. Each maximal run of identical characters "
            "becomes the character followed by its count as a decimal "
            "integer. Example: 'aaabbc' -> 'a3b2c1'. Empty input returns ''."
        ),
        "tests": (
            "from solution import rle_encode\n"
            "def test_basic():\n"
            "    assert rle_encode('aaabbc') == 'a3b2c1'\n"
            "def test_singletons():\n"
            "    assert rle_encode('abc') == 'a1b1c1'\n"
            "def test_empty():\n"
            "    assert rle_encode('') == ''\n"
            "def test_long_run():\n"
            "    assert rle_encode('xxxxxxxxxx') == 'x10'\n"
            "def test_alternating():\n"
            "    assert rle_encode('ababab') == 'a1b1a1b1a1b1'\n"
        ),
    },
    {
        "seed": 100,
        "split": "test",
        "example_id": "balanced_brackets",
        "signature": "def balanced_brackets(s: str) -> bool:",
        "spec": (
            "Return True iff every opening bracket in `s` (one of (, [, {) "
            "is closed by the matching closer in the correct order. Ignore "
            "any non-bracket characters."
        ),
        "tests": (
            "from solution import balanced_brackets\n"
            "def test_basic():\n"
            "    assert balanced_brackets('()')\n"
            "    assert balanced_brackets('([{}])')\n"
            "    assert not balanced_brackets('([)]')\n"
            "def test_with_text():\n"
            "    assert balanced_brackets('a(b)c[d]')\n"
            "    assert not balanced_brackets('a(b]c')\n"
            "def test_empty_and_single():\n"
            "    assert balanced_brackets('')\n"
            "    assert not balanced_brackets('(')\n"
        ),
    },
    {
        "seed": 101,
        "split": "test",
        "example_id": "merge_sorted",
        "signature": "def merge_sorted(a: list[int], b: list[int]) -> list[int]:",
        "spec": (
            "Merge two ascending-sorted lists into a single ascending-sorted "
            "list in O(len(a) + len(b)). Both inputs are pre-sorted."
        ),
        "tests": (
            "from solution import merge_sorted\n"
            "def test_basic():\n"
            "    assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\n"
            "def test_empty():\n"
            "    assert merge_sorted([], [1, 2, 3]) == [1, 2, 3]\n"
            "    assert merge_sorted([1, 2, 3], []) == [1, 2, 3]\n"
            "    assert merge_sorted([], []) == []\n"
            "def test_dupes():\n"
            "    assert merge_sorted([1, 2, 2], [2, 3]) == [1, 2, 2, 2, 3]\n"
            "def test_negatives():\n"
            "    assert merge_sorted([-3, -1, 0], [-2, 5]) == [-3, -2, -1, 0, 5]\n"
        ),
    },
]


_openai_client: Any = None


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    if OpenAI is None:
        raise HTTPException(
            status_code=503,
            detail=f"openai package not installed; container deps in pyproject.toml. {_OPENAI_IMPORT_ERROR!r}",
        )
    if "OPENAI_API_KEY" not in os.environ:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY not set in container env; cannot serve live rollouts.",
        )
    _openai_client = OpenAI()
    return _openai_client


# --- Agent + verifier ---------------------------------------------------------


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _call_agent(client: Any, system_prompt: str, row: dict[str, Any]) -> tuple[str, dict[str, int]]:
    user_content = (
        f"# Task\n"
        f"{row['spec']}\n\n"
        f"# Required signature (match exactly)\n"
        f"{row['signature']}\n\n"
        f"Return only the function body and signature. No imports unless required. "
        f"No usage examples. No markdown fences."
    )
    resp = client.chat.completions.create(
        model=POLICY_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    usage = {
        "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
    }
    return _strip_code_fences(text), usage


def _run_pytest(solution_code: str, tests_code: str) -> tuple[float, dict[str, Any]]:
    """Drop solution.py + test_solution.py into a temp dir, run pytest, parse result."""
    with tempfile.TemporaryDirectory(prefix="tblite_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "solution.py").write_text(solution_code)
        (tmp_path / "test_solution.py").write_text(tests_code)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=short", "test_solution.py"],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=TEST_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return 0.0, {
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "timeout": True,
                "returncode": -1,
                "stdout_tail": "",
                "stderr_tail": "timeout",
            }
        out = proc.stdout or ""
        err = proc.stderr or ""
        # Parse pytest summary line like "1 failed, 3 passed in 0.05s" or "4 passed in 0.05s"
        passed = _count(out, r"(\d+)\s+passed")
        failed = _count(out, r"(\d+)\s+failed")
        errors = _count(out, r"(\d+)\s+error")
        total = passed + failed + errors
        reward = (passed / total) if total > 0 else 0.0
        return reward, {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total": total,
            "timeout": False,
            "returncode": proc.returncode,
            "stdout_tail": out[-800:],
            "stderr_tail": err[-400:],
        }


def _count(text: str, pattern: str) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0


# --- FastAPI app --------------------------------------------------------------

app = FastAPI(title="tblite-gepa-container")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metadata")
@app.get("/info")
async def metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": "tblite_gepa_live",
            "name": "Terminal-Bench-Lite GEPA (real pytest verifier, OpenAI agent)",
            "description": "Public coding-agent cookbook: generate Python implementations and verify against hidden pytest suites.",
        },
        "capabilities": {
            "contract_version": "container_contract.v1",
            "rollout_modes": ["blocking"],
            "metadata": {"trace_schema": "prompt_calls.llm_request.messages.v1"},
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
            "name": "Terminal-Bench-Lite Python function impl",
            "description": (
                "Optimize a starting_prompt for an OpenAI coding agent that writes one Python "
                "function and is scored by hidden pytest tests."
            ),
            "objective": "Maximize the fraction of hidden pytest tests passed across coding tasks.",
            "domain": "single-function Python code generation with edge-case-sensitive hidden tests",
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "visible_splits": ["train", "test"],
            "default_split": "train",
            "row_count": len(ROWS),
            "seed_semantics": (
                "Seeds select deterministic tasks from the public task catalog. The catalog is small, "
                "so profiles should prefer more proposal search over fake extra rows."
            ),
            "task_types": sorted({row["example_id"] for row in ROWS}),
        },
        "prompt_program": {
            "mutable_modules": ["starting_prompt"],
            "candidate_field": "starting_prompt",
            "output_contract": "The policy must output only Python function source code with the exact requested signature.",
        },
        "evaluation": {
            "primary_metric": "outcome_reward",
            "reward_definition": "passed_tests / total_tests from a real pytest subprocess",
            "rollout_trace_contains": ["agent_solution", "pytest_verdict", "stdout_tail"],
        },
        "proposal_guidance": {
            "premises": [
                "The user prompt provides a function signature, natural-language spec, and no hidden tests.",
                "The generated answer is written to solution.py and imported by pytest.",
                "Markdown fences, prose, wrong signatures, imports with side effects, or example usage usually fail.",
            ],
            "constraints": [
                "Do not optimize for one literal task only; propose general coding-agent behavior.",
                "Keep the output contract strict: source code only, exact signature, standard library.",
                "Prefer robust edge-case handling over clever short code.",
            ],
            "high_leverage_heuristics": [
                "Tell the agent to infer input invariants and handle empty, singleton, duplicate, negative, and malformed-ish cases.",
                "Tell it to preserve the signature exactly and return deterministic values.",
                "Tell it to implement the simplest complete algorithm before adding polish.",
                "Tell it to avoid prints, global mutable state, file/network I/O, and test-specific hacks.",
            ],
            "anti_patterns": [
                "Memorized mappings from visible task names to solutions.",
                "Verbose explanations or markdown around the function.",
                "Instructions that encourage broad frameworks instead of a direct function body.",
            ],
        },
        "metadata": {
            "policy_model": POLICY_MODEL,
            "verifier": "pytest_subprocess",
            "test_timeout_seconds": TEST_TIMEOUT_SECONDS,
            "trace_schema": "prompt_calls.llm_request.messages.v1",
        },
    }


@app.get("/program")
async def program() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "tblite_starting_prompt_gepa",
        "modules": [
            {
                "module_id": "starting_prompt",
                "role": "system",
                "content": DEFAULT_STARTING_PROMPT,
                "mutable": True,
                "candidate_field": "starting_prompt",
                "template_variables": [],
                "metadata": {"verifier": "pytest_subprocess"},
            }
        ],
        "target_modules": [
            {
                "module_id": "starting_prompt",
                "candidate_field": "starting_prompt",
                "objective": "fraction_tests_passing",
            }
        ],
        "seed_candidate": {"starting_prompt": DEFAULT_STARTING_PROMPT},
        "rollout_overlay_schema": {"candidate_fields": ["starting_prompt"]},
        "metadata": {
            "task_id": TASK_ID,
            "dataset_id": DATASET_ID,
            "trace_schema": "prompt_calls.llm_request.messages.v1",
        },
    }


@app.get("/dataset")
async def dataset() -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "splits": {
            "train": sum(1 for row in ROWS if row["split"] == "train"),
            "test": sum(1 for row in ROWS if row["split"] == "test"),
        },
        "source": "tblite_public_pytest_tasks",
    }


@app.post("/dataset/rows")
async def dataset_rows(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    seeds = [int(seed) for seed in payload.get("seeds") or []]
    return {
        "rows": [
            # Hide tests from the optimizer side — only spec + signature are exposed.
            {k: v for k, v in _row_for_seed(split=split, seed=seed).items() if k != "tests"}
            for seed in seeds
        ]
    }


@app.post("/rollout")
@app.post("/rollouts")
def rollout(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    incoming_row = payload.get("dataset_row") if isinstance(payload.get("dataset_row"), dict) else None
    if incoming_row and "tests" in incoming_row:
        row = incoming_row
    else:
        # /dataset/rows strips `tests` from outgoing rows, so fall through to
        # the local catalog using the row's seed (or the payload's seed) and
        # the payload's declared split. This is the GEPA-platform path too.
        seed_hint = int((incoming_row or {}).get("seed", payload.get("seed") or 0))
        split_hint = str(
            (incoming_row or {}).get("split")
            or payload.get("split")
            or "train"
        )
        row = _row_for_seed(split=split_hint, seed=seed_hint)
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    system_prompt = str(candidate.get("starting_prompt") or DEFAULT_STARTING_PROMPT)
    seed = int(row.get("seed") or 0)

    client = _get_openai_client()
    solution_code, usage = _call_agent(client, system_prompt, row)
    reward, verdict = _run_pytest(solution_code, row["tests"])

    rollout_id = str(payload.get("rollout_id") or f"rollout_{uuid.uuid4().hex[:12]}")
    now = _now()
    return {
        "rollout_id": rollout_id,
        "status": "completed",
        "success_status": "succeeded" if reward >= 1.0 else "failed",
        "task_id": TASK_ID,
        "seed": seed,
        "reward_info": {
            "outcome_reward": reward,
            "event_rewards": [reward],
            "details": {
                "example_id": row.get("example_id"),
                "passed": verdict["passed"],
                "failed": verdict["failed"],
                "errors": verdict["errors"],
                "total": verdict.get("total"),
                "pytest_timeout": verdict.get("timeout"),
                "pytest_returncode": verdict.get("returncode"),
                "policy_model": POLICY_MODEL,
            },
        },
        "summary": {
            "outcome_reward": reward,
            "example_id": row.get("example_id"),
            "passed": verdict["passed"],
            "failed": verdict["failed"],
            "solution_chars": len(solution_code),
            "pytest_stdout_tail": verdict["stdout_tail"][-300:],
        },
        "usage": {**usage, "cost_usd": 0.0},
        "trace": {
            "event_history": [
                {
                    "type": "agent_solution",
                    "example_id": row.get("example_id"),
                    "solution_chars": len(solution_code),
                },
                {
                    "type": "pytest_verdict",
                    "example_id": row.get("example_id"),
                    "passed": verdict["passed"],
                    "failed": verdict["failed"],
                    "errors": verdict["errors"],
                    "stdout_tail": verdict["stdout_tail"][-300:],
                },
            ],
            "metadata": {
                "example_id": row.get("example_id"),
                "call_site_id": "tblite.python_function_impl",
            },
        },
        "metadata": {"candidate": candidate},
        "created_at": now,
        "updated_at": now,
    }


def _row_for_seed(*, split: str, seed: int) -> dict[str, Any]:
    normalized_split = "test" if split in {"heldout", "test", "validation", "val"} else "train"
    rows = [row for row in ROWS if row["split"] == normalized_split]
    if not rows:
        rows = list(ROWS)
    match = next((row for row in rows if int(row["seed"]) == int(seed)), None)
    row = match or rows[int(seed) % len(rows)]
    return dict(row)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
