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
  OPENAI_API_KEY               — required when rollout.policy.credential_mode=byok.
  TBLITE_TEST_TIMEOUT_SECONDS  — default: 30 (pytest subprocess hard cap)
"""
from __future__ import annotations

import argparse
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
    GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"

try:
    from openai import OpenAI
except Exception as _openai_err:
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = _openai_err
else:
    _OPENAI_IMPORT_ERROR = None


TASK_ID = "tblite.python_function_impl"
DATASET_ID = "tblite_public_pytest_tasks"

TEST_TIMEOUT_SECONDS = int(os.environ.get("TBLITE_TEST_TIMEOUT_SECONDS", "30"))

DEFAULT_STARTING_PROMPT = (
    "Wrap your code in ```python ... ``` fences. Explain your reasoning briefly "
    "before the code block."
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
    # ── Harder train problems ────────────────────────────────────────────────
    # These were added so the seed prompt doesn't saturate on
    # gpt-4.1-nano. They exercise DP, sliding-window, string parsing, and
    # tricky-edge-case territory where small models routinely fail.
    {
        "seed": 3,
        "split": "train",
        "example_id": "text_justification",
        "signature": "def text_justification(words: list[str], maxWidth: int) -> list[str]:",
        "spec": (
            "Greedily pack `words` into lines of width `maxWidth`, then fully "
            "justify every line *except* the last. Within a non-last line, "
            "distribute extra spaces between words from left to right so the "
            "left gaps are >= the right gaps. A line with a single word is "
            "left-justified and padded with trailing spaces. The last line is "
            "always left-justified with a single space between words and "
            "trailing spaces to width."
        ),
        "tests": (
            "from solution import text_justification\n"
            "def test_canonical():\n"
            "    out = text_justification(['This', 'is', 'an', 'example', 'of', 'text', 'justification.'], 16)\n"
            "    assert out == ['This    is    an', 'example  of text', 'justification.  ']\n"
            "def test_uneven_distribution():\n"
            "    out = text_justification(['What', 'must', 'be', 'acknowledgment', 'shall', 'be'], 16)\n"
            "    assert out == ['What   must   be', 'acknowledgment  ', 'shall be        ']\n"
            "def test_single_short_line():\n"
            "    assert text_justification(['hi'], 5) == ['hi   ']\n"
        ),
    },
    {
        "seed": 4,
        "split": "train",
        "example_id": "decode_ways",
        "signature": "def decode_ways(s: str) -> int:",
        "spec": (
            "`s` is a non-empty digit string. Each character of an original "
            "message was encoded by mapping 'A'->'1', 'B'->'2', ..., 'Z'->'26'. "
            "Return the number of ways to decode `s`. Treat '0' as un-decodable "
            "by itself: it must combine with a preceding '1' or '2'. Anything "
            "else (e.g. '30', '06') contributes 0 ways."
        ),
        "tests": (
            "from solution import decode_ways\n"
            "def test_two_digit():\n"
            "    assert decode_ways('12') == 2  # 'AB' or 'L'\n"
            "def test_with_zero():\n"
            "    assert decode_ways('226') == 3\n"
            "    assert decode_ways('06') == 0\n"
            "    assert decode_ways('10') == 1\n"
            "def test_invalid_zero():\n"
            "    assert decode_ways('100') == 0\n"
            "    assert decode_ways('301') == 0\n"
            "def test_long():\n"
            "    assert decode_ways('11106') == 2  # 'AAJF' or 'KJF'\n"
        ),
    },
    {
        "seed": 5,
        "split": "train",
        "example_id": "min_window_substring",
        "signature": "def min_window_substring(s: str, t: str) -> str:",
        "spec": (
            "Return the minimum-length contiguous substring of `s` that "
            "contains every character of `t` *including duplicates* (as a "
            "multiset). Return '' if no such window exists. If multiple "
            "windows tie on length, return the one whose left index is "
            "smallest."
        ),
        "tests": (
            "from solution import min_window_substring\n"
            "def test_basic():\n"
            "    assert min_window_substring('ADOBECODEBANC', 'ABC') == 'BANC'\n"
            "def test_no_window():\n"
            "    assert min_window_substring('a', 'aa') == ''\n"
            "    assert min_window_substring('abc', 'xy') == ''\n"
            "def test_full_string():\n"
            "    assert min_window_substring('a', 'a') == 'a'\n"
            "def test_duplicates():\n"
            "    assert min_window_substring('aaflslflsldkalskaaa', 'aaa') == 'aaa'\n"
        ),
    },
    {
        "seed": 6,
        "split": "train",
        "example_id": "valid_number",
        "signature": "def valid_number(s: str) -> bool:",
        "spec": (
            "Return True iff `s` is a valid number per the following rules: "
            "an optional sign, then either (1) an integer (one or more digits), "
            "(2) a decimal (digits before and/or after a '.', with at least "
            "one digit overall), optionally followed by 'e' or 'E' and a "
            "signed integer exponent. Leading and trailing whitespace are "
            "*not* allowed. Examples of invalid: '', '.', '+.', '1e', '1e+', "
            "'4e+', 'e3', '99e2.5', '+-3'."
        ),
        "tests": (
            "from solution import valid_number\n"
            "def test_integers_and_decimals():\n"
            "    for x in ['0', '0.1', '.1', '1.', '+3', '-2.5', '3e10', '+3.14e-2']:\n"
            "        assert valid_number(x), x\n"
            "def test_invalid():\n"
            "    for x in ['', '.', '+.', '1e', '1e+', '4e+', 'e3', '99e2.5', '+-3', ' 1', '1 ', 'abc', '1..0', '1e1.5']:\n"
            "        assert not valid_number(x), x\n"
        ),
    },
    {
        "seed": 7,
        "split": "train",
        "example_id": "longest_palindromic_subseq",
        "signature": "def longest_palindromic_subseq(s: str) -> int:",
        "spec": (
            "Return the length of the longest subsequence of `s` (characters "
            "in order, not necessarily contiguous) that reads the same "
            "forwards and backwards. Empty string returns 0. Single character "
            "returns 1."
        ),
        "tests": (
            "from solution import longest_palindromic_subseq\n"
            "def test_basic():\n"
            "    assert longest_palindromic_subseq('bbbab') == 4  # 'bbbb'\n"
            "def test_no_repeat():\n"
            "    assert longest_palindromic_subseq('abcde') == 1\n"
            "def test_already_palindrome():\n"
            "    assert longest_palindromic_subseq('character') == 5  # 'carac'\n"
            "def test_empty_and_one():\n"
            "    assert longest_palindromic_subseq('') == 0\n"
            "    assert longest_palindromic_subseq('a') == 1\n"
        ),
    },
    # ── Harder heldout problems ──────────────────────────────────────────────
    {
        "seed": 102,
        "split": "test",
        "example_id": "word_break",
        "signature": "def word_break(s: str, word_dict: list[str]) -> bool:",
        "spec": (
            "Return True iff `s` can be segmented into a sequence of one or "
            "more dictionary words from `word_dict`. Words from the "
            "dictionary may be reused. Empty `s` returns True."
        ),
        "tests": (
            "from solution import word_break\n"
            "def test_basic():\n"
            "    assert word_break('leetcode', ['leet', 'code'])\n"
            "def test_repeat():\n"
            "    assert word_break('applepenapple', ['apple', 'pen'])\n"
            "def test_fail():\n"
            "    assert not word_break('catsandog', ['cats', 'dog', 'sand', 'and', 'cat'])\n"
            "def test_empty():\n"
            "    assert word_break('', ['a'])\n"
        ),
    },
    {
        "seed": 103,
        "split": "test",
        "example_id": "spiral_matrix",
        "signature": "def spiral_matrix(matrix: list[list[int]]) -> list[int]:",
        "spec": (
            "Return all elements of an m×n matrix in clockwise spiral order, "
            "starting at the top-left and turning right/down/left/up as the "
            "boundary contracts. Handle non-square (including 1×n and m×1) "
            "matrices."
        ),
        "tests": (
            "from solution import spiral_matrix\n"
            "def test_square():\n"
            "    assert spiral_matrix([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]\n"
            "def test_rect_wide():\n"
            "    assert spiral_matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1,2,3,4,8,12,11,10,9,5,6,7]\n"
            "def test_single_row():\n"
            "    assert spiral_matrix([[1,2,3]]) == [1,2,3]\n"
            "def test_single_col():\n"
            "    assert spiral_matrix([[1],[2],[3]]) == [1,2,3]\n"
        ),
    },
    {
        "seed": 104,
        "split": "test",
        "example_id": "regex_match",
        "signature": "def regex_match(s: str, p: str) -> bool:",
        "spec": (
            "Return True iff the entire string `s` matches the pattern `p`. "
            "Pattern syntax: '.' matches any single character; 'x*' matches "
            "zero or more of the preceding character `x` (where `x` is a "
            "literal or '.'). No other metacharacters. Anchored at both "
            "ends — partial matches don't count."
        ),
        "tests": (
            "from solution import regex_match\n"
            "def test_literal():\n"
            "    assert regex_match('aa', 'aa')\n"
            "    assert not regex_match('aa', 'a')\n"
            "def test_star():\n"
            "    assert regex_match('aa', 'a*')\n"
            "    assert regex_match('aaa', 'a*')\n"
            "    assert regex_match('', 'a*')\n"
            "def test_dot_star():\n"
            "    assert regex_match('ab', '.*')\n"
            "    assert regex_match('aab', 'c*a*b')\n"
            "    assert not regex_match('mississippi', 'mis*is*p*.')\n"
        ),
    },
]


_openai_clients: dict[tuple[str, str, str], Any] = {}
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
    max_tokens = policy.get("max_tokens")
    if max_tokens is not None:
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
                detail="rollout.policy.max_tokens must be positive when set.",
            )
    return {
        "provider": provider,
        "model": model,
        "base_url": _strip_openai_endpoint_suffix(raw_base_url) if raw_base_url else None,
        "credential_mode": credential_mode,
        "max_tokens": max_tokens,
    }


def _policy_api_key(policy: dict[str, Any]) -> str:
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


def _get_openai_client(policy: dict[str, Any]) -> Any:
    if OpenAI is None:
        raise HTTPException(
            status_code=503,
            detail=f"openai package not installed; container deps in pyproject.toml. {_OPENAI_IMPORT_ERROR!r}",
        )
    base_url = policy.get("base_url")
    key = (policy["provider"].lower(), policy["credential_mode"], str(base_url or ""))
    client = _openai_clients.get(key)
    if client is None:
        client_kwargs = {"api_key": _policy_api_key(policy)}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        _openai_clients[key] = client
    return client


# --- Agent + verifier ---------------------------------------------------------


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _call_agent(
    client: Any,
    policy: dict[str, Any],
    system_prompt: str,
    row: dict[str, Any],
) -> tuple[str, dict[str, int]]:
    user_content = (
        f"# Task\n"
        f"{row['spec']}\n\n"
        f"# Required signature (match exactly)\n"
        f"{row['signature']}\n\n"
        f"Return only the function body and signature. No imports unless required. "
        f"No usage examples. No markdown fences."
    )
    request_kwargs = {
        "model": policy["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if policy["max_tokens"] is not None:
        request_kwargs["max_tokens"] = policy["max_tokens"]
    resp = client.chat.completions.create(**request_kwargs)
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
            "policy_model_source": "rollout.policy.model",
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
    policy = _require_policy(payload)
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

    client = _get_openai_client(policy)
    solution_code, usage = _call_agent(client, policy, system_prompt, row)
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
                "policy_model": policy["model"],
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
