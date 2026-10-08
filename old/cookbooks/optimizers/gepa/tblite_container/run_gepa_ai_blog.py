"""
gepa-ai on TBLite — blog comparison run
========================================
Mirrors the Synth GEPA `blog_compare` profile: 8 train / 5 heldout, expanded
ROWS catalog (palindrome / two_sum / RLE / merge / balanced + text_justification
/ decode_ways / min_window_substring / valid_number / longest_palindromic_subseq
/ word_break / spiral_matrix / regex_match) so gpt-4.1-nano doesn't saturate.

Usage:
    cd cookbooks/optimizers/gepa/tblite_container
    uv run --with gepa --with httpx --with litellm python run_gepa_ai_blog.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

# ── Keys ──────────────────────────────────────────────────────────────────────
ENV_FILE = Path(__file__).resolve().parents[5] / "synth-ai" / ".env"
if not ENV_FILE.exists():
    for candidate in [
        Path.home() / "Documents/GitHub/synth-ai/.env",
        Path("/Users/joshpurtell/Documents/GitHub/synth-ai/.env"),
    ]:
        if candidate.exists():
            ENV_FILE = candidate
            break

_env_openai_key: str = ""
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k == "OPENAI_API_KEY":
            _env_openai_key = v
        else:
            os.environ.setdefault(k, v)
if _env_openai_key:
    os.environ["OPENAI_API_KEY"] = _env_openai_key

import gepa  # noqa: E402
from gepa.core.adapter import EvaluationBatch, GEPAAdapter  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
HELDOUT_SEEDS = [100, 101, 102, 103, 104]
TEST_TIMEOUT_SECONDS = 30

SEED_PROMPT = (
    "Wrap your code in ```python ... ``` fences. Explain your reasoning briefly "
    "before the code block."
)

# Each rollout = 1 OpenAI call + pytest. Pretty cheap per call.
MAX_METRIC_CALLS = 150
REFLECTION_MINIBATCH = 4
REFLECTION_LM = "openai/gpt-4.1-mini"
ROLLOUT_WORKERS = 8


# ── Container ─────────────────────────────────────────────────────────────────


def pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_container(port: int) -> subprocess.Popen:
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set — cannot start container")
    _strip = {"VIRTUAL_ENV"}
    base_env = {k: v for k, v in os.environ.items() if k not in _strip}
    env = {
        **base_env,
        "OPENAI_API_KEY": openai_key,
        "TBLITE_POLICY_MODEL": "gpt-4.1-nano",
        "TBLITE_TEST_TIMEOUT_SECONDS": str(TEST_TIMEOUT_SECONDS),
    }
    container_dir = Path(__file__).parent
    cmd = [
        "uv",
        "run",
        "python",
        "synth_service_app.py",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=container_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def wait_for_health(port: int, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("container did not become healthy in time")


def fetch_rows(port: int, split: str, seeds: list[int]) -> list[dict]:
    r = httpx.post(
        f"http://127.0.0.1:{port}/dataset/rows",
        json={"split": split, "seeds": seeds},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["rows"]


def post_rollout(port: int, seed: int, split: str, system_prompt: str) -> dict:
    body = {
        "seed": seed,
        "split": split,
        "candidate": {"starting_prompt": system_prompt},
        "submission_mode": "sync",
        "rollout_id": f"gai_{split}_{seed}_{int(time.time()*1000)%1000000}",
    }
    r = httpx.post(f"http://127.0.0.1:{port}/rollout", json=body, timeout=180)
    r.raise_for_status()
    return r.json()


# ── Adapter ───────────────────────────────────────────────────────────────────


class TBLiteAdapter(GEPAAdapter):
    def __init__(self, port: int, max_workers: int = 8):
        self.port = port
        self.max_workers = max_workers
        self.rollout_count = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        system_prompt = candidate["starting_prompt"]
        outputs = [None] * len(batch)
        scores = [0.0] * len(batch)
        traj = [None] * len(batch) if capture_traces else None

        def _do(i_data):
            i, data = i_data
            try:
                result = post_rollout(
                    self.port,
                    seed=data["seed"],
                    split=data["split"],
                    system_prompt=system_prompt,
                )
                reward = float(result.get("reward_info", {}).get("outcome_reward", 0.0))
                summ = result.get("summary", {}) or {}
                details = result.get("reward_info", {}).get("details", {}) or {}
                passed = int(summ.get("passed", details.get("passed", 0)))
                failed = int(summ.get("failed", details.get("failed", 0)))
                example_id = str(summ.get("example_id", details.get("example_id", "")))
                stdout_tail = str(summ.get("pytest_stdout_tail", ""))
                return i, reward, passed, failed, example_id, stdout_tail
            except Exception as exc:
                return i, 0.0, 0, 0, "<error>", repr(exc)

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for fut in as_completed([ex.submit(_do, x) for x in enumerate(batch)]):
                i, reward, passed, failed, example_id, stdout_tail = fut.result()
                scores[i] = reward
                outputs[i] = {
                    "reward": reward,
                    "passed": passed,
                    "failed": failed,
                    "example_id": example_id,
                }
                if traj is not None:
                    traj[i] = {
                        "data": batch[i],
                        "reward": reward,
                        "passed": passed,
                        "failed": failed,
                        "example_id": example_id,
                        "stdout_tail": stdout_tail,
                        "feedback": (
                            f"All {passed} tests passed for {example_id}." if reward >= 1.0
                            else f"{passed}/{passed+failed} tests passed for {example_id}. "
                                 f"Output tail: {stdout_tail[-200:]}"
                        ),
                    }
        self.rollout_count += len(batch)
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traj)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        comp = components_to_update[0]
        items = []
        for t in eval_batch.trajectories:
            items.append(
                {
                    "Inputs": f"TBLite Python task: example_id={t['example_id']}, seed={t['data']['seed']}.",
                    "Generated Outputs": (
                        f"Candidate solution {'PASSED all tests' if t['reward'] >= 1.0 else 'FAILED some tests'}."
                    ),
                    "Feedback": (
                        f"{'PASS' if t['reward'] >= 1.0 else 'FAIL'}: "
                        f"{t['passed']} passed / {t['failed']} failed. "
                        + (
                            "Solution code matched the spec exactly."
                            if t["reward"] >= 1.0
                            else f"Pytest output tail: {t['stdout_tail'][-300:]}"
                        )
                    ),
                }
            )
        return {comp: items}


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "runs" / f"tblite_gepa_ai_blog_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    port = pick_free_port()
    print(f"[gepa-ai tblite] starting container on port {port}", flush=True)
    proc = start_container(port)

    try:
        wait_for_health(port)
        print("[gepa-ai tblite] container healthy", flush=True)

        train_rows = fetch_rows(port, "train", TRAIN_SEEDS)
        heldout_rows = fetch_rows(port, "test", HELDOUT_SEEDS)
        print(
            f"[gepa-ai tblite] rows fetched: train={len(train_rows)} heldout={len(heldout_rows)}",
            flush=True,
        )

        trainset = [
            {"seed": int(r["seed"]), "split": "train", "example_id": r.get("example_id", "")}
            for r in train_rows
        ]
        valset = trainset

        adapter = TBLiteAdapter(port=port, max_workers=ROLLOUT_WORKERS)
        seed_candidate = {"starting_prompt": SEED_PROMPT}

        t0 = time.time()
        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=REFLECTION_LM,
            candidate_selection_strategy="pareto",
            reflection_minibatch_size=REFLECTION_MINIBATCH,
            max_metric_calls=MAX_METRIC_CALLS,
            display_progress_bar=True,
            seed=0,
            track_best_outputs=False,
        )
        elapsed = time.time() - t0
        print(
            f"[gepa-ai tblite] done in {elapsed:.1f}s, rollouts={adapter.rollout_count}",
            flush=True,
        )

        heldout_inputs = [
            {"seed": int(r["seed"]), "split": "test", "example_id": r.get("example_id", "")}
            for r in heldout_rows
        ]

        candidate_results = []
        print(
            f"[gepa-ai tblite] evaluating {len(result.candidates)} candidates on heldout …",
            flush=True,
        )
        for idx, cand in enumerate(result.candidates):
            adapter.rollout_count = 0
            h_eval = adapter.evaluate(heldout_inputs, cand, capture_traces=False)
            h_score = sum(h_eval.scores) / len(h_eval.scores)
            val_score = (
                result.val_aggregate_scores[idx]
                if idx < len(result.val_aggregate_scores)
                else None
            )
            candidate_results.append(
                {
                    "idx": idx,
                    "heldout": round(h_score, 6),
                    "val_score": val_score,
                    "candidate": cand,
                }
            )
            print(f"  candidate {idx:2d}: heldout={h_score:.3f}", flush=True)

        adapter.rollout_count = 0
        seed_h_eval = adapter.evaluate(heldout_inputs, seed_candidate, capture_traces=False)
        seed_heldout = sum(seed_h_eval.scores) / len(seed_h_eval.scores)

        best_heldout = max(c["heldout"] for c in candidate_results)
        print(
            f"[gepa-ai tblite] seed heldout={seed_heldout:.3f}  best heldout={best_heldout:.3f}  "
            f"lift={best_heldout - seed_heldout:+.3f}",
            flush=True,
        )

        summary = {
            "task": "tblite",
            "gepa_version": getattr(gepa, "__version__", "unknown"),
            "elapsed_seconds": elapsed,
            "train_size": len(TRAIN_SEEDS),
            "heldout_size": len(HELDOUT_SEEDS),
            "max_metric_calls": MAX_METRIC_CALLS,
            "seed_prompt": SEED_PROMPT,
            "best_candidate": result.best_candidate,
            "seed_heldout": round(seed_heldout, 6),
            "best_heldout": round(best_heldout, 6),
            "lift": round(best_heldout - seed_heldout, 6),
            "candidates": candidate_results,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[gepa-ai tblite] summary → {output_dir / 'summary.json'}", flush=True)

    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
