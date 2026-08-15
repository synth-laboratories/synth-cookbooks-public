"""
gepa-ai on HotpotQA — blog comparison run
==========================================
Mirrors the Synth GEPA `blog_compare` profile: n=24 train, n=100 heldout,
gpt-4.1-nano policy (matches banking77 for parity). Writes results to
`../runs/hotpotqa_gepa_ai_blog_<timestamp>/summary.json`.

Usage:
    cd cookbooks/optimizers/gepa/hotpotqa_container
    uv run --with gepa --with httpx python run_gepa_ai_blog.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
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
TRAIN_SIZE = 24
HELDOUT_SIZE = 100
TRAIN_SEED_START = 0
HELDOUT_SEED_START = 1000

SEED_PROMPT = "Answer the question using the context. Reply with just the answer."

MAX_METRIC_CALLS = 700
REFLECTION_MINIBATCH = 12
REFLECTION_LM = "openai/gpt-4.1-mini"
POLICY_CONCURRENCY = 64


# ── Container ─────────────────────────────────────────────────────────────────


def pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_container(port: int) -> subprocess.Popen:
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set — cannot start container")
    # Strip OPENROUTER_API_KEY so the container's HOTPOTQA_POLICY_API_KEY_ENV
    # resolution lands on OPENAI_API_KEY (we're forcing OpenAI for blog parity).
    _strip = {"VIRTUAL_ENV", "OPENROUTER_API_KEY"}
    base_env = {k: v for k, v in os.environ.items() if k not in _strip}
    env = {
        **base_env,
        "OPENAI_API_KEY": openai_key,
        "HOTPOTQA_POLICY_MODEL": "gpt-4.1-nano",
        "HOTPOTQA_POLICY_BASE_URL": "https://api.openai.com/v1",
        "HOTPOTQA_POLICY_API_KEY_ENV": "OPENAI_API_KEY",
        "HOTPOTQA_POLICY_TIMEOUT_SECONDS": "25",
        "HOTPOTQA_ROLLOUT_TIMEOUT_SECONDS": "30",
        "HOTPOTQA_POLICY_CONCURRENCY": str(POLICY_CONCURRENCY),
        "HOTPOTQA_POLICY_RETRIES": "2",
        "HOTPOTQA_POLICY_MAX_TOKENS": "64",
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


def wait_for_health(port: int, timeout: float = 120.0) -> None:
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
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["rows"]


def post_rollout(port: int, seed: int, split: str, system_prompt: str) -> dict:
    body = {
        "seed": seed,
        "split": split,
        "candidate": {"stage1_system": system_prompt},
        "submission_mode": "sync",
        "rollout_id": f"gai_{split}_{seed}_{int(time.time()*1000)%1000000}",
    }
    r = httpx.post(f"http://127.0.0.1:{port}/rollout", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


# ── Adapter ───────────────────────────────────────────────────────────────────


class HotpotQAAdapter(GEPAAdapter):
    def __init__(self, port: int, max_workers: int = 64):
        self.port = port
        self.max_workers = max_workers
        self.rollout_count = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        system_prompt = candidate["stage1_system"]
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
                pred = str(summ.get("prediction", ""))
                expected = str(summ.get("expected", ""))
                em = float(summ.get("exact_match", 0.0))
                return i, reward, pred, expected, em
            except Exception as exc:
                return i, 0.0, "", f"<error: {exc!r}>", 0.0

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for fut in as_completed([ex.submit(_do, x) for x in enumerate(batch)]):
                i, reward, pred, expected, em = fut.result()
                scores[i] = reward
                outputs[i] = {"prediction": pred, "expected": expected, "exact_match": em}
                if traj is not None:
                    traj[i] = {
                        "data": batch[i],
                        "prediction": pred,
                        "expected": expected,
                        "f1": reward,
                        "exact_match": em,
                        "feedback": (
                            f"F1={reward:.2f}. Predicted '{pred}', expected '{expected}'. "
                            + ("Exact match." if em >= 1.0
                               else "Partial overlap." if reward > 0
                               else "No overlap.")
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
                    "Inputs": f"HotpotQA question: {t['data'].get('question', '<missing>')}",
                    "Generated Outputs": f"Predicted answer: {t['prediction']}",
                    "Feedback": (
                        f"Expected '{t['expected']}'. Predicted '{t['prediction']}'. "
                        f"Token F1 = {t['f1']:.2f}. "
                        + ("Exact match — correct." if t["exact_match"] >= 1.0
                           else "Partial overlap; check answer type and surface form."
                           if t["f1"] > 0
                           else "No overlap — wrong answer entity.")
                    ),
                }
            )
        return {comp: items}


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "runs" / f"hotpotqa_gepa_ai_blog_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    port = pick_free_port()
    print(f"[gepa-ai hotpotqa] starting container on port {port}", flush=True)
    proc = start_container(port)

    try:
        wait_for_health(port)
        print("[gepa-ai hotpotqa] container healthy", flush=True)

        train_seeds = list(range(TRAIN_SEED_START, TRAIN_SEED_START + TRAIN_SIZE))
        heldout_seeds = list(range(HELDOUT_SEED_START, HELDOUT_SEED_START + HELDOUT_SIZE))
        train_rows = fetch_rows(port, "train", train_seeds)
        heldout_rows = fetch_rows(port, "validation", heldout_seeds)
        print(
            f"[gepa-ai hotpotqa] rows fetched: train={len(train_rows)} heldout={len(heldout_rows)}",
            flush=True,
        )

        trainset = [
            {
                "seed": int(r["seed"]),
                "split": "train",
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
            }
            for r in train_rows
        ]
        valset = trainset

        adapter = HotpotQAAdapter(port=port, max_workers=POLICY_CONCURRENCY)
        seed_candidate = {"stage1_system": SEED_PROMPT}

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
            f"[gepa-ai hotpotqa] done in {elapsed:.1f}s, rollouts={adapter.rollout_count}",
            flush=True,
        )

        # ── Per-candidate heldout evals ──────────────────────────────────────
        heldout_inputs = [
            {
                "seed": int(r["seed"]),
                "split": "validation",
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
            }
            for r in heldout_rows
        ]

        candidate_results = []
        print(
            f"[gepa-ai hotpotqa] evaluating {len(result.candidates)} candidates on heldout …",
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
            f"[gepa-ai hotpotqa] seed heldout={seed_heldout:.3f}  best heldout={best_heldout:.3f}  "
            f"lift={best_heldout - seed_heldout:+.3f}",
            flush=True,
        )

        summary = {
            "task": "hotpotqa",
            "gepa_version": getattr(gepa, "__version__", "unknown"),
            "elapsed_seconds": elapsed,
            "train_size": TRAIN_SIZE,
            "heldout_size": HELDOUT_SIZE,
            "max_metric_calls": MAX_METRIC_CALLS,
            "seed_prompt": SEED_PROMPT,
            "best_candidate": result.best_candidate,
            "seed_heldout": round(seed_heldout, 6),
            "best_heldout": round(best_heldout, 6),
            "lift": round(best_heldout - seed_heldout, 6),
            "candidates": candidate_results,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[gepa-ai hotpotqa] summary → {output_dir / 'summary.json'}", flush=True)

    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
