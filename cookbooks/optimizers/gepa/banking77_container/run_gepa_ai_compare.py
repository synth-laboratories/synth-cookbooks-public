"""Run the original gepa-ai (PyPI) optimizer against the same banking77
container the rust platform run used, so we can compare proposed prompts
and lift apples-to-apples.

Usage:
    /opt/homebrew/bin/python3.11 run_gepa_ai_compare.py

Loads OPENROUTER_API_KEY (policy) and OPENAI_API_KEY (reflector) from
synth-ai/.env. Spins up the banking77 container, fetches 100 train +
100 heldout rows that mirror the rust run, and runs gepa.optimize with
a 300-call train budget.
"""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

# Load env from synth-ai/.env so we get both keys.
ENV_FILE = Path("/Users/joshpurtell/Documents/GitHub/synth-ai/.env")
for line in ENV_FILE.read_text().splitlines():
    if "=" not in line or line.lstrip().startswith("#"):
        continue
    k, _, v = line.partition("=")
    v = v.strip().strip('"').strip("'")
    os.environ.setdefault(k.strip(), v)

import gepa  # noqa: E402
from gepa.core.adapter import EvaluationBatch, GEPAAdapter  # noqa: E402


SEED_PROMPT = (
    "Classify the customer banking query into exactly one Banking77 intent. "
    "Return exactly one label from the allowed label list, preserving the "
    "label's spelling, underscores, capitalization, and punctuation. Use the "
    "full query, not one keyword. Prefer the label for the user's concrete "
    "banking action, status, or problem: separate physical-card ordering "
    "from delivery timing, virtual-card creation from virtual-card problems, "
    "card payments from cash withdrawals, top-ups from incoming transfers, "
    "pending from failed/declined/reverted, passcodes from card PINs, and "
    "phone loss from card compromise. Return only the label."
)


def pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_container(port: int) -> subprocess.Popen[bytes]:
    env = {
        **os.environ,
        "BANKING77_TRAIN_SAMPLE": "100",
        "BANKING77_TEST_SAMPLE": "200",
        "BANKING77_POLICY_CONCURRENCY": "60",
        "BANKING77_POLICY_MODEL": "qwen/qwen-2.5-7b-instruct",
        "BANKING77_POLICY_BASE_URL": "https://openrouter.ai/api/v1",
        "BANKING77_POLICY_API_KEY_ENV": "OPENROUTER_API_KEY",
        "BANKING77_POLICY_DISABLE_REASONING": "auto",
        "BANKING77_POLICY_MAX_TOKENS": "16",
        "BANKING77_POLICY_RETRIES": "1",
        "BANKING77_POLICY_API_MODE": "auto",
        "BANKING77_ROLLOUT_TIMEOUT_SECONDS": "25",
        "BANKING77_POLICY_TIMEOUT_SECONDS": "20",
        "BANKING77_TRAIN_SHUFFLE_SEED": "664514459",
        "BANKING77_TEST_SHUFFLE_SEED": "4174331086",
    }
    cmd = [
        "uv", "run", "--project", "banking77_container",
        "python", "banking77_container/synth_service_app.py",
        "--host", "127.0.0.1", "--port", str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd="..",
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def wait_for_health(port: int, timeout: float = 60.0) -> None:
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
        "candidate": {"stage2_system": system_prompt},
        "submission_mode": "sync",
        "rollout_id": f"gepaai_{split}_{seed}_{int(time.time()*1000)%1000000}",
    }
    r = httpx.post(
        f"http://127.0.0.1:{port}/rollout",
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


class Banking77Adapter(GEPAAdapter):
    def __init__(self, port: int, max_workers: int = 40):
        self.port = port
        self.max_workers = max_workers
        self.rollout_count = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        system_prompt = candidate["stage2_system"]
        outputs: list[dict] = [None] * len(batch)
        scores: list[float] = [0.0] * len(batch)
        traj: list[dict] = [None] * len(batch) if capture_traces else None

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
                pred = (result.get("summary", {}) or {}).get("prediction", "")
                expected = (result.get("summary", {}) or {}).get("expected", "")
                return i, reward, pred, expected
            except Exception as exc:
                return i, 0.0, "", f"<error: {exc!r}>"

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(_do, (i, d)) for i, d in enumerate(batch)]
            for fut in as_completed(futures):
                i, reward, pred, expected = fut.result()
                scores[i] = reward
                outputs[i] = {"prediction": pred, "expected": expected}
                if traj is not None:
                    data = batch[i]
                    feedback = (
                        f"Correct. Predicted '{pred}'."
                        if reward >= 1.0
                        else f"Wrong. Predicted '{pred}', expected '{expected}'."
                    )
                    traj[i] = {
                        "data": data,
                        "prediction": pred,
                        "expected": expected,
                        "feedback": feedback,
                    }
        self.rollout_count += len(batch)
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traj)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        comp = components_to_update[0]
        items = []
        for t in eval_batch.trajectories:
            items.append({
                "Inputs": f"Customer query: {t['data']['text']}",
                "Generated Outputs": f"Predicted label: {t['prediction']}",
                "Feedback": (
                    f"Correct label was '{t['expected']}'. "
                    f"Predicted '{t['prediction']}'. "
                    + ("Correct." if t['prediction'] == t['expected'] else
                       "Wrong — the predicted label is a close but different intent.")
                ),
            })
        return {comp: items}


def main() -> None:
    output_dir = Path("../runs/banking77_gepa_ai_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    port = pick_free_port()
    print(f"[gepa-ai] starting container on port {port}", flush=True)
    proc = start_container(port)
    try:
        wait_for_health(port)
        print(f"[gepa-ai] container healthy", flush=True)

        train_seeds = list(range(100))
        heldout_seeds = list(range(100))
        train_rows = fetch_rows(port, "train", train_seeds)
        heldout_rows = fetch_rows(port, "test", heldout_seeds)
        print(f"[gepa-ai] fetched train={len(train_rows)} heldout={len(heldout_rows)}", flush=True)

        # gepa-ai uses an unconstrained DataInst — pass dicts with required fields.
        trainset = [
            {"seed": int(r["seed"]), "split": "train", "text": r["text"], "label": r["label"]}
            for r in train_rows
        ]
        # Use a 50-example subset of train as the pareto valset (gepa-ai re-evals every accept).
        valset = trainset[:50]

        adapter = Banking77Adapter(port=port, max_workers=40)

        seed_candidate = {"stage2_system": SEED_PROMPT}

        # Reflector via openai
        from gepa import optimize

        reflection_lm = "openai/gpt-5.4-mini"  # litellm-style

        t0 = time.time()
        result = optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm,
            candidate_selection_strategy="pareto",
            reflection_minibatch_size=8,
            max_metric_calls=300,
            display_progress_bar=False,
            seed=0,
            track_best_outputs=False,
        )
        elapsed = time.time() - t0
        print(f"[gepa-ai] optimize done in {elapsed:.1f}s, total rollouts={adapter.rollout_count}", flush=True)

        # Pull every candidate from the result.
        candidates_dump = []
        try:
            for i, cand in enumerate(result.candidates):
                val_scores = result.val_aggregate_scores[i] if i < len(result.val_aggregate_scores) else None
                candidates_dump.append({
                    "idx": i,
                    "candidate": cand,
                    "val_aggregate_score": val_scores,
                    "parent_idxs": result.parents[i] if i < len(result.parents) else None,
                })
        except Exception as exc:
            print(f"[gepa-ai] could not enumerate candidates: {exc}", flush=True)

        # Heldout eval on best candidate and seed.
        best = result.best_candidate
        print(f"[gepa-ai] best candidate: {json.dumps(best, indent=2)[:500]}", flush=True)
        heldout_inputs = [
            {"seed": int(r["seed"]), "split": "test", "text": r["text"], "label": r["label"]}
            for r in heldout_rows
        ]
        adapter.rollout_count = 0
        seed_heldout = adapter.evaluate(heldout_inputs, seed_candidate, capture_traces=False)
        best_heldout = adapter.evaluate(heldout_inputs, best, capture_traces=False)
        seed_h = sum(seed_heldout.scores) / len(seed_heldout.scores)
        best_h = sum(best_heldout.scores) / len(best_heldout.scores)
        print(f"[gepa-ai] heldout: seed={seed_h:.3f}  best={best_h:.3f}  lift={best_h - seed_h:+.3f}", flush=True)

        summary = {
            "elapsed_seconds": elapsed,
            "train_budget": 300,
            "heldout_size": len(heldout_rows),
            "seed_prompt": SEED_PROMPT,
            "best_candidate": best,
            "seed_heldout": seed_h,
            "best_heldout": best_h,
            "lift": best_h - seed_h,
            "candidates": candidates_dump,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[gepa-ai] summary written to {output_dir/'summary.json'}", flush=True)

    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
