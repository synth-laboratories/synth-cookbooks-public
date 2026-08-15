"""
gepa-ai on Crafter — blog comparison run
=========================================
Mirrors `blog_compare` profile: n=12 train / n=16 heldout, max_turns=15,
deliberately-weak seed prompt so the seed score doesn't saturate at +2 achievements.

Usage:
    cd cookbooks/optimizers/gepa/crafter_container
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
TRAIN_SIZE = 12
HELDOUT_SIZE = 16
TRAIN_SEED_START = 11
HELDOUT_SEED_START = 101
MAX_TURNS = 15

SEED_PROMPT = (
    "You are a Crafter agent. Respond with "
    '<tool_call>{"name":"crafter_interact","arguments":{"actions_list":["<action>"]}}</tool_call>. '
    "Pick reasonable actions."
)

# Each rollout = full episode of up to MAX_TURNS LLM calls
MAX_METRIC_CALLS = 96
REFLECTION_MINIBATCH = 4
REFLECTION_LM = "openai/gpt-4.1-mini"
ROLLOUT_WORKERS = 2  # Avoid JAX JIT thrash; each rollout fresh-compiles env


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
        "CRAFTER_POLICY_MODEL": "gpt-4.1-nano",
        "CRAFTER_MAX_TURNS": str(MAX_TURNS),
        "CRAFTER_MIN_BATCH": "1",
        "CRAFTER_MAX_BATCH": "5",
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


def wait_for_health(port: int, timeout: float = 180.0) -> None:
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
        "candidate": {"react_system_prompt": system_prompt},
        "submission_mode": "sync",
        "rollout_id": f"gai_{split}_{seed}_{int(time.time()*1000)%1000000}",
    }
    r = httpx.post(f"http://127.0.0.1:{port}/rollout", json=body, timeout=360)
    r.raise_for_status()
    return r.json()


class CrafterAdapter(GEPAAdapter):
    def __init__(self, port: int, max_workers: int = 12):
        self.port = port
        self.max_workers = max_workers
        self.rollout_count = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        system_prompt = candidate["react_system_prompt"]
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
                achievements = list(summ.get("achievements_unlocked", details.get("achievements", [])) or [])
                n_steps = int(summ.get("n_steps", details.get("n_steps", 0)))
                actions = list(summ.get("actions_taken", []) or [])
                return i, reward, achievements, n_steps, actions
            except Exception as exc:
                return i, 0.0, [], 0, [f"<error: {exc!r}>"]

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for fut in as_completed([ex.submit(_do, x) for x in enumerate(batch)]):
                i, reward, achievements, n_steps, actions = fut.result()
                scores[i] = reward
                outputs[i] = {
                    "reward": reward,
                    "achievements": achievements,
                    "n_steps": n_steps,
                }
                if traj is not None:
                    traj[i] = {
                        "data": batch[i],
                        "reward": reward,
                        "achievements": achievements,
                        "n_steps": n_steps,
                        "actions": actions,
                        "feedback": (
                            f"Episode ended after {n_steps} steps with {reward:.2f} total reward "
                            f"and achievements: {achievements}"
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
                    "Inputs": (
                        f"Crafter survival episode (seed={t['data']['seed']}, "
                        f"max_turns={MAX_TURNS}). Agent needs to collect resources and craft tools."
                    ),
                    "Generated Outputs": (
                        f"Actions taken: {t['actions'][:30]}"
                        + ("..." if len(t['actions']) > 30 else "")
                    ),
                    "Feedback": (
                        f"Reward={t['reward']:.2f}, achievements={t['achievements']}, "
                        f"steps={t['n_steps']}/{MAX_TURNS}. "
                        + (
                            "Reasonable progression."
                            if t["reward"] >= 2.0
                            else "Limited progression — common failures: not collecting wood "
                            "first, walking into water/lava, failing to place a crafting table."
                        )
                    ),
                }
            )
        return {comp: items}


def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "runs" / f"crafter_gepa_ai_blog_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    port = pick_free_port()
    print(f"[gepa-ai crafter] starting container on port {port}", flush=True)
    proc = start_container(port)

    try:
        wait_for_health(port)
        print("[gepa-ai crafter] container healthy", flush=True)

        train_seeds = list(range(TRAIN_SEED_START, TRAIN_SEED_START + TRAIN_SIZE))
        heldout_seeds = list(range(HELDOUT_SEED_START, HELDOUT_SEED_START + HELDOUT_SIZE))
        train_rows = fetch_rows(port, "train", train_seeds)
        heldout_rows = fetch_rows(port, "test", heldout_seeds)
        print(
            f"[gepa-ai crafter] rows fetched: train={len(train_rows)} heldout={len(heldout_rows)}",
            flush=True,
        )

        trainset = [
            {"seed": int(r["seed"]), "split": "train", "example_id": r.get("example_id", "")}
            for r in train_rows
        ]
        valset = trainset

        adapter = CrafterAdapter(port=port, max_workers=ROLLOUT_WORKERS)
        seed_candidate = {"react_system_prompt": SEED_PROMPT}

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
            f"[gepa-ai crafter] done in {elapsed:.1f}s, rollouts={adapter.rollout_count}",
            flush=True,
        )

        heldout_inputs = [
            {"seed": int(r["seed"]), "split": "test", "example_id": r.get("example_id", "")}
            for r in heldout_rows
        ]

        candidate_results = []
        print(
            f"[gepa-ai crafter] evaluating {len(result.candidates)} candidates on heldout …",
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
            f"[gepa-ai crafter] seed heldout={seed_heldout:.3f}  best heldout={best_heldout:.3f}  "
            f"lift={best_heldout - seed_heldout:+.3f}",
            flush=True,
        )

        summary = {
            "task": "crafter",
            "max_turns": MAX_TURNS,
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
        print(f"[gepa-ai crafter] summary → {output_dir / 'summary.json'}", flush=True)

    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
