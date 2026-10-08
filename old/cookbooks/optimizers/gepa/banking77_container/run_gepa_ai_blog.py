"""
gepa-ai on Banking77 — blog comparison run
==========================================
Same container, same n=24 train / n=200 heldout seeds as the Synth GEPA blog_n24 profile.
Writes results to ../runs/banking77_gepa_ai_blog_<timestamp>/summary.json.

Usage:
    cd cookbooks/optimizers/gepa/banking77_container
    uv run --project . python run_gepa_ai_blog.py

The script loads API keys from synth-ai/.env (OPENROUTER_API_KEY for policy,
OPENAI_API_KEY for proposer/reflector). Shuffles with fixed seeds so both
runs hit the same 24 train examples and 200 heldout examples.
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

# ── Keys ──────────────────────────────────────────────────────────────────────
ENV_FILE = Path(__file__).resolve().parents[5] / "synth-ai" / ".env"
if not ENV_FILE.exists():
    # fallback search
    for candidate in [
        Path.home() / "Documents/GitHub/synth-ai/.env",
        Path("/Users/joshpurtell/Documents/GitHub/synth-ai/.env"),
    ]:
        if candidate.exists():
            ENV_FILE = candidate
            break

# Keys loaded from .env: use setdefault so shell exports are not clobbered,
# EXCEPT for OPENAI_API_KEY which we ALWAYS want to read from .env because
# the shell may have inherited an OpenRouter key (sk-or-...) from a prior
# context and setdefault would silently keep the wrong key.
_env_openai_key: str = ""
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k == "OPENAI_API_KEY":
            _env_openai_key = v   # take last occurrence
        else:
            os.environ.setdefault(k, v)
# Force the .env value for OPENAI_API_KEY so sk-or-v1 never leaks in.
if _env_openai_key:
    os.environ["OPENAI_API_KEY"] = _env_openai_key

import gepa  # noqa: E402
from gepa.core.adapter import EvaluationBatch, GEPAAdapter  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_SIZE    = 24
HELDOUT_SIZE  = 200
TRAIN_SHUFFLE_SEED   = 664514459   # fixed so both arms see same rows
HELDOUT_SHUFFLE_SEED = 4174331086

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

MAX_METRIC_CALLS = 600   # ≈ train budget (gepa-ai mixes train+val internally)
REFLECTION_MINIBATCH = 8
REFLECTION_LM = "openai/gpt-4.1-mini"   # litellm-style
POLICY_CONCURRENCY = 128


# ── Container ─────────────────────────────────────────────────────────────────

def pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_container(port: int) -> subprocess.Popen:
    # Explicitly resolve the API key value so the container subprocess always
    # has it even if uv's env inheritance is unreliable.
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set — cannot start container")
    # Build a clean env: inherit host env but strip:
    #   VIRTUAL_ENV       — uv sets this to its build cache; conflicts with the
    #                        container's own .venv and breaks the Python import env.
    #   OPENROUTER_API_KEY — synth_service_app.py checks OPENROUTER_API_KEY before
    #                        OPENAI_API_KEY (line 61). If present, the OpenRouter key
    #                        is used against api.openai.com → 401. Remove it so the
    #                        container falls through to OPENAI_API_KEY.
    _strip = {"VIRTUAL_ENV", "OPENROUTER_API_KEY"}
    base_env = {k: v for k, v in os.environ.items() if k not in _strip}
    env = {
        **base_env,
        # Explicit key so container can always find it regardless of inheritance
        "OPENAI_API_KEY": openai_key,
        "BANKING77_TRAIN_SAMPLE": str(TRAIN_SIZE),
        "BANKING77_TEST_SAMPLE": str(HELDOUT_SIZE),
        "BANKING77_POLICY_CONCURRENCY": str(POLICY_CONCURRENCY),
        "BANKING77_POLICY_MODEL": "gpt-4.1-nano",
        "BANKING77_POLICY_BASE_URL": "https://api.openai.com/v1",
        "BANKING77_POLICY_API_KEY_ENV": "OPENAI_API_KEY",
        "BANKING77_POLICY_DISABLE_REASONING": "auto",
        "BANKING77_POLICY_MAX_TOKENS": "64",
        "BANKING77_POLICY_RETRIES": "2",
        "BANKING77_POLICY_API_MODE": "auto",
        "BANKING77_ROLLOUT_TIMEOUT_SECONDS": "30",
        "BANKING77_POLICY_TIMEOUT_SECONDS": "25",
        "BANKING77_TRAIN_SHUFFLE_SEED": str(TRAIN_SHUFFLE_SEED),
        "BANKING77_TEST_SHUFFLE_SEED": str(HELDOUT_SHUFFLE_SEED),
    }
    # Must run from inside banking77_container/ so uv finds pyproject.toml
    # and synth_service_app.py is at cwd level.
    container_dir = Path(__file__).parent
    cmd = [
        "uv", "run",
        "python", "synth_service_app.py",
        "--host", "127.0.0.1", "--port", str(port),
    ]
    proc = subprocess.Popen(cmd, cwd=container_dir, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    r = httpx.post(f"http://127.0.0.1:{port}/dataset/rows",
                   json={"split": split, "seeds": seeds}, timeout=30)
    r.raise_for_status()
    return r.json()["rows"]


def post_rollout(port: int, seed: int, split: str, system_prompt: str) -> dict:
    body = {
        "seed": seed, "split": split,
        "candidate": {"stage2_system": system_prompt},
        "submission_mode": "sync",
        "rollout_id": f"gai_{split}_{seed}_{int(time.time()*1000)%1000000}",
    }
    r = httpx.post(f"http://127.0.0.1:{port}/rollout", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


# ── Adapter ───────────────────────────────────────────────────────────────────

class Banking77Adapter(GEPAAdapter):
    def __init__(self, port: int, max_workers: int = 40):
        self.port = port
        self.max_workers = max_workers
        self.rollout_count = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        system_prompt = candidate["stage2_system"]
        outputs = [None] * len(batch)
        scores  = [0.0]  * len(batch)
        traj    = [None] * len(batch) if capture_traces else None

        def _do(i_data):
            i, data = i_data
            try:
                result = post_rollout(self.port, seed=data["seed"],
                                      split=data["split"], system_prompt=system_prompt)
                reward = float(result.get("reward_info", {}).get("outcome_reward", 0.0))
                pred     = (result.get("summary", {}) or {}).get("prediction", "")
                expected = (result.get("summary", {}) or {}).get("expected", "")
                return i, reward, pred, expected
            except Exception as exc:
                return i, 0.0, "", f"<error: {exc!r}>"

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for fut in as_completed([ex.submit(_do, x) for x in enumerate(batch)]):
                i, reward, pred, expected = fut.result()
                scores[i]  = reward
                outputs[i] = {"prediction": pred, "expected": expected}
                if traj is not None:
                    traj[i] = {
                        "data": batch[i], "prediction": pred, "expected": expected,
                        "feedback": (
                            f"Correct. Predicted '{pred}'." if reward >= 1.0
                            else f"Wrong. Predicted '{pred}', expected '{expected}'."
                        ),
                    }
        self.rollout_count += len(batch)
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traj)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        comp  = components_to_update[0]
        items = []
        for t in eval_batch.trajectories:
            items.append({
                "Inputs":            f"Customer query: {t['data']['text']}",
                "Generated Outputs": f"Predicted label: {t['prediction']}",
                "Feedback": (
                    f"Correct label was '{t['expected']}'. "
                    f"Predicted '{t['prediction']}'. "
                    + ("Correct." if t["prediction"] == t["expected"]
                       else "Wrong — the predicted label is a close but different intent.")
                ),
            })
        return {comp: items}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "runs" / f"banking77_gepa_ai_blog_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    port = pick_free_port()
    print(f"[gepa-ai blog] starting container on port {port}", flush=True)
    proc = start_container(port)

    try:
        wait_for_health(port)
        print("[gepa-ai blog] container healthy", flush=True)

        train_seeds   = list(range(TRAIN_SIZE))
        heldout_seeds = list(range(HELDOUT_SIZE))
        train_rows    = fetch_rows(port, "train", train_seeds)
        heldout_rows  = fetch_rows(port, "test",  heldout_seeds)
        print(f"[gepa-ai blog] rows fetched: train={len(train_rows)} heldout={len(heldout_rows)}", flush=True)

        trainset = [
            {"seed": int(r["seed"]), "split": "train",
             "text": r["text"], "label": r["label"]}
            for r in train_rows
        ]
        valset = trainset  # gepa-ai uses full trainset as val too

        adapter = Banking77Adapter(port=port, max_workers=POLICY_CONCURRENCY)
        seed_candidate = {"stage2_system": SEED_PROMPT}

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
        print(f"[gepa-ai blog] done in {elapsed:.1f}s, rollouts={adapter.rollout_count}", flush=True)

        # ── Per-candidate heldout evals ──────────────────────────────────────
        heldout_inputs = [
            {"seed": int(r["seed"]), "split": "test",
             "text": r["text"], "label": r["label"]}
            for r in heldout_rows
        ]

        candidate_results = []
        print(f"[gepa-ai blog] evaluating {len(result.candidates)} candidates on heldout …", flush=True)
        for idx, cand in enumerate(result.candidates):
            adapter.rollout_count = 0
            h_eval = adapter.evaluate(heldout_inputs, cand, capture_traces=False)
            h_score = sum(h_eval.scores) / len(h_eval.scores)
            val_score = (result.val_aggregate_scores[idx]
                         if idx < len(result.val_aggregate_scores) else None)
            candidate_results.append({
                "idx":        idx,
                "heldout":    round(h_score, 6),
                "val_score":  val_score,
                "candidate":  cand,
            })
            print(f"  candidate {idx:2d}: heldout={h_score:.3f}", flush=True)

        # Seed baseline heldout
        adapter.rollout_count = 0
        seed_h_eval = adapter.evaluate(heldout_inputs, seed_candidate, capture_traces=False)
        seed_heldout = sum(seed_h_eval.scores) / len(seed_h_eval.scores)

        best_heldout = max(c["heldout"] for c in candidate_results)
        print(f"[gepa-ai blog] seed heldout={seed_heldout:.3f}  best heldout={best_heldout:.3f}  "
              f"lift={best_heldout - seed_heldout:+.3f}", flush=True)

        summary = {
            "gepa_version":   getattr(gepa, "__version__", "unknown"),
            "elapsed_seconds": elapsed,
            "train_size":     TRAIN_SIZE,
            "heldout_size":   HELDOUT_SIZE,
            "max_metric_calls": MAX_METRIC_CALLS,
            "seed_prompt":    SEED_PROMPT,
            "best_candidate": result.best_candidate,
            "seed_heldout":   round(seed_heldout, 6),
            "best_heldout":   round(best_heldout, 6),
            "lift":           round(best_heldout - seed_heldout, 6),
            "candidates":     candidate_results,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[gepa-ai blog] summary → {output_dir / 'summary.json'}", flush=True)

    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
