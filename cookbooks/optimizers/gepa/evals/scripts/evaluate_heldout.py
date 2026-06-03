"""Re-evaluate every candidate from proposal_timeline.jsonl on the heldout set.

Starts the banking77 container, runs each unique candidate payload against
every heldout seed, and writes two files:

  evidence/heldout_evaluations.jsonl — one row per (candidate, heldout_seed)
  evidence/candidate_timeline.jsonl  — one row per candidate, aggregated

Candidates already in candidate_timeline.jsonl are skipped (idempotent).

Usage (from evals/):
    python scripts/evaluate_heldout.py --benchmark banking77
    python scripts/evaluate_heldout.py --benchmark banking77 --concurrency 60
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import socket
import subprocess
import sys
import time
import tomllib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = EVALS_DIR / "evidence"
SYNTH_AI_ENV = EVALS_DIR.parents[3].parent / "synth-ai" / ".env"


def load_env() -> None:
    if SYNTH_AI_ENV.is_file():
        for line in SYNTH_AI_ENV.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(benchmark: str) -> dict:
    cfg_path = EVALS_DIR / "configs" / f"{benchmark}.toml"
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)


def pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def eval_concurrency(cfg: dict) -> int:
    return int((cfg.get("limits") or {}).get("eval_concurrency", 256))


def eval_workers(cfg: dict) -> int:
    return int((cfg.get("limits") or {}).get("eval_workers", 8))


def benchmark_container_dir(cfg: dict) -> Path:
    bench = cfg["benchmark"]
    dirname = bench.get("container_dir") or f"{bench['name']}_container"
    return EVALS_DIR.parent / dirname


def benchmark_env_prefix(cfg: dict) -> str:
    bench = cfg["benchmark"]
    return bench.get("env_prefix") or bench["name"].upper().replace("-", "_")


def start_container(port: int, cfg: dict) -> subprocess.Popen:
    bench = cfg["benchmark"]
    pc = cfg["parity_controls"]
    ds = cfg["dataset"]
    heldout_seeds = ds["heldout_seeds"]
    prefix = benchmark_env_prefix(cfg)
    workers = eval_workers(cfg) if bench["name"] == "banking77" else 1
    container_dir = benchmark_container_dir(cfg)
    env = {
        **os.environ,
        # Heldout/train re-eval fans out hard across multiple uvicorn workers so
        # no single event loop is saturated (keeps /health responsive and pushes
        # more concurrent policy calls). Effective concurrency = workers ×
        # per-worker semaphore ≈ eval_concurrency.
        f"{prefix}_POLICY_CONCURRENCY": str(max(1, eval_concurrency(cfg) // workers)),
        f"{prefix}_POLICY_MODEL": pc["policy_model"],
        f"{prefix}_POLICY_API_KEY_ENV": pc["policy_api_key_env"],
        f"{prefix}_POLICY_MAX_TOKENS": str(pc.get("policy_max_tokens", 16)),
        f"{prefix}_POLICY_RETRIES": str(pc.get("eval_policy_retries", pc.get("policy_retries", 4))),
        f"{prefix}_ROLLOUT_TIMEOUT_SECONDS": str(pc.get("rollout_timeout_seconds", 30)),
        f"{prefix}_POLICY_TIMEOUT_SECONDS": str(pc.get("policy_timeout_seconds", 25)),
    }
    if pc.get("judge_model"):
        env[f"{prefix}_JUDGE_MODEL"] = pc["judge_model"]
    if pc.get("judge_max_tokens"):
        env[f"{prefix}_JUDGE_MAX_TOKENS"] = str(pc["judge_max_tokens"])
    if pc.get("policy_base_url"):
        env[f"{prefix}_POLICY_BASE_URL"] = pc["policy_base_url"]
        env["OPENAI_BASE_URL"] = pc["policy_base_url"]
    if bench["name"] == "tau2_retail":
        env["TAU2_RETAIL_AGENT_MODEL"] = pc["policy_model"]
    if bench.get("synthetic_train_cap"):
        env[f"{prefix}_TRAIN_CAP"] = str(bench["synthetic_train_cap"])
    if bench["name"] == "banking77":
        env.update({
            "BANKING77_TRAIN_SAMPLE": str(max(ds["train_seeds"]) + 1),
            "BANKING77_TEST_SAMPLE": str(max(heldout_seeds) + 1),
            "BANKING77_TRAIN_SHUFFLE_SEED": str(bench["train_shuffle_seed"]),
            "BANKING77_TEST_SHUFFLE_SEED": str(bench["test_shuffle_seed"]),
            "BANKING77_WORKERS": str(workers),
            "BANKING77_POLICY_DISABLE_REASONING": "auto",
            "BANKING77_POLICY_API_MODE": "auto",
        })
    cmd = [
        "uv", "run", "--project", str(container_dir),
        "python", str(container_dir / "synth_service_app.py"),
        "--host", "127.0.0.1", "--port", str(port),
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(container_dir.parent),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_health(port: int, timeout: float = 180.0) -> None:
    import httpx
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=3.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("container did not become healthy in time")


def post_rollout(client, port: int, seed: int, split: str, payload: dict) -> dict:
    body = {
        "seed": seed,
        "split": split,
        "candidate": payload,
        "submission_mode": "sync",
        "rollout_id": f"eval_{split}_{seed}_{uuid.uuid4().hex[:8]}",
    }
    r = client.post(f"http://127.0.0.1:{port}/rollout", json=body, timeout=120)
    r.raise_for_status()
    return r.json()


def load_proposal_timeline() -> list[dict]:
    path = EVIDENCE_DIR / "proposal_timeline.jsonl"
    if not path.exists():
        raise SystemExit("evidence/proposal_timeline.jsonl not found. Run extract_candidates.py first.")
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def load_already_evaluated(path: Path) -> set[str]:
    """Candidate keys already present in a per-row evaluations file."""
    done: set[str] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add(f"{r.get('stack')}:{r.get('run_id')}:{r.get('candidate_id')}")
            except json.JSONDecodeError:
                pass
    return done


def candidate_key(row: dict) -> str:
    return f"{row.get('stack')}:{row.get('run_id')}:{row.get('candidate_id')}"


MAX_ROLLOUT_ATTEMPTS = 8


def _run_one(client, port: int, split: str, payload: dict, seed: int) -> dict:
    """One (candidate, seed) rollout. Returns a row dict (never raises).

    Transient failures (HTTP 5xx/timeout/connection errors from the container or
    OpenRouter) are RETRIED with jittered backoff, so they do not corrupt the
    score as a fake reward=0. A successful rollout that is merely a wrong
    prediction (HTTP 200, reward 0) is a real data point and is NOT retried —
    only exceptions are. Only after exhausting all attempts is an error recorded.
    """
    last_exc: Exception | None = None
    attempts = 0
    for attempt in range(1, MAX_ROLLOUT_ATTEMPTS + 1):
        attempts = attempt
        try:
            r = post_rollout(client, port, seed, split, payload)
            reward = float((r.get("reward_info") or {}).get("outcome_reward", 0.0))
            details = (r.get("reward_info") or {}).get("details") or {}
            pred = details.get("prediction") or (r.get("summary") or {}).get("prediction", "")
            expected = details.get("expected") or (r.get("summary") or {}).get("expected", "")
            usage = r.get("usage") or {}
            return {
                "seed": seed, "reward": reward, "prediction": pred, "expected": expected,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "reward_details": details,
                "error": None, "attempts": attempt,
            }
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_ROLLOUT_ATTEMPTS:
                # Exponential backoff + jitter so retries spread out and let a
                # rate-limited / saturated provider recover. Runs in a worker
                # thread, so sleeping does not block other rollouts.
                delay = min(0.5 * (2 ** (attempt - 1)), 10.0) + random.uniform(0, 0.5)
                time.sleep(delay)
    return {
        "seed": seed, "reward": 0.0, "prediction": "", "expected": "",
        "prompt_tokens": 0, "completion_tokens": 0,
        "reward_details": {},
        "error": f"{type(last_exc).__name__}: {last_exc}", "attempts": attempts,
    }


def main() -> int:
    load_env()
    p = argparse.ArgumentParser(description="Re-evaluate all candidates on a split.")
    p.add_argument("--benchmark", default="banking77")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--stacks", nargs="*", default=None, help="Filter by stack names")
    p.add_argument("--split", choices=["test", "train"], default="test",
                   help="Which split to score every candidate on. test=heldout (canonical), "
                        "train=the seeds the optimizer searched over.")
    args = p.parse_args()

    import httpx

    cfg = load_config(args.benchmark)
    is_heldout = args.split == "test"
    eval_seeds = cfg["dataset"]["heldout_seeds"] if is_heldout else cfg["dataset"]["train_seeds"]
    label = "heldout" if is_heldout else "train"
    concurrency = args.concurrency or eval_concurrency(cfg)

    candidates = load_proposal_timeline()
    candidates = [c for c in candidates if c.get("benchmark") == args.benchmark]
    if args.stacks:
        candidates = [c for c in candidates if c["stack"] in args.stacks]

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    evals_path = EVIDENCE_DIR / f"{label}_evaluations.jsonl"
    candidate_timeline_path = EVIDENCE_DIR / "candidate_timeline.jsonl"

    # Idempotency keyed off the per-row evaluations file for this split.
    already_done = load_already_evaluated(evals_path)
    to_eval = [
        c for c in candidates
        if candidate_key(c) not in already_done
    ]

    if not to_eval:
        print(f"All candidates already evaluated on {label}. Nothing to do.", flush=True)
        return 0

    # Flatten EVERY (candidate, seed) into one work list — they are all independent
    # rollouts — and fan out across one big pool. Wall-clock scales with
    # total_rollouts / concurrency instead of summing per-candidate passes.
    payloads = {id(c): json.loads(c["candidate_payload_json"]) for c in to_eval}
    work = [(c, seed) for c in to_eval for seed in eval_seeds]
    total = len(work)
    print(
        f"Evaluating {len(to_eval)} candidates × {len(eval_seeds)} {label} seeds "
        f"= {total} rollouts at concurrency {concurrency}.",
        flush=True,
    )

    rows_by_cand: dict[str, list[dict]] = {candidate_key(c): [] for c in to_eval}
    port = pick_free_port()
    proc = start_container(port, cfg)
    t0 = time.time()
    try:
        wait_for_health(port)
        print(f"Container healthy on port {port}", flush=True)

        limits = httpx.Limits(max_connections=concurrency + 50,
                              max_keepalive_connections=concurrency + 50)
        done = 0
        with httpx.Client(limits=limits) as client, ThreadPoolExecutor(max_workers=concurrency) as pool:
            fut_to_cand = {
                pool.submit(_run_one, client, port, args.split, payloads[id(c)], seed): c
                for c, seed in work
            }
            for fut in as_completed(fut_to_cand):
                cand = fut_to_cand[fut]
                rows_by_cand[candidate_key(cand)].append(fut.result())
                done += 1
                if done % 200 == 0 or done == total:
                    rate = done / max(1e-6, time.time() - t0)
                    print(f"  {done}/{total} rollouts ({rate:.0f}/s)", flush=True)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    elapsed = time.time() - t0
    errors = 0
    retried = 0
    with open(evals_path, "a") as ef, open(candidate_timeline_path, "a") as cf:
        for cand in to_eval:
            cid = cand["candidate_id"]
            per_row = rows_by_cand[candidate_key(cand)]
            scores = [r["reward"] for r in per_row]
            mean_score = sum(scores) / len(scores) if scores else 0.0
            errors += sum(1 for r in per_row if r["error"])
            retried += sum(1 for r in per_row if (r.get("attempts") or 1) > 1)
            for r in per_row:
                ef.write(json.dumps({
                    "stack": cand["stack"], "benchmark": args.benchmark,
                    "run_id": cand["run_id"], "candidate_id": cid, "split": args.split,
                    "heldout_seed": r["seed"], "reward": r["reward"],
                    "prediction": r["prediction"], "expected": r["expected"],
                    "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                    "reward_details": r.get("reward_details") or {},
                    "error": r["error"],
                }) + "\n")
            print(f"  {cand['stack']:11s} {cid[:18]} {label}={mean_score:.3f} (n={len(per_row)})", flush=True)
            if not is_heldout:
                continue
            cf.write(json.dumps({
                "stack": cand["stack"], "benchmark": args.benchmark, "run_id": cand["run_id"],
                "candidate_id": cid, "candidate_index": cand.get("candidate_index"),
                "generated_at": cand.get("created_at"), "evaluated_at": cand.get("evaluated_at"),
                "elapsed_seconds": cand.get("elapsed_seconds"),
                "cumulative_rollout_count": cand.get("cumulative_rollout_count"),
                "cumulative_rollout_cost_usd": cand.get("cumulative_rollout_cost_usd"),
                "cumulative_proposer_cost_usd": cand.get("cumulative_proposer_cost_usd"),
                "cumulative_cost_usd": cand.get("cumulative_cost_usd"),
                "proposer_model": cand.get("proposer_model"),
                "policy_model": cfg["parity_controls"]["policy_model"],
                "train_score": cand.get("train_score"),
                "train_heldout_score": cand.get("train_heldout_score"),
                "heldout_score": mean_score, "heldout_n": len(per_row),
                "accepted": cand.get("acceptance_score") is not None,
                "parent_candidate_id": cand.get("parent_candidate_id"),
                "generation": cand.get("generation"), "proposal_source": cand.get("proposal_source"),
                "candidate_payload_json": cand.get("candidate_payload_json"),
            }) + "\n")

    print(f"\nWrote {total} rows → {evals_path} in {elapsed:.0f}s "
          f"({total/max(1e-6,elapsed):.0f} rollouts/s, {retried} needed retries, "
          f"{errors} still errored after {MAX_ROLLOUT_ATTEMPTS} attempts)", flush=True)
    if is_heldout:
        print(f"Updated candidate timeline → {candidate_timeline_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
