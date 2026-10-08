"""Bigger live evals for Banking77 and HealthBench. Does not start MLX."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from synth_containers.platform import create_compat_app

ROOT = Path(__file__).resolve().parents[1]
ENV_CANDIDATES = [
    Path("/Users/joshuapurtell/Documents/GitHub/backend/.env.local"),
    Path("/Users/joshuapurtell/Documents/GitHub/synth-ai/.env"),
]
TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
BANKING77_N = 32
HEALTHBENCH_N = 8


def load_env() -> None:
    for path in ENV_CANDIDATES:
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and value and key not in os.environ:
                os.environ[key] = value
        return
    raise SystemExit("no .env found")


def merge_evidence(cookbook_id: str, key: str, payload: dict) -> None:
    path = ROOT / "cookbooks" / cookbook_id / "evidence" / "evidence.json"
    data = json.loads(path.read_text()) if path.is_file() else {"cookbook_id": cookbook_id}
    data[key] = payload
    path.write_text(json.dumps(data, indent=2) + "\n")


def banking77_batch() -> dict:
    client = TestClient(create_compat_app("banking77_classify"))
    registered = client.post(
        "/policy-configs",
        json={
            "config_id": "live_nano",
            "harness": "classify",
            "config": {
                "provider": "openai",
                "model": "gpt-4.1-nano",
                "api_key_env": "OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1",
                "temperature": 0,
                "max_tokens": 32,
            },
        },
    )
    assert registered.status_code == 200, registered.text
    rows = []
    for seed in range(BANKING77_N):
        rid = f"b77_batch_{seed}"
        client.post("/rollouts/prepare", json={"rollout_id": rid, "telemetry": TELEMETRY})
        started = client.post(
            "/rollouts",
            json={
                "rollout_id": rid,
                "telemetry": TELEMETRY,
                "slot": "stream",
                "world_ref": "world:banking77@heldout",
                "task_instance_id": f"seed:{seed}",
                "evaluation_plan_ref": "banking77_eval.v1",
                "policy_ref": {"harness": "classify", "config": "live_nano"},
            },
        )
        scored = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"}).json()
        rows.append(
            {
                "seed": seed,
                "http": started.status_code,
                "reward": scored.get("reward"),
                "status": scored.get("status"),
                "tokens": (started.json().get("usage") or {}).get("total_tokens")
                if started.status_code == 200
                else None,
            }
        )
        print(f"banking77 seed={seed} reward={scored.get('reward')} status={scored.get('status')}", flush=True)
    scored_rows = [row for row in rows if row["status"] == "scored" and row["reward"] is not None]
    hits = sum(1 for row in scored_rows if row["reward"] == 1.0)
    return {
        "proved_at": NOW,
        "split": "heldout",
        "n": BANKING77_N,
        "n_scored": len(scored_rows),
        "hits": hits,
        "accuracy": hits / len(scored_rows) if scored_rows else None,
        "model": "gpt-4.1-nano",
        "rows": rows,
    }


def _healthbench_one(seed: int) -> dict:
    os.environ["HEALTHBENCH_GRADER_MODEL"] = "gpt-4.1-mini-2025-04-14"
    client = TestClient(create_compat_app("healthbench_chat"))
    rid = f"hb_batch_{seed}"
    client.post("/rollouts/prepare", json={"rollout_id": rid, "telemetry": TELEMETRY})
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rid,
            "slot": "stream",
            "telemetry": TELEMETRY,
            "task_instance_id": f"seed:{seed}",
            "world_ref": "world:healthbench@eval",
            "evaluation_plan_ref": "healthbench_scaled_grader.v1",
            "policy_ref": {"harness": "chat_completion", "config": "openai_gpt41_mini"},
        },
    )
    scored = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"}).json()
    usage = started.json().get("usage") or {} if started.status_code == 200 else {}
    row = {
        "seed": seed,
        "http": started.status_code,
        "reward": scored.get("reward"),
        "status": scored.get("status"),
        "cost_usd": usage.get("cost_usd"),
    }
    print(f"healthbench seed={seed} reward={row['reward']} status={row['status']}", flush=True)
    return row


def healthbench_batch() -> dict:
    os.environ["HEALTHBENCH_GRADER_MODEL"] = "gpt-4.1-mini-2025-04-14"
    rows = []
    # Sequential: each row is a bundle of rubric calls; parallelism can stampede spend.
    for seed in range(HEALTHBENCH_N):
        rows.append(_healthbench_one(seed))
    scored_rows = [row for row in rows if row["status"] == "scored" and isinstance(row["reward"], (int, float))]
    mean = sum(float(row["reward"]) for row in scored_rows) / len(scored_rows) if scored_rows else None
    cost = sum(float(row["cost_usd"] or 0) for row in rows)
    return {
        "proved_at": NOW,
        "n": HEALTHBENCH_N,
        "n_scored": len(scored_rows),
        "mean_reward": mean,
        "estimated_cost_usd": cost,
        "grader": "gpt-4.1-mini-2025-04-14",
        "plan": "healthbench_scaled_grader.v1",
        "rows": rows,
    }


def hosted_catalog() -> dict:
    import urllib.request

    key = os.environ.get("SYNTH_API_KEY", "").strip()
    request = urllib.request.Request(
        "https://api.usesynth.ai/api/v1/optimizers/startup",
        headers={"authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode())
    algorithms = [
        row.get("algorithm")
        for row in payload.get("available_algorithms") or []
        if isinstance(row, dict)
    ]
    return {
        "proved_at": NOW,
        "ok": True,
        "algorithms": algorithms,
        "cispo_listed": any("cispo" in str(item) for item in algorithms),
        "n_algorithms": len(algorithms),
    }


def main() -> int:
    load_env()
    print("== banking77 heldout 32 ==")
    banking = banking77_batch()
    merge_evidence("gepa-banking77", "live_batch", banking)
    print(json.dumps({k: banking[k] for k in ("n", "n_scored", "hits", "accuracy")}, indent=2))
    print("== healthbench 8 ==")
    health = healthbench_batch()
    merge_evidence("evals-healthbench", "live_batch", health)
    print(
        json.dumps(
            {k: health[k] for k in ("n", "n_scored", "mean_reward", "estimated_cost_usd")},
            indent=2,
        )
    )
    print("== hosted catalog ==")
    hosted = hosted_catalog()
    merge_evidence("remote-cispo-gsm8k", "live_batch", hosted)
    print(json.dumps({k: hosted[k] for k in ("ok", "algorithms", "cispo_listed")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
