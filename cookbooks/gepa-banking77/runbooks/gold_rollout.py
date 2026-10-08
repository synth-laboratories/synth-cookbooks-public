from __future__ import annotations

import argparse
import json
import sys
import urllib.request


TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}


def _json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"content-type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    health = _json("GET", f"{base}/health")
    if health.get("status") != "ok":
        print(json.dumps(health), file=sys.stderr)
        return 1
    rollout_id = "cookbook_gold_0"
    _json(
        "POST",
        f"{base}/rollouts/prepare",
        {"rollout_id": rollout_id, "telemetry": TELEMETRY},
    )
    started = _json(
        "POST",
        f"{base}/rollouts",
        {
            "rollout_id": rollout_id,
            "telemetry": TELEMETRY,
            "slot": "stream",
            "world_ref": "world:banking77@train",
            "task_instance_id": "seed:0",
            "evaluation_plan_ref": "banking77_eval.v1",
            "policy_ref": {"harness": "dataset_gold", "config": "dataset_gold"},
        },
    )
    scored = _json(
        "POST",
        f"{base}/reward",
        {"rollout_id": rollout_id, "mode": "terminal"},
    )
    receipt = {
        "target": health.get("target"),
        "rollout_id": started.get("rollout_id"),
        "reward": scored.get("reward"),
        "status": scored.get("status"),
    }
    print(json.dumps(receipt, indent=2))
    return 0 if scored.get("reward") == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
