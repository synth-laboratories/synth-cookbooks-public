from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
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
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8114")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = tomllib.loads(Path(args.config).read_text())
    base = args.url.rstrip("/")
    task_ids = list(config["taskset"]["train_ids"]) + list(config["taskset"]["heldout_ids"])
    rows = []
    for index, task_id in enumerate(task_ids):
        rollout_id = f"hb_eval_{index}"
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
                "task_instance_id": task_id,
                "policy_ref": {
                    "harness": "chat_completion",
                    "config": config["policy"]["config_id"],
                },
            },
        )
        scored = _json(
            "POST",
            f"{base}/reward",
            {"rollout_id": rollout_id, "mode": "terminal"},
        )
        rows.append(
            {
                "task_id": task_id,
                "status": scored.get("status"),
                "reward": scored.get("reward"),
                "rollout_id": started.get("rollout_id"),
            }
        )
    print(json.dumps({"n": len(rows), "results": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
