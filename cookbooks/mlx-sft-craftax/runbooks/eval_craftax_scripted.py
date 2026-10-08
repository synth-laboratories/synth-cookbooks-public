from __future__ import annotations

import json

from fastapi.testclient import TestClient
from synth_containers.platform import create_compat_app

TELEMETRY = {"enabled": True, "transport": "sse"}
WORLD_REF = "world:craftax_default@symbolic_survival"


def run(*, seeds: int = 2) -> dict:
    client = TestClient(create_compat_app("craftax_engine"))
    client.post(
        "/policy-configs",
        json={
            "config_id": "scripted",
            "config": {"model": "scripted", "observation_mode": "text"},
        },
    )
    rows = []
    for seed in range(seeds):
        rollout_id = f"craftax_sft_eval_{seed}"
        prepared = client.post(
            "/rollouts/prepare",
            json={"rollout_id": rollout_id, "telemetry": TELEMETRY},
        )
        assert prepared.status_code == 200, prepared.text
        started = client.post(
            "/rollouts",
            json={
                "rollout_id": rollout_id,
                "telemetry": TELEMETRY,
                "slot": "stream",
                "world_ref": WORLD_REF,
                "task_instance_id": f"seed:{seed}",
                "policy_ref": {"harness": "react", "config": "scripted"},
            },
        )
        assert started.status_code == 200, started.text
        scored = client.post("/reward", json={"rollout_id": rollout_id, "mode": "terminal"})
        assert scored.status_code == 200, scored.text
        body = scored.json()
        rows.append(
            {
                "seed": seed,
                "status": body.get("status"),
                "reward": body.get("reward"),
                "environment_ref": started.json().get("environment_ref"),
            }
        )
    return {
        "target": "craftax_engine",
        "n": len(rows),
        "rows": rows,
        "note": "scripted fixture eval; not /evals/craftax gold CLI",
    }


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2))
    assert all(row["environment_ref"] == "env:craftax_fixture" for row in result["rows"])
    assert all(row["status"] in {"scored", "absent"} for row in result["rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
