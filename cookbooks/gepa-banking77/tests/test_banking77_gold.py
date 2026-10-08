from __future__ import annotations

from fastapi.testclient import TestClient
from synth_containers.platform import create_compat_app


TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}


def test_banking77_gold_seed0_scores_one() -> None:
    client = TestClient(create_compat_app("banking77_classify"))
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["target"] == "banking77_classify"
    meta = client.get("/metadata").json()
    assert meta["runtime_family"] == "banking77"
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": "gold0", "telemetry": TELEMETRY},
    )
    assert prepared.status_code == 200, prepared.text
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "gold0",
            "telemetry": TELEMETRY,
            "slot": "stream",
            "world_ref": "world:banking77@train",
            "task_instance_id": "seed:0",
            "evaluation_plan_ref": "banking77_eval.v1",
            "policy_ref": {"harness": "dataset_gold", "config": "dataset_gold"},
        },
    )
    assert started.status_code == 200, started.text
    scored = client.post("/reward", json={"rollout_id": "gold0", "mode": "terminal"}).json()
    assert scored["status"] == "scored"
    assert scored["reward"] == 1.0
