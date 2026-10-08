from __future__ import annotations

from fastapi.testclient import TestClient
from synth_containers.platform import create_compat_app
from synth_mlx_rl.api.app import create_app
from synth_mlx_rl.testing import FakeEngine


TELEMETRY = {"enabled": True, "transport": "sse"}


def test_craftax_fixture_scripted_eval() -> None:
    client = TestClient(create_compat_app("craftax_engine"))
    posted = client.post(
        "/policy-configs",
        json={"config_id": "scripted", "config": {"model": "scripted"}},
    )
    assert posted.status_code == 200, posted.text
    rollout_id = "craftax_sft_eval_0"
    assert (
        client.post(
            "/rollouts/prepare",
            json={"rollout_id": rollout_id, "telemetry": TELEMETRY},
        ).status_code
        == 200
    )
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rollout_id,
            "telemetry": TELEMETRY,
            "slot": "stream",
            "world_ref": "world:craftax_default@symbolic_survival",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "react", "config": "scripted"},
        },
    )
    assert started.status_code == 200, started.text
    assert started.json()["environment_ref"] == "env:craftax_fixture"
    scored = client.post("/reward", json={"rollout_id": rollout_id, "mode": "terminal"})
    assert scored.status_code == 200, scored.text
    assert scored.json()["status"] in {"scored", "absent"}


def test_mlx_sft_cross_entropy_checkpoint(tmp_path) -> None:
    with TestClient(create_app(engine=FakeEngine(tmp_path))) as client:
        cap = client.get("/v1/synth/capability").json()
        assert "cross_entropy" in cap["objectives"]["supported"]
        datum = {"input_ids": [1, 2, 3], "target_ids": [2, 3, 4], "weights": [0.0, 1.0, 1.0]}
        forward = client.post(
            "/v1/forward_backward", json={"data": [datum], "loss_fn": "cross_entropy"}
        )
        assert forward.status_code == 200, forward.text
        step = client.post("/v1/optim_step", json={"params": {"learning_rate": 5e-5}})
        assert step.status_code == 200
        saved = client.post("/v1/checkpoints/save", json={"name": "sft-craftax-smoke"})
        assert saved.status_code == 200
