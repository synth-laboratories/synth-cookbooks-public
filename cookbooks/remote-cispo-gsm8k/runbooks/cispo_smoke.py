"""One CISPO minimax step on the current synth-mlx-rl HTTP contract."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from synth_mlx_rl.api.app import create_app
from synth_mlx_rl.testing import FakeEngine


def run() -> dict:
    with TestClient(create_app(engine=FakeEngine())) as client:
        cap = client.get("/v1/synth/capability").json()
        assert "cispo_minimax" in cap["objectives"]["supported"]
        datum = {
            "input_ids": [1, 2, 3],
            "target_ids": [2, 3, 4],
            "weights": [0.0, 1.0, 1.0],
            "behavior_logprobs": [0.0, -0.5, -0.4],
            "advantages": 1.5,
        }
        response = client.post(
            "/v1/forward_backward",
            json={
                "data": [datum],
                "loss_fn": "cispo_minimax",
                "eps_low": 1.0,
                "eps_high": 4.0,
            },
        )
        assert response.status_code == 200, response.text
        refused = client.post(
            "/v1/forward_backward",
            json={"data": [datum], "loss_fn": "cispo_minimax", "eps_low": 0.2},
        )
        assert refused.status_code == 422, refused.text
        step = client.post("/v1/optim_step", json={"params": {"learning_rate": 1e-5}})
        assert step.status_code == 200, step.text
        return {
            "objective": "cispo_minimax",
            "loss": response.json()["loss"],
            "training_version": step.json()["training_version"],
            "minimax_refusal": 422,
            "dataset": "gsm8k-fixture",
        }


def main() -> int:
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
