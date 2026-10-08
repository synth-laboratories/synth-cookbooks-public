"""In-process SFT identity smoke using the FakeEngine (no Apple GPU required)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from synth_mlx_rl.api.app import create_app
from synth_mlx_rl.testing import FakeEngine


def main() -> int:
    engine = FakeEngine()
    with TestClient(create_app(engine=engine)) as client:
        cap = client.get("/v1/synth/capability").json()
        assert "cross_entropy" in cap["objectives"]["supported"]
        datum = {"input_ids": [1, 2, 3], "target_ids": [2, 3, 4], "weights": [0.0, 1.0, 1.0]}
        forward = client.post(
            "/v1/forward_backward", json={"data": [datum], "loss_fn": "cross_entropy"}
        )
        assert forward.status_code == 200, forward.text
        step = client.post("/v1/optim_step", json={"params": {"learning_rate": 5e-5}})
        assert step.status_code == 200, step.text
        saved = client.post("/v1/checkpoints/save", json={"name": "sft-craftax-smoke"})
        assert saved.status_code == 200, saved.text
        print(
            json.dumps(
                {
                    "loss_fn": "cross_entropy",
                    "loss": forward.json()["loss"],
                    "training_version": step.json()["training_version"],
                    "checkpoint": saved.json().get("name") or "sft-craftax-smoke",
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
