from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from synth_mlx_rl.api.app import create_app
from synth_mlx_rl.testing import FakeEngine


def test_gsm8k_fixture_present() -> None:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "gsm8k_tiny.json"
    payload = path.read_text()
    assert "Natalia" in payload


def test_cispo_minimax_http(tmp_path) -> None:
    with TestClient(create_app(engine=FakeEngine(tmp_path))) as client:
        cap = client.get("/v1/synth/capability").json()
        assert "cispo_minimax" in cap["objectives"]["supported"]
        datum = {
            "input_ids": [1, 2, 3],
            "target_ids": [2, 3, 4],
            "weights": [0.0, 1.0, 1.0],
            "behavior_logprobs": [0.0, -0.5, -0.4],
            "advantages": 1.5,
        }
        ok = client.post(
            "/v1/forward_backward",
            json={
                "data": [datum],
                "loss_fn": "cispo_minimax",
                "eps_low": 1.0,
                "eps_high": 4.0,
            },
        )
        assert ok.status_code == 200, ok.text
        refused = client.post(
            "/v1/forward_backward",
            json={"data": [datum], "loss_fn": "cispo_minimax", "eps_low": 0.2},
        )
        assert refused.status_code == 422
