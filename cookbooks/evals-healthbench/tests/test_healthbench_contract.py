from __future__ import annotations

from fastapi.testclient import TestClient
from synth_containers.platform import create_compat_app


def test_healthbench_eval_contract() -> None:
    client = TestClient(create_compat_app("healthbench_chat"))
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["target"] == "healthbench_chat"
    info = client.get("/info").json()
    gepa = info["capabilities"]["optimizer_contracts"]["gepa"]
    assert gepa["version"] == "synth_optimizers.gepa.v2"
    roles = info["metadata"]["model_roles"]
    assert roles["policy"]["usage_lane"] == "policy"
    assert roles["scorer"]["usage_lane"] == "grader"
    assert roles["scorer"]["canonical"] is True
    configs = {row["config"] for row in info["policy_refs"]}
    assert "openai_gpt41_mini" in configs
