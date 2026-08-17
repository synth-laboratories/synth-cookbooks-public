from fastapi.testclient import TestClient

from synth_service_app import create_app


def test_healthbench2_contract_is_truthful_and_gepa_ready(tmp_path):
    client = TestClient(create_app(storage_root=tmp_path))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["target"] == "healthbench_chat"

    metadata = client.get("/metadata").json()
    assert metadata["runtime_family"] == "healthbench"
    assert metadata["live_frames"] == "unsupported"
    assert metadata["metadata"]["optimizer_contracts"]["gepa"]["version"] == "synth_optimizers.gepa.v2"

    program = client.get("/program").json()
    assert program["target_modules"][0]["candidate_field"] == "system_prompt"
    assert set(program["seed_candidate"]) == {"system_prompt"}

    task = client.get("/task_info").json()
    assert task["output_kind"] == "open_text"
    assert task["literal_training_targets"] == "forbidden"


def test_healthbench2_unknown_rollout_is_not_fabricated(tmp_path):
    client = TestClient(create_app(storage_root=tmp_path))
    response = client.post("/reward", json={"rollout_id": "missing", "mode": "terminal"})
    assert response.status_code == 404
