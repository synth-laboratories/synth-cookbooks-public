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
    roles = metadata["metadata"]["model_roles"]
    assert roles["policy"]["configuration_authority"] == "policy_ref"
    assert roles["policy"]["usage_lane"] == "policy"
    assert roles["scorer"]["model"] == "gpt-4.1-2025-04-14"
    assert roles["scorer"]["api_key_env"] == "OPENAI_API_KEY"
    assert roles["scorer"]["usage_lane"] == "grader"
    assert roles["scorer"]["canonical"] is True
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


def test_healthbench2_terminal_rollout_survives_service_reconstruction(tmp_path, monkeypatch):
    from synth_containers.platform.runtimes import healthbench

    monkeypatch.setattr(healthbench, "load_row", lambda _seed: None)
    first = TestClient(create_app(storage_root=tmp_path))
    prepared = first.post(
        "/rollouts/prepare",
        json={
            "rollout_id": "healthbench2-restart",
            "telemetry": {"enabled": True, "transport": "poll", "retention": "run"},
        },
    )
    assert prepared.status_code == 200
    terminal = first.post(
        "/rollouts",
        json={
            "rollout_id": "healthbench2-restart",
            "task_instance_id": "seed:0",
            "slot": "stream",
            "telemetry": {"enabled": True, "transport": "poll", "retention": "run"},
            "policy_ref": {"harness": "chat_completion", "config": "groq_llama31_8b"},
        },
    )
    assert terminal.status_code == 200
    assert terminal.json()["status"] == "failed"

    restarted = TestClient(create_app(storage_root=tmp_path))
    recovered = restarted.get("/rollouts/healthbench2-restart")
    assert recovered.status_code == 200
    assert recovered.json()["rollout_id"] == "healthbench2-restart"
