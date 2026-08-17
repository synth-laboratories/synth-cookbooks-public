import asyncio
import json

import pytest

import synth_service_app as service


class JsonRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


@pytest.fixture
def isolated_rollout_store(tmp_path, monkeypatch):
    monkeypatch.setenv("BANKING77_STREAM_ROOT", str(tmp_path))
    service._ASYNC_ROLLOUTS.clear()
    service._STREAMS.clear()
    yield tmp_path
    service._ASYNC_ROLLOUTS.clear()
    service._STREAMS.clear()


def test_synchronous_rollout_is_retrievable_by_record_and_reward(
    isolated_rollout_store, monkeypatch
):
    completed = {
        "rollout_id": "sync-retained-1",
        "status": "completed",
        "success_status": "succeeded",
        "summary": {
            "outcome_reward": 1.0,
            "prediction": "activate_my_card",
            "expected": "activate_my_card",
        },
        "usage": {"total_tokens": 42},
        "metadata": {"candidate": {"stage2_system": "test"}},
    }

    async def execute(_payload):
        return dict(completed)

    monkeypatch.setattr(service, "_execute_rollout_payload_with_timeout", execute)
    result = asyncio.run(
        service.rollout(
            JsonRequest(
                {
                    "rollout_id": "sync-retained-1",
                    "submission_mode": "sync",
                    "telemetry": {"transport": "poll"},
                }
            )
        )
    )

    assert result["metadata"]["submission_mode"] == "sync"
    assert (
        asyncio.run(service.rollout_record("sync-retained-1"))["summary"]
        == completed["summary"]
    )
    assert asyncio.run(service.reward("sync-retained-1"))["reward"] == 1.0
    assert asyncio.run(
        service.reward_post(JsonRequest({"rollout_id": "sync-retained-1"}))
    )["reward"] == 1.0
    assert (
        asyncio.run(
            service.rollout(
                JsonRequest(
                    {
                        "rollout_id": "sync-retained-1",
                        "submission_mode": "sync",
                        "telemetry": {"transport": "sse"},
                    }
                )
            )
        )["summary"]
        == completed["summary"]
    )


def test_synchronous_rollout_survives_process_memory_loss(
    isolated_rollout_store, monkeypatch
):
    completed = {
        "rollout_id": "sync-restart-1",
        "status": "completed",
        "success_status": "failed",
        "summary": {
            "outcome_reward": 0.0,
            "prediction": "card_arrival",
            "expected": "activate_my_card",
        },
        "usage": {"total_tokens": 43},
        "metadata": {},
    }

    async def execute(_payload):
        return dict(completed)

    monkeypatch.setattr(service, "_execute_rollout_payload_with_timeout", execute)
    asyncio.run(
        service.rollout(
            JsonRequest(
                {
                    "rollout_id": "sync-restart-1",
                    "submission_mode": "sync",
                    "telemetry": {"transport": "poll"},
                }
            )
        )
    )
    record_path = service._rollout_record_path("sync-restart-1")
    envelope = json.loads(record_path.read_text())
    assert envelope["schema"] == service._ROLLOUT_RECORD_SCHEMA

    service._ASYNC_ROLLOUTS.clear()
    restored = asyncio.run(service.rollout_record("sync-restart-1"))
    assert restored["summary"]["prediction"] == "card_arrival"
    assert asyncio.run(service.reward("sync-restart-1"))["reward"] == 0.0


def test_desktop_contract_is_truthfully_advertised_and_resolved():
    info = asyncio.run(service.metadata())
    capabilities = info["capabilities"]
    assert capabilities["protocol"] == "synth.container.live-eval.v1"
    assert capabilities["operations"] == {
        "rollouts.prepare": True,
        "rollouts.start_prepared": True,
        "rollouts.get": True,
        "rollouts.poll": True,
        "reward.get": True,
        "trace_v5.capture": False,
    }
    assert capabilities["policy_refs"] == [service.DESKTOP_EVAL_POLICY_REF]

    policy = service._require_policy(
        {"policy_ref": dict(service.DESKTOP_EVAL_POLICY_REF)}
    )
    assert policy["provider"] == "openrouter"
    assert policy["model"] == "openai/gpt-4.1-nano"
    assert policy["base_url"] == "https://openrouter.ai/api/v1"

    stream = service._stream_descriptor("prepared-1")
    assert stream["transports"]["poll"]["url"].endswith("/events")
    assert stream["transports"]["sse"]["url"].endswith("/events/sse")


def test_unadvertised_desktop_policy_ref_is_rejected():
    with pytest.raises(service.HTTPException) as raised:
        service._require_policy(
            {"policy_ref": {"harness": "desktop_eval", "config": "made_up"}}
        )
    assert raised.value.status_code == 422
