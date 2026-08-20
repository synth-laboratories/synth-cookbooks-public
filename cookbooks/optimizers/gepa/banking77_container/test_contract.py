from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, ClassVar


class _Labels:
    names: ClassVar[list[str]] = ["card_arrival", "cash_withdrawal"]


class _Features:
    def __getitem__(self, key: str) -> _Labels:
        if key != "label":
            raise KeyError(key)
        return _Labels()


class _Split(list[dict[str, Any]]):
    features = _Features()


def _fake_load_dataset(*_args: Any, **_kwargs: Any) -> dict[str, _Split]:
    return {
        "train": _Split(
            [
                {"text": "When will my card arrive?", "label": 0},
                {"text": "Why did the ATM decline me?", "label": 1},
            ]
        ),
        "test": _Split(
            [
                {"text": "Card delivery estimate", "label": 0},
                {"text": "Cash machine issue", "label": 1},
            ]
        ),
    }


def _load_service() -> types.ModuleType:
    os.environ["BANKING77_TRAIN_SAMPLE"] = "2"
    os.environ["BANKING77_TEST_SAMPLE"] = "2"
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = _fake_load_dataset  # type: ignore[attr-defined]
    sys.modules["datasets"] = datasets
    path = Path(__file__).with_name("synth_service_app.py")
    spec = importlib.util.spec_from_file_location("banking77_contract_service", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Request:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def json(self) -> dict[str, Any]:
        return self._payload


class Banking77GepaV2ContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.service = _load_service()
        self.tempdir = tempfile.TemporaryDirectory()
        os.environ["BANKING77_STREAM_ROOT"] = self.tempdir.name
        self.service._STREAMS.clear()
        self.service._ASYNC_ROLLOUTS.clear()

    async def asyncTearDown(self) -> None:
        self.tempdir.cleanup()
        os.environ.pop("BANKING77_STREAM_ROOT", None)

    async def test_metadata_advertises_only_implemented_gepa_v2_routes(self) -> None:
        metadata = await self.service.metadata()
        contract = metadata["metadata"]["optimizer_contracts"]["gepa"]
        self.assertEqual(contract["version"], "synth_optimizers.gepa.v2")
        for route_key in (
            "program_route",
            "taskset_route",
            "taskset_tasks_route",
            "dataset_route",
            "dataset_rows_route",
            "rollout_route",
        ):
            self.assertTrue(contract[route_key].startswith("/"))

    async def test_taskset_resolves_stable_split_ids(self) -> None:
        descriptor = await self.service.taskset()
        self.assertEqual(descriptor["splits"], {"train": 2, "test": 2})
        response = await self.service.taskset_tasks(
            _Request({"split": "train", "task_ids": ["train:0", "train:1"]})
        )
        self.assertEqual(
            [item["task_id"] for item in response["tasks"]], ["train:0", "train:1"]
        )
        self.assertEqual(response["tasks"][0]["task_instance_id"], "banking77:train:0")

    async def test_prepare_poll_and_restart_replay_use_a_monotonic_cursor(self) -> None:
        prepared = await self.service.prepare_rollout(
            _Request(
                {"rollout_id": "rollout-replay", "telemetry": {"transport": "poll"}}
            )
        )
        self.assertEqual(prepared["stream"]["schema"], "synth.rollout.stream.v1")
        initial = await self.service.rollout_events("rollout-replay", after=0)
        self.assertEqual(initial["events"][0]["kind"], "stream.subscribed")
        self.assertIsNone(initial["events"][0]["sequence"])

        await self.service._append_stream_event(
            "rollout-replay", "trace.opened", {"trace_id": "t1"}
        )
        await self.service._append_stream_event(
            "rollout-replay", "trace.closed", {"trace_id": "t1"}
        )
        tail = await self.service.rollout_events("rollout-replay", after=1)
        self.assertEqual(tail["cursor"]["high_water"], 2)
        self.assertEqual([event["sequence"] for event in tail["events"]], [2])

        self.service._STREAMS.clear()
        replayed = await self.service.rollout_events("rollout-replay", after=0)
        self.assertEqual(
            [event["sequence"] for event in replayed["events"]],
            [None, 1, 2],
        )

    async def test_async_rollout_emits_terminal_events_and_authoritative_reward(
        self,
    ) -> None:
        async def fake_predict(
            _text: str, *, policy: dict[str, Any], system_prompt: str
        ) -> tuple[str, dict[str, int]]:
            self.assertEqual(policy["model"], "test-model")
            self.assertTrue(system_prompt)
            return "card_arrival", {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
            }

        self.service._predict_label = fake_predict
        queued = await self.service.rollout(
            _Request(
                {
                    "rollout_id": "rollout-terminal",
                    "submission_mode": "async",
                    "task": {
                        "task_id": "train:0",
                        "seed": 0,
                        "split": "train",
                        "text": "When will my card arrive?",
                        "label": "card_arrival",
                    },
                    "policy": {
                        "provider": "openai",
                        "model": "test-model",
                        "api_family": "responses",
                        "credential_mode": "byok",
                    },
                    "telemetry": {"transport": "poll"},
                }
            )
        )
        self.assertEqual(queued["rollout_id"], "rollout-terminal")
        for _ in range(100):
            state = await self.service.rollout_state("rollout-terminal")
            if state["status"] in self.service._TERMINAL_ROLLOUT_STATUSES:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["rollout_id"], "rollout-terminal")

        events = await self.service.rollout_events("rollout-terminal", after=0)
        kinds = [event["kind"] for event in events["events"]]
        self.assertEqual(kinds[0], "stream.subscribed")
        self.assertEqual(kinds[-1], "trace.closed")
        self.assertIn("span.llm.closed", kinds)
        sequences = [
            event["sequence"]
            for event in events["events"]
            if event["sequence"] is not None
        ]
        self.assertEqual(sequences, list(range(1, len(sequences) + 1)))

        reward = await self.service.reward("rollout-terminal")
        self.assertEqual(reward["status"], "scored")
        self.assertEqual(reward["reward"], 1.0)
        self.assertEqual(reward["node_results"][0]["authority"], "environment")

    async def test_sync_rollout_keeps_reward_available_at_its_declared_url(
        self,
    ) -> None:
        async def fake_predict(
            _text: str, *, policy: dict[str, Any], system_prompt: str
        ) -> tuple[str, dict[str, int]]:
            return "cash_withdrawal", {
                "prompt_tokens": 8,
                "completion_tokens": 1,
                "total_tokens": 9,
            }

        self.service._predict_label = fake_predict
        completed = await self.service.rollout(
            _Request(
                {
                    "rollout_id": "rollout-sync",
                    "submission_mode": "sync",
                    "task": {
                        "task_id": "train:1",
                        "seed": 1,
                        "split": "train",
                        "text": "Why did the ATM decline me?",
                        "label": "cash_withdrawal",
                    },
                    "policy": {
                        "provider": "openai",
                        "model": "test-model",
                        "api_family": "responses",
                        "credential_mode": "byok",
                    },
                    "telemetry": {"transport": "poll"},
                }
            )
        )
        self.assertEqual(
            completed["stream"]["reward"]["url"], "/reward?rollout_id=rollout-sync"
        )
        reward = await self.service.reward("rollout-sync")
        self.assertEqual(reward["reward"], 1.0)


if __name__ == "__main__":
    unittest.main()


class _Headers:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)


class _Response:
    def __init__(self, status_code: int, headers: dict[str, str]) -> None:
        self.status_code = status_code
        self.headers = _Headers(headers)


class _FakeStatusError(Exception):
    """Shape-compatible stand-in for `openai.APIStatusError`."""

    def __init__(
        self,
        message: str,
        status_code: int,
        headers: dict[str, str] | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = _Response(status_code, headers or {})
        self.body = body


class PolicyFailureDiagnosticsTests(unittest.TestCase):
    """The v0.7 Banking77 boundary.

    Ten rollouts and two preflights recorded exactly
    ``Policy model 'openai/gpt-4.1-nano' failed through Chat Completions API
    after 1/1 attempts, timeout=20.0s.`` and a span carrying only
    ``error_type: HTTPException`` — so a Workshop proxy failure, an OpenRouter
    rejection, and a relayed upstream 502 were indistinguishable. These tests
    pin the cause to the record.
    """

    service: ClassVar[types.ModuleType]

    @classmethod
    def setUpClass(cls) -> None:
        cls.service = _load_service()

    def test_upstream_status_and_relay_origin_survive_into_the_detail(self) -> None:
        error = _FakeStatusError(
            "Bad gateway",
            502,
            headers={"x-workshop-proxy-origin": "proxy"},
            body={"error": {"code": "upstream_unreachable"}},
        )
        detail = self.service._policy_failure_detail(
            {"model": "openai/gpt-4.1-nano"}, error, 1, "Chat Completions API"
        )
        self.assertIn("upstream_status=502", detail)
        self.assertIn("relay_origin=proxy", detail)
        self.assertIn("upstream_code=upstream_unreachable", detail)

    def test_a_relayed_upstream_status_is_distinguishable_from_a_proxy_one(
        self,
    ) -> None:
        proxy_side = self.service._classify_policy_error(
            _FakeStatusError("x", 502, headers={"x-workshop-proxy-origin": "proxy"})
        )
        upstream_side = self.service._classify_policy_error(
            _FakeStatusError("x", 502, headers={"x-workshop-proxy-origin": "upstream"})
        )
        self.assertEqual(proxy_side["upstream_status"], 502)
        self.assertEqual(upstream_side["upstream_status"], 502)
        self.assertNotEqual(proxy_side["relay_origin"], upstream_side["relay_origin"])

    def test_diagnostics_never_publish_a_capability_handle_or_key(self) -> None:
        leaky = _FakeStatusError(
            "Error connecting to "
            "http://127.0.0.1:8791/cap/wcap_3e51db83d8ed4123b5b42b84f59c70cb"
            "/v1/providers/openrouter/chat/completions "
            "with Authorization: Bearer sk-or-v1-deadbeefdeadbeef",
            502,
        )
        detail = self.service._policy_failure_detail(
            {"model": "openai/gpt-4.1-nano"}, leaky, 1, "Chat Completions API"
        )
        self.assertNotIn("wcap_3e51db83d8ed4123b5b42b84f59c70cb", detail)
        self.assertNotIn("sk-or-v1-deadbeefdeadbeef", detail)
        self.assertIn("wcap_<redacted>", detail)
        self.assertIn("upstream_status=502", detail)

    def test_a_transport_failure_without_a_status_still_names_its_type(self) -> None:
        classified = self.service._classify_policy_error(
            ConnectionRefusedError("[Errno 61] Connection refused")
        )
        self.assertEqual(classified["error_type"], "ConnectionRefusedError")
        self.assertIsNone(classified["upstream_status"])
        self.assertIn("Connection refused", classified["message"])

    def test_disable_reasoning_auto_survives_the_proxy_hop(self) -> None:
        """`auto` keyed off the base URL, which the proxy replaces.

        Under ``credential_mode=proxy`` the base URL is Workshop's capability
        route, so an OpenRouter policy silently stopped disabling reasoning.
        """
        direct = self.service._policy_chat_extra_body(
            {
                "provider": "openrouter",
                "disable_reasoning": "auto",
                "base_url": "https://openrouter.ai/api/v1",
            }
        )
        proxied = self.service._policy_chat_extra_body(
            {
                "provider": "openrouter",
                "disable_reasoning": "auto",
                "base_url": "http://127.0.0.1:8791/cap/wcap_x/v1/providers/openrouter",
            }
        )
        self.assertIsNotNone(direct)
        self.assertEqual(direct, proxied)

    def test_a_non_openrouter_provider_still_leaves_reasoning_alone(self) -> None:
        self.assertIsNone(
            self.service._policy_chat_extra_body(
                {
                    "provider": "openai",
                    "disable_reasoning": "auto",
                    "base_url": "https://api.openai.com/v1",
                }
            )
        )
