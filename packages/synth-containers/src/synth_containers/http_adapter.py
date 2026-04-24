from __future__ import annotations

from inspect import isawaitable
from typing import Any, Protocol, runtime_checkable

from fastapi import FastAPI, HTTPException, Request

from .capabilities import RuntimeMetadata, TaskCatalog, TaskInfo
from .compatibility import compatibility_matrix, evaluate_consumer_support
from .formats import (
    ExecutionControlSurface,
    execution_to_rollout_payload,
    execution_to_state_payload,
    metadata_to_http_payload,
    task_info_to_http_payload,
)
from .http_models import (
    CheckpointLabelsRequestModel,
    CreateCheckpointRequestModel,
    PauseRequestModel,
    ResumeRequestModel,
    RolloutRequestModel,
    TerminateRequestModel,
)
from .nouns import CheckpointDescriptor, ExecutionRecord
from .ontology import CONTRACT_VERSION
from .serde import JsonObject


@runtime_checkable
class ManagedRuntime(Protocol):
    def metadata(self) -> RuntimeMetadata: ...

    def task_info(self) -> TaskInfo: ...

    def task_catalog(self) -> TaskCatalog: ...

    async def submit_rollout(self, request: JsonObject) -> ExecutionRecord: ...

    async def get_execution(self, rollout_id: str) -> ExecutionRecord | None: ...

    async def get_execution_state(self, rollout_id: str) -> ExecutionRecord | None: ...

    async def pause_execution(self, rollout_id: str, request: JsonObject) -> ExecutionRecord | None: ...

    async def terminate_execution(self, rollout_id: str, request: JsonObject) -> ExecutionRecord | None: ...

    async def create_checkpoint(self, rollout_id: str, request: JsonObject) -> CheckpointDescriptor | None: ...

    async def get_checkpoint(self, checkpoint_id: str) -> CheckpointDescriptor | None: ...

    async def list_checkpoints(self, rollout_id: str | None = None) -> list[CheckpointDescriptor]: ...

    async def get_rollout_checkpoint(self, rollout_id: str, checkpoint_id: str) -> CheckpointDescriptor | None: ...

    async def update_checkpoint_labels(self, checkpoint_id: str, request: JsonObject) -> CheckpointDescriptor | None: ...

    async def resume_execution(self, rollout_id: str, request: JsonObject) -> ExecutionRecord | None: ...


def _metadata(runtime: ManagedRuntime) -> RuntimeMetadata:
    value = runtime.metadata()
    if isinstance(value, RuntimeMetadata):
        return value
    raise TypeError("runtime.metadata() must return RuntimeMetadata")


def _task_info(runtime: ManagedRuntime) -> TaskInfo:
    return runtime.task_info()


async def _task_info_for_request(runtime: ManagedRuntime, query: dict[str, Any]) -> TaskInfo:
    handler = getattr(runtime, "task_info_for_request", None)
    if callable(handler):
        value = handler(query)
        if isawaitable(value):
            value = await value
        if isinstance(value, TaskInfo):
            return value
        raise TypeError("runtime.task_info_for_request() must return TaskInfo")
    return _task_info(runtime)


def _task_catalog(runtime: ManagedRuntime) -> TaskCatalog:
    return runtime.task_catalog()


def _coerce_rollout_payload(value: ExecutionRecord) -> dict[str, Any]:
    return execution_to_rollout_payload(value)


def _coerce_state_payload(runtime: ManagedRuntime, value: ExecutionRecord) -> dict[str, Any]:
    capabilities = _metadata(runtime).capabilities
    return execution_to_state_payload(
        value,
        capabilities=capabilities,
        control=ExecutionControlSurface(
            pause_supported=capabilities.pause_support,
            terminate_supported=capabilities.terminate_support,
            resume_supported=capabilities.resume_support,
            checkpoint_supported=capabilities.checkpoint_support,
        )
    )


def _coerce_checkpoint_payload(value: CheckpointDescriptor) -> dict[str, Any]:
    return value.to_dict()


def _coerce_checkpoint_list(value: list[CheckpointDescriptor]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("runtime.list_checkpoints() must return a list")
    return [item.to_dict() for item in value]



def create_reference_app(runtime: ManagedRuntime, *, title: str = "synth-containers-reference") -> FastAPI:
    app = FastAPI(title=title)

    @app.get("/")
    async def root() -> dict[str, Any]:
        metadata = _metadata(runtime)
        return {
            "status": "ok",
            "contract_version": CONTRACT_VERSION,
            "runtime": metadata_to_http_payload(metadata),
            "task_info": task_info_to_http_payload(await _task_info_for_request(runtime, {})),
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "contract_version": CONTRACT_VERSION}

    @app.get("/metadata")
    @app.get("/info")
    async def metadata() -> dict[str, Any]:
        return metadata_to_http_payload(_metadata(runtime))

    @app.get("/task_info")
    async def task_info(request: Request) -> dict[str, Any]:
        query = {key: value for key, value in request.query_params.multi_items()}
        return task_info_to_http_payload(await _task_info_for_request(runtime, query))

    @app.get("/task_catalog")
    async def task_catalog() -> dict[str, Any]:
        return _task_catalog(runtime).to_dict()

    @app.get("/compatibility")
    async def compatibility(target: str | None = None) -> dict[str, Any]:
        metadata = _metadata(runtime)
        if target is None or not str(target).strip():
            return compatibility_matrix(metadata)
        normalized_target = str(target).strip()
        try:
            return evaluate_consumer_support(metadata, normalized_target).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid_compatibility_target:{normalized_target}:{exc}") from exc

    @app.post("/rollout")
    @app.post("/rollouts")
    async def rollout(request: RolloutRequestModel) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        result = await runtime.submit_rollout(request=payload)
        return _coerce_rollout_payload(result)

    @app.get("/rollouts/{rollout_id}")
    async def get_rollout(rollout_id: str) -> dict[str, Any]:
        result = await runtime.get_execution(rollout_id=rollout_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_rollout_payload(result)

    @app.get("/rollouts/{rollout_id}/state")
    async def get_rollout_state(rollout_id: str) -> dict[str, Any]:
        result = await runtime.get_execution_state(rollout_id=rollout_id)
        if result is None:
            result = await runtime.get_execution(rollout_id=rollout_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_state_payload(runtime, result)

    @app.get("/rollouts/{rollout_id}/summary")
    async def get_rollout_summary(rollout_id: str) -> dict[str, Any]:
        payload = await get_rollout(rollout_id)
        return {
            "rollout_id": rollout_id,
            "trace_correlation_id": payload.get("trace_correlation_id"),
            "summary": payload.get("summary") or {},
            "outcome_reward": ((payload.get("reward_info") or {}).get("outcome_reward")),
            "parent_rollout_id": payload.get("parent_rollout_id"),
            "parent_checkpoint_id": payload.get("parent_checkpoint_id"),
        }

    @app.get("/rollouts/{rollout_id}/usage")
    async def get_rollout_usage(rollout_id: str) -> dict[str, Any]:
        payload = await get_rollout(rollout_id)
        return {
            "rollout_id": rollout_id,
            "trace_correlation_id": payload.get("trace_correlation_id"),
            "usage": payload.get("usage") or {},
        }

    @app.get("/rollouts/{rollout_id}/artifacts")
    async def get_rollout_artifacts(rollout_id: str) -> dict[str, Any]:
        payload = await get_rollout(rollout_id)
        return {"rollout_id": rollout_id, "artifacts": payload["artifacts"]}

    @app.get("/rollouts/{rollout_id}/events")
    async def get_rollout_events(rollout_id: str) -> dict[str, Any]:
        payload = await get_rollout(rollout_id)
        raw_trace = payload.get("trace")
        trace = dict(raw_trace) if isinstance(raw_trace, dict) else {}
        return {"rollout_id": rollout_id, "events": trace.get("events") or trace.get("event_history") or []}

    @app.get("/rollouts/{rollout_id}/trace")
    async def get_rollout_trace(rollout_id: str) -> dict[str, Any]:
        payload = await get_rollout(rollout_id)
        raw_trace = payload.get("trace")
        trace = dict(raw_trace) if isinstance(raw_trace, dict) else {}
        return {"rollout_id": rollout_id, **trace}

    @app.post("/rollouts/{rollout_id}/pause")
    async def pause_rollout(rollout_id: str, request: PauseRequestModel) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        result = await runtime.pause_execution(rollout_id=rollout_id, request=payload)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_state_payload(runtime, result)

    @app.post("/rollouts/{rollout_id}/terminate")
    async def terminate_rollout(rollout_id: str, request: TerminateRequestModel) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        result = await runtime.terminate_execution(rollout_id=rollout_id, request=payload)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_state_payload(runtime, result)

    @app.post("/rollouts/{rollout_id}/checkpoints")
    async def create_checkpoint(rollout_id: str, request: CreateCheckpointRequestModel) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        result = await runtime.create_checkpoint(rollout_id=rollout_id, request=payload)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_checkpoint_payload(result)

    @app.get("/rollouts/{rollout_id}/checkpoints")
    async def list_rollout_checkpoints(rollout_id: str) -> dict[str, Any]:
        rows = await runtime.list_checkpoints(rollout_id=rollout_id)
        return {
            "rollout_id": rollout_id,
            "checkpoints": _coerce_checkpoint_list(rows),
        }

    @app.get("/rollouts/{rollout_id}/checkpoints/{checkpoint_id}")
    async def get_rollout_checkpoint(rollout_id: str, checkpoint_id: str) -> dict[str, Any]:
        result = await runtime.get_rollout_checkpoint(
            rollout_id=rollout_id,
            checkpoint_id=checkpoint_id,
        )
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_checkpoint:{checkpoint_id}")
        return _coerce_checkpoint_payload(result)

    @app.get("/checkpoints")
    async def list_checkpoints() -> dict[str, Any]:
        rows = await runtime.list_checkpoints(rollout_id=None)
        return {"checkpoints": _coerce_checkpoint_list(rows)}

    @app.get("/checkpoints/{checkpoint_id}")
    async def get_checkpoint(checkpoint_id: str) -> dict[str, Any]:
        result = await runtime.get_checkpoint(checkpoint_id=checkpoint_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_checkpoint:{checkpoint_id}")
        return _coerce_checkpoint_payload(result)

    @app.post("/checkpoints/{checkpoint_id}/labels")
    async def update_checkpoint_labels(
        checkpoint_id: str,
        request: CheckpointLabelsRequestModel,
    ) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        result = await runtime.update_checkpoint_labels(
            checkpoint_id=checkpoint_id,
            request=payload,
        )
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_checkpoint:{checkpoint_id}")
        return _coerce_checkpoint_payload(result)

    @app.post("/rollouts/{rollout_id}/resume")
    async def resume_rollout(rollout_id: str, request: ResumeRequestModel) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        result = await runtime.resume_execution(rollout_id=rollout_id, request=payload)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_rollout_payload(result)

    return app
