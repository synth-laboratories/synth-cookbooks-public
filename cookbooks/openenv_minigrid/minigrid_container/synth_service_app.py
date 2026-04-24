from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import uvicorn
from fastapi import HTTPException

from synth_containers import (
    CheckpointDescriptor,
    DatasetDescriptor,
    ExecutionRecord,
    RuntimeMetadata,
    ResourceRef,
    RuntimeCapabilitySurface,
    StateSnapshot,
    TaskCatalog,
    TaskDefinition,
    TaskInfo,
    TaskInstance,
    create_reference_app,
)
from synth_containers.compat.base import execution_from_service_result
from synth_containers.compat.openenv import (
    OpenEnvCompatRuntime,
    openenv_capability_surface,
    openenv_registry_resource_refs,
)
from synth_containers.ontology import CheckpointSemantics
from synth_containers.serde import JsonObject

import service_app as legacy


CONTAINER_ROOT = Path(__file__).resolve().parent
TASK_REGISTRY_PATH = CONTAINER_ROOT / "task_registry.json"
TASK_ID = "minigrid_prompt_opt"
TASK_NAME = "MiniGrid Prompt Optimization"
TASK_VERSION = "v1"
RUNTIME_ID = "minigrid_go_ex_synth_runtime"


def _optional_int(value: Any) -> int | None:
    if value is None or not str(value).strip():
        return None
    return int(value)


def _registry_payload() -> dict[str, Any]:
    return legacy._load_registry()


def _registry_entries() -> list[legacy.RegistryEntry]:
    return legacy._registry_entries()


def _dataset_id() -> str:
    dataset = dict(_registry_payload().get("dataset") or {})
    return str(dataset.get("id") or "minigrid_curated_goex_v1")


def _resource_refs() -> list[ResourceRef]:
    return openenv_registry_resource_refs(
        registry_path=TASK_REGISTRY_PATH,
        environment_name="minigrid",
        dataset_id=_dataset_id(),
    )


def _runtime_capabilities() -> RuntimeCapabilitySurface:
    return openenv_capability_surface(
        checkpointable=True,
        metadata={
            "task_family": "minigrid",
            "task_registry_path": str(TASK_REGISTRY_PATH),
            "runtime_import_error": legacy.MINIGRID_IMPORT_ERROR,
        },
    )


def _runtime_metadata() -> RuntimeMetadata:
    return RuntimeMetadata(
        runtime_id=RUNTIME_ID,
        name="MiniGrid Go-Explore synth_containers runtime",
        description=(
            "MiniGrid long-horizon runtime with true snapshot checkpointing for Go-Explore."
        ),
        capabilities=_runtime_capabilities(),
        metadata={
            "task_family": "minigrid",
            "service": "evals_minigrid_goex_synth",
            "task_registry_path": str(TASK_REGISTRY_PATH),
            "runtime_available": bool(legacy.MINIGRID_AVAILABLE),
            "runtime_import_error": legacy.MINIGRID_IMPORT_ERROR,
            "resource_refs": [item.to_dict() for item in _resource_refs()],
        },
    )


def _dataset_descriptor() -> DatasetDescriptor:
    registry = _registry_payload()
    dataset = dict(registry.get("dataset") or {})
    entries = registry.get("entries")
    row_count = len(entries) if isinstance(entries, list) else None
    visible_splits = [
        str(item)
        for item in [dataset.get("train_split"), dataset.get("heldout_split")]
        if item
    ]
    return DatasetDescriptor(
        dataset_id=_dataset_id(),
        split=str(dataset.get("train_split") or "train"),
        visible_splits=visible_splits,
        default_split=str(dataset.get("train_split") or "train"),
        row_count=row_count,
        source="evals.containers.minigrid.task_registry",
        path=str(TASK_REGISTRY_PATH),
        metadata={
            "name": str(dataset.get("name") or "MiniGrid Curated Go-Ex Slice"),
            "version": str(dataset.get("version") or TASK_VERSION),
            "task_registry": registry,
        },
    )


def _task_definition() -> TaskDefinition:
    return TaskDefinition(
        task_id=TASK_ID,
        task_name=TASK_NAME,
        task_family="minigrid",
        description="Optimize MiniGrid prompts with real mission progress and checkpoints.",
        version=TASK_VERSION,
        benchmark="minigrid",
        metadata={"task_registry_path": str(TASK_REGISTRY_PATH)},
        resource_refs=_resource_refs(),
    )


def _task_instance(entry: legacy.RegistryEntry) -> TaskInstance:
    return TaskInstance(
        task_instance_id=entry.task_instance_id,
        task_id=TASK_ID,
        split=entry.split_group,
        seed=entry.seed,
        tags=[entry.family, entry.label],
        metadata={
            "registry_task_id": entry.registry_task_id,
            "env_id": entry.env_id,
            "family": entry.family,
            "label": entry.label,
        },
        resource_refs=_resource_refs(),
    )


def _task_catalog() -> TaskCatalog:
    return TaskCatalog(
        catalog_id="minigrid:catalog",
        tasks=[_task_definition()],
        instances=[_task_instance(entry) for entry in _registry_entries()],
        metadata={
            "task_family": "minigrid",
            "task_registry_path": str(TASK_REGISTRY_PATH),
        },
        resource_refs=_resource_refs(),
    )


def _task_info_for_entry(entry: legacy.RegistryEntry) -> TaskInfo:
    return TaskInfo(
        task=_task_definition(),
        dataset=_dataset_descriptor(),
        capabilities=_runtime_capabilities(),
        limits={"default_max_steps": legacy.DEFAULT_MAX_STEPS},
        inference={
            "default_inference_url": legacy.DEFAULT_INFERENCE_URL,
            "default_model": legacy.DEFAULT_MODEL,
            "api_key_env": "OPENAI_API_KEY",
        },
        task_metadata={
            "task_family": "minigrid",
            "task_registry_path": str(TASK_REGISTRY_PATH),
            "task_preview": {
                "task_instance_id": entry.task_instance_id,
                "registry_task_id": entry.registry_task_id,
                "seed": entry.seed,
                "split_group": entry.split_group,
                "family": entry.family,
                "env_id": entry.env_id,
                "label": entry.label,
            },
        },
        environment="minigrid",
        metadata={
            "status": "ok",
            "reward_type": "env_progress_reward",
            "restore_semantics": CheckpointSemantics.TRUE_ENVIRONMENT_SNAPSHOT.value,
            "true_environment_snapshot": True,
            "supports_branching": True,
            "runtime_available": bool(legacy.MINIGRID_AVAILABLE),
            "runtime_import_error": legacy.MINIGRID_IMPORT_ERROR,
        },
        resource_refs=_resource_refs(),
    )


def _task_info(
    *,
    seed: int | None = None,
    split_group: str | None = None,
    family: str | None = None,
    task_instance_id: str | None = None,
    task_id: str | None = None,
) -> TaskInfo:
    entry = legacy._entry_by_identity(
        seed=seed,
        task_instance_id=task_instance_id,
        task_id=task_id,
        split_group=split_group,
        family=family,
    )
    return _task_info_for_entry(entry)


def _rollout_request_payload(request: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(request)
    rollout_id = str(
        payload.get("rollout_id") or payload.get("trace_correlation_id") or ""
    ).strip()
    if rollout_id:
        payload["rollout_id"] = rollout_id
    return payload


def _checkpoint_from_descriptor(payload: Mapping[str, Any]) -> CheckpointDescriptor:
    return CheckpointDescriptor(
        checkpoint_id=str(payload["checkpoint_id"]),
        rollout_id=str(payload.get("rollout_id") or ""),
        checkpoint_uri=str(payload.get("checkpoint_uri") or ""),
        created_at=str(payload.get("created_at") or ""),
        checkpoint_version=str(
            payload.get("checkpoint_version") or "minigrid.true_env_snapshot.v1"
        ),
        restore_eligible=bool(payload.get("restore_eligible", True)),
        label=str(payload.get("label") or "") or None,
        labels=[str(item) for item in payload.get("labels") or []],
        source=str(payload.get("source") or "minigrid_goex_container"),
        actor_ids=[str(item) for item in payload.get("actor_ids") or []],
        metadata=dict(payload.get("metadata") or {}),
        annotations=dict(payload.get("annotations") or {}),
        branchable=bool(payload.get("supports_branching", True)),
        checkpoint_semantics=CheckpointSemantics.TRUE_ENVIRONMENT_SNAPSHOT,
        restore_semantics=CheckpointSemantics.TRUE_ENVIRONMENT_SNAPSHOT.value,
        true_environment_snapshot=bool(payload.get("true_environment_snapshot", True)),
    )


def _execution_from_session(session: Mapping[str, Any]) -> ExecutionRecord:
    response_payload = dict(
        session.get("response_payload") or legacy._response_payload(dict(session))
    )
    request_payload = dict(session.get("request_payload") or {})
    execution = execution_from_service_result(
        result=response_payload,
        request=request_payload,
        task_info=_task_info_for_entry(session["entry"]),
        task_catalog=_task_catalog(),
    )
    state = legacy._state_payload(dict(session))
    execution.state = StateSnapshot(
        state_id=f"{execution.execution_id}:minigrid_state",
        values=dict(state.get("observation") or {}),
        created_at=str(state.get("updated_at") or ""),
        authoritative=True,
        metadata=state,
    )
    execution.parent_rollout_id = (
        str(session.get("parent_rollout_id")) if session.get("parent_rollout_id") else None
    )
    execution.parent_checkpoint_id = (
        str(session.get("parent_checkpoint_id")) if session.get("parent_checkpoint_id") else None
    )
    checkpoints = state.get("checkpoints") if isinstance(state.get("checkpoints"), list) else []
    if checkpoints:
        execution.checkpoint = _checkpoint_from_descriptor(checkpoints[-1])
    execution.metadata.setdefault("status_detail", str(session.get("status_detail") or ""))
    return execution


class MiniGridRuntime(OpenEnvCompatRuntime):
    def __init__(self) -> None:
        super().__init__(
            metadata=_runtime_metadata(),
            task_info=_task_info(),
            task_catalog=_task_catalog(),
        )

    def task_info_for_request(self, query: dict[str, Any]) -> TaskInfo:
        return _task_info(
            seed=_optional_int(query.get("seed")),
            split_group=str(query.get("split_group") or "").strip() or None,
            family=str(query.get("family") or "").strip() or None,
            task_instance_id=str(query.get("task_instance_id") or "").strip() or None,
            task_id=str(query.get("task_id") or "").strip() or None,
        )

    async def submit_rollout(self, request: JsonObject) -> ExecutionRecord:
        legacy._ensure_runtime_available()
        payload = _rollout_request_payload(request)
        parsed = legacy.RolloutRequest(**payload)
        env_config = dict(parsed.env.config or {})
        entry = legacy._entry_by_identity(
            seed=parsed.env.seed,
            task_instance_id=parsed.task_instance_id
            or str(parsed.task_metadata.get("task_instance_id") or "").strip()
            or None,
            task_id=parsed.task_id
            or str(parsed.task_metadata.get("task_id") or "").strip()
            or None,
            split_group=str(env_config.get("split_group") or "").strip() or None,
            family=str(env_config.get("family") or "").strip() or None,
        )
        max_steps = max(
            int(
                env_config.get("max_steps")
                or env_config.get("segment_steps")
                or legacy.DEFAULT_MAX_STEPS
            ),
            1,
        )
        seed = int(parsed.env.seed if parsed.env.seed is not None else entry.seed)
        env = legacy._make_env(entry, max_steps=max_steps, seed=seed)
        obs, _info = env.reset(seed=seed)
        session = legacy._build_session(request=parsed, entry=entry, env=env, obs=obs)
        async with legacy._STORE_LOCK:
            legacy._ROLLOUTS[session["rollout_id"]] = session
        if str(parsed.submission_mode or "sync").strip().lower() == "async":
            await legacy._schedule_rollout(session["rollout_id"])
            async with legacy._STORE_LOCK:
                execution = _execution_from_session(legacy._ROLLOUTS[session["rollout_id"]])
        else:
            await legacy._run_rollout(session["rollout_id"])
            async with legacy._STORE_LOCK:
                execution = _execution_from_session(legacy._ROLLOUTS[session["rollout_id"]])
        self._executions[execution.execution_id] = execution
        return execution

    async def get_execution(self, rollout_id: str) -> ExecutionRecord | None:
        async with legacy._STORE_LOCK:
            session = legacy._ROLLOUTS.get(str(rollout_id))
            if session is None:
                return None
            execution = _execution_from_session(session)
        self._executions[execution.execution_id] = execution
        return execution

    async def get_execution_state(self, rollout_id: str) -> ExecutionRecord | None:
        return await self.get_execution(rollout_id)

    async def pause_execution(self, rollout_id: str, request: JsonObject) -> ExecutionRecord | None:
        async with legacy._STORE_LOCK:
            session = legacy._ROLLOUTS.get(str(rollout_id))
            if session is None:
                return None
            if session["status"] in {"queued", "running"}:
                session["pause_requested"] = True
                session["pause_reason"] = str(request.get("reason") or "")
                session["updated_at"] = legacy._utc_now_iso()
            return _execution_from_session(session)

    async def terminate_execution(
        self,
        rollout_id: str,
        request: JsonObject,
    ) -> ExecutionRecord | None:
        async with legacy._STORE_LOCK:
            session = legacy._ROLLOUTS.get(str(rollout_id))
            if session is None:
                return None
            if session["status"] not in {"completed", "failed", "cancelled"}:
                session["terminate_requested"] = True
                session["status_detail"] = str(request.get("reason") or "terminate_requested")
                session["updated_at"] = legacy._utc_now_iso()
            return _execution_from_session(session)

    async def create_checkpoint(
        self,
        rollout_id: str,
        request: JsonObject,
    ) -> CheckpointDescriptor | None:
        async with legacy._STORE_LOCK:
            session = legacy._ROLLOUTS.get(str(rollout_id))
            if session is None:
                return None
            record = legacy._create_checkpoint_locked(
                session,
                checkpoint_id=str(request.get("checkpoint_id") or "").strip() or None,
                label=str(request.get("label") or "").strip()
                or next(
                    (str(item) for item in request.get("labels") or [] if str(item).strip()),
                    None,
                ),
                source=str(request.get("source") or "").strip() or None,
                actor_ids=[str(item) for item in request.get("actor_ids") or []],
                metadata=dict(request.get("metadata") or {}),
                annotations=dict(request.get("annotations") or {}),
            )
            session["response_payload"] = legacy._response_payload(session)
            return _checkpoint_from_descriptor(legacy._checkpoint_descriptor(record))

    async def get_checkpoint(self, checkpoint_id: str) -> CheckpointDescriptor | None:
        async with legacy._STORE_LOCK:
            record = legacy._CHECKPOINTS.get(str(checkpoint_id))
            if record is None:
                return None
            return _checkpoint_from_descriptor(legacy._checkpoint_descriptor(record))

    async def list_checkpoints(self, rollout_id: str | None = None) -> list[CheckpointDescriptor]:
        async with legacy._STORE_LOCK:
            rows = list(legacy._CHECKPOINTS.values())
        if rollout_id is not None:
            rows = [
                item
                for item in rows
                if str(item.get("rollout_id") or "") == str(rollout_id)
            ]
        return [_checkpoint_from_descriptor(legacy._checkpoint_descriptor(item)) for item in rows]

    async def get_rollout_checkpoint(
        self,
        rollout_id: str,
        checkpoint_id: str,
    ) -> CheckpointDescriptor | None:
        checkpoint = await self.get_checkpoint(checkpoint_id)
        if checkpoint is None or checkpoint.rollout_id != str(rollout_id):
            return None
        return checkpoint

    async def update_checkpoint_labels(
        self,
        checkpoint_id: str,
        request: JsonObject,
    ) -> CheckpointDescriptor | None:
        async with legacy._STORE_LOCK:
            record = legacy._CHECKPOINTS.get(str(checkpoint_id))
            if record is None:
                return None
            labels = [
                str(item).strip()
                for item in request.get("labels") or []
                if str(item).strip()
            ]
            if labels:
                record["labels"] = labels
                record["label"] = labels[0]
            metadata = dict(record.get("metadata") or {})
            metadata.update(dict(request.get("metadata") or {}))
            record["metadata"] = metadata
            annotations = dict(record.get("annotations") or {})
            annotations.update(dict(request.get("annotations") or {}))
            record["annotations"] = annotations
            return _checkpoint_from_descriptor(legacy._checkpoint_descriptor(record))

    async def resume_execution(
        self,
        rollout_id: str,
        request: JsonObject,
    ) -> ExecutionRecord | None:
        legacy._ensure_runtime_available()
        parsed = legacy.ResumeRequest(
            rollout_id=rollout_id,
            checkpoint_id=request.get("checkpoint_id"),
            target_rollout_id=request.get("target_rollout_id"),
            mode=request.get("mode") or "new_rollout",
            submission_mode=request.get("submission_mode") or "sync",
            overrides=dict(request.get("overrides") or {}),
        )
        checkpoint_id = str(parsed.checkpoint_id or "").strip()
        if not checkpoint_id:
            raise HTTPException(
                status_code=400,
                detail="checkpoint_id is required for minigrid resume",
            )
        async with legacy._STORE_LOCK:
            checkpoint = legacy._CHECKPOINTS.get(checkpoint_id)
            if checkpoint is None:
                return None
        entry = legacy._entry_by_identity(
            task_instance_id=parsed.overrides.task_instance_id or None,
            task_id=parsed.overrides.task_id or None,
        )
        if not parsed.overrides.task_instance_id and not parsed.overrides.task_id:
            async with legacy._STORE_LOCK:
                source_rollout = legacy._ROLLOUTS.get(str(checkpoint.get("rollout_id") or ""))
            if source_rollout is not None:
                entry = source_rollout["entry"]
        target_rollout_id = str(
            parsed.target_rollout_id or f"minigrid_resume_{legacy.uuid.uuid4().hex[:10]}"
        )
        session = legacy._restore_session_from_checkpoint(
            checkpoint=checkpoint,
            request=parsed,
            entry=entry,
            rollout_id=target_rollout_id,
        )
        async with legacy._STORE_LOCK:
            legacy._ROLLOUTS[target_rollout_id] = session
        if str(parsed.submission_mode or "sync").strip().lower() == "async":
            await legacy._schedule_rollout(target_rollout_id)
            async with legacy._STORE_LOCK:
                execution = _execution_from_session(legacy._ROLLOUTS[target_rollout_id])
        else:
            await legacy._run_rollout(target_rollout_id)
            async with legacy._STORE_LOCK:
                execution = _execution_from_session(legacy._ROLLOUTS[target_rollout_id])
        self._executions[execution.execution_id] = execution
        return execution


app = create_reference_app(MiniGridRuntime(), title="minigrid-synth-container")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8922"))
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)
