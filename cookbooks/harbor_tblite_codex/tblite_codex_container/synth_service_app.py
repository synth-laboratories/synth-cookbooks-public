from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import uvicorn

from synth_containers import (
    DatasetDescriptor,
    RuntimeMetadata,
    ResourceRef,
    RuntimeCapabilitySurface,
    TaskCatalog,
    TaskDefinition,
    TaskInfo,
    TaskInstance,
    create_reference_app,
)
from synth_containers.compat.harbor import (
    HarborCompatRuntime,
    harbor_capability_surface,
    harbor_resource_refs,
)
from synth_containers.nouns import (
    Actor,
    ExecutionRecord,
    Observation,
    Outcome,
    TraceEvent,
    Trajectory,
    TurnRecord,
    VerifierResult,
)
from synth_containers.ontology import OutcomeKind

from tb_lite_dataset import OPEN_THOUGHTS_TBLITE_DATASET, package_open_thoughts_task


TASK_ID = "tblite.codex_harbor"
DEFAULT_OPEN_THOUGHTS_TASK = "application-debug"
CONTAINER_ROOT = Path(__file__).resolve().parent


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class TBLiteRuntime(HarborCompatRuntime):
    def __init__(self) -> None:
        super().__init__(
            metadata=self.metadata(),
            task_info=self.task_info(),
            task_catalog=self.task_catalog(),
        )

    def _capabilities(self) -> RuntimeCapabilitySurface:
        return harbor_capability_surface(
            metadata={"task_family": "terminal_bench_lite", "runner": "codex_harbor"}
        )

    def _resource_refs(self) -> list[ResourceRef]:
        return harbor_resource_refs(
            container_root=CONTAINER_ROOT,
            dataset_id=OPEN_THOUGHTS_TBLITE_DATASET,
            runner_path=CONTAINER_ROOT / "codex_harbor_runner.py",
        )

    def metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(
            runtime_id="tblite.codex_harbor.synth_containers",
            name="TBLite Codex Harbor synth-containers runtime",
            description="Public contract wrapper for packaging TBLite Codex Harbor rollouts.",
            capabilities=self._capabilities(),
            metadata={
                "source_dataset": OPEN_THOUGHTS_TBLITE_DATASET,
                "resource_refs": [item.to_dict() for item in self._resource_refs()],
            },
        )

    def _task_definition(self) -> TaskDefinition:
        return TaskDefinition(
            task_id=TASK_ID,
            task_name="Terminal-Bench Lite Codex Harbor",
            task_family="terminal_bench_lite",
            description="Package a TBLite task for a Codex Harbor worker loop.",
            version="v1",
            benchmark="terminal_bench_lite",
            resource_refs=self._resource_refs(),
        )

    def task_info(self) -> TaskInfo:
        return TaskInfo(
            task=self._task_definition(),
            dataset=DatasetDescriptor(
                dataset_id=OPEN_THOUGHTS_TBLITE_DATASET,
                split="main",
                default_split="main",
                source="huggingface",
            ),
            capabilities=self._capabilities(),
            inference={"agent": "codex", "default_model": "openai/gpt-5.4-nano"},
            task_metadata={"default_open_thoughts_task": DEFAULT_OPEN_THOUGHTS_TASK},
            environment="harbor",
            resource_refs=self._resource_refs(),
        )

    def task_catalog(self) -> TaskCatalog:
        return TaskCatalog(
            catalog_id="tblite:catalog",
            tasks=[self._task_definition()],
            instances=[
                TaskInstance(
                    task_instance_id=f"tblite:{DEFAULT_OPEN_THOUGHTS_TASK}",
                    task_id=TASK_ID,
                    seed=0,
                    metadata={"open_thoughts_task": DEFAULT_OPEN_THOUGHTS_TASK},
                    resource_refs=self._resource_refs(),
                )
            ],
            resource_refs=self._resource_refs(),
        )

    async def submit_rollout(self, request: Mapping[str, Any]) -> ExecutionRecord:
        payload = dict(request)
        env = payload.get("env") if isinstance(payload.get("env"), dict) else {}
        env_config = env.get("config") if isinstance(env.get("config"), dict) else {}
        task_id = str(
            payload.get("task_instance_id")
            or payload.get("task_id")
            or env_config.get("open_thoughts_task")
            or DEFAULT_OPEN_THOUGHTS_TASK
        )
        if task_id == TASK_ID:
            task_id = DEFAULT_OPEN_THOUGHTS_TASK
        trace_id = str(
            payload.get("trace_correlation_id") or f"tblite_{uuid.uuid4().hex[:10]}"
        )
        now = _utc_now_iso()
        packaged = package_open_thoughts_task(
            task_id,
            runner_path=CONTAINER_ROOT / "codex_harbor_runner.py",
        )
        reward = 1.0 if packaged.context_tar_base64 and packaged.dockerfile else 0.0
        turn = TurnRecord(
            turn_index=1,
            actor_id="packager",
            assistant_text="package_open_thoughts_task",
            actions=[{"name": "package_open_thoughts_task", "task_id": task_id}],
            executed_actions=[{"name": "package_open_thoughts_task", "task_id": task_id}],
            observation=Observation(
                content={
                    "task_id": packaged.task_id,
                    "dockerfile_bytes": len(packaged.dockerfile.encode("utf-8")),
                    "context_tar_base64_bytes": len(
                        packaged.context_tar_base64.encode("ascii")
                    ),
                },
                channels={"reward": reward, "packaged": reward >= 1.0},
                actor_id="packager",
                created_at=now,
            ),
            event_rewards=[reward],
            outcome_reward=reward,
            metadata={"source_dataset": OPEN_THOUGHTS_TBLITE_DATASET},
        )
        execution = ExecutionRecord(
            execution_id=trace_id,
            trace_correlation_id=trace_id,
            status="completed",
            success_status="success" if reward >= 1.0 else "failed",
            created_at=now,
            updated_at=now,
            task=self._task_definition(),
            task_instance=TaskInstance(
                task_instance_id=f"tblite:{packaged.task_id}",
                task_id=TASK_ID,
                seed=env.get("seed") if isinstance(env.get("seed"), int) else None,
                metadata=packaged.metadata,
                resource_refs=self._resource_refs(),
            ),
            actors=[
                Actor(actor_id="packager", role="runtime", display_name="TBLite Packager")
            ],
            trajectory=Trajectory(
                turns=[turn],
                events=[
                    TraceEvent(
                        event_type="tblite_task_packaged",
                        at=now,
                        step_index=1,
                        actor_id="packager",
                        payload={
                            "source_task_id": packaged.task_id,
                            "dockerfile_bytes": len(packaged.dockerfile.encode("utf-8")),
                            "context_tar_base64_bytes": len(
                                packaged.context_tar_base64.encode("ascii")
                            ),
                        },
                    )
                ],
                metadata={"source_dataset": OPEN_THOUGHTS_TBLITE_DATASET},
            ),
            outcome=Outcome(
                kind=OutcomeKind.REWARD,
                reward=reward,
                passed=reward >= 1.0,
                verifier=VerifierResult(
                    verdict="packaged" if reward >= 1.0 else "failed",
                    score=reward,
                    passed=reward >= 1.0,
                ),
            ),
            summary={
                "outcome_reward": reward,
                "open_thoughts_task": packaged.task_id,
                "dockerfile_bytes": len(packaged.dockerfile.encode("utf-8")),
                "context_tar_base64_bytes": len(
                    packaged.context_tar_base64.encode("ascii")
                ),
                "codex_execution": "not_run_by_public_contract_smoke",
            },
            metadata={
                "status_detail": "packaged",
                "reward_source": "packaging_smoke",
                "source_dataset": OPEN_THOUGHTS_TBLITE_DATASET,
                "codex_execution": "not_run_by_public_contract_smoke",
                "resource_refs": [item.to_dict() for item in self._resource_refs()],
            },
        )
        self._executions[trace_id] = execution
        return execution


app = create_reference_app(TBLiteRuntime(), title="tblite-codex-harbor-synth-container")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8952"))
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)
