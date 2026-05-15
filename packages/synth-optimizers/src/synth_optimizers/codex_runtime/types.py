from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class ParticipantRole(StrEnum):
    WORKER = "worker"
    ORCHESTRATOR = "orchestrator"


class WorkerHostKind(StrEnum):
    LOCAL = "local"
    DOCKER = "docker"


class ExecutionResultStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    TIMED_OUT = "timed_out"


class ExecutionSessionStatus(StrEnum):
    STARTING = "starting"
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELED = "canceled"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class ExecutionControlState(StrEnum):
    NONE = "none"
    INTERRUPT_REQUESTED = "interrupt_requested"


class CodexSessionEventKind(StrEnum):
    THREAD_STARTED = "thread_started"
    THREAD_STATUS_CHANGED = "thread_status_changed"
    THREAD_CLOSED = "thread_closed"
    TURN_STARTED = "turn_started"
    TURN_COMPLETED = "turn_completed"
    TURN_FAILED = "turn_failed"
    TURN_INTERRUPTED = "turn_interrupted"
    AGENT_MESSAGE = "agent_message"
    REASONING = "reasoning"
    COMMAND_EXECUTION = "command_execution"
    FILE_CHANGE = "file_change"
    MCP_TOOL_CALL = "mcp_tool_call"
    MESSAGE = "message"
    ERROR = "error"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class CodexProfile:
    profile_id: str
    model: str
    reasoning_effort: str | None = None
    approval_policy: str | None = None
    lora_id: str | None = None


@dataclass(frozen=True, slots=True)
class SandboxProfile:
    profile_id: str
    sandbox_mode: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    profile_id: str
    codex: CodexProfile | None = None
    sandbox: SandboxProfile | None = None


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    participant_role: ParticipantRole
    host_kind: WorkerHostKind
    profile: ExecutionProfile | None = None
    policy: Any | None = None

    def effective_profile(self) -> ExecutionProfile | None:
        return self.profile


@dataclass(frozen=True, slots=True)
class ExecutionSessionSpec:
    run_id: str
    session_id: str
    instructions: str
    execution_config: ExecutionConfig


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    session_id: str
    status: ExecutionResultStatus
    started_at: datetime
    failure_reason: str | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    event_count: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CodexSessionSnapshot:
    session_id: str
    status: ExecutionSessionStatus
    control_state: ExecutionControlState
    thread_id: str | None
    active_turn_id: str | None
    thread_loaded: bool
    raw_event_count: int
    started_at: datetime
    updated_at: datetime
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CodexSessionEvent:
    session_id: str
    kind: CodexSessionEventKind
    payload: dict[str, Any]
    occurred_at: datetime | None
