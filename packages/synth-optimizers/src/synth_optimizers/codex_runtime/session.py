from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timezone
from typing import Any

from .protocol import (
    build_codex_thread_start_params,
    build_codex_turn_start_params,
    classify_event_kind,
    extract_failure_reason,
    extract_thread_id,
    extract_turn_id,
    is_retryable_error_notification,
    thread_or_turn_status,
)
from .stdio import CodexAppServerStdioClient
from .types import (
    CodexSessionEvent,
    CodexSessionEventKind,
    CodexSessionSnapshot,
    ExecutionConfig,
    ExecutionControlState,
    ExecutionProfile,
    ExecutionResult,
    ExecutionResultStatus,
    ExecutionSessionSpec,
    ExecutionSessionStatus,
)

_REQUEST_TIMEOUT_SECONDS = 60.0
_RESUME_THREAD_TIMEOUT_SECONDS = 60.0
_TURN_START_TIMEOUT_SECONDS = 60.0
_QUEUE_CLOSED = object()


class AppServerCodexSession:
    """Minimal worker-focused Codex app-server session controller.

    This intentionally omits Horizons-specific orchestration logic. It owns only
    the worker session lifecycle needed by MIPROv2 codex workspace proposer runs.
    """

    def __init__(self, client: CodexAppServerStdioClient) -> None:
        self._client = client
        self._command_lock = asyncio.Lock()
        self._spec: ExecutionSessionSpec | None = None
        self._session_id = ""
        self._thread_id: str | None = None
        self._thread_loaded = False
        self._active_turn_id: str | None = None
        self._active_turn_done: asyncio.Future[Mapping[str, Any]] | None = None
        self._started_at: datetime = _utcnow()
        self._updated_at: datetime = self._started_at
        self._raw_event_count = 0
        self._status = ExecutionSessionStatus.STARTING
        self._control_state = ExecutionControlState.NONE
        self._usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }
        self._closed = False
        self._reader_task: asyncio.Task[None] | None = None
        self._event_queue: asyncio.Queue[CodexSessionEvent | object] = asyncio.Queue()
        self._pending_requests: dict[str, asyncio.Future[Mapping[str, Any]]] = {}
        self._last_turn_result: ExecutionResult | None = None
        self._thread_closed_waiter: asyncio.Future[None] | None = None
        self._last_notification_method: str | None = None
        self._last_notification_payload: dict[str, Any] | None = None
        self._last_notification_at: datetime | None = None
        self._close_diagnostics: dict[str, Any] = {}

    async def start(self, spec: ExecutionSessionSpec) -> CodexSessionSnapshot:
        async with self._command_lock:
            if self._spec is not None:
                raise RuntimeError("codex session has already been started")
            self._spec = spec
            self._session_id = spec.session_id
            self._started_at = _utcnow()
            self._touch(status=ExecutionSessionStatus.STARTING, control_state=ExecutionControlState.NONE)
            await self._client.start()
            await self._client.initialize()
            self._start_reader()
            thread_response = await self._request_with_timeout(
                stage="thread_start",
                timeout_seconds=_REQUEST_TIMEOUT_SECONDS,
                method="thread/start",
                params=self._thread_start_params(spec),
            )
            self._thread_id = extract_thread_id(thread_response)
            if not self._thread_id:
                raise RuntimeError(f"codex app-server thread/start response missing thread id: {thread_response}")
            self._thread_loaded = True
            self._reset_thread_closed_waiter()
            self._touch(status=ExecutionSessionStatus.IDLE, control_state=ExecutionControlState.NONE)
            await self._start_turn(spec.instructions)
            return self._snapshot_locked()

    async def snapshot(self) -> CodexSessionSnapshot:
        async with self._command_lock:
            return self._snapshot_locked()

    async def stream_events(self) -> AsyncIterator[CodexSessionEvent]:
        while True:
            event = await self._event_queue.get()
            if event is _QUEUE_CLOSED:
                return
            yield event  # type: ignore[misc]

    async def send_message(self, *, instructions: str) -> CodexSessionSnapshot:
        async with self._command_lock:
            self._require_started()
            if self._thread_id is None:
                raise RuntimeError("codex session has no thread to continue")
            if self._active_turn_id is not None:
                raise RuntimeError("active turn is running; use steer() or interrupt()")
            if not self._thread_loaded:
                await self._resume_thread()
            await self._start_turn(instructions)
            return self._snapshot_locked()

    async def steer(self, *, instructions: str) -> CodexSessionSnapshot:
        async with self._command_lock:
            self._require_started()
            if not self._thread_id or not self._active_turn_id:
                raise RuntimeError("codex session has no active turn to steer")
            await self._request(
                method="turn/steer",
                params={
                    "threadId": self._thread_id,
                    "expectedTurnId": self._active_turn_id,
                    "input": [{"type": "text", "text": instructions, "textElements": []}],
                },
            )
            self._touch()
            return self._snapshot_locked()

    async def interrupt(self, *, message: str | None = None) -> CodexSessionSnapshot:
        self._require_started()
        no_active_turn = False
        idle_snapshot: CodexSessionSnapshot | None = None
        async with self._command_lock:
            thread_id = self._thread_id
            active_turn_id = self._active_turn_id
            active_turn_done = self._active_turn_done
            if not thread_id or not active_turn_id or active_turn_done is None:
                no_active_turn = True
                idle_snapshot = self._snapshot_locked()
            else:
                self._touch(control_state=ExecutionControlState.INTERRUPT_REQUESTED)
                await self._request(method="turn/interrupt", params={"threadId": thread_id, "turnId": active_turn_id})
        if no_active_turn:
            if message:
                return await self.send_message(instructions=message)
            return idle_snapshot if idle_snapshot is not None else await self.snapshot()
        try:
            await asyncio.shield(active_turn_done)
        except asyncio.CancelledError as exc:
            async with self._command_lock:
                turn_still_active = self._active_turn_id is not None
                future_cancelled = active_turn_done.cancelled()
            if turn_still_active or not future_cancelled:
                raise RuntimeError("codex turn interrupt was cancelled before the turn settled") from exc
        if message:
            return await self.send_message(instructions=message)
        async with self._command_lock:
            return self._snapshot_locked()

    async def pause(self, *, interrupt_if_running: bool = False) -> CodexSessionSnapshot:
        self._require_started()
        if interrupt_if_running and self._active_turn_id is not None:
            await self.interrupt()
        async with self._command_lock:
            if self._active_turn_id is not None:
                raise RuntimeError("cannot pause while a turn is active; set interrupt_if_running=True")
            if not self._thread_id:
                raise RuntimeError("codex session has no thread to pause")
            if self._thread_loaded:
                await self._request(method="thread/unsubscribe", params={"threadId": self._thread_id})
            self._touch(status=ExecutionSessionStatus.PAUSED, control_state=ExecutionControlState.NONE)
            thread_closed_waiter = self._thread_closed_waiter
        if thread_closed_waiter is not None and not thread_closed_waiter.done():
            await thread_closed_waiter
        async with self._command_lock:
            self._thread_loaded = False
            return self._snapshot_locked()

    async def cancel(self, *, reason: str | None = None) -> CodexSessionSnapshot:
        self._require_started()
        if self._active_turn_id is not None:
            await self.interrupt()
        reader_task: asyncio.Task[None] | None
        async with self._command_lock:
            reader_task = self._reader_task
            await self._client.terminate()
            self._closed = True
            self._thread_loaded = False
            self._touch(status=ExecutionSessionStatus.CANCELED, control_state=ExecutionControlState.NONE)
            self._last_turn_result = ExecutionResult(
                session_id=self._session_id,
                status=ExecutionResultStatus.CANCELED,
                started_at=self._started_at,
                failure_reason=reason,
                thread_id=self._thread_id,
                turn_id=self._active_turn_id,
                event_count=self._raw_event_count,
                diagnostics=dict(self._close_diagnostics),
            )
        if reader_task is not None:
            await reader_task
        else:
            await self._close_streams()
        async with self._command_lock:
            return self._snapshot_locked()

    async def resume(self, *, instructions: str | None = None) -> CodexSessionSnapshot:
        async with self._command_lock:
            self._require_started()
            if not self._thread_id:
                raise RuntimeError("codex session has no thread to resume")
            if not self._thread_loaded:
                await self._resume_thread()
            if instructions:
                if self._active_turn_id is not None:
                    raise RuntimeError("cannot resume with new input while a turn is active")
                await self._start_turn(instructions)
            return self._snapshot_locked()

    async def wait_for_idle(self) -> CodexSessionSnapshot:
        while True:
            async with self._command_lock:
                turn_done = self._active_turn_done
                if turn_done is None:
                    return self._snapshot_locked()
            await asyncio.shield(turn_done)

    async def wait(self) -> ExecutionResult:
        await self.wait_for_idle()
        async with self._command_lock:
            if self._last_turn_result is not None:
                return self._last_turn_result
            self._require_started()
            return ExecutionResult(
                session_id=self._session_id,
                status=_result_status_from_session(self._status),
                started_at=self._started_at,
                thread_id=self._thread_id,
                turn_id=self._active_turn_id,
                event_count=self._raw_event_count,
                diagnostics=dict(self._close_diagnostics),
            )

    async def shutdown(self) -> CodexSessionSnapshot:
        self._require_started()
        async with self._command_lock:
            if self._active_turn_id is not None:
                raise RuntimeError("cannot shutdown while a turn is active")
            if self._closed:
                return self._snapshot_locked()
            self._closed = True
            self._thread_loaded = False
            reader_task = self._reader_task
            await self._client.terminate()
        if reader_task is not None:
            await reader_task
        else:
            await self._close_streams()
        async with self._command_lock:
            return self._snapshot_locked()

    def close_diagnostics(self) -> Mapping[str, Any]:
        return dict(self._close_diagnostics)

    def _start_reader(self) -> None:
        if self._reader_task is None:
            self._reader_task = asyncio.create_task(self._reader_loop())

    async def _reader_loop(self) -> None:
        try:
            while True:
                message = await self._client.read_message()
                if message is None:
                    self._close_diagnostics = self._build_close_diagnostics(stage="reader_eof")
                    break
                self._raw_event_count += 1
                if _is_response(message):
                    self._resolve_request_waiter(message)
                    continue
                await self._handle_notification(message)
        except Exception as exc:
            self._fail_waiters(exc)
            self._close_diagnostics = self._build_close_diagnostics(stage="reader_loop", failure_reason=str(exc), exc=exc)
            if self._spec is not None:
                self._last_turn_result = ExecutionResult(
                    session_id=self._session_id,
                    status=ExecutionResultStatus.FAILED,
                    started_at=self._started_at,
                    failure_reason=str(exc),
                    thread_id=self._thread_id,
                    turn_id=self._active_turn_id,
                    event_count=self._raw_event_count,
                    diagnostics=dict(self._close_diagnostics),
                )
        finally:
            await self._close_streams()

    async def _handle_notification(self, message: Mapping[str, Any]) -> None:
        method = str(message.get("method") or "").strip()
        thread_id = extract_thread_id(message)
        if thread_id:
            self._thread_id = thread_id
            self._thread_loaded = method != "thread/closed"
        self._update_usage_from_message(message)
        self._last_notification_method = method or None
        self._last_notification_payload = dict(message)
        self._last_notification_at = _utcnow()
        if method == "turn/started":
            turn_id = extract_turn_id(message)
            if turn_id:
                self._active_turn_id = turn_id
            self._touch(status=ExecutionSessionStatus.RUNNING, control_state=ExecutionControlState.NONE)
        elif method in {"turn/completed", "turn/failed", "turn/interrupted"}:
            self._finalize_turn_from_message(message, method=method)
        elif method == "thread/closed":
            self._thread_loaded = False
            if self._thread_closed_waiter is not None and not self._thread_closed_waiter.done():
                self._thread_closed_waiter.set_result(None)
            if self._status == ExecutionSessionStatus.PAUSED:
                self._touch(status=ExecutionSessionStatus.PAUSED, control_state=ExecutionControlState.NONE)
            else:
                self._touch()
        elif method == "thread/status/changed":
            thread_status = thread_or_turn_status(message)
            if thread_status == "idle" and self._active_turn_id is None:
                if self._status not in {ExecutionSessionStatus.PAUSED, ExecutionSessionStatus.CANCELED, ExecutionSessionStatus.FAILED}:
                    self._touch(status=ExecutionSessionStatus.IDLE)
        elif method == "error":
            if is_retryable_error_notification(message):
                self._touch()
            else:
                self._touch(status=ExecutionSessionStatus.FAILED, control_state=ExecutionControlState.NONE)
        else:
            self._touch()
        await self._event_queue.put(
            CodexSessionEvent(
                session_id=self._session_id,
                kind=classify_event_kind(message),
                payload=dict(message),
                occurred_at=self._last_notification_at,
            )
        )

    def _finalize_turn_from_message(self, message: Mapping[str, Any], *, method: str) -> None:
        result_status, failure_reason = _turn_result_from_message(message, method=method)
        session_status = _session_status_from_result(result_status)
        turn_id = extract_turn_id(message) or self._active_turn_id
        self._last_turn_result = ExecutionResult(
            session_id=self._session_id,
            status=result_status,
            started_at=self._started_at,
            failure_reason=failure_reason,
            thread_id=self._thread_id,
            turn_id=turn_id,
            event_count=self._raw_event_count,
            diagnostics=dict(self._close_diagnostics),
        )
        turn_done = self._active_turn_done
        self._active_turn_id = None
        self._active_turn_done = None
        self._touch(status=session_status, control_state=ExecutionControlState.NONE)
        if turn_done is not None and not turn_done.done():
            turn_done.set_result(dict(message))

    async def _resume_thread(self) -> None:
        if not self._thread_id:
            raise RuntimeError("codex session has no thread to resume")
        try:
            resume_response = await self._request_with_timeout(
                stage="resume_thread",
                timeout_seconds=_RESUME_THREAD_TIMEOUT_SECONDS,
                method="thread/resume",
                params={"threadId": self._thread_id},
            )
        except RuntimeError as exc:
            if "is closing" not in str(exc):
                raise
            thread_closed_waiter = self._thread_closed_waiter
            if thread_closed_waiter is not None and not thread_closed_waiter.done():
                await thread_closed_waiter
            resume_response = await self._request_with_timeout(
                stage="resume_thread",
                timeout_seconds=_RESUME_THREAD_TIMEOUT_SECONDS,
                method="thread/resume",
                params={"threadId": self._thread_id},
            )
        resumed_thread_id = extract_thread_id(resume_response)
        if resumed_thread_id:
            self._thread_id = resumed_thread_id
        self._thread_loaded = True
        self._reset_thread_closed_waiter()
        self._touch(status=ExecutionSessionStatus.IDLE, control_state=ExecutionControlState.NONE)

    async def _start_turn(self, instructions: str) -> None:
        if not self._thread_id:
            raise RuntimeError("codex session has no thread for turn/start")
        turn_response = await self._request_with_timeout(
            stage="turn_start",
            timeout_seconds=_TURN_START_TIMEOUT_SECONDS,
            method="turn/start",
            params=build_codex_turn_start_params(
                thread_id=self._thread_id,
                instructions=instructions,
                model=self._resolve_model(),
                reasoning_effort=self._resolve_reasoning_effort(),
                approval_policy=self._resolve_approval_policy(),
                sandbox_mode=self._resolve_sandbox_mode(),
            ),
        )
        turn_id = extract_turn_id(turn_response)
        if not turn_id:
            raise RuntimeError(f"codex app-server turn/start response missing turn id: {turn_response}")
        loop = asyncio.get_running_loop()
        self._active_turn_id = turn_id
        self._active_turn_done = loop.create_future()
        self._touch(status=ExecutionSessionStatus.RUNNING, control_state=ExecutionControlState.NONE)

    def _thread_start_params(self, spec: ExecutionSessionSpec) -> dict[str, Any]:
        return build_codex_thread_start_params(
            model=self._resolve_model(spec.execution_config),
            run_instructions=spec.instructions,
            approval_policy=self._resolve_approval_policy(spec.execution_config),
            sandbox_mode=self._resolve_sandbox_mode(spec.execution_config),
        )

    def _resolve_model(self, execution_config: ExecutionConfig | None = None) -> str:
        config = execution_config or (self._spec.execution_config if self._spec is not None else None)
        profile = None if config is None else config.effective_profile()
        if profile is not None and profile.codex is not None and str(profile.codex.model or "").strip():
            return str(profile.codex.model)
        raise RuntimeError("codex session missing model configuration")

    def _resolve_reasoning_effort(self) -> str | None:
        profile = None if self._spec is None else self._spec.execution_config.effective_profile()
        if profile is None or profile.codex is None:
            return None
        effort = str(profile.codex.reasoning_effort or "").strip()
        return effort or None

    def _resolve_approval_policy(self, execution_config: ExecutionConfig | None = None) -> str | None:
        config = execution_config or (self._spec.execution_config if self._spec is not None else None)
        profile = None if config is None else config.effective_profile()
        if profile is None or profile.codex is None:
            return None
        approval = str(profile.codex.approval_policy or "").strip()
        return approval or None

    def _resolve_sandbox_mode(self, execution_config: ExecutionConfig | None = None) -> str | None:
        config = execution_config or (self._spec.execution_config if self._spec is not None else None)
        profile = None if config is None else config.effective_profile()
        if profile is None or profile.sandbox is None:
            return None
        sandbox_mode = str(profile.sandbox.sandbox_mode or "").strip()
        return sandbox_mode or None

    async def _request(self, *, method: str, params: Any) -> Mapping[str, Any]:
        request_id = self._client.reserve_request_id()
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        self._pending_requests[str(request_id)] = waiter
        await self._client.send_request_with_id(request_id, method=method, params=params)
        return await waiter

    async def _request_with_timeout(
        self,
        *,
        stage: str,
        timeout_seconds: float,
        method: str,
        params: Any,
    ) -> Mapping[str, Any]:
        request_id = self._client.reserve_request_id()
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[Mapping[str, Any]] = loop.create_future()
        self._pending_requests[str(request_id)] = waiter
        await self._client.send_request_with_id(request_id, method=method, params=params)
        try:
            return await asyncio.wait_for(asyncio.shield(waiter), timeout=timeout_seconds)
        except TimeoutError as exc:
            if str(request_id) in self._pending_requests:
                self._pending_requests.pop(str(request_id), None)
            raise RuntimeError(f"codex app-server {stage} timed out after {timeout_seconds:.1f}s") from exc

    def _resolve_request_waiter(self, message: Mapping[str, Any]) -> None:
        request_id = str(message.get("id") or "").strip()
        if not request_id:
            return
        waiter = self._pending_requests.pop(request_id, None)
        if waiter is None or waiter.done():
            return
        if isinstance(message.get("error"), Mapping):
            waiter.set_exception(RuntimeError(str(message["error"].get("message") or "codex app-server request failed")))
            return
        waiter.set_result(dict(message))

    def _fail_waiters(self, exc: Exception) -> None:
        for request_id, waiter in list(self._pending_requests.items()):
            if not waiter.done():
                waiter.set_exception(exc)
            self._pending_requests.pop(request_id, None)
        if self._active_turn_done is not None and not self._active_turn_done.done():
            self._active_turn_done.set_exception(exc)

    async def _close_streams(self) -> None:
        if self._reader_task is not None and self._reader_task.done():
            self._reader_task = None
        await self._event_queue.put(_QUEUE_CLOSED)

    def _build_close_diagnostics(self, *, stage: str, failure_reason: str | None = None, exc: Exception | None = None) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "failure_stage": stage,
            "last_notification_method": self._last_notification_method,
            "app_server_returncode": self._client.returncode,
        }
        stderr_tail = self._client.stderr_tail()
        if stderr_tail:
            diagnostics["app_server_stderr_tail"] = list(stderr_tail)
        if self._last_notification_at is not None:
            diagnostics["last_notification_at"] = self._last_notification_at.isoformat()
        if self._last_notification_payload is not None:
            diagnostics["last_notification_excerpt"] = dict(self._last_notification_payload)
        if failure_reason:
            diagnostics["failure_reason"] = failure_reason
        if exc is not None:
            diagnostics["reader_failure_type"] = type(exc).__name__
            diagnostics["reader_failure_message"] = str(exc)
        return diagnostics

    def _reset_thread_closed_waiter(self) -> None:
        loop = asyncio.get_running_loop()
        self._thread_closed_waiter = loop.create_future()

    def _touch(
        self,
        *,
        status: ExecutionSessionStatus | None = None,
        control_state: ExecutionControlState | None = None,
    ) -> None:
        if status is not None:
            self._status = status
        if control_state is not None:
            self._control_state = control_state
        self._updated_at = _utcnow()

    def _snapshot_locked(self) -> CodexSessionSnapshot:
        return CodexSessionSnapshot(
            session_id=self._session_id,
            status=self._status,
            control_state=self._control_state,
            thread_id=self._thread_id,
            active_turn_id=self._active_turn_id,
            thread_loaded=self._thread_loaded,
            raw_event_count=self._raw_event_count,
            started_at=self._started_at,
            updated_at=self._updated_at,
            usage=dict(self._usage),
        )

    def _require_started(self) -> None:
        if self._spec is None:
            raise RuntimeError("codex session has not been started")

    def _update_usage_from_message(self, message: Mapping[str, Any]) -> None:
        total_usage = _extract_total_token_usage(message)
        if total_usage is None:
            return
        self._usage = {
            "prompt_tokens": _int_value(total_usage.get("input_tokens")),
            "completion_tokens": _int_value(total_usage.get("output_tokens")),
            "total_tokens": _int_value(total_usage.get("total_tokens")),
            "cached_prompt_tokens": _int_value(total_usage.get("cached_input_tokens")),
            "reasoning_tokens": _int_value(total_usage.get("reasoning_output_tokens")),
        }


def _turn_result_from_message(message: Mapping[str, Any], *, method: str) -> tuple[ExecutionResultStatus, str | None]:
    if method == "turn/interrupted":
        return ExecutionResultStatus.CANCELED, "turn interrupted"
    params = message.get("params")
    if isinstance(params, Mapping):
        turn = params.get("turn")
        if isinstance(turn, Mapping):
            status = str(turn.get("status") or "").strip().lower()
            if status == "completed":
                return ExecutionResultStatus.COMPLETED, None
            if status == "failed":
                return ExecutionResultStatus.FAILED, extract_failure_reason(message)
            if status in {"interrupted", "cancelled"}:
                return ExecutionResultStatus.CANCELED, f"turn {status}"
    if method == "turn/completed":
        return ExecutionResultStatus.COMPLETED, None
    if method == "turn/failed":
        return ExecutionResultStatus.FAILED, extract_failure_reason(message)
    return ExecutionResultStatus.FAILED, "unrecognized terminal turn event"


def _session_status_from_result(status: ExecutionResultStatus) -> ExecutionSessionStatus:
    if status == ExecutionResultStatus.COMPLETED:
        return ExecutionSessionStatus.IDLE
    if status == ExecutionResultStatus.CANCELED:
        return ExecutionSessionStatus.IDLE
    if status == ExecutionResultStatus.TIMED_OUT:
        return ExecutionSessionStatus.TIMED_OUT
    return ExecutionSessionStatus.FAILED


def _result_status_from_session(status: ExecutionSessionStatus) -> ExecutionResultStatus:
    if status == ExecutionSessionStatus.TIMED_OUT:
        return ExecutionResultStatus.TIMED_OUT
    if status in {ExecutionSessionStatus.CANCELED, ExecutionSessionStatus.PAUSED}:
        return ExecutionResultStatus.CANCELED
    if status == ExecutionSessionStatus.FAILED:
        return ExecutionResultStatus.FAILED
    return ExecutionResultStatus.COMPLETED


def _extract_total_token_usage(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates: list[Mapping[str, Any]] = []
    params = message.get("params")
    if isinstance(params, Mapping):
        for key in ("msg", "item", "info"):
            candidate = params.get(key)
            if isinstance(candidate, Mapping):
                candidates.append(candidate)
    payload = message.get("payload")
    if isinstance(payload, Mapping):
        candidates.append(payload)
    candidates.append(message)
    for candidate in candidates:
        if str(candidate.get("type") or "").strip().lower() == "token_count":
            info = candidate.get("info")
            if isinstance(info, Mapping):
                total_usage = info.get("total_token_usage")
                if isinstance(total_usage, Mapping):
                    return total_usage
        total_usage = candidate.get("total_token_usage")
        if isinstance(total_usage, Mapping):
            return total_usage
        info = candidate.get("info")
        if isinstance(info, Mapping):
            nested_total = info.get("total_token_usage")
            if isinstance(nested_total, Mapping):
                return nested_total
    return None


def _int_value(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _is_response(message: Mapping[str, Any]) -> bool:
    return "id" in message and "method" not in message


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
