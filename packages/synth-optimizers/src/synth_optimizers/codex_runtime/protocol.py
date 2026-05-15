from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .types import CodexSessionEventKind


def build_codex_app_server_command(
    *,
    binary: str = "codex",
    config_overrides: Sequence[str] = (),
    extra_args: Sequence[str] = (),
) -> tuple[str, ...]:
    command = [str(binary).strip() or "codex", "app-server"]
    for override in tuple(str(item).strip() for item in config_overrides if str(item).strip()):
        command.extend(("-c", override))
    command.extend(str(arg) for arg in extra_args)
    return tuple(command)


def build_codex_thread_start_params(
    *,
    model: str,
    run_instructions: str | None = None,
    approval_policy: str | None = None,
    sandbox_mode: str | None = None,
    extra_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"model": str(model)}
    instructions = str(run_instructions or "").strip()
    if instructions:
        params["instructions"] = instructions
    approval = str(approval_policy or "").strip()
    if approval:
        params["approvalPolicy"] = approval
    sandbox_policy = _sandbox_policy_for_mode(sandbox_mode)
    if sandbox_policy is not None:
        params["sandbox"] = str(sandbox_mode)
    if extra_config:
        params["config"] = dict(extra_config)
    return params


def build_codex_turn_start_params(
    *,
    thread_id: str,
    instructions: str,
    model: str,
    reasoning_effort: str | None = None,
    approval_policy: str | None = None,
    sandbox_mode: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "threadId": str(thread_id),
        "model": str(model),
        "input": [_text_input(instructions)],
    }
    effort = str(reasoning_effort or "").strip()
    if effort:
        params["effort"] = effort
    approval = str(approval_policy or "").strip()
    if approval:
        params["approvalPolicy"] = approval
    sandbox_policy = _sandbox_policy_for_mode(sandbox_mode)
    if sandbox_policy is not None:
        params["sandboxPolicy"] = sandbox_policy
    return params


def extract_thread_id(payload: Mapping[str, Any]) -> str | None:
    result = payload.get("result")
    if isinstance(result, Mapping):
        thread = result.get("thread")
        if isinstance(thread, Mapping):
            thread_id = str(thread.get("id") or "").strip()
            if thread_id:
                return thread_id
        thread_id = str(result.get("threadId") or "").strip()
        if thread_id:
            return thread_id
    params = payload.get("params")
    if isinstance(params, Mapping):
        thread = params.get("thread")
        if isinstance(thread, Mapping):
            thread_id = str(thread.get("id") or "").strip()
            if thread_id:
                return thread_id
        thread_id = str(params.get("threadId") or "").strip()
        if thread_id:
            return thread_id
    return None


def extract_turn_id(payload: Mapping[str, Any]) -> str | None:
    result = payload.get("result")
    if isinstance(result, Mapping):
        turn = result.get("turn")
        if isinstance(turn, Mapping):
            turn_id = str(turn.get("id") or "").strip()
            if turn_id:
                return turn_id
        turn_id = str(result.get("turnId") or "").strip()
        if turn_id:
            return turn_id
    params = payload.get("params")
    if isinstance(params, Mapping):
        turn = params.get("turn")
        if isinstance(turn, Mapping):
            turn_id = str(turn.get("id") or "").strip()
            if turn_id:
                return turn_id
        turn_id = str(params.get("turnId") or "").strip()
        if turn_id:
            return turn_id
    return None


def message_is_terminal_for_turn(payload: Mapping[str, Any], turn_id: str) -> bool:
    method = str(payload.get("method") or "").strip()
    params = payload.get("params")
    if not isinstance(params, Mapping):
        return False
    observed_turn_id = extract_turn_id(payload)
    if observed_turn_id and observed_turn_id != str(turn_id):
        return False
    if method in {"turn/completed", "turn/failed", "turn/interrupted"}:
        return True
    turn = params.get("turn")
    if isinstance(turn, Mapping):
        status = str(turn.get("status") or "").strip().lower()
        if status in {"completed", "failed", "interrupted", "cancelled"}:
            return True
    return False


def extract_failure_reason(message: Mapping[str, Any]) -> str | None:
    error = message.get("error")
    if isinstance(error, Mapping):
        failure_reason = str(error.get("message") or "").strip()
        if failure_reason:
            return failure_reason
    params = message.get("params")
    if isinstance(params, Mapping):
        turn = params.get("turn")
        if isinstance(turn, Mapping):
            failure_reason = str(turn.get("failureReason") or "").strip()
            if failure_reason:
                return failure_reason
            turn_error = turn.get("error")
            if isinstance(turn_error, Mapping):
                failure_reason = str(turn_error.get("message") or "").strip()
                if failure_reason:
                    return failure_reason
    return None


def classify_event_kind(message: Mapping[str, Any]) -> CodexSessionEventKind:
    method = str(message.get("method") or "").strip()
    if method == "thread/started":
        return CodexSessionEventKind.THREAD_STARTED
    if method == "thread/status/changed":
        return CodexSessionEventKind.THREAD_STATUS_CHANGED
    if method == "thread/closed":
        return CodexSessionEventKind.THREAD_CLOSED
    if method == "turn/started":
        return CodexSessionEventKind.TURN_STARTED
    if method == "turn/completed":
        status = thread_or_turn_status(message)
        if status == "failed":
            return CodexSessionEventKind.TURN_FAILED
        if status in {"interrupted", "cancelled"}:
            return CodexSessionEventKind.TURN_INTERRUPTED
        return CodexSessionEventKind.TURN_COMPLETED
    if method == "turn/failed":
        return CodexSessionEventKind.TURN_FAILED
    if method == "turn/interrupted":
        return CodexSessionEventKind.TURN_INTERRUPTED
    if method.startswith("item/"):
        item_type = extract_item_type(message)
        if item_type == "agent_message":
            return CodexSessionEventKind.AGENT_MESSAGE
        if item_type == "reasoning":
            return CodexSessionEventKind.REASONING
        if item_type == "command_execution":
            return CodexSessionEventKind.COMMAND_EXECUTION
        if item_type == "file_change":
            return CodexSessionEventKind.FILE_CHANGE
        if item_type == "mcp_tool_call":
            return CodexSessionEventKind.MCP_TOOL_CALL
        return CodexSessionEventKind.MESSAGE
    if method == "error":
        return CodexSessionEventKind.ERROR
    return CodexSessionEventKind.OTHER


def thread_or_turn_status(message: Mapping[str, Any]) -> str:
    params = message.get("params")
    if not isinstance(params, Mapping):
        return ""
    for key in ("thread", "turn"):
        target = params.get(key)
        if isinstance(target, Mapping):
            status = str(target.get("status") or "").strip().lower()
            if status:
                return status
    return ""


def extract_item_type(message: Mapping[str, Any]) -> str:
    params = message.get("params")
    if not isinstance(params, Mapping):
        return ""
    item = params.get("item")
    if isinstance(item, Mapping):
        return str(item.get("type") or "").strip().lower()
    return ""


def is_retryable_error_notification(message: Mapping[str, Any]) -> bool:
    params = message.get("params")
    if not isinstance(params, Mapping):
        return False
    if not bool(params.get("willRetry")):
        return False
    error = params.get("error")
    if not isinstance(error, Mapping):
        return False
    additional_details = str(error.get("additionalDetails") or "").strip().lower()
    codex_error_info = error.get("codexErrorInfo")
    if "stream disconnected before completion" in additional_details:
        return True
    if isinstance(codex_error_info, Mapping) and isinstance(codex_error_info.get("responseStreamDisconnected"), Mapping):
        return True
    return False


def render_json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _text_input(text: str) -> dict[str, Any]:
    return {"type": "text", "text": str(text), "textElements": []}


def _sandbox_policy_for_mode(sandbox_mode: str | None) -> dict[str, Any] | None:
    mode = str(sandbox_mode or "").strip()
    if not mode:
        return None
    if mode == "danger-full-access":
        return {"type": "dangerFullAccess"}
    if mode == "read-only":
        return {"type": "readOnly", "access": {"type": "fullAccess"}, "networkAccess": True}
    if mode == "workspace-write":
        return {"type": "workspaceWrite", "readOnlyAccess": {"type": "fullAccess"}, "networkAccess": True}
    return None
