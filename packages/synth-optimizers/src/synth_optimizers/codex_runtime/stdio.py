from __future__ import annotations

import asyncio
import json
import os
import signal
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEFAULT_STDIO_STREAM_LIMIT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class CodexAppServerLaunchSpec:
    command: tuple[str, ...]
    working_dir: Path
    env: Mapping[str, str] = field(default_factory=dict)


class CodexAppServerStdioClient:
    """Minimal JSON-RPC stdio wrapper for `codex app-server`."""

    def __init__(self, spec: CodexAppServerLaunchSpec) -> None:
        self._spec = spec
        self._process: asyncio.subprocess.Process | None = None
        self._start_lock = asyncio.Lock()
        self._buffered_messages: deque[dict[str, Any]] = deque()
        self._stderr_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=50)
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_closed = False
        self._next_request_id = 1

    @property
    def spec(self) -> CodexAppServerLaunchSpec:
        return self._spec

    @property
    def returncode(self) -> int | None:
        return None if self._process is None else self._process.returncode

    def stderr_tail(self) -> list[str]:
        return list(self._stderr_tail)

    async def start(self) -> None:
        async with self._start_lock:
            if self._process is not None and self._process.returncode is None:
                return
            self._process = await asyncio.create_subprocess_exec(
                *self._spec.command,
                cwd=str(self._spec.working_dir),
                env=dict(self._spec.env),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_DEFAULT_STDIO_STREAM_LIMIT_BYTES,
                start_new_session=True,
            )
            self._stderr_closed = False
            self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def send_message(self, payload: Mapping[str, Any]) -> None:
        process = self._require_process()
        if process.stdin is None:
            raise RuntimeError("codex app-server stdin is not available")
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        process.stdin.write(encoded)
        process.stdin.write(b"\n")
        await process.stdin.drain()

    async def send_request(self, *, method: str, params: Any) -> int:
        request_id = self.reserve_request_id()
        await self.send_request_with_id(request_id, method=method, params=params)
        return request_id

    def reserve_request_id(self) -> int:
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    async def send_request_with_id(self, request_id: int, *, method: str, params: Any) -> None:
        await self.send_message({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})

    async def send_notification(self, *, method: str, params: Any | None = None) -> None:
        await self.send_message({"jsonrpc": "2.0", "method": method, "params": params})

    async def send_response(self, response_id: Any, *, result: Any | None = None, error: Any | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": response_id}
        if error is not None:
            payload["error"] = error
        else:
            payload["result"] = result
        await self.send_message(payload)

    async def read_message(self) -> dict[str, Any] | None:
        if self._buffered_messages:
            return self._buffered_messages.popleft()
        return await self._read_process_message()

    async def _read_process_message(self) -> dict[str, Any] | None:
        process = self._require_process()
        if process.stdout is None:
            raise RuntimeError("codex app-server stdout is not available")
        payload = await self._read_jsonrpc_message(process.stdout)
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise RuntimeError("codex app-server emitted a non-object JSON message")
        return payload

    async def _read_jsonrpc_message(self, stream: asyncio.StreamReader) -> dict[str, Any] | None:
        headers: dict[str, str] = {}
        while True:
            raw_line = await stream.readline()
            if not raw_line:
                return None
            stripped = raw_line.strip()
            if not stripped:
                if headers:
                    break
                continue
            if not headers and stripped[:1] in (b"{", b"["):
                payload = json.loads(stripped.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise RuntimeError("codex app-server emitted a non-object JSON message")
                return payload
            text = raw_line.decode("utf-8")
            if ":" not in text:
                continue
            key, value = text.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        raw_length = headers.get("content-length")
        if raw_length is None:
            raise RuntimeError("codex app-server stdout message missing Content-Length")
        try:
            content_length = int(raw_length)
        except ValueError as exc:
            raise RuntimeError(f"codex app-server emitted invalid Content-Length header: {raw_length}") from exc
        if content_length < 0:
            raise RuntimeError("codex app-server emitted negative Content-Length")
        payload_bytes = await stream.readexactly(content_length)
        payload = json.loads(payload_bytes.decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("codex app-server emitted a non-object JSON message")
        return payload

    async def initialize(
        self,
        *,
        client_name: str = "mipro-codex-runtime",
        client_title: str = "mipro-codex-runtime",
        client_version: str = "0.1.0",
    ) -> dict[str, Any]:
        request_id = await self.send_request(
            method="initialize",
            params={"clientInfo": {"name": client_name, "title": client_title, "version": client_version}},
        )
        response = await self.wait_for_response(request_id)
        await self.send_notification(method="initialized", params=None)
        return response

    async def wait_for_response(self, request_id: int) -> dict[str, Any]:
        buffered_while_waiting: list[dict[str, Any]] = []
        while True:
            if self._buffered_messages:
                message = self._buffered_messages.popleft()
            else:
                message = await self._read_process_message()
            if message is None:
                raise RuntimeError(f"codex app-server exited before responding to request {request_id}")
            if str(message.get("id") or "") == str(request_id) and "method" not in message:
                self._buffered_messages.extend(buffered_while_waiting)
                return message
            buffered_while_waiting.append(message)

    async def read_stderr_line(self) -> str | None:
        return await self._stderr_queue.get()

    async def terminate(self, *, grace_seconds: float = 5.0) -> int | None:
        process = self._process
        if process is None:
            return None
        if process.returncode is not None:
            return process.returncode
        self._signal_process_tree(signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        except asyncio.TimeoutError:
            self._signal_process_tree(signal.SIGKILL)
            try:
                await asyncio.wait_for(process.wait(), timeout=max(1.0, grace_seconds))
            except asyncio.TimeoutError:
                return process.returncode
        return process.returncode

    def _signal_process_tree(self, sig: signal.Signals) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        pid = int(process.pid or 0)
        if pid <= 0:
            return
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        except Exception:
            try:
                if sig == signal.SIGTERM:
                    process.terminate()
                else:
                    process.kill()
            except ProcessLookupError:
                return

    async def _drain_stderr(self) -> None:
        process = self._require_process()
        if process.stderr is None:
            self._stderr_closed = True
            await self._stderr_queue.put(None)
            return
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            self._stderr_tail.append(text)
            await self._stderr_queue.put(text)
        self._stderr_closed = True
        await self._stderr_queue.put(None)

    def _require_process(self) -> asyncio.subprocess.Process:
        if self._process is None:
            raise RuntimeError("codex app-server process has not been started")
        return self._process
