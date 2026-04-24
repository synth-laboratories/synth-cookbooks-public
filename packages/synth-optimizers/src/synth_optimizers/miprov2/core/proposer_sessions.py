"""File-backed MIPRO proposer session storage."""

from __future__ import annotations

import json
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4


def _now() -> float:
    return float(time.time())


def new_session_id() -> str:
    return f"mps_{uuid4().hex}"


def new_event_id() -> str:
    return f"evt_{uuid4().hex}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(slots=True)
class MiproProposerSession:
    session_id: str
    run_id: str | None = None
    round_idx: int = 0
    source_kind: str = "checkpoint"
    source_ref: str | None = None
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    status: str = "open"
    base_version: int = 0
    current_version: int = 0
    variant: dict[str, Any] = field(default_factory=dict)
    workspace_root: str | None = None
    session_dir: str | None = None
    event_log_path: str | None = None
    pre_state_ref: str | None = None
    current_state_ref: str | None = None
    committed_state_ref: str | None = None
    event_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "round_idx": int(self.round_idx),
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "status": self.status,
            "base_version": int(self.base_version),
            "current_version": int(self.current_version),
            "variant": dict(self.variant),
            "workspace_root": self.workspace_root,
            "session_dir": self.session_dir,
            "event_log_path": self.event_log_path,
            "pre_state_ref": self.pre_state_ref,
            "current_state_ref": self.current_state_ref,
            "committed_state_ref": self.committed_state_ref,
            "event_count": int(self.event_count),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MiproProposerSession":
        return cls(
            session_id=str(payload["session_id"]),
            run_id=str(payload["run_id"]) if payload.get("run_id") is not None else None,
            round_idx=int(payload.get("round_idx") or 0),
            source_kind=str(payload.get("source_kind") or "checkpoint"),
            source_ref=str(payload["source_ref"]) if payload.get("source_ref") is not None else None,
            created_at=float(payload.get("created_at") or _now()),
            updated_at=float(payload.get("updated_at") or _now()),
            status=str(payload.get("status") or "open"),
            base_version=int(payload.get("base_version") or 0),
            current_version=int(payload.get("current_version") or 0),
            variant=dict(payload.get("variant") or {}),
            workspace_root=(
                str(payload["workspace_root"]) if payload.get("workspace_root") is not None else None
            ),
            session_dir=(
                str(payload["session_dir"]) if payload.get("session_dir") is not None else None
            ),
            event_log_path=(
                str(payload["event_log_path"]) if payload.get("event_log_path") is not None else None
            ),
            pre_state_ref=(
                str(payload["pre_state_ref"]) if payload.get("pre_state_ref") is not None else None
            ),
            current_state_ref=(
                str(payload["current_state_ref"])
                if payload.get("current_state_ref") is not None
                else None
            ),
            committed_state_ref=(
                str(payload["committed_state_ref"])
                if payload.get("committed_state_ref") is not None
                else None
            ),
            event_count=int(payload.get("event_count") or 0),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True, frozen=True)
class MiproProposerEvent:
    event_id: str
    session_id: str
    created_at: float
    event_type: str
    actor_id: str
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    state_version_before: int | None = None
    state_version_after: int | None = None
    mutation_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "created_at": float(self.created_at),
            "event_type": self.event_type,
            "actor_id": self.actor_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "result": dict(self.result),
            "state_version_before": self.state_version_before,
            "state_version_after": self.state_version_after,
            "mutation_summary": dict(self.mutation_summary),
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class MiproProposerSessionStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        return self.root / str(session_id)

    def create(
        self,
        *,
        session: MiproProposerSession,
        pre_state: dict[str, Any],
        current_state: dict[str, Any],
        actor_id: str,
    ) -> MiproProposerSession:
        session_dir = self.session_dir(session.session_id)
        if session_dir.exists():
            raise FileExistsError(str(session_dir))
        session_dir.mkdir(parents=True)
        session.session_dir = str(session_dir)
        session.event_log_path = str(session_dir / "events.jsonl")
        session.pre_state_ref = str(session_dir / "pre_state.json")
        session.current_state_ref = str(session_dir / "current_state.json")
        _write_json(session_dir / "pre_state.json", pre_state)
        _write_json(session_dir / "current_state.json", current_state)
        self.save_session(session)
        self.append_event(
            session,
            MiproProposerEvent(
                event_id=new_event_id(),
                session_id=session.session_id,
                created_at=_now(),
                event_type="session_created",
                actor_id=actor_id,
                state_version_before=0,
                state_version_after=0,
                metadata={"source_kind": session.source_kind, "source_ref": session.source_ref},
            ),
        )
        return session

    def load_session(self, session_id: str) -> MiproProposerSession:
        return MiproProposerSession.from_dict(_read_json(self.session_dir(session_id) / "session.json"))

    def save_session(self, session: MiproProposerSession) -> None:
        session.updated_at = _now()
        if session.session_dir is None:
            session.session_dir = str(self.session_dir(session.session_id))
        _write_json(Path(session.session_dir) / "session.json", session.to_dict())

    def load_current_state(self, session: MiproProposerSession) -> dict[str, Any]:
        if session.current_state_ref is None:
            raise ValueError("session is missing current_state_ref")
        return _read_json(Path(session.current_state_ref))

    def save_current_state(
        self,
        session: MiproProposerSession,
        payload: dict[str, Any],
    ) -> None:
        if session.current_state_ref is None:
            raise ValueError("session is missing current_state_ref")
        _write_json(Path(session.current_state_ref), payload)

    def append_event(
        self,
        session: MiproProposerSession,
        event: MiproProposerEvent,
    ) -> None:
        if session.event_log_path is None:
            raise ValueError("session is missing event_log_path")
        path = Path(session.event_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True, ensure_ascii=True) + "\n")
        session.event_count += 1
        self.save_session(session)

    def write_checkpoint(
        self,
        session: MiproProposerSession,
        payload: dict[str, Any],
    ) -> Path:
        session_dir = self.session_dir(session.session_id)
        checkpoint_dir = session_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"checkpoint_{int(session.event_count):06d}.json"
        _write_json(path, payload)
        return path

    def write_commit(
        self,
        session: MiproProposerSession,
        *,
        state_payload: dict[str, Any],
        commit_payload: dict[str, Any],
    ) -> tuple[Path, Path]:
        session_dir = self.session_dir(session.session_id)
        committed_state_path = session_dir / "committed_state.json"
        commit_path = session_dir / "commit.json"
        commit_payload["committed_state_ref"] = str(committed_state_path)
        commit_payload["commit_ref"] = str(commit_path)
        _write_json(committed_state_path, state_payload)
        _write_json(commit_path, commit_payload)
        session.committed_state_ref = str(committed_state_path)
        self.save_session(session)
        return committed_state_path, commit_path

    @contextmanager
    def lock(self, session_id: str) -> Iterator[None]:
        lock_path = self.session_dir(session_id) / ".lock"
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at": _now(),
        }
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"proposer session is locked: {session_id}") from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
            yield
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
