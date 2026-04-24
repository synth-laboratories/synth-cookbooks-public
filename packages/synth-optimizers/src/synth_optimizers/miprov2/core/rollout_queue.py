"""Rollout queue models for MIPROv2 proposer planning."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


def _now() -> float:
    return float(time.time())


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


@dataclass(slots=True)
class MiproCandidateInterventionRef:
    candidate_id: str
    lever_bundle_hash: str
    source_config: dict[str, str]
    parent_candidate_id: str | None = None
    plugin_kind: str = "prompt"
    plugin_id: str | None = None
    prompt_intervention: dict[str, Any] = field(default_factory=dict)
    sft_intervention: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parent_candidate_id": self.parent_candidate_id,
            "lever_bundle_hash": self.lever_bundle_hash,
            "plugin_kind": self.plugin_kind,
            "plugin_id": self.plugin_id,
            "source_config": dict(self.source_config),
            "prompt_intervention": dict(self.prompt_intervention),
            "sft_intervention": dict(self.sft_intervention),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MiproCandidateInterventionRef":
        return cls(
            candidate_id=str(payload.get("candidate_id") or ""),
            parent_candidate_id=(
                str(payload["parent_candidate_id"])
                if payload.get("parent_candidate_id") is not None
                else None
            ),
            lever_bundle_hash=str(payload.get("lever_bundle_hash") or ""),
            plugin_kind=str(payload.get("plugin_kind") or "prompt"),
            plugin_id=(
                str(payload["plugin_id"]) if payload.get("plugin_id") is not None else None
            ),
            source_config={
                str(key): str(value)
                for key, value in _dict(payload.get("source_config")).items()
            },
            prompt_intervention=_dict(payload.get("prompt_intervention")),
            sft_intervention=_dict(payload.get("sft_intervention")),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class MiproQueuedRollout:
    rollout_id: str
    candidate_id: str
    candidate_interventions: list[MiproCandidateInterventionRef]
    split: str = "train"
    row_id: str | None = None
    seed: int | None = None
    task_instance_id: str | None = None
    evaluator_config: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    priority: float = 0.0
    created_by: str = "tpe"
    created_at: float = field(default_factory=_now)
    started_at: float | None = None
    completed_at: float | None = None
    result_ref: str | None = None
    linked_hypothesis_refs: list[str] = field(default_factory=list)
    linked_bet_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "candidate_id": self.candidate_id,
            "candidate_interventions": [
                item.to_dict() for item in self.candidate_interventions
            ],
            "split": self.split,
            "row_id": self.row_id,
            "seed": self.seed,
            "task_instance_id": self.task_instance_id,
            "evaluator_config": dict(self.evaluator_config),
            "status": self.status,
            "priority": float(self.priority),
            "created_by": self.created_by,
            "created_at": float(self.created_at),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result_ref": self.result_ref,
            "linked_hypothesis_refs": list(self.linked_hypothesis_refs),
            "linked_bet_refs": list(self.linked_bet_refs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MiproQueuedRollout":
        return cls(
            rollout_id=str(payload.get("rollout_id") or ""),
            candidate_id=str(payload.get("candidate_id") or ""),
            candidate_interventions=[
                MiproCandidateInterventionRef.from_dict(dict(item))
                for item in list(payload.get("candidate_interventions") or [])
                if isinstance(item, dict)
            ],
            split=str(payload.get("split") or "train"),
            row_id=str(payload["row_id"]) if payload.get("row_id") is not None else None,
            seed=int(payload["seed"]) if payload.get("seed") is not None else None,
            task_instance_id=(
                str(payload["task_instance_id"])
                if payload.get("task_instance_id") is not None
                else None
            ),
            evaluator_config=_dict(payload.get("evaluator_config")),
            status=str(payload.get("status") or "queued"),
            priority=float(payload.get("priority") or 0.0),
            created_by=str(payload.get("created_by") or "tpe"),
            created_at=float(payload.get("created_at") or _now()),
            started_at=(
                float(payload["started_at"])
                if payload.get("started_at") is not None
                else None
            ),
            completed_at=(
                float(payload["completed_at"])
                if payload.get("completed_at") is not None
                else None
            ),
            result_ref=(
                str(payload["result_ref"]) if payload.get("result_ref") is not None else None
            ),
            linked_hypothesis_refs=[
                str(item) for item in list(payload.get("linked_hypothesis_refs") or [])
            ],
            linked_bet_refs=[str(item) for item in list(payload.get("linked_bet_refs") or [])],
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class MiproRolloutQueue:
    queue_id: str
    queue_kind: str
    task_id: str | None
    split: str
    created_by: str
    candidates: list[MiproCandidateInterventionRef]
    rollouts: list[MiproQueuedRollout]
    created_at: float = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "queue_kind": self.queue_kind,
            "task_id": self.task_id,
            "split": self.split,
            "created_by": self.created_by,
            "candidate_count": len(self.candidates),
            "rollout_count": len(self.rollouts),
            "candidates": [item.to_dict() for item in self.candidates],
            "rollouts": [item.to_dict() for item in self.rollouts],
            "created_at": float(self.created_at),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MiproRolloutQueue":
        return cls(
            queue_id=str(payload.get("queue_id") or ""),
            queue_kind=str(payload.get("queue_kind") or "tentative"),
            task_id=str(payload["task_id"]) if payload.get("task_id") is not None else None,
            split=str(payload.get("split") or "train"),
            created_by=str(payload.get("created_by") or "tpe"),
            candidates=[
                MiproCandidateInterventionRef.from_dict(dict(item))
                for item in list(payload.get("candidates") or [])
                if isinstance(item, dict)
            ],
            rollouts=[
                MiproQueuedRollout.from_dict(dict(item))
                for item in list(payload.get("rollouts") or [])
                if isinstance(item, dict)
            ],
            created_at=float(payload.get("created_at") or _now()),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class MiproRolloutQueueOverride:
    override_id: str
    queue_id: str
    override_kind: str
    reason: str
    target_rollout_id: str | None = None
    replacement_rollout: MiproQueuedRollout | None = None
    target_index: int | None = None
    expected_information_gain: str | None = None
    linked_hypothesis_refs: list[str] = field(default_factory=list)
    linked_bet_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "override_id": self.override_id,
            "queue_id": self.queue_id,
            "override_kind": self.override_kind,
            "target_rollout_id": self.target_rollout_id,
            "replacement_rollout": (
                self.replacement_rollout.to_dict()
                if self.replacement_rollout is not None
                else None
            ),
            "target_index": self.target_index,
            "reason": self.reason,
            "expected_information_gain": self.expected_information_gain,
            "linked_hypothesis_refs": list(self.linked_hypothesis_refs),
            "linked_bet_refs": list(self.linked_bet_refs),
            "metadata": dict(self.metadata),
            "created_at": float(self.created_at),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MiproRolloutQueueOverride":
        replacement = payload.get("replacement_rollout")
        return cls(
            override_id=str(payload.get("override_id") or ""),
            queue_id=str(payload.get("queue_id") or ""),
            override_kind=str(payload.get("override_kind") or ""),
            target_rollout_id=(
                str(payload["target_rollout_id"])
                if payload.get("target_rollout_id") is not None
                else None
            ),
            replacement_rollout=(
                MiproQueuedRollout.from_dict(dict(replacement))
                if isinstance(replacement, dict)
                else None
            ),
            target_index=(
                int(payload["target_index"]) if payload.get("target_index") is not None else None
            ),
            reason=str(payload.get("reason") or ""),
            expected_information_gain=(
                str(payload["expected_information_gain"])
                if payload.get("expected_information_gain") is not None
                else None
            ),
            linked_hypothesis_refs=[
                str(item) for item in list(payload.get("linked_hypothesis_refs") or [])
            ],
            linked_bet_refs=[str(item) for item in list(payload.get("linked_bet_refs") or [])],
            metadata=_dict(payload.get("metadata")),
            created_at=float(payload.get("created_at") or _now()),
        )


@dataclass(slots=True)
class MiproRolloutQueueCommit:
    commit_id: str
    queue_id: str
    committed_queue_id: str
    accept_tpe_defaults: bool
    proposer_override_refs: list[str] = field(default_factory=list)
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit_id": self.commit_id,
            "queue_id": self.queue_id,
            "committed_queue_id": self.committed_queue_id,
            "accept_tpe_defaults": bool(self.accept_tpe_defaults),
            "proposer_override_refs": list(self.proposer_override_refs),
            "reason": self.reason,
            "metadata": dict(self.metadata),
            "created_at": float(self.created_at),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MiproRolloutQueueCommit":
        return cls(
            commit_id=str(payload.get("commit_id") or ""),
            queue_id=str(payload.get("queue_id") or ""),
            committed_queue_id=str(payload.get("committed_queue_id") or ""),
            accept_tpe_defaults=bool(payload.get("accept_tpe_defaults")),
            proposer_override_refs=[
                str(item) for item in list(payload.get("proposer_override_refs") or [])
            ],
            reason=str(payload["reason"]) if payload.get("reason") is not None else None,
            metadata=_dict(payload.get("metadata")),
            created_at=float(payload.get("created_at") or _now()),
        )


@dataclass(slots=True)
class MiproQueuedRolloutResult:
    rollout_id: str
    candidate_id: str
    score: float
    status: str = "completed"
    result_ref: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "candidate_id": self.candidate_id,
            "score": float(self.score),
            "status": self.status,
            "result_ref": self.result_ref,
            "details": dict(self.details),
            "created_at": float(self.created_at),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MiproQueuedRolloutResult":
        return cls(
            rollout_id=str(payload.get("rollout_id") or ""),
            candidate_id=str(payload.get("candidate_id") or ""),
            score=float(payload.get("score") or 0.0),
            status=str(payload.get("status") or "completed"),
            result_ref=(
                str(payload["result_ref"]) if payload.get("result_ref") is not None else None
            ),
            details=_dict(payload.get("details")),
            created_at=float(payload.get("created_at") or _now()),
        )


def queue_id_for(*, run_id: str, round_idx: int, kind: str, suffix: str = "") -> str:
    return _stable_id(
        "queue",
        {
            "run_id": run_id,
            "round_idx": int(round_idx),
            "kind": kind,
            "suffix": suffix,
        },
    )


def rollout_id_for(
    *,
    queue_id: str,
    candidate_id: str,
    row_id: str | None,
    index: int,
) -> str:
    return _stable_id(
        "rollout",
        {
            "queue_id": queue_id,
            "candidate_id": candidate_id,
            "row_id": row_id,
            "index": int(index),
        },
    )


__all__ = [
    "MiproCandidateInterventionRef",
    "MiproQueuedRollout",
    "MiproRolloutQueue",
    "MiproRolloutQueueOverride",
    "MiproRolloutQueueCommit",
    "MiproQueuedRolloutResult",
    "queue_id_for",
    "rollout_id_for",
]
