"""Durable proposer memory models for MIPROv2."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


def _now() -> float:
    return float(time.time())


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


class MiproHypothesisStatus(StrEnum):
    ACTIVE = "active"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    SUPERSEDED = "superseded"
    ABANDONED = "abandoned"


class MiproBetStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"
    VOID = "void"


class MiproBetResolution(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    MIXED = "mixed"
    INCONCLUSIVE = "inconclusive"


class MiproRolloutLabelDefinitionStatus(StrEnum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class MiproRolloutLabelAssignmentSource(StrEnum):
    PROPOSER = "proposer"
    LABELLER = "labeller"
    MANUAL = "manual"


@dataclass(slots=True)
class MiproHypothesis:
    hypothesis_id: str
    summary: str
    rationale: str
    status: MiproHypothesisStatus = MiproHypothesisStatus.ACTIVE
    task_id: str | None = None
    dataset_id: str | None = None
    agent_id: str | None = None
    proposer_id: str | None = None
    preference_model_notes: str | None = None
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    candidate_refs: list[str] = field(default_factory=list)
    rollout_refs: list[str] = field(default_factory=list)
    task_refs: list[str] = field(default_factory=list)
    queue_refs: list[str] = field(default_factory=list)
    queue_override_refs: list[str] = field(default_factory=list)
    adjustment_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "task_id": self.task_id,
            "dataset_id": self.dataset_id,
            "agent_id": self.agent_id,
            "proposer_id": self.proposer_id,
            "summary": self.summary,
            "rationale": self.rationale,
            "preference_model_notes": self.preference_model_notes,
            "status": self.status.value,
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "candidate_refs": list(self.candidate_refs),
            "rollout_refs": list(self.rollout_refs),
            "task_refs": list(self.task_refs),
            "queue_refs": list(self.queue_refs),
            "queue_override_refs": list(self.queue_override_refs),
            "adjustment_refs": list(self.adjustment_refs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MiproHypothesis":
        return cls(
            hypothesis_id=str(payload.get("hypothesis_id") or ""),
            task_id=str(payload["task_id"]) if payload.get("task_id") is not None else None,
            dataset_id=str(payload["dataset_id"]) if payload.get("dataset_id") is not None else None,
            agent_id=str(payload["agent_id"]) if payload.get("agent_id") is not None else None,
            proposer_id=(
                str(payload["proposer_id"]) if payload.get("proposer_id") is not None else None
            ),
            summary=str(payload.get("summary") or ""),
            rationale=str(payload.get("rationale") or ""),
            preference_model_notes=(
                str(payload["preference_model_notes"])
                if payload.get("preference_model_notes") is not None
                else None
            ),
            status=MiproHypothesisStatus(str(payload.get("status") or "active")),
            created_at=float(payload.get("created_at") or _now()),
            updated_at=float(payload.get("updated_at") or _now()),
            candidate_refs=_str_list(payload.get("candidate_refs")),
            rollout_refs=_str_list(payload.get("rollout_refs")),
            task_refs=_str_list(payload.get("task_refs")),
            queue_refs=_str_list(payload.get("queue_refs")),
            queue_override_refs=_str_list(payload.get("queue_override_refs")),
            adjustment_refs=_str_list(payload.get("adjustment_refs")),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class MiproHypothesisAdjustment:
    adjustment_id: str
    hypothesis_id: str
    summary: str
    reason: str
    created_at: float = field(default_factory=_now)
    created_by: str | None = None
    diff_payload: dict[str, Any] = field(default_factory=dict)
    linked_candidate_refs: list[str] = field(default_factory=list)
    linked_rollout_refs: list[str] = field(default_factory=list)
    linked_task_refs: list[str] = field(default_factory=list)
    linked_queue_refs: list[str] = field(default_factory=list)
    linked_queue_override_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adjustment_id": self.adjustment_id,
            "hypothesis_id": self.hypothesis_id,
            "created_at": float(self.created_at),
            "created_by": self.created_by,
            "summary": self.summary,
            "diff_payload": dict(self.diff_payload),
            "reason": self.reason,
            "linked_candidate_refs": list(self.linked_candidate_refs),
            "linked_rollout_refs": list(self.linked_rollout_refs),
            "linked_task_refs": list(self.linked_task_refs),
            "linked_queue_refs": list(self.linked_queue_refs),
            "linked_queue_override_refs": list(self.linked_queue_override_refs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MiproHypothesisAdjustment":
        return cls(
            adjustment_id=str(payload.get("adjustment_id") or ""),
            hypothesis_id=str(payload.get("hypothesis_id") or ""),
            created_at=float(payload.get("created_at") or _now()),
            created_by=(
                str(payload["created_by"]) if payload.get("created_by") is not None else None
            ),
            summary=str(payload.get("summary") or ""),
            diff_payload=_dict(payload.get("diff_payload")),
            reason=str(payload.get("reason") or ""),
            linked_candidate_refs=_str_list(payload.get("linked_candidate_refs")),
            linked_rollout_refs=_str_list(payload.get("linked_rollout_refs")),
            linked_task_refs=_str_list(payload.get("linked_task_refs")),
            linked_queue_refs=_str_list(payload.get("linked_queue_refs")),
            linked_queue_override_refs=_str_list(payload.get("linked_queue_override_refs")),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class MiproBet:
    bet_id: str
    summary: str
    prediction: str
    status: MiproBetStatus = MiproBetStatus.OPEN
    hypothesis_id: str | None = None
    proposer_id: str | None = None
    created_at: float = field(default_factory=_now)
    resolved_at: float | None = None
    rollout_refs: list[str] = field(default_factory=list)
    candidate_refs: list[str] = field(default_factory=list)
    task_refs: list[str] = field(default_factory=list)
    queue_refs: list[str] = field(default_factory=list)
    queue_override_refs: list[str] = field(default_factory=list)
    success_criteria: str | None = None
    expected_outcome: str | None = None
    confidence: float | None = None
    resolution: MiproBetResolution | None = None
    resolution_comment: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bet_id": self.bet_id,
            "hypothesis_id": self.hypothesis_id,
            "proposer_id": self.proposer_id,
            "created_at": float(self.created_at),
            "status": self.status.value,
            "summary": self.summary,
            "prediction": self.prediction,
            "rollout_refs": list(self.rollout_refs),
            "candidate_refs": list(self.candidate_refs),
            "task_refs": list(self.task_refs),
            "queue_refs": list(self.queue_refs),
            "queue_override_refs": list(self.queue_override_refs),
            "success_criteria": self.success_criteria,
            "expected_outcome": self.expected_outcome,
            "confidence": self.confidence,
            "resolved_at": self.resolved_at,
            "resolution": self.resolution.value if self.resolution is not None else None,
            "resolution_comment": self.resolution_comment,
            "evidence_refs": list(self.evidence_refs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MiproBet":
        resolution = payload.get("resolution")
        return cls(
            bet_id=str(payload.get("bet_id") or ""),
            hypothesis_id=(
                str(payload["hypothesis_id"]) if payload.get("hypothesis_id") is not None else None
            ),
            proposer_id=(
                str(payload["proposer_id"]) if payload.get("proposer_id") is not None else None
            ),
            created_at=float(payload.get("created_at") or _now()),
            status=MiproBetStatus(str(payload.get("status") or "open")),
            summary=str(payload.get("summary") or ""),
            prediction=str(payload.get("prediction") or ""),
            rollout_refs=_str_list(payload.get("rollout_refs")),
            candidate_refs=_str_list(payload.get("candidate_refs")),
            task_refs=_str_list(payload.get("task_refs")),
            queue_refs=_str_list(payload.get("queue_refs")),
            queue_override_refs=_str_list(payload.get("queue_override_refs")),
            success_criteria=(
                str(payload["success_criteria"])
                if payload.get("success_criteria") is not None
                else None
            ),
            expected_outcome=(
                str(payload["expected_outcome"])
                if payload.get("expected_outcome") is not None
                else None
            ),
            confidence=(
                float(payload["confidence"]) if payload.get("confidence") is not None else None
            ),
            resolved_at=(
                float(payload["resolved_at"]) if payload.get("resolved_at") is not None else None
            ),
            resolution=(
                MiproBetResolution(str(resolution)) if resolution is not None else None
            ),
            resolution_comment=(
                str(payload["resolution_comment"])
                if payload.get("resolution_comment") is not None
                else None
            ),
            evidence_refs=_str_list(payload.get("evidence_refs")),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class MiproRolloutLabelDefinition:
    label_id: str
    name: str
    description: str
    allowed_values: list[str]
    status: MiproRolloutLabelDefinitionStatus = MiproRolloutLabelDefinitionStatus.ACTIVE
    task_id: str | None = None
    created_by_proposer_id: str | None = None
    created_at: float = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "task_id": self.task_id,
            "created_by_proposer_id": self.created_by_proposer_id,
            "name": self.name,
            "description": self.description,
            "allowed_values": list(self.allowed_values),
            "created_at": float(self.created_at),
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MiproRolloutLabelDefinition":
        return cls(
            label_id=str(payload.get("label_id") or ""),
            task_id=str(payload["task_id"]) if payload.get("task_id") is not None else None,
            created_by_proposer_id=(
                str(payload["created_by_proposer_id"])
                if payload.get("created_by_proposer_id") is not None
                else None
            ),
            name=str(payload.get("name") or ""),
            description=str(payload.get("description") or ""),
            allowed_values=_str_list(payload.get("allowed_values")),
            created_at=float(payload.get("created_at") or _now()),
            status=MiproRolloutLabelDefinitionStatus(str(payload.get("status") or "active")),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class MiproRolloutLabel:
    rollout_label_id: str
    rollout_id: str
    label_id: str
    value: str
    assigned_by: str | None = None
    assignment_source: MiproRolloutLabelAssignmentSource = (
        MiproRolloutLabelAssignmentSource.PROPOSER
    )
    assigned_at: float = field(default_factory=_now)
    candidate_id: str | None = None
    task_id: str | None = None
    confidence: float | None = None
    rationale: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    linked_hypothesis_refs: list[str] = field(default_factory=list)
    linked_bet_refs: list[str] = field(default_factory=list)
    queue_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_label_id": self.rollout_label_id,
            "rollout_id": self.rollout_id,
            "label_id": self.label_id,
            "value": self.value,
            "assigned_by": self.assigned_by,
            "assignment_source": self.assignment_source.value,
            "assigned_at": float(self.assigned_at),
            "candidate_id": self.candidate_id,
            "task_id": self.task_id,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "evidence_refs": list(self.evidence_refs),
            "linked_hypothesis_refs": list(self.linked_hypothesis_refs),
            "linked_bet_refs": list(self.linked_bet_refs),
            "queue_refs": list(self.queue_refs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MiproRolloutLabel":
        return cls(
            rollout_label_id=str(payload.get("rollout_label_id") or ""),
            rollout_id=str(payload.get("rollout_id") or ""),
            label_id=str(payload.get("label_id") or ""),
            value=str(payload.get("value") or ""),
            assigned_by=(
                str(payload["assigned_by"]) if payload.get("assigned_by") is not None else None
            ),
            assignment_source=MiproRolloutLabelAssignmentSource(
                str(payload.get("assignment_source") or "proposer")
            ),
            assigned_at=float(payload.get("assigned_at") or _now()),
            candidate_id=(
                str(payload["candidate_id"]) if payload.get("candidate_id") is not None else None
            ),
            task_id=str(payload["task_id"]) if payload.get("task_id") is not None else None,
            confidence=(
                float(payload["confidence"]) if payload.get("confidence") is not None else None
            ),
            rationale=str(payload["rationale"]) if payload.get("rationale") is not None else None,
            evidence_refs=_str_list(payload.get("evidence_refs")),
            linked_hypothesis_refs=_str_list(payload.get("linked_hypothesis_refs")),
            linked_bet_refs=_str_list(payload.get("linked_bet_refs")),
            queue_refs=_str_list(payload.get("queue_refs")),
            metadata=_dict(payload.get("metadata")),
        )


def hypothesis_id_for(payload: dict[str, Any]) -> str:
    return _stable_id(
        "hyp",
        {
            "summary": str(payload.get("summary") or ""),
            "rationale": str(payload.get("rationale") or ""),
            "task_id": payload.get("task_id"),
            "dataset_id": payload.get("dataset_id"),
            "candidate_refs": _str_list(payload.get("candidate_refs")),
            "rollout_refs": _str_list(payload.get("rollout_refs")),
        },
    )


def hypothesis_adjustment_id_for(payload: dict[str, Any]) -> str:
    return _stable_id(
        "hadj",
        {
            "hypothesis_id": str(payload.get("hypothesis_id") or ""),
            "summary": str(payload.get("summary") or ""),
            "reason": str(payload.get("reason") or ""),
            "diff_payload": _dict(payload.get("diff_payload")),
        },
    )


def bet_id_for(payload: dict[str, Any]) -> str:
    return _stable_id(
        "bet",
        {
            "hypothesis_id": payload.get("hypothesis_id"),
            "summary": str(payload.get("summary") or ""),
            "prediction": str(payload.get("prediction") or ""),
            "rollout_refs": _str_list(payload.get("rollout_refs")),
            "candidate_refs": _str_list(payload.get("candidate_refs")),
            "task_refs": _str_list(payload.get("task_refs")),
        },
    )


def rollout_label_definition_id_for(payload: dict[str, Any]) -> str:
    return _stable_id(
        "rldef",
        {
            "task_id": payload.get("task_id"),
            "name": str(payload.get("name") or ""),
            "allowed_values": _str_list(payload.get("allowed_values")),
        },
    )


def rollout_label_id_for(payload: dict[str, Any]) -> str:
    return _stable_id(
        "rlabel",
        {
            "rollout_id": str(payload.get("rollout_id") or ""),
            "label_id": str(payload.get("label_id") or ""),
            "value": str(payload.get("value") or ""),
            "candidate_id": payload.get("candidate_id"),
        },
    )


def empty_memory_state() -> dict[str, Any]:
    return {
        "hypotheses": {},
        "adjustments": {},
        "bets": {},
        "label_definitions": {},
        "rollout_labels": {},
    }


def normalize_memory_state(memory_state: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(memory_state or {})
    return {
        "hypotheses": {
            str(key): dict(value)
            for key, value in _dict(payload.get("hypotheses")).items()
            if isinstance(value, dict)
        },
        "adjustments": {
            str(key): dict(value)
            for key, value in _dict(payload.get("adjustments")).items()
            if isinstance(value, dict)
        },
        "bets": {
            str(key): dict(value)
            for key, value in _dict(payload.get("bets")).items()
            if isinstance(value, dict)
        },
        "label_definitions": {
            str(key): dict(value)
            for key, value in _dict(payload.get("label_definitions")).items()
            if isinstance(value, dict)
        },
        "rollout_labels": {
            str(key): dict(value)
            for key, value in _dict(payload.get("rollout_labels")).items()
            if isinstance(value, dict)
        },
    }


def proposer_memory_summary(memory_state: dict[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_memory_state(memory_state)
    hypotheses = normalized["hypotheses"]
    bets = normalized["bets"]
    active_hypotheses = [
        item
        for item in hypotheses.values()
        if str(item.get("status") or "active") == MiproHypothesisStatus.ACTIVE.value
    ]
    open_bets = [
        item for item in bets.values() if str(item.get("status") or "open") == MiproBetStatus.OPEN.value
    ]
    label_definitions = normalized["label_definitions"]
    rollout_labels = normalized["rollout_labels"]
    active_label_definitions = [
        item
        for item in label_definitions.values()
        if str(item.get("status") or "active")
        == MiproRolloutLabelDefinitionStatus.ACTIVE.value
    ]
    return {
        "hypothesis_count": len(hypotheses),
        "active_hypothesis_count": len(active_hypotheses),
        "adjustment_count": len(normalized["adjustments"]),
        "bet_count": len(bets),
        "open_bet_count": len(open_bets),
        "label_definition_count": len(label_definitions),
        "active_label_definition_count": len(active_label_definitions),
        "rollout_label_count": len(rollout_labels),
        "active_hypotheses": active_hypotheses[:10],
        "open_bets": open_bets[:10],
        "active_label_definitions": active_label_definitions[:10],
    }
