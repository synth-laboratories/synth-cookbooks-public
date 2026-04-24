"""Shared MIPRO proposer tool execution primitives."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from synth_optimizers.miprov2.core.proposer_memory import (
    MiproBet,
    MiproBetResolution,
    MiproBetStatus,
    MiproHypothesis,
    MiproHypothesisAdjustment,
    MiproHypothesisStatus,
    MiproOpenEndednessScoreSource,
    MiproRolloutLabel,
    MiproRolloutLabelAssignmentSource,
    MiproRolloutLabelDefinition,
    MiproRolloutLabelDefinitionStatus,
    MiproRolloutOpenEndednessScore,
    bet_id_for,
    empty_memory_state,
    hypothesis_adjustment_id_for,
    hypothesis_id_for,
    open_endedness_score_id_for,
    normalize_memory_state,
    proposer_memory_summary,
    rollout_label_definition_id_for,
    rollout_label_id_for,
)


@dataclass(slots=True)
class MiproProposerToolState:
    compiled_space: Any
    context: Any
    config: Any
    instruction_patches: list[Any] = field(default_factory=list)
    demo_patches: list[Any] = field(default_factory=list)
    memory_state: dict[str, Any] = field(default_factory=dict)
    queue_state: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class MiproProposerToolResult:
    raw_result: dict[str, Any]
    stop_session: bool = False
    made_progress: bool = False
    state_mutated: bool = False
    mutation_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.raw_result.get("status") or "ok"),
            "action": str(self.raw_result.get("action") or ""),
            "result": dict(self.raw_result),
            "stop_session": bool(self.stop_session),
            "made_progress": bool(self.made_progress),
            "state_mutated": bool(self.state_mutated),
            "mutation_summary": dict(self.mutation_summary),
        }


_EVIDENCE_TOOLS = {
    "get_grounding_summary",
    "list_components",
    "list_component_options",
    "list_registered_instruction_transforms",
    "query_transform_compatibility",
    "list_registered_candidates",
    "list_recent_trial_rows",
    "get_recent_trial_row",
    "list_sampled_train_rows",
    "get_sampled_train_row",
    "query_candidates",
    "query_rollouts",
    "query_evidence_files",
    "read_evidence_file",
    "query_candidate_rollout_deltas",
    "query_candidate_verdict_digest",
    "get_rollout_trace",
}
_SEARCH_SPACE_TOOLS = {
    "add_instruction_candidate",
    "add_instruction_transform",
    "add_static_demo_candidate",
    "add_trajectory_demo_candidate",
}
_CONTROL_TOOLS = {"finish"}
_QUEUE_TOOLS = {
    "preview_tpe_rollout_queue",
    "get_rollout_queue",
    "override_rollout_queue",
    "commit_rollout_queue",
}
_MEMORY_TOOLS = {
    "register_hypothesis",
    "append_hypothesis_adjustment",
    "query_hypotheses",
    "register_bet",
    "resolve_bet",
    "query_bets",
    "register_rollout_label_definition",
    "query_rollout_label_definitions",
    "assign_rollout_label",
    "query_rollouts_by_label",
    "score_rollout_open_endedness",
    "query_open_ended_rollouts",
}


def tool_category(name: str) -> str:
    text = str(name)
    if text in _SEARCH_SPACE_TOOLS:
        return "search_space"
    if text in _CONTROL_TOOLS:
        return "control"
    if text in _QUEUE_TOOLS:
        return "planning"
    if text in _MEMORY_TOOLS:
        return "memory"
    if text in _EVIDENCE_TOOLS:
        return "evidence"
    return "other"


def tool_mutates_state(name: str) -> bool:
    return str(name) in _SEARCH_SPACE_TOOLS or str(name) in {
        "override_rollout_queue",
        "commit_rollout_queue",
        "register_hypothesis",
        "append_hypothesis_adjustment",
        "register_bet",
        "resolve_bet",
        "register_rollout_label_definition",
        "assign_rollout_label",
        "score_rollout_open_endedness",
    }


def _memory_maps(memory_state: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    normalized = normalize_memory_state(memory_state)
    memory_state.clear()
    memory_state.update(normalized)
    return {
        "hypotheses": normalized["hypotheses"],
        "adjustments": normalized["adjustments"],
        "bets": normalized["bets"],
        "label_definitions": normalized["label_definitions"],
        "rollout_labels": normalized["rollout_labels"],
        "open_endedness_scores": normalized["open_endedness_scores"],
    }


def _missing_memory_refs(
    *,
    memory_state: dict[str, Any],
    hypothesis_refs: list[str],
    bet_refs: list[str],
) -> dict[str, list[str]]:
    maps = _memory_maps(memory_state)
    return {
        "hypothesis_refs": [item for item in hypothesis_refs if item not in maps["hypotheses"]],
        "bet_refs": [item for item in bet_refs if item not in maps["bets"]],
    }


def _require_known_memory_refs(
    *,
    memory_state: dict[str, Any],
    hypothesis_refs: list[str],
    bet_refs: list[str],
) -> dict[str, Any] | None:
    missing = _missing_memory_refs(
        memory_state=memory_state,
        hypothesis_refs=hypothesis_refs,
        bet_refs=bet_refs,
    )
    if missing["hypothesis_refs"] or missing["bet_refs"]:
        return {
            "status": "error",
            "reason": "unknown memory refs",
            "missing_refs": missing,
        }
    return None


def _definition_by_id_or_name(
    label_definitions: dict[str, dict[str, Any]],
    *,
    label_id: str,
    name: str,
) -> dict[str, Any] | None:
    if label_id and label_id in label_definitions:
        return dict(label_definitions[label_id])
    if name:
        for definition in label_definitions.values():
            if str(definition.get("name") or "") == name:
                return dict(definition)
    return None


def _rollout_lookup_from_context(state: MiproProposerToolState) -> dict[str, dict[str, Any]]:
    read_model = getattr(state.context, "read_model_payload", {}) or {}
    if not isinstance(read_model, dict):
        return {}
    rollouts = read_model.get("rollouts")
    if not isinstance(rollouts, list):
        return {}
    return {
        str(item.get("rollout_id") or ""): dict(item)
        for item in rollouts
        if isinstance(item, dict) and str(item.get("rollout_id") or "").strip()
    }


def _memory_tool_result(action: Any, state: MiproProposerToolState) -> dict[str, Any] | None:
    name = str(getattr(action, "name", "") or "")
    args = dict(getattr(action, "arguments", {}) or {})
    if name not in _MEMORY_TOOLS:
        return None

    if not state.memory_state:
        state.memory_state.update(empty_memory_state())
    maps = _memory_maps(state.memory_state)
    hypotheses = maps["hypotheses"]
    adjustments = maps["adjustments"]
    bets = maps["bets"]
    label_definitions = maps["label_definitions"]
    rollout_labels = maps["rollout_labels"]
    open_endedness_scores = maps["open_endedness_scores"]

    if name == "register_hypothesis":
        summary = str(args.get("summary") or "").strip()
        rationale = str(args.get("rationale") or "").strip()
        if not summary or not rationale:
            return {
                "status": "error",
                "action": name,
                "reason": "summary and rationale are required",
            }
        try:
            hypothesis = MiproHypothesis.from_dict(
                {
                    **args,
                    "hypothesis_id": str(args.get("hypothesis_id") or hypothesis_id_for(args)),
                    "summary": summary,
                    "rationale": rationale,
                    "status": str(args.get("status") or MiproHypothesisStatus.ACTIVE.value),
                }
            )
        except ValueError as exc:
            return {"status": "error", "action": name, "reason": str(exc)}
        hypotheses[hypothesis.hypothesis_id] = hypothesis.to_dict()
        return {
            "status": "ok",
            "action": name,
            "hypothesis": hypothesis.to_dict(),
            "memory_summary": proposer_memory_summary(state.memory_state),
        }

    if name == "append_hypothesis_adjustment":
        hypothesis_id = str(args.get("hypothesis_id") or "").strip()
        if hypothesis_id not in hypotheses:
            return {
                "status": "error",
                "action": name,
                "reason": f"unknown hypothesis_id: {hypothesis_id}",
            }
        summary = str(args.get("summary") or "").strip()
        reason = str(args.get("reason") or "").strip()
        if not summary or not reason:
            return {
                "status": "error",
                "action": name,
                "reason": "summary and reason are required",
            }
        adjustment = MiproHypothesisAdjustment.from_dict(
            {
                **args,
                "adjustment_id": str(
                    args.get("adjustment_id") or hypothesis_adjustment_id_for(args)
                ),
                "hypothesis_id": hypothesis_id,
                "summary": summary,
                "reason": reason,
            }
        )
        adjustments[adjustment.adjustment_id] = adjustment.to_dict()
        hypothesis = MiproHypothesis.from_dict(dict(hypotheses[hypothesis_id]))
        hypothesis.adjustment_refs = list(
            dict.fromkeys([*hypothesis.adjustment_refs, adjustment.adjustment_id])
        )
        hypothesis.candidate_refs = list(
            dict.fromkeys([*hypothesis.candidate_refs, *adjustment.linked_candidate_refs])
        )
        hypothesis.rollout_refs = list(
            dict.fromkeys([*hypothesis.rollout_refs, *adjustment.linked_rollout_refs])
        )
        hypothesis.task_refs = list(
            dict.fromkeys([*hypothesis.task_refs, *adjustment.linked_task_refs])
        )
        hypothesis.queue_refs = list(
            dict.fromkeys([*hypothesis.queue_refs, *adjustment.linked_queue_refs])
        )
        hypothesis.queue_override_refs = list(
            dict.fromkeys(
                [*hypothesis.queue_override_refs, *adjustment.linked_queue_override_refs]
            )
        )
        hypothesis.updated_at = adjustment.created_at
        hypotheses[hypothesis_id] = hypothesis.to_dict()
        return {
            "status": "ok",
            "action": name,
            "adjustment": adjustment.to_dict(),
            "hypothesis": hypothesis.to_dict(),
            "memory_summary": proposer_memory_summary(state.memory_state),
        }

    if name == "query_hypotheses":
        status_filter = str(args.get("status") or "").strip()
        candidate_id = str(args.get("candidate_id") or "").strip()
        rollout_id = str(args.get("rollout_id") or "").strip()
        task_id = str(args.get("task_id") or "").strip()
        limit = int(args.get("limit") or 20)
        rows = list(hypotheses.values())
        if status_filter:
            rows = [item for item in rows if str(item.get("status") or "") == status_filter]
        if candidate_id:
            rows = [
                item for item in rows if candidate_id in list(item.get("candidate_refs") or [])
            ]
        if rollout_id:
            rows = [item for item in rows if rollout_id in list(item.get("rollout_refs") or [])]
        if task_id:
            rows = [
                item
                for item in rows
                if str(item.get("task_id") or "") == task_id
                or task_id in list(item.get("task_refs") or [])
            ]
        return {
            "status": "ok",
            "action": name,
            "hypotheses": rows[: max(0, limit)],
            "memory_summary": proposer_memory_summary(state.memory_state),
        }

    if name == "register_bet":
        hypothesis_id = str(args.get("hypothesis_id") or "").strip()
        if hypothesis_id and hypothesis_id not in hypotheses:
            return {
                "status": "error",
                "action": name,
                "reason": f"unknown hypothesis_id: {hypothesis_id}",
            }
        summary = str(args.get("summary") or "").strip()
        prediction = str(args.get("prediction") or "").strip()
        if not summary or not prediction:
            return {
                "status": "error",
                "action": name,
                "reason": "summary and prediction are required",
            }
        bet = MiproBet.from_dict(
            {
                **args,
                "bet_id": str(args.get("bet_id") or bet_id_for(args)),
                "hypothesis_id": hypothesis_id or None,
                "summary": summary,
                "prediction": prediction,
                "status": MiproBetStatus.OPEN.value,
            }
        )
        bets[bet.bet_id] = bet.to_dict()
        if hypothesis_id:
            hypothesis = MiproHypothesis.from_dict(dict(hypotheses[hypothesis_id]))
            hypothesis.candidate_refs = list(
                dict.fromkeys([*hypothesis.candidate_refs, *bet.candidate_refs])
            )
            hypothesis.rollout_refs = list(
                dict.fromkeys([*hypothesis.rollout_refs, *bet.rollout_refs])
            )
            hypothesis.task_refs = list(dict.fromkeys([*hypothesis.task_refs, *bet.task_refs]))
            hypothesis.queue_refs = list(
                dict.fromkeys([*hypothesis.queue_refs, *bet.queue_refs])
            )
            hypothesis.queue_override_refs = list(
                dict.fromkeys([*hypothesis.queue_override_refs, *bet.queue_override_refs])
            )
            hypothesis.updated_at = bet.created_at
            hypotheses[hypothesis_id] = hypothesis.to_dict()
        return {
            "status": "ok",
            "action": name,
            "bet": bet.to_dict(),
            "memory_summary": proposer_memory_summary(state.memory_state),
        }

    if name == "resolve_bet":
        bet_id = str(args.get("bet_id") or "").strip()
        if bet_id not in bets:
            return {
                "status": "error",
                "action": name,
                "reason": f"unknown bet_id: {bet_id}",
            }
        resolution = str(args.get("resolution") or "").strip()
        if not resolution:
            return {"status": "error", "action": name, "reason": "resolution is required"}
        bet = MiproBet.from_dict(dict(bets[bet_id]))
        bet.status = MiproBetStatus.RESOLVED
        try:
            bet.resolution = MiproBetResolution(resolution)
        except ValueError as exc:
            return {"status": "error", "action": name, "reason": str(exc)}
        bet.resolved_at = float(args.get("resolved_at") or 0.0) or None
        bet.resolved_at = bet.resolved_at if bet.resolved_at is not None else time.time()
        bet.resolution_comment = (
            str(args["resolution_comment"])
            if args.get("resolution_comment") is not None
            else None
        )
        bet.evidence_refs = [
            str(item) for item in list(args.get("evidence_refs") or []) if str(item).strip()
        ]
        if isinstance(args.get("metadata"), dict):
            bet.metadata = {**bet.metadata, **dict(args["metadata"])}
        bets[bet_id] = bet.to_dict()
        return {
            "status": "ok",
            "action": name,
            "bet": bet.to_dict(),
            "memory_summary": proposer_memory_summary(state.memory_state),
        }

    if name == "query_bets":
        status_filter = str(args.get("status") or "").strip()
        hypothesis_id = str(args.get("hypothesis_id") or "").strip()
        candidate_id = str(args.get("candidate_id") or "").strip()
        rollout_id = str(args.get("rollout_id") or "").strip()
        task_id = str(args.get("task_id") or "").strip()
        limit = int(args.get("limit") or 20)
        rows = list(bets.values())
        if status_filter:
            rows = [item for item in rows if str(item.get("status") or "") == status_filter]
        if hypothesis_id:
            rows = [
                item for item in rows if str(item.get("hypothesis_id") or "") == hypothesis_id
            ]
        if candidate_id:
            rows = [
                item for item in rows if candidate_id in list(item.get("candidate_refs") or [])
            ]
        if rollout_id:
            rows = [item for item in rows if rollout_id in list(item.get("rollout_refs") or [])]
        if task_id:
            rows = [
                item for item in rows if task_id in list(item.get("task_refs") or [])
            ]
        return {
            "status": "ok",
            "action": name,
            "bets": rows[: max(0, limit)],
            "memory_summary": proposer_memory_summary(state.memory_state),
        }

    if name == "register_rollout_label_definition":
        label_name = str(args.get("name") or "").strip()
        description = str(args.get("description") or "").strip()
        allowed_values = [
            str(item) for item in list(args.get("allowed_values") or []) if str(item).strip()
        ]
        if not label_name or not description or not allowed_values:
            return {
                "status": "error",
                "action": name,
                "reason": "name, description, and allowed_values are required",
            }
        try:
            definition = MiproRolloutLabelDefinition.from_dict(
                {
                    **args,
                    "label_id": str(
                        args.get("label_id") or rollout_label_definition_id_for(args)
                    ),
                    "name": label_name,
                    "description": description,
                    "allowed_values": allowed_values,
                    "status": str(
                        args.get("status")
                        or MiproRolloutLabelDefinitionStatus.ACTIVE.value
                    ),
                }
            )
        except ValueError as exc:
            return {"status": "error", "action": name, "reason": str(exc)}
        label_definitions[definition.label_id] = definition.to_dict()
        return {
            "status": "ok",
            "action": name,
            "label_definition": definition.to_dict(),
            "memory_summary": proposer_memory_summary(state.memory_state),
        }

    if name == "query_rollout_label_definitions":
        status_filter = str(args.get("status") or "").strip()
        task_id = str(args.get("task_id") or "").strip()
        label_name = str(args.get("name") or "").strip()
        limit = int(args.get("limit") or 20)
        rows = list(label_definitions.values())
        if status_filter:
            rows = [item for item in rows if str(item.get("status") or "") == status_filter]
        if task_id:
            rows = [item for item in rows if str(item.get("task_id") or "") == task_id]
        if label_name:
            rows = [item for item in rows if str(item.get("name") or "") == label_name]
        return {
            "status": "ok",
            "action": name,
            "label_definitions": rows[: max(0, limit)],
            "memory_summary": proposer_memory_summary(state.memory_state),
        }

    if name == "assign_rollout_label":
        label_id = str(args.get("label_id") or "").strip()
        label_name = str(args.get("name") or "").strip()
        definition = _definition_by_id_or_name(
            label_definitions,
            label_id=label_id,
            name=label_name,
        )
        if definition is None:
            return {
                "status": "error",
                "action": name,
                "reason": "unknown label definition",
            }
        if (
            str(definition.get("status") or "active")
            != MiproRolloutLabelDefinitionStatus.ACTIVE.value
        ):
            return {
                "status": "error",
                "action": name,
                "reason": "label definition is not active",
                "label_definition": definition,
            }
        value = str(args.get("value") or "").strip()
        allowed_values = [str(item) for item in list(definition.get("allowed_values") or [])]
        if value not in allowed_values:
            return {
                "status": "error",
                "action": name,
                "reason": "value is not allowed for label definition",
                "allowed_values": allowed_values,
            }
        rollout_id = str(args.get("rollout_id") or "").strip()
        if not rollout_id:
            return {"status": "error", "action": name, "reason": "rollout_id is required"}
        linked_hypothesis_refs = [
            str(item)
            for item in list(args.get("linked_hypothesis_refs") or [])
            if str(item).strip()
        ]
        linked_bet_refs = [
            str(item) for item in list(args.get("linked_bet_refs") or []) if str(item).strip()
        ]
        ref_error = _require_known_memory_refs(
            memory_state=state.memory_state,
            hypothesis_refs=linked_hypothesis_refs,
            bet_refs=linked_bet_refs,
        )
        if ref_error is not None:
            return {"action": name, **ref_error}
        maps = _memory_maps(state.memory_state)
        rollout_labels = maps["rollout_labels"]
        label_payload = {
            **args,
            "label_id": str(definition["label_id"]),
            "value": value,
            "rollout_id": rollout_id,
            "assignment_source": str(
                args.get("assignment_source")
                or MiproRolloutLabelAssignmentSource.PROPOSER.value
            ),
            "linked_hypothesis_refs": linked_hypothesis_refs,
            "linked_bet_refs": linked_bet_refs,
        }
        label_payload["rollout_label_id"] = str(
            args.get("rollout_label_id") or rollout_label_id_for(label_payload)
        )
        try:
            label = MiproRolloutLabel.from_dict(label_payload)
        except ValueError as exc:
            return {"status": "error", "action": name, "reason": str(exc)}
        rollout_labels[label.rollout_label_id] = label.to_dict()
        return {
            "status": "ok",
            "action": name,
            "rollout_label": label.to_dict(),
            "memory_summary": proposer_memory_summary(state.memory_state),
        }

    if name == "query_rollouts_by_label":
        label_id = str(args.get("label_id") or "").strip()
        label_name = str(args.get("name") or "").strip()
        value = str(args.get("value") or "").strip()
        candidate_id = str(args.get("candidate_id") or "").strip()
        split = str(args.get("split") or "").strip()
        limit = int(args.get("limit") or 20)
        definition = _definition_by_id_or_name(
            label_definitions,
            label_id=label_id,
            name=label_name,
        )
        resolved_label_id = str(definition.get("label_id") or "") if definition else label_id
        rollout_lookup = _rollout_lookup_from_context(state)
        rows = list(rollout_labels.values())
        if resolved_label_id:
            rows = [item for item in rows if str(item.get("label_id") or "") == resolved_label_id]
        if value:
            rows = [item for item in rows if str(item.get("value") or "") == value]
        if candidate_id:
            rows = [item for item in rows if str(item.get("candidate_id") or "") == candidate_id]
        enriched: list[dict[str, Any]] = []
        for label in rows:
            rollout = rollout_lookup.get(str(label.get("rollout_id") or ""))
            if split and rollout is not None and str(rollout.get("split") or "") != split:
                continue
            enriched.append(
                {
                    "label": dict(label),
                    "label_definition": dict(definition or {}),
                    "rollout": dict(rollout or {}),
                }
            )
        return {
            "status": "ok",
            "action": name,
            "matches": enriched[: max(0, limit)],
            "memory_summary": proposer_memory_summary(state.memory_state),
        }

    if name == "score_rollout_open_endedness":
        rollout_id = str(args.get("rollout_id") or "").strip()
        if not rollout_id:
            return {"status": "error", "action": name, "reason": "rollout_id is required"}
        missing_scores = [
            key
            for key in ("novelty_score", "unexpectedness_score", "learnability_score")
            if args.get(key) is None
        ]
        if missing_scores:
            return {
                "status": "error",
                "action": name,
                "reason": "novelty_score, unexpectedness_score, and learnability_score are required",
                "missing": missing_scores,
            }
        linked_hypothesis_refs = [
            str(item)
            for item in list(args.get("linked_hypothesis_refs") or [])
            if str(item).strip()
        ]
        linked_bet_refs = [
            str(item) for item in list(args.get("linked_bet_refs") or []) if str(item).strip()
        ]
        linked_label_refs = [
            str(item)
            for item in list(args.get("linked_label_refs") or [])
            if str(item).strip()
        ]
        ref_error = _require_known_memory_refs(
            memory_state=state.memory_state,
            hypothesis_refs=linked_hypothesis_refs,
            bet_refs=linked_bet_refs,
        )
        if ref_error is not None:
            return {"action": name, **ref_error}
        unknown_label_refs = [item for item in linked_label_refs if item not in rollout_labels]
        if unknown_label_refs:
            return {
                "status": "error",
                "action": name,
                "reason": "unknown label refs",
                "missing_refs": {"linked_label_refs": unknown_label_refs},
            }
        score_payload = {
            **args,
            "rollout_id": rollout_id,
            "score_source": str(
                args.get("score_source") or MiproOpenEndednessScoreSource.PROPOSER.value
            ),
            "linked_hypothesis_refs": linked_hypothesis_refs,
            "linked_bet_refs": linked_bet_refs,
            "linked_label_refs": linked_label_refs,
        }
        score_payload["score_id"] = str(
            args.get("score_id") or open_endedness_score_id_for(score_payload)
        )
        try:
            score = MiproRolloutOpenEndednessScore.from_dict(score_payload)
        except ValueError as exc:
            return {"status": "error", "action": name, "reason": str(exc)}
        maps = _memory_maps(state.memory_state)
        open_endedness_scores = maps["open_endedness_scores"]
        open_endedness_scores[score.score_id] = score.to_dict()
        return {
            "status": "ok",
            "action": name,
            "open_endedness_score": score.to_dict(),
            "memory_summary": proposer_memory_summary(state.memory_state),
        }

    if name == "query_open_ended_rollouts":
        min_score = (
            float(args["min_open_endedness_score"])
            if args.get("min_open_endedness_score") is not None
            else None
        )
        rollout_id = str(args.get("rollout_id") or "").strip()
        candidate_id = str(args.get("candidate_id") or "").strip()
        hypothesis_id = str(args.get("hypothesis_id") or "").strip()
        bet_id = str(args.get("bet_id") or "").strip()
        label_id = str(args.get("label_id") or "").strip()
        label_name = str(args.get("name") or "").strip()
        task_id = str(args.get("task_id") or "").strip()
        limit = int(args.get("limit") or 20)
        definition = _definition_by_id_or_name(
            label_definitions,
            label_id=label_id,
            name=label_name,
        )
        resolved_label_id = str(definition.get("label_id") or "") if definition else label_id
        label_refs_for_definition = {
            str(item.get("rollout_label_id") or "")
            for item in rollout_labels.values()
            if str(item.get("label_id") or "") == resolved_label_id
        }
        rollout_lookup = _rollout_lookup_from_context(state)
        rows = list(open_endedness_scores.values())
        if min_score is not None:
            rows = [
                item
                for item in rows
                if float(item.get("open_endedness_score") or 0.0) >= min_score
            ]
        if rollout_id:
            rows = [item for item in rows if str(item.get("rollout_id") or "") == rollout_id]
        if candidate_id:
            rows = [
                item for item in rows if str(item.get("candidate_id") or "") == candidate_id
            ]
        if task_id:
            rows = [item for item in rows if str(item.get("task_id") or "") == task_id]
        if hypothesis_id:
            rows = [
                item
                for item in rows
                if hypothesis_id in list(item.get("linked_hypothesis_refs") or [])
            ]
        if bet_id:
            rows = [
                item for item in rows if bet_id in list(item.get("linked_bet_refs") or [])
            ]
        if resolved_label_id:
            rows = [
                item
                for item in rows
                if bool(
                    set(str(ref) for ref in list(item.get("linked_label_refs") or []))
                    & label_refs_for_definition
                )
            ]
        rows = sorted(
            rows,
            key=lambda item: float(item.get("open_endedness_score") or 0.0),
            reverse=True,
        )
        return {
            "status": "ok",
            "action": name,
            "matches": [
                {
                    "open_endedness_score": dict(item),
                    "rollout": dict(rollout_lookup.get(str(item.get("rollout_id") or "")) or {}),
                }
                for item in rows[: max(0, limit)]
            ],
            "memory_summary": proposer_memory_summary(state.memory_state),
        }

    return None


def _queue_state_summary(queue_state: dict[str, Any]) -> dict[str, Any]:
    queues = {
        str(key): dict(value)
        for key, value in dict(queue_state.get("queues") or {}).items()
        if isinstance(value, dict)
    }
    committed = queue_state.get("committed_queue_id")
    return {
        "queue_count": len(queues),
        "queue_ids": sorted(queues),
        "tentative_queue_id": queue_state.get("tentative_queue_id"),
        "committed_queue_id": str(committed) if committed is not None else None,
        "override_count": len(list(queue_state.get("overrides") or [])),
    }


def _queue_tool_result(action: Any, state: MiproProposerToolState) -> dict[str, Any] | None:
    name = str(getattr(action, "name", "") or "")
    args = dict(getattr(action, "arguments", {}) or {})
    if name not in _QUEUE_TOOLS:
        return None

    queues = state.queue_state.setdefault("queues", {})
    if not isinstance(queues, dict):
        queues = {}
        state.queue_state["queues"] = queues

    def get_queue(queue_id: str | None) -> dict[str, Any] | None:
        resolved_id = str(queue_id or state.queue_state.get("tentative_queue_id") or "").strip()
        if not resolved_id:
            return None
        value = queues.get(resolved_id)
        return dict(value) if isinstance(value, dict) else None

    if name == "preview_tpe_rollout_queue":
        queue = get_queue(args.get("queue_id"))
        if queue is None:
            return {
                "status": "error",
                "action": name,
                "reason": "no tentative rollout queue is available in proposer state",
                "queue_state": _queue_state_summary(state.queue_state),
            }
        return {
            "status": "ok",
            "action": name,
            "queue": queue,
            "queue_state": _queue_state_summary(state.queue_state),
            "memory_summary": proposer_memory_summary(state.memory_state),
        }

    if name == "get_rollout_queue":
        queue = get_queue(args.get("queue_id"))
        if queue is None:
            return {
                "status": "error",
                "action": name,
                "reason": "queue_id not found",
                "queue_state": _queue_state_summary(state.queue_state),
            }
        return {
            "status": "ok",
            "action": name,
            "queue": queue,
            "memory_summary": proposer_memory_summary(state.memory_state),
        }

    if name == "override_rollout_queue":
        queue = get_queue(args.get("queue_id"))
        if queue is None:
            return {"status": "error", "action": name, "reason": "queue_id not found"}
        queue_id = str(queue.get("queue_id") or "")
        rollouts = [
            dict(item)
            for item in list(queue.get("rollouts") or [])
            if isinstance(item, dict)
        ]
        override_payloads = [
            dict(item)
            for item in list(args.get("overrides") or [])
            if isinstance(item, dict)
        ]
        if not override_payloads:
            return {
                "status": "error",
                "action": name,
                "reason": "overrides must be a non-empty list",
            }
        applied: list[dict[str, Any]] = []
        for idx, override in enumerate(override_payloads):
            linked_hypothesis_refs = [
                str(item)
                for item in list(override.get("linked_hypothesis_refs") or [])
                if str(item).strip()
            ]
            linked_bet_refs = [
                str(item)
                for item in list(override.get("linked_bet_refs") or [])
                if str(item).strip()
            ]
            ref_error = _require_known_memory_refs(
                memory_state=state.memory_state,
                hypothesis_refs=linked_hypothesis_refs,
                bet_refs=linked_bet_refs,
            )
            if ref_error is not None:
                return {"action": name, **ref_error}
            kind = str(override.get("override_kind") or "").strip()
            target = str(override.get("target_rollout_id") or "").strip()
            replacement = override.get("replacement_rollout")
            if kind == "remove_rollout":
                before = len(rollouts)
                rollouts = [item for item in rollouts if str(item.get("rollout_id") or "") != target]
                if len(rollouts) == before:
                    return {
                        "status": "error",
                        "action": name,
                        "reason": f"target_rollout_id not found: {target}",
                    }
            elif kind == "replace_rollout":
                if not isinstance(replacement, dict):
                    return {
                        "status": "error",
                        "action": name,
                        "reason": "replace_rollout requires replacement_rollout",
                    }
                replaced = False
                for rollout_idx, rollout in enumerate(rollouts):
                    if str(rollout.get("rollout_id") or "") == target:
                        rollouts[rollout_idx] = dict(replacement)
                        replaced = True
                        break
                if not replaced:
                    return {
                        "status": "error",
                        "action": name,
                        "reason": f"target_rollout_id not found: {target}",
                    }
            elif kind == "insert_rollout":
                if not isinstance(replacement, dict):
                    return {
                        "status": "error",
                        "action": name,
                        "reason": "insert_rollout requires replacement_rollout",
                    }
                insert_at = int(override.get("target_index") or len(rollouts))
                rollouts.insert(max(0, min(insert_at, len(rollouts))), dict(replacement))
            elif kind == "reorder":
                order = [str(item) for item in list(override.get("rollout_ids") or [])]
                if not order:
                    return {
                        "status": "error",
                        "action": name,
                        "reason": "reorder requires rollout_ids",
                    }
                by_id = {str(item.get("rollout_id") or ""): item for item in rollouts}
                missing = [item for item in order if item not in by_id]
                if missing:
                    return {
                        "status": "error",
                        "action": name,
                        "reason": f"reorder references unknown rollout ids: {missing}",
                    }
                ordered = [by_id[item] for item in order]
                remaining = [
                    item for item in rollouts if str(item.get("rollout_id") or "") not in set(order)
                ]
                rollouts = ordered + remaining
            else:
                return {
                    "status": "error",
                    "action": name,
                    "reason": f"unsupported override_kind: {kind}",
                }
            override_id = str(override.get("override_id") or f"override_{len(state.queue_state.get('overrides') or []) + idx + 1:04d}")
            normalized = {
                **override,
                "override_id": override_id,
                "queue_id": queue_id,
                "linked_hypothesis_refs": linked_hypothesis_refs,
                "linked_bet_refs": linked_bet_refs,
            }
            applied.append(normalized)

        committed_queue_id = str(args.get("committed_queue_id") or f"{queue_id}_overridden")
        edited_queue = {
            **queue,
            "queue_id": committed_queue_id,
            "queue_kind": "edited",
            "rollouts": rollouts,
            "rollout_count": len(rollouts),
            "metadata": {
                **dict(queue.get("metadata") or {}),
                "source_queue_id": queue_id,
                "override_count": len(applied),
            },
        }
        queues[committed_queue_id] = edited_queue
        state.queue_state.setdefault("overrides", [])
        state.queue_state["overrides"] = list(state.queue_state.get("overrides") or []) + applied
        state.queue_state["active_queue_id"] = committed_queue_id
        return {
            "status": "ok",
            "action": name,
            "source_queue_id": queue_id,
            "edited_queue_id": committed_queue_id,
            "applied_override_count": len(applied),
            "queue": edited_queue,
        }

    if name == "commit_rollout_queue":
        queue = get_queue(args.get("queue_id") or state.queue_state.get("active_queue_id"))
        if queue is None:
            return {"status": "error", "action": name, "reason": "queue_id not found"}
        linked_hypothesis_refs = [
            str(item)
            for item in list(args.get("linked_hypothesis_refs") or [])
            if str(item).strip()
        ]
        linked_bet_refs = [
            str(item) for item in list(args.get("linked_bet_refs") or []) if str(item).strip()
        ]
        ref_error = _require_known_memory_refs(
            memory_state=state.memory_state,
            hypothesis_refs=linked_hypothesis_refs,
            bet_refs=linked_bet_refs,
        )
        if ref_error is not None:
            return {"action": name, **ref_error}
        queue_id = str(queue.get("queue_id") or "")
        commit_id = str(args.get("commit_id") or f"commit_{queue_id}")
        commit = {
            "commit_id": commit_id,
            "queue_id": str(state.queue_state.get("tentative_queue_id") or queue_id),
            "committed_queue_id": queue_id,
            "accept_tpe_defaults": bool(args.get("accept_tpe_defaults", queue_id == state.queue_state.get("tentative_queue_id"))),
            "proposer_override_refs": [
                str(item.get("override_id"))
                for item in list(state.queue_state.get("overrides") or [])
                if isinstance(item, dict) and item.get("override_id")
            ],
            "reason": str(args.get("reason") or "").strip() or None,
            "linked_hypothesis_refs": linked_hypothesis_refs,
            "linked_bet_refs": linked_bet_refs,
            "metadata": dict(args.get("metadata") or {}),
        }
        state.queue_state["committed_queue_id"] = queue_id
        state.queue_state["commit"] = commit
        return {
            "status": "ok",
            "action": name,
            "commit": commit,
            "committed_queue": queue,
        }
    return None


def annotate_mipro_tool(tool: dict[str, Any]) -> dict[str, Any]:
    name = str(tool.get("name") or "")
    payload = dict(tool)
    payload.setdefault("output_schema", {"type": "object"})
    payload["category"] = tool_category(name)
    payload["mutates_state"] = tool_mutates_state(name)
    payload.setdefault("available_in_modes", ["autonomous", "interactive"])
    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("interactive_proposer_v1", True)
    payload["metadata"] = metadata
    return payload


def _component_option_counts(compiled_space: Any) -> dict[str, int]:
    search_space = getattr(compiled_space, "search_space", {}) or {}
    return {str(key): len(list(value)) for key, value in search_space.items()}


def _mutation_summary(
    *,
    state: MiproProposerToolState,
    before_instruction_patch_count: int,
    before_demo_patch_count: int,
    before_option_counts: dict[str, int],
) -> dict[str, Any]:
    after_option_counts = _component_option_counts(state.compiled_space)
    added_options = {
        key: int(after_option_counts.get(key, 0)) - int(before_option_counts.get(key, 0))
        for key in sorted(set(before_option_counts) | set(after_option_counts))
        if int(after_option_counts.get(key, 0)) != int(before_option_counts.get(key, 0))
    }
    return {
        "instruction_patches_added": max(
            0, len(state.instruction_patches) - before_instruction_patch_count
        ),
        "demo_patches_added": max(0, len(state.demo_patches) - before_demo_patch_count),
        "component_options_added": added_options,
    }


def execute_mipro_tool(
    *,
    action: Any,
    state: MiproProposerToolState,
) -> MiproProposerToolResult:
    """Execute one existing MIPRO proposer tool against mutable session state."""

    from synth_optimizers.miprov2.core import proposer_openenv as openenv

    before_instruction_patch_count = len(state.instruction_patches)
    before_demo_patch_count = len(state.demo_patches)
    before_option_counts = _component_option_counts(state.compiled_space)
    queue_before = jsonable_queue_state = _queue_state_summary(state.queue_state)
    memory_before = proposer_memory_summary(state.memory_state)
    memory_result = _memory_tool_result(action, state)
    if memory_result is not None:
        memory_after = proposer_memory_summary(state.memory_state)
        state_mutated = memory_before != memory_after
        return MiproProposerToolResult(
            raw_result=dict(memory_result),
            stop_session=False,
            made_progress=state_mutated,
            state_mutated=state_mutated,
            mutation_summary={
                "memory_state_before": memory_before,
                "memory_state_after": memory_after,
            },
        )
    queue_result = _queue_tool_result(action, state)
    if queue_result is not None:
        queue_after = _queue_state_summary(state.queue_state)
        state_mutated = queue_before != queue_after
        return MiproProposerToolResult(
            raw_result=dict(queue_result),
            stop_session=False,
            made_progress=state_mutated,
            state_mutated=state_mutated,
            mutation_summary={
                "queue_state_before": jsonable_queue_state,
                "queue_state_after": queue_after,
            },
        )
    raw_result, stop_session, made_progress = openenv._execute_action(  # noqa: SLF001
        action=action,
        compiled_space=state.compiled_space,
        context=state.context,
        config=state.config,
        instruction_patches=state.instruction_patches,
        demo_patches=state.demo_patches,
    )
    mutation_summary = _mutation_summary(
        state=state,
        before_instruction_patch_count=before_instruction_patch_count,
        before_demo_patch_count=before_demo_patch_count,
        before_option_counts=before_option_counts,
    )
    state_mutated = bool(
        mutation_summary["instruction_patches_added"]
        or mutation_summary["demo_patches_added"]
        or mutation_summary["component_options_added"]
    )
    return MiproProposerToolResult(
        raw_result=dict(raw_result),
        stop_session=bool(stop_session),
        made_progress=bool(made_progress),
        state_mutated=state_mutated,
        mutation_summary=mutation_summary,
    )
