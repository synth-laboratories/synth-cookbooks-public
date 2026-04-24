"""Shared MIPRO proposer tool execution primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


def tool_category(name: str) -> str:
    text = str(name)
    if text in _SEARCH_SPACE_TOOLS:
        return "search_space"
    if text in _CONTROL_TOOLS:
        return "control"
    if text in _QUEUE_TOOLS:
        return "planning"
    if text in _EVIDENCE_TOOLS:
        return "evidence"
    return "other"


def tool_mutates_state(name: str) -> bool:
    return str(name) in _SEARCH_SPACE_TOOLS or str(name) in {
        "override_rollout_queue",
        "commit_rollout_queue",
    }


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
        return {"status": "ok", "action": name, "queue": queue}

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
