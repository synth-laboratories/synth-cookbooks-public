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


def tool_category(name: str) -> str:
    text = str(name)
    if text in _SEARCH_SPACE_TOOLS:
        return "search_space"
    if text in _CONTROL_TOOLS:
        return "control"
    if text in _EVIDENCE_TOOLS:
        return "evidence"
    return "other"


def tool_mutates_state(name: str) -> bool:
    return str(name) in _SEARCH_SPACE_TOOLS


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
