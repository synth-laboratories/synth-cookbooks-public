"""Phase-3 OpenEnv ReAct-style proposer lane for MIPROv2.

This module provides a library-first proposer loop that mirrors the OpenEnv
pattern used in OpenEnv-style proposer loops:

- build a typed runtime tool catalog
- run a bounded turn loop with one action per turn
- execute actions against a local stateful compiler space
- register new discrete options for TPE to explore
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import httpx

from synth_optimizers.miprov2.core.optimizer import DiscreteMiproOptimizer
from synth_optimizers.miprov2.core.instruction_transforms import (
    compile_instruction_text_from_payloads,
)
from synth_optimizers.miprov2.core.program_compiler import (
    CompiledMiproSpace,
    demo_component_key,
    instruction_component_key,
    list_registered_instruction_transforms,
    query_instruction_transform_compatibility,
    register_instruction_candidate,
    register_instruction_transform,
)
from synth_optimizers.miprov2.core.program_model import (
    DemoMessage,
    MiproDemo,
    StaticFewShotDemo,
    TrajectorySnippetDemo,
)

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_GEMINI_CHAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
_GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_MAX_TOOL_RESULT_CHARS = 10_000
_PARSE_FAILURE_REASONS = {
    "empty_model_response",
    "non_json_model_response",
    "non_object_model_response",
    "missing_action",
}
_VARIANT_FIELDS = {
    "variant_id",
    "description",
    "system_prompt",
    "system_prompt_append",
    "orientation_prompt",
    "orientation_prompt_append",
    "enabled_tools",
    "disabled_tools",
    "tool_order",
    "tool_description_overrides",
    "tool_schema_overrides",
    "tool_extra_overrides",
    "proposer_config_overrides",
    "metadata",
}


def _jsonable_dict(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    output = dict(value)
    try:
        json.dumps(output, sort_keys=True, ensure_ascii=True, default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-serializable") from exc
    return output


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise ValueError(f"{field_name} must be a list of strings")
    seen: set[str] = set()
    output: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text:
            raise ValueError(f"{field_name} must not contain empty strings")
        if text in seen:
            continue
        seen.add(text)
        output.append(text)
    return tuple(output)


def _string_mapping(value: Any, *, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    output: dict[str, str] = {}
    for key, item in value.items():
        name = str(key).strip()
        if not name:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        output[name] = str(item)
    return output


def _object_mapping(value: Any, *, field_name: str) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    output: dict[str, dict[str, Any]] = {}
    for key, item in value.items():
        name = str(key).strip()
        if not name:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name}.{name} must be an object")
        output[name] = _jsonable_dict(item, field_name=f"{field_name}.{name}")
    return output


@dataclass(slots=True, frozen=True)
class MiproOpenEnvProposerVariant:
    """Declarative proposer override used for checkpoint replay experiments."""

    variant_id: str = "default"
    description: str = ""
    system_prompt: str | None = None
    system_prompt_append: str | None = None
    orientation_prompt: str | None = None
    orientation_prompt_append: str | None = None
    enabled_tools: tuple[str, ...] = ()
    disabled_tools: tuple[str, ...] = ()
    tool_order: tuple[str, ...] = ()
    tool_description_overrides: dict[str, str] = field(default_factory=dict)
    tool_schema_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_extra_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    proposer_config_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variant_id",
            _require_non_empty(self.variant_id, field_name="MiproOpenEnvProposerVariant.variant_id"),
        )
        object.__setattr__(self, "description", str(self.description))
        for attr in (
            "system_prompt",
            "system_prompt_append",
            "orientation_prompt",
            "orientation_prompt_append",
        ):
            value = getattr(self, attr)
            object.__setattr__(self, attr, str(value) if value is not None else None)
        object.__setattr__(
            self,
            "enabled_tools",
            _string_tuple(self.enabled_tools, field_name="MiproOpenEnvProposerVariant.enabled_tools"),
        )
        object.__setattr__(
            self,
            "disabled_tools",
            _string_tuple(self.disabled_tools, field_name="MiproOpenEnvProposerVariant.disabled_tools"),
        )
        object.__setattr__(
            self,
            "tool_order",
            _string_tuple(self.tool_order, field_name="MiproOpenEnvProposerVariant.tool_order"),
        )
        object.__setattr__(
            self,
            "tool_description_overrides",
            _string_mapping(
                self.tool_description_overrides,
                field_name="MiproOpenEnvProposerVariant.tool_description_overrides",
            ),
        )
        object.__setattr__(
            self,
            "tool_schema_overrides",
            _object_mapping(
                self.tool_schema_overrides,
                field_name="MiproOpenEnvProposerVariant.tool_schema_overrides",
            ),
        )
        object.__setattr__(
            self,
            "tool_extra_overrides",
            _object_mapping(
                self.tool_extra_overrides,
                field_name="MiproOpenEnvProposerVariant.tool_extra_overrides",
            ),
        )
        if not isinstance(self.proposer_config_overrides, Mapping):
            raise ValueError("MiproOpenEnvProposerVariant.proposer_config_overrides must be an object")
        object.__setattr__(
            self,
            "proposer_config_overrides",
            _jsonable_dict(
                self.proposer_config_overrides,
                field_name="MiproOpenEnvProposerVariant.proposer_config_overrides",
            ),
        )
        if not isinstance(self.metadata, Mapping):
            raise ValueError("MiproOpenEnvProposerVariant.metadata must be an object")
        object.__setattr__(
            self,
            "metadata",
            _jsonable_dict(self.metadata, field_name="MiproOpenEnvProposerVariant.metadata"),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "MiproOpenEnvProposerVariant":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("MiproOpenEnvProposerVariant.from_dict requires an object")
        unknown = sorted(set(payload) - _VARIANT_FIELDS)
        if unknown:
            raise ValueError(f"unknown proposer variant field(s): {', '.join(unknown)}")
        return cls(
            variant_id=str(payload.get("variant_id") or "default"),
            description=str(payload.get("description") or ""),
            system_prompt=(
                str(payload["system_prompt"]) if payload.get("system_prompt") is not None else None
            ),
            system_prompt_append=(
                str(payload["system_prompt_append"])
                if payload.get("system_prompt_append") is not None
                else None
            ),
            orientation_prompt=(
                str(payload["orientation_prompt"]) if payload.get("orientation_prompt") is not None else None
            ),
            orientation_prompt_append=(
                str(payload["orientation_prompt_append"])
                if payload.get("orientation_prompt_append") is not None
                else None
            ),
            enabled_tools=_string_tuple(payload.get("enabled_tools"), field_name="enabled_tools"),
            disabled_tools=_string_tuple(payload.get("disabled_tools"), field_name="disabled_tools"),
            tool_order=_string_tuple(payload.get("tool_order"), field_name="tool_order"),
            tool_description_overrides=_string_mapping(
                payload.get("tool_description_overrides"),
                field_name="tool_description_overrides",
            ),
            tool_schema_overrides=_object_mapping(
                payload.get("tool_schema_overrides"),
                field_name="tool_schema_overrides",
            ),
            tool_extra_overrides=_object_mapping(
                payload.get("tool_extra_overrides"),
                field_name="tool_extra_overrides",
            ),
            proposer_config_overrides=dict(payload.get("proposer_config_overrides") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "system_prompt_append": self.system_prompt_append,
            "orientation_prompt": self.orientation_prompt,
            "orientation_prompt_append": self.orientation_prompt_append,
            "enabled_tools": list(self.enabled_tools),
            "disabled_tools": list(self.disabled_tools),
            "tool_order": list(self.tool_order),
            "tool_description_overrides": dict(self.tool_description_overrides),
            "tool_schema_overrides": {
                key: dict(value) for key, value in self.tool_schema_overrides.items()
            },
            "tool_extra_overrides": {
                key: dict(value) for key, value in self.tool_extra_overrides.items()
            },
            "proposer_config_overrides": dict(self.proposer_config_overrides),
            "metadata": dict(self.metadata),
        }

    def stable_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_non_empty(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _truncate(text: str, *, limit: int = 240) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _demo_payload_key(demo: MiproDemo) -> str:
    return json.dumps(demo.to_dict(), sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _coerce_proposer_variant(
    variant: MiproOpenEnvProposerVariant | Mapping[str, Any] | None,
) -> MiproOpenEnvProposerVariant:
    if variant is None:
        return MiproOpenEnvProposerVariant()
    if isinstance(variant, MiproOpenEnvProposerVariant):
        return variant
    return MiproOpenEnvProposerVariant.from_dict(variant)


def _validate_tool_schema(tool_name: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    payload = _jsonable_dict(schema, field_name=f"tool_schema_overrides.{tool_name}")
    if payload.get("type") != "object":
        raise ValueError(f"tool_schema_overrides.{tool_name} must be a JSON schema object")
    properties = payload.get("properties")
    if properties is not None and not isinstance(properties, Mapping):
        raise ValueError(f"tool_schema_overrides.{tool_name}.properties must be an object")
    required = payload.get("required")
    if required is not None and not isinstance(required, list):
        raise ValueError(f"tool_schema_overrides.{tool_name}.required must be a list")
    return payload


def _ensure_known_tool_names(
    *,
    names: set[str],
    requested: tuple[str, ...] | Mapping[str, Any],
    field_name: str,
) -> None:
    values = requested.keys() if isinstance(requested, Mapping) else requested
    unknown = sorted(str(name) for name in values if str(name) not in names)
    if unknown:
        raise ValueError(f"{field_name} references unknown tool(s): {', '.join(unknown)}")


def _apply_tool_catalog_variant(
    runtime_tools: list[dict[str, Any]],
    variant: MiproOpenEnvProposerVariant,
) -> list[dict[str, Any]]:
    if not runtime_tools:
        return []
    tool_names = {str(item.get("name") or "") for item in runtime_tools}
    tool_names.discard("")
    for field_name, requested in (
        ("enabled_tools", variant.enabled_tools),
        ("disabled_tools", variant.disabled_tools),
        ("tool_order", variant.tool_order),
        ("tool_description_overrides", variant.tool_description_overrides),
        ("tool_schema_overrides", variant.tool_schema_overrides),
        ("tool_extra_overrides", variant.tool_extra_overrides),
    ):
        _ensure_known_tool_names(names=tool_names, requested=requested, field_name=field_name)

    output: list[dict[str, Any]] = []
    for tool in runtime_tools:
        name = str(tool.get("name") or "")
        patched = dict(tool)
        if name in variant.tool_description_overrides:
            patched["description"] = variant.tool_description_overrides[name]
        if name in variant.tool_schema_overrides:
            patched["input_schema"] = _validate_tool_schema(
                name, variant.tool_schema_overrides[name]
            )
        if name in variant.tool_extra_overrides:
            extras = dict(variant.tool_extra_overrides[name])
            if "name" in extras:
                raise ValueError("tool_extra_overrides cannot change executable tool names")
            patched.update(extras)
        output.append(patched)

    if variant.enabled_tools:
        enabled = set(variant.enabled_tools)
        output = [tool for tool in output if str(tool.get("name") or "") in enabled]
    if variant.disabled_tools:
        disabled = set(variant.disabled_tools)
        output = [tool for tool in output if str(tool.get("name") or "") not in disabled]
    if variant.tool_order:
        order = {name: idx for idx, name in enumerate(variant.tool_order)}
        indexed_output = list(enumerate(output))
        indexed_output.sort(
            key=lambda pair: (order.get(str(pair[1].get("name") or ""), len(order)), pair[0])
        )
        output = [tool for _, tool in indexed_output]
    return output


def clone_compiled_space(compiled_space: CompiledMiproSpace) -> CompiledMiproSpace:
    """Clone a compiled space so proposer mutations do not mutate caller state."""

    return CompiledMiproSpace(
        program_template=compiled_space.program_template,
        search_space={k: list(v) for k, v in compiled_space.search_space.items()},
        instruction_lookup={k: dict(v) for k, v in compiled_space.instruction_lookup.items()},
        instruction_base_lookup={
            key: dict(value)
            for key, value in compiled_space.instruction_base_lookup.items()
        },
        instruction_transforms={
            key: {transform_id: transform for transform_id, transform in value.items()}
            for key, value in compiled_space.instruction_transforms.items()
        },
        demo_lookup={k: dict(v) for k, v in compiled_space.demo_lookup.items()},
        component_order=tuple(compiled_space.component_order),
        instruction_metadata={
            key: {option_id: dict(payload) for option_id, payload in value.items()}
            for key, value in compiled_space.instruction_metadata.items()
        },
        instruction_base_metadata={
            key: {option_id: dict(payload) for option_id, payload in value.items()}
            for key, value in compiled_space.instruction_base_metadata.items()
        },
        demo_metadata={
            key: {option_id: dict(payload) for option_id, payload in value.items()}
            for key, value in compiled_space.demo_metadata.items()
        },
    )


def build_openenv_tool_catalog(
    compiled_space: CompiledMiproSpace,
    variant: MiproOpenEnvProposerVariant | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an OpenEnv-like runtime tool catalog for proposer sessions."""

    _ = compiled_space
    runtime_tools = [
        {
            "name": "list_components",
            "description": "List available optimizer components and current option counts.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "list_component_options",
            "description": (
                "Read the exact text of all current instruction bundle options for one component, "
                "including base option ids and applied transform ids. CALL THIS FIRST — you must "
                "know the verbatim text and bundle lineage of the current instruction before you "
                "write a transform against it."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "component_key": {"type": "string"},
                },
                "required": ["component_key"],
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "list_registered_instruction_transforms",
            "description": (
                "List the reusable atomic instruction transforms registered for one module. "
                "Use this to understand the transform library that TPE can compose into bundles."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "module_id": {"type": "string"},
                },
                "required": ["module_id"],
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "query_transform_compatibility",
            "description": (
                "Check whether a set of transform ids compiles cleanly against one base instruction "
                "for a module. Use this when you want to reason about composed transform bundles."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "module_id": {"type": "string"},
                    "base_option_id": {"type": "string"},
                    "transform_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["module_id", "base_option_id", "transform_ids"],
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "get_grounding_summary",
            "description": (
                "Get the task group distribution and aggregate performance summary for the current round. "
                "Shows how many training examples are in each task group and recent score trends. "
                "Call this to understand which task groups are largest and how performance is distributed."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "list_recent_trial_rows",
            "description": (
                "List recent trial results: score per trial and, when available, reasoning_trace for "
                "failed examples — the patient model's internal chain-of-thought before each wrong answer. "
                "This is your primary failure-diagnosis tool. Call with limit≥3 to see multiple trials. "
                "Look for PATTERNS across traces: repeated format confusion, wrong dialect, extra commentary, "
                "uncertainty about a specific task group. A rule that directly addresses a recurring pattern "
                "in the traces is almost always the right fix."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "get_recent_trial_row",
            "description": (
                "Read the full details of one recent trial result by index, including all failing traces. "
                "Use after list_recent_trial_rows to inspect a specific trial more deeply."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "row_idx": {"type": "integer"},
                },
                "required": ["row_idx"],
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "list_sampled_train_rows",
            "description": (
                "Sample training examples stratified across task groups. Returns prompt and gold answer for each. "
                "THIS IS YOUR WINDOW INTO THE TASK — call with limit≥12 to see examples from each task group "
                "and understand the full range of input formats, question types, and expected output formats "
                "the instruction must handle. Do not rely on traces alone: traces only show failures from "
                "recent trials; list_sampled_train_rows shows the breadth of the task across all groups."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "query_candidates",
            "description": (
                "Query candidate summaries from the evidence-rich read model, including rollout counts, "
                "average scores, parent linkage, and candidate metadata."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "query_rollouts",
            "description": (
                "Query recorded rollout rows with candidate linkage, split, reward-aligned score components, "
                "full verifier_verdict metadata, and workspace metadata. Use this to inspect why a rollout "
                "earned its score before you patch."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "rollout_id": {"type": "string"},
                    "split": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "query_evidence_files",
            "description": (
                "Query materialized proposer evidence files for rollout summaries, traces, verdicts, "
                "and artifact indexes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "rollout_id": {"type": "string"},
                    "candidate_id": {"type": "string"},
                    "kind": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "read_evidence_file",
            "description": (
                "Read one materialized proposer evidence file. Use after query_evidence_files or a delta/verdict "
                "digest query when you need the full content."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer"},
                    "offset": {"type": "integer"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "query_candidate_rollout_deltas",
            "description": (
                "Query candidate-vs-baseline or candidate-vs-parent delta digests. "
                "This is the fastest way to see which examples improved or regressed."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "compare_to_candidate_id": {"type": "string"},
                    "split": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "query_candidate_verdict_digest",
            "description": (
                "Query verifier/verdict motif digests for a candidate. Use this to understand repeated failure "
                "patterns without inferring them from raw logs. Then drill into query_rollouts or "
                "verifier_verdict.json evidence files for representative wins/regressions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "split": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "get_rollout_trace",
            "description": (
                "Read the normalized stored rollout trace for one rollout, including evidence file paths and "
                "trace previews."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "rollout_id": {"type": "string"},
                    "max_chars": {"type": "integer"},
                    "offset": {"type": "integer"},
                },
                "required": ["rollout_id"],
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "get_sampled_train_row",
            "description": (
                "Read the full details of one training example by index. "
                "Use after list_sampled_train_rows to inspect a specific example in depth."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "row_idx": {"type": "integer"},
                },
                "required": ["row_idx"],
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "add_instruction_candidate",
            "description": (
                "Add a new instruction text candidate to one module. "
                "The instruction_text you provide becomes the COMPLETE system prompt sent verbatim to the "
                "downstream model for every evaluation example — it is not a diff or suffix. "
                "The downstream model receives exactly: system=instruction_text, user=<question>, "
                "and its response is exact-match compared to the gold answer. "
                "A good instruction_text must work correctly across ALL task types and question formats in the dataset. "
                "Use add_instruction_transform instead when you want to make a targeted edit to an existing instruction."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "module_id": {"type": "string"},
                    "instruction_text": {"type": "string"},
                },
                "required": ["module_id", "instruction_text"],
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "add_instruction_transform",
            "description": (
                "Apply line-level edits to an existing instruction option to produce a new candidate. "
                "Prefer this over add_instruction_candidate when you want to add a reusable atomic "
                "transform to the transform library instead of rewriting a full prompt from scratch. "
                "Use 'replace' to swap one specific line for another. "
                "Use 'follow' to insert a new line immediately after a specific anchor line — "
                "prefer 'follow' when adding a new rule or heuristic that does not replace existing content. "
                "Each transform must include an instruction_type classifying what kind of content the new line adds. "
                "The engine will compose compatible transforms into bundle candidates up to the module's K limit, "
                "and each resulting bundle becomes a full system prompt stored verbatim. "
                "Call list_component_options first to see the exact text of available options."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "module_id": {"type": "string"},
                    "base_option_id": {
                        "type": "string",
                        "description": "Option ID of the instruction to transform (e.g. 'baseline' or 'i0'). Use list_component_options to find available IDs and their exact text.",
                    },
                    "transforms": {
                        "type": "array",
                        "description": "Ordered list of line-level edits to apply sequentially.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "description": "'replace' to fix/improve an existing line; 'follow' to insert a new rule or fact after an anchor line.",
                                },
                                "instruction_type": {
                                    "type": "string",
                                    "description": (
                                        "Typology of the new content: "
                                        "'premise' (a fact about the task the model needs to know); "
                                        "'context' (broader background); "
                                        "'task_priority' (what outcome to optimise for); "
                                        "'core_task_description' (what the task is); "
                                        "'heuristics' (strategy or approach for a specific case); "
                                        "'constraints' (soft limit — avoid or minimise violations); "
                                        "'rules' (hard constraint — must or must not); "
                                        "'input_description' (what the input looks like); "
                                        "'output_description' (what the output should look like); "
                                        "'other'."
                                    ),
                                },
                                "prev_line": {
                                    "type": "string",
                                    "description": "Sentence or line to replace (for 'replace'). Matched by suffix then substring if an exact stripped match is not found.",
                                },
                                "line_to_follow": {
                                    "type": "string",
                                    "description": "Anchor sentence or line to insert after (for 'follow'). Matched by suffix then substring.",
                                },
                                "replacement_line": {
                                    "type": "string",
                                    "description": "The new line text to substitute or insert.",
                                },
                            },
                            "required": ["type", "instruction_type", "replacement_line"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["module_id", "base_option_id", "transforms"],
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "add_static_demo_candidate",
            "description": "Add a static few-shot demo candidate to one module slot.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "module_id": {"type": "string"},
                    "slot_id": {"type": "string"},
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                            "additionalProperties": False,
                        },
                    },
                    "demo_label": {"type": "string"},
                },
                "required": ["module_id", "slot_id", "messages"],
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "add_trajectory_demo_candidate",
            "description": "Add a trajectory snippet demo candidate to one module slot.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "module_id": {"type": "string"},
                    "slot_id": {"type": "string"},
                    "rollout_id": {"type": "string"},
                    "start_step": {"type": "integer"},
                    "end_step": {"type": "integer"},
                    "snippet_label": {"type": "string"},
                },
                "required": ["module_id", "slot_id", "rollout_id", "start_step", "end_step"],
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "list_registered_candidates",
            "description": (
                "List all instruction candidates already registered in the search space, "
                "including their option IDs and full text. "
                "Call this when patches are being rejected to see what has already been registered "
                "and avoid proposing duplicate or near-duplicate content. "
                "If everything novel has already been registered, call finish immediately."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "component_key": {"type": "string"},
                },
                "required": ["component_key"],
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
        {
            "name": "finish",
            "description": "Finish proposer session and return control to orchestrator.",
            "input_schema": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "additionalProperties": False,
            },
            "source": "mipro_openenv_action",
        },
    ]
    variant_model = _coerce_proposer_variant(variant)
    runtime_tools = _apply_tool_catalog_variant(runtime_tools, variant_model)
    return {
        "runtime_tools": runtime_tools,
        "tool_count": len(runtime_tools),
    }


def _next_option_id(existing: Mapping[str, Any], *, prefix: str) -> str:
    max_idx = -1
    for key in existing:
        text = str(key)
        if not text.startswith(prefix):
            continue
        try:
            idx = int(text[len(prefix) :])
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return f"{prefix}{max_idx + 1}"


def _instruction_component_lookup(compiled_space: CompiledMiproSpace) -> dict[str, str]:
    return {
        module.module_id: instruction_component_key(module.module_id)
        for module in compiled_space.program_template.modules
    }


def _demo_component_lookup(compiled_space: CompiledMiproSpace) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for module in compiled_space.program_template.modules:
        for slot in module.demo_slots:
            out[(module.module_id, slot.slot_id)] = demo_component_key(module.module_id, slot.slot_id)
    return out


def summarize_compiled_space(compiled_space: CompiledMiproSpace) -> dict[str, Any]:
    """Return a compact, model-safe space summary for ReAct state prompts."""

    instruction_components = _instruction_component_lookup(compiled_space)
    demo_components = _demo_component_lookup(compiled_space)
    components: list[dict[str, Any]] = []
    for module in sorted(compiled_space.program_template.modules, key=lambda item: item.module_id):
        instr_key = instruction_components[module.module_id]
        instr_options = compiled_space.search_space.get(instr_key, [])
        transform_registry = compiled_space.instruction_transforms.get(instr_key, {})
        components.append(
            {
                "component_key": instr_key,
                "kind": "instruction",
                "module_id": module.module_id,
                "slot_id": None,
                "option_count": len(instr_options),
                "base_option_count": len(
                    compiled_space.instruction_base_lookup.get(instr_key, {})
                ),
                "transform_count": len(transform_registry),
                "max_instruction_transforms_per_candidate": int(
                    module.max_instruction_transforms_per_candidate
                ),
                "option_ids": list(instr_options),
            }
        )
        for slot in sorted(module.demo_slots, key=lambda item: item.slot_id):
            demo_key = demo_components[(module.module_id, slot.slot_id)]
            demo_options = compiled_space.search_space.get(demo_key, [])
            components.append(
                {
                    "component_key": demo_key,
                    "kind": "demo",
                    "module_id": module.module_id,
                    "slot_id": slot.slot_id,
                    "option_count": len(demo_options),
                    "option_ids": list(demo_options),
                }
            )
    return {
        "program_id": compiled_space.program_template.program_id,
        "component_order": list(compiled_space.component_order),
        "components": components,
    }


def _list_component_options(
    compiled_space: CompiledMiproSpace,
    *,
    component_key: str,
) -> dict[str, Any]:
    options = list(compiled_space.search_space.get(component_key, []))
    if not options:
        return {"component_key": component_key, "options": [], "status": "empty_or_unknown_component"}

    if component_key in compiled_space.instruction_lookup:
        lookup = compiled_space.instruction_lookup[component_key]
        metadata_map = compiled_space.instruction_metadata.get(component_key, {})
        payload = [
            {
                "option_id": option_id,
                "preview": _truncate(lookup.get(option_id, ""), limit=200),
                "base_instruction_option_id": str(
                    (metadata_map.get(option_id) or {}).get(
                        "base_instruction_option_id"
                    )
                    or option_id
                ),
                "applied_transform_ids": list(
                    (metadata_map.get(option_id) or {}).get("applied_transform_ids")
                    or []
                ),
                "transform_count": int(
                    (metadata_map.get(option_id) or {}).get("transform_count") or 0
                ),
            }
            for option_id in options
        ]
        return {"component_key": component_key, "kind": "instruction", "options": payload, "status": "ok"}

    if component_key in compiled_space.demo_lookup:
        lookup = compiled_space.demo_lookup[component_key]
        payload = [
            {
                "option_id": option_id,
                "preview": _truncate(_demo_payload_key(lookup[option_id]), limit=200)
                if option_id in lookup
                else "",
            }
            for option_id in options
        ]
        return {"component_key": component_key, "kind": "demo", "options": payload, "status": "ok"}

    return {"component_key": component_key, "options": [], "status": "unknown_component"}


def _list_registered_instruction_transforms(
    compiled_space: CompiledMiproSpace,
    *,
    module_id: str,
) -> dict[str, Any]:
    transforms = list_registered_instruction_transforms(
        compiled_space, module_id=module_id
    )
    payload = []
    for transform in transforms:
        payload.append(
            {
                "transform_id": str(transform.get("transform_id") or ""),
                "module_id": str(transform.get("module_id") or ""),
                "base_instruction_anchor_id": str(
                    transform.get("base_instruction_anchor_id") or ""
                )
                or None,
                "localizer_type": str(transform.get("localizer_type") or ""),
                "target_text": str(transform.get("target_text") or ""),
                "replacement_text": str(transform.get("replacement_text") or ""),
                "priority": int(transform.get("priority") or 0),
                "instruction_type": str(
                    (dict(transform.get("metadata") or {})).get("instruction_type") or ""
                )
                or None,
            }
        )
    return {"module_id": module_id, "transforms": payload, "status": "ok"}


@dataclass(slots=True, frozen=True)
class MiproOpenEnvAction:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_non_empty(self.name, field_name="MiproOpenEnvAction.name"))
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(slots=True, frozen=True)
class MiproOpenEnvToolInvocation:
    tool_call_id: str
    action: MiproOpenEnvAction

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tool_call_id",
            _require_non_empty(
                self.tool_call_id,
                field_name="MiproOpenEnvToolInvocation.tool_call_id",
            ),
        )


@dataclass(slots=True, frozen=True)
class MiproOpenEnvTurnResponse:
    assistant_message: dict[str, Any]
    tool_calls: tuple[MiproOpenEnvToolInvocation, ...] = ()
    usage: Mapping[str, Any] = field(default_factory=dict)
    provider_trace: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "assistant_message", dict(self.assistant_message))
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        object.__setattr__(self, "usage", dict(self.usage))
        object.__setattr__(self, "provider_trace", dict(self.provider_trace))


class MiproOpenEnvReactAgent(Protocol):
    async def next_turn(
        self,
        *,
        objective: str,
        runtime_tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        proposer_state: Mapping[str, Any],
        config: MiproOpenEnvProposerConfig | None = None,
    ) -> MiproOpenEnvTurnResponse:
        ...

    async def next_action(
        self,
        *,
        objective: str,
        runtime_tools: list[dict[str, Any]],
        transcript: list[dict[str, Any]],
        proposer_state: Mapping[str, Any],
    ) -> MiproOpenEnvAction:
        ...


@dataclass(slots=True, frozen=True)
class MiproOpenEnvProposerContext:
    objective: str
    round_idx: int = 0
    recent_failures: tuple[str, ...] = ()
    recent_successes: tuple[str, ...] = ()
    grounding_payload: Mapping[str, Any] = field(default_factory=dict)
    run_metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_summary_counts: Mapping[str, Any] = field(default_factory=dict)
    current_best_candidate_id: str | None = None
    baseline_candidate_id: str | None = None
    delta_digest_paths: Mapping[str, Any] = field(default_factory=dict)
    workspace_locations: Mapping[str, Any] = field(default_factory=dict)
    read_model_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective",
            _require_non_empty(self.objective, field_name="MiproOpenEnvProposerContext.objective"),
        )
        object.__setattr__(self, "round_idx", int(self.round_idx))
        object.__setattr__(self, "recent_failures", tuple(str(item) for item in self.recent_failures))
        object.__setattr__(self, "recent_successes", tuple(str(item) for item in self.recent_successes))
        object.__setattr__(self, "grounding_payload", dict(self.grounding_payload))
        object.__setattr__(self, "run_metadata", dict(self.run_metadata))
        object.__setattr__(
            self, "candidate_summary_counts", dict(self.candidate_summary_counts)
        )
        object.__setattr__(
            self,
            "current_best_candidate_id",
            (
                _require_non_empty(
                    self.current_best_candidate_id,
                    field_name="MiproOpenEnvProposerContext.current_best_candidate_id",
                )
                if self.current_best_candidate_id is not None
                else None
            ),
        )
        object.__setattr__(
            self,
            "baseline_candidate_id",
            (
                _require_non_empty(
                    self.baseline_candidate_id,
                    field_name="MiproOpenEnvProposerContext.baseline_candidate_id",
                )
                if self.baseline_candidate_id is not None
                else None
            ),
        )
        object.__setattr__(self, "delta_digest_paths", dict(self.delta_digest_paths))
        object.__setattr__(
            self, "workspace_locations", dict(self.workspace_locations)
        )
        object.__setattr__(self, "read_model_payload", dict(self.read_model_payload))


@dataclass(slots=True, frozen=True)
class MiproOpenEnvProposerConfig:
    max_turns: int = 32
    max_noop_turns: int = 12
    min_read_actions_before_patch: int = 1
    max_patch_actions_per_session: int = 8
    max_consecutive_patch_actions: int = 2
    require_distinct_read_tools_before_patch: bool = False
    max_instruction_patches: int = 8
    max_demo_patches: int = 8
    max_instruction_chars: int = 1800
    max_messages_per_demo: int = 10
    max_message_chars: int = 1800
    context_budget_tokens: int = 200_000
    keep_tail_tokens: int = 50_000
    chat_retry_attempts: int = 4
    chat_retry_base_seconds: float = 2.0
    count_successful_reads_as_progress: bool = True
    archive_root: str | None = None

    def __post_init__(self) -> None:
        if int(self.max_turns) <= 0:
            raise ValueError("MiproOpenEnvProposerConfig.max_turns must be > 0")
        if int(self.max_noop_turns) <= 0:
            raise ValueError("MiproOpenEnvProposerConfig.max_noop_turns must be > 0")
        if int(self.min_read_actions_before_patch) < 0:
            raise ValueError("MiproOpenEnvProposerConfig.min_read_actions_before_patch must be >= 0")
        if int(self.max_patch_actions_per_session) < 0:
            raise ValueError("MiproOpenEnvProposerConfig.max_patch_actions_per_session must be >= 0")
        if int(self.max_consecutive_patch_actions) < 0:
            raise ValueError("MiproOpenEnvProposerConfig.max_consecutive_patch_actions must be >= 0")
        if int(self.max_instruction_patches) < 0:
            raise ValueError("MiproOpenEnvProposerConfig.max_instruction_patches must be >= 0")
        if int(self.max_demo_patches) < 0:
            raise ValueError("MiproOpenEnvProposerConfig.max_demo_patches must be >= 0")
        if int(self.max_instruction_chars) <= 0:
            raise ValueError("MiproOpenEnvProposerConfig.max_instruction_chars must be > 0")
        if int(self.max_messages_per_demo) <= 0:
            raise ValueError("MiproOpenEnvProposerConfig.max_messages_per_demo must be > 0")
        if int(self.max_message_chars) <= 0:
            raise ValueError("MiproOpenEnvProposerConfig.max_message_chars must be > 0")
        if int(self.context_budget_tokens) <= 0:
            raise ValueError("MiproOpenEnvProposerConfig.context_budget_tokens must be > 0")
        if int(self.keep_tail_tokens) <= 0:
            raise ValueError("MiproOpenEnvProposerConfig.keep_tail_tokens must be > 0")
        if int(self.keep_tail_tokens) >= int(self.context_budget_tokens):
            raise ValueError(
                "MiproOpenEnvProposerConfig.keep_tail_tokens must be < context_budget_tokens"
            )
        if int(self.chat_retry_attempts) <= 0:
            raise ValueError("MiproOpenEnvProposerConfig.chat_retry_attempts must be > 0")
        if float(self.chat_retry_base_seconds) < 0.0:
            raise ValueError(
                "MiproOpenEnvProposerConfig.chat_retry_base_seconds must be >= 0"
            )
        if self.archive_root is not None and not str(self.archive_root).strip():
            raise ValueError(
                "MiproOpenEnvProposerConfig.archive_root must be non-empty when provided"
            )


@dataclass(slots=True, frozen=True)
class MiproInstructionPatch:
    module_id: str
    component_key: str
    option_id: str
    instruction_text: str
    base_option_id: str | None = None
    transform_id: str | None = None
    bundle_option_ids: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class MiproDemoPatch:
    module_id: str
    slot_id: str
    component_key: str
    option_id: str
    demo: MiproDemo


@dataclass(slots=True)
class MiproOpenEnvProposerOutcome:
    compiled_space: CompiledMiproSpace
    instruction_patches: list[MiproInstructionPatch] = field(default_factory=list)
    demo_patches: list[MiproDemoPatch] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    action_counts: dict[str, int] = field(default_factory=dict)
    read_action_count: int = 0
    patch_action_count: int = 0
    duplicate_patch_count: int = 0
    ignored_patch_count: int = 0
    policy_violation_count: int = 0
    grounding_read_action_count: int = 0
    evidence_read_action_count: int = 0
    read_tools_used: tuple[str, ...] = ()
    stop_reason: str = "unknown"
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_turn_count: int = 0
    tool_call_count: int = 0
    archive_spill_count: int = 0
    archived_message_count: int = 0
    archive_path: str | None = None


def _cached_prompt_tokens_from_usage(usage: Mapping[str, Any]) -> int:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        return int(details.get("cached_tokens") or 0)
    details = usage.get("input_tokens_details")
    if isinstance(details, Mapping):
        return int(details.get("cached_tokens") or 0)
    return int(usage.get("cached_prompt_tokens") or usage.get("cached_input_tokens") or 0)


def _parse_groq_json_action(text: str) -> MiproOpenEnvAction:
    stripped = str(text or "").strip()
    if not stripped:
        return MiproOpenEnvAction(name="finish", arguments={"reason": "empty_model_response"})

    fence_match = _JSON_FENCE_RE.search(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()
    else:
        obj_match = _JSON_OBJECT_RE.search(stripped)
        if obj_match:
            stripped = obj_match.group(0).strip()

    payload = _decode_first_json_object(stripped)
    if payload is None:
        return MiproOpenEnvAction(name="finish", arguments={"reason": "non_json_model_response"})
    if not isinstance(payload, Mapping):
        return MiproOpenEnvAction(name="finish", arguments={"reason": "non_object_model_response"})

    action_name = str(payload.get("action") or "").strip()
    if not action_name:
        return MiproOpenEnvAction(name="finish", arguments={"reason": "missing_action"})
    args = payload.get("arguments") or {}
    if not isinstance(args, Mapping):
        args = {}
    return MiproOpenEnvAction(name=action_name, arguments=dict(args))


def _decode_first_json_object(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[start_idx:])
        except Exception:
            continue
        if isinstance(obj, Mapping):
            return dict(obj)
    return None


def _coerce_arguments_map(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if text:
            try:
                parsed = json.loads(text)
            except Exception:
                return {}
            if isinstance(parsed, Mapping):
                return dict(parsed)
    return {}


def _coerce_action_from_payload(
    payload: Any,
    *,
    allowed_actions: set[str],
) -> MiproOpenEnvAction | None:
    if not isinstance(payload, Mapping):
        return None

    if "action" in payload:
        action_name = str(payload.get("action") or "").strip()
        if action_name and action_name in allowed_actions:
            return MiproOpenEnvAction(
                name=action_name,
                arguments=_coerce_arguments_map(payload.get("arguments")),
            )

    action_name = str(payload.get("name") or "").strip()
    args = _coerce_arguments_map(payload.get("arguments"))
    if action_name and action_name in allowed_actions:
        return MiproOpenEnvAction(name=action_name, arguments=args)

    nested_name = str(args.get("name") or "").strip()
    nested_args = _coerce_arguments_map(args.get("arguments"))
    if nested_name and nested_name in allowed_actions:
        return MiproOpenEnvAction(name=nested_name, arguments=nested_args)
    return None


def _coerce_action_from_raw_json(
    raw_text: str,
    *,
    allowed_actions: set[str],
) -> MiproOpenEnvAction | None:
    payload = _decode_first_json_object(raw_text)
    if payload is None:
        return None
    return _coerce_action_from_payload(payload, allowed_actions=allowed_actions)


def _tool_call_payload_from_invocation(
    invocation: MiproOpenEnvToolInvocation,
) -> dict[str, Any]:
    return {
        "id": str(invocation.tool_call_id),
        "type": "function",
        "function": {
            "name": str(invocation.action.name),
            "arguments": json.dumps(
                dict(invocation.action.arguments),
                sort_keys=True,
                ensure_ascii=True,
            ),
        },
    }


def _assistant_message_from_invocations(
    *,
    invocations: list[MiproOpenEnvToolInvocation],
    content: str = "",
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": str(content or ""),
        "tool_calls": [
            _tool_call_payload_from_invocation(invocation) for invocation in invocations
        ],
    }


def _tool_invocations_from_openai_message(
    message: Mapping[str, Any],
    *,
    allowed_actions: set[str],
) -> tuple[MiproOpenEnvToolInvocation, ...]:
    invocations: list[MiproOpenEnvToolInvocation] = []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for idx, item in enumerate(tool_calls, start=1):
            if not isinstance(item, Mapping):
                continue
            item_payload = cast(Mapping[str, Any], item)
            raw_id = str(item_payload.get("id") or f"call_{idx}").strip() or f"call_{idx}"
            function_payload = item_payload.get("function")
            action = None
            if isinstance(function_payload, Mapping):
                action = _coerce_action_from_payload(
                    function_payload, allowed_actions=allowed_actions
                )
            if action is None:
                action = _coerce_action_from_payload(
                    item_payload, allowed_actions=allowed_actions
                )
            if action is not None:
                invocations.append(
                    MiproOpenEnvToolInvocation(tool_call_id=raw_id, action=action)
                )
    if invocations:
        return tuple(invocations)
    single_function_call = message.get("function_call")
    if isinstance(single_function_call, Mapping):
        action = _coerce_action_from_payload(
            single_function_call, allowed_actions=allowed_actions
        )
        if action is not None:
            return (
                MiproOpenEnvToolInvocation(
                    tool_call_id="call_function_1",
                    action=action,
                ),
            )
    return ()


def _is_parse_failure_action(action: MiproOpenEnvAction) -> bool:
    if action.name != "finish":
        return False
    reason = str(action.arguments.get("reason") or "").strip()
    return reason in _PARSE_FAILURE_REASONS


def _openai_tools_payload_from_runtime_tools(runtime_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in runtime_tools:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        input_schema = item.get("input_schema")
        schema = dict(input_schema) if isinstance(input_schema, Mapping) else {"type": "object"}
        if str(schema.get("type") or "").strip() == "":
            schema["type"] = "object"
        payload.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(item.get("description") or name),
                    "parameters": schema,
                },
            }
        )
    return payload


def _compact_proposer_transcript(transcript: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in list(transcript)[-max(1, int(limit)) :]:
        if not isinstance(item, Mapping):
            continue
        tool_results = item.get("tool_results")
        result_rows = [
            dict(tool_item)
            for tool_item in list(tool_results or [])
            if isinstance(tool_item, Mapping)
        ]
        first_result = result_rows[0] if result_rows else {}
        compact.append(
            {
                "turn": item.get("turn"),
                "tool_call_count": len(result_rows),
                "action_names": [
                    str((dict(tool_item.get("action") or {})).get("name") or "")
                    for tool_item in result_rows[:4]
                ],
                "result_status": str((dict(first_result.get("result") or {})).get("status") or ""),
                "result_reason": str((dict(first_result.get("result") or {})).get("reason") or ""),
                "result_added": (dict(first_result.get("result") or {})).get("added"),
                "turns_remaining_after_action": first_result.get("turns_remaining_after_action"),
            }
        )
    return compact


def _serialize_tool_result(result: Mapping[str, Any]) -> str:
    text = json.dumps(dict(result), sort_keys=True, ensure_ascii=True)
    if len(text) <= _MAX_TOOL_RESULT_CHARS:
        return text
    truncated = text[:_MAX_TOOL_RESULT_CHARS]
    return (
        truncated
        + f"... [truncated {len(text) - _MAX_TOOL_RESULT_CHARS} chars; "
        + f"limit={_MAX_TOOL_RESULT_CHARS}]"
    )


def _synthesized_assistant_tool_message(
    *,
    action: Mapping[str, Any],
    turn_idx: int,
) -> dict[str, Any]:
    action_name = str(action.get("name") or "").strip()
    arguments = action.get("arguments")
    args_map = dict(arguments) if isinstance(arguments, Mapping) else {}
    invocation = MiproOpenEnvToolInvocation(
        tool_call_id=f"call_turn_{turn_idx}",
        action=MiproOpenEnvAction(name=action_name, arguments=args_map),
    )
    return _assistant_message_from_invocations(invocations=[invocation])


def _conversation_messages(
    *,
    system_prompt: str,
    initial_user_message: str,
    transcript: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_message},
    ]
    for turn_idx, item in enumerate(transcript, start=1):
        if not isinstance(item, Mapping):
            continue
        assistant_payload = item.get("assistant_message")
        assistant_message = (
            dict(assistant_payload) if isinstance(assistant_payload, Mapping) else {}
        )
        tool_results = item.get("tool_results")
        tool_result_rows = [
            dict(tool_item)
            for tool_item in list(tool_results or [])
            if isinstance(tool_item, Mapping)
        ]
        if not assistant_message:
            agent_trace = item.get("agent_trace")
            trace_map = dict(agent_trace) if isinstance(agent_trace, Mapping) else {}
            response = trace_map.get("response")
            response_map = dict(response) if isinstance(response, Mapping) else {}
            raw_message = response_map.get("message")
            assistant_message = dict(raw_message) if isinstance(raw_message, Mapping) else {}
        if not assistant_message and tool_result_rows:
            first_action = dict(tool_result_rows[0].get("action") or {})
            assistant_message = _synthesized_assistant_tool_message(
                action=first_action,
                turn_idx=turn_idx,
            )
        elif not assistant_message:
            action = item.get("action")
            action_map = dict(action) if isinstance(action, Mapping) else {}
            if action_map:
                assistant_message = _synthesized_assistant_tool_message(
                    action=action_map,
                    turn_idx=turn_idx,
                )
        if assistant_message:
            assistant_message["role"] = "assistant"
            messages.append(assistant_message)
        if tool_result_rows:
            for tool_idx, tool_row in enumerate(tool_result_rows, start=1):
                action_map = dict(tool_row.get("action") or {})
                result_map = dict(tool_row.get("result") or {})
                tool_call_id = str(
                    tool_row.get("tool_call_id")
                    or f"call_turn_{turn_idx}_{tool_idx}"
                )
                tool_name = str(action_map.get("name") or "").strip()
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": _serialize_tool_result(result_map),
                }
                if tool_name:
                    tool_message["name"] = tool_name
                messages.append(tool_message)
            continue
        action = item.get("action")
        action_map = dict(action) if isinstance(action, Mapping) else {}
        result = item.get("result")
        result_map = dict(result) if isinstance(result, Mapping) else {}
        if not action_map:
            continue
        tool_call_id = str(
            (((assistant_message.get("tool_calls") or [{}])[0]).get("id") or f"call_turn_{turn_idx}")
            if assistant_message
            else f"call_turn_{turn_idx}"
        )
        tool_name = str(action_map.get("name") or "").strip()
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": _serialize_tool_result(result_map),
        }
        if tool_name:
            tool_message["name"] = tool_name
        messages.append(tool_message)
    return messages


def _orientation_user_message(
    *,
    objective: str,
    proposer_state: Mapping[str, Any],
    transcript: list[dict[str, Any]],
    variant: MiproOpenEnvProposerVariant | Mapping[str, Any] | None = None,
) -> str:
    variant_model = _coerce_proposer_variant(variant)
    components = list(proposer_state.get("components") or [])
    compact_components: list[dict[str, Any]] = []
    for item in components[:16]:
        if not isinstance(item, Mapping):
            continue
        compact_components.append(
            {
                "component_key": str(item.get("component_key") or ""),
                "kind": str(item.get("kind") or ""),
                "module_id": str(item.get("module_id") or ""),
                "slot_id": item.get("slot_id"),
                "option_count": int(item.get("option_count") or 0),
            }
        )
    summary_payload = {
        "objective": str(objective),
        "round_idx": int(proposer_state.get("round_idx") or 0),
        "turns_remaining_before_action": int(proposer_state.get("turns_remaining_before_action") or 0),
        "instruction_patch_count": int(proposer_state.get("instruction_patch_count") or 0),
        "demo_patch_count": int(proposer_state.get("demo_patch_count") or 0),
        "recent_failures": list(proposer_state.get("recent_failures") or []),
        "recent_successes": list(proposer_state.get("recent_successes") or []),
        "grounding_summary": dict(proposer_state.get("grounding_summary") or {}),
        "grounding_counts": dict(proposer_state.get("grounding_counts") or {}),
        "run_metadata": dict(proposer_state.get("run_metadata") or {}),
        "candidate_summary_counts": dict(
            proposer_state.get("candidate_summary_counts") or {}
        ),
        "current_best_candidate_id": proposer_state.get("current_best_candidate_id"),
        "baseline_candidate_id": proposer_state.get("baseline_candidate_id"),
        "delta_digest_paths": dict(proposer_state.get("delta_digest_paths") or {}),
        "workspace_locations": dict(proposer_state.get("workspace_locations") or {}),
        "components": compact_components,
        "recent_transcript_tail": _compact_proposer_transcript(transcript, limit=8),
    }
    message = (
        "MIPRO proposer context (summary-first). "
        "Use tools for detailed evidence before patching.\n"
        + json.dumps(summary_payload, sort_keys=True, ensure_ascii=True)
    )
    if variant_model.orientation_prompt is not None:
        message = variant_model.orientation_prompt
    if variant_model.orientation_prompt_append:
        message = f"{message}\n\n{variant_model.orientation_prompt_append}"
    return message


async def _next_action_via_openai_compatible(
    *,
    provider_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_tokens_field: str,
    timeout_s: float,
    runtime_tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    config: MiproOpenEnvProposerConfig,
) -> MiproOpenEnvTurnResponse:
    allowed_actions = {str(item.get("name") or "").strip() for item in runtime_tools}
    allowed_actions.discard("")
    provider_tools = _openai_tools_payload_from_runtime_tools(runtime_tools)
    if not provider_tools:
        raise RuntimeError("mipro_proposer_empty_runtime_tool_catalog")
    body: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "tools": provider_tools,
        "tool_choice": "required",
        "temperature": temperature,
    }
    body[str(max_tokens_field)] = int(max_tokens)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = await _post_json_with_retries(
        provider_url=provider_url,
        headers=headers,
        body=body,
        timeout_s=timeout_s,
        retry_attempts=int(config.chat_retry_attempts),
        retry_base_seconds=float(config.chat_retry_base_seconds),
    )
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("mipro_proposer_provider_empty_choices")
    message = choices[0].get("message") or {}
    if not isinstance(message, Mapping):
        raise RuntimeError("mipro_proposer_provider_missing_message_payload")
    invocations = _tool_invocations_from_openai_message(
        message, allowed_actions=allowed_actions
    )
    if not invocations:
        raise RuntimeError(
            "mipro_proposer_provider_no_native_tool_call:"
            f"{_truncate(str(message.get('content') or ''), limit=220)}"
        )
    assistant_message = dict(message)
    assistant_message["role"] = "assistant"
    if not isinstance(assistant_message.get("tool_calls"), list):
        assistant_message = _assistant_message_from_invocations(
            invocations=list(invocations),
            content=str(message.get("content") or ""),
        )
    usage = dict(payload.get("usage") or {}) if isinstance(payload.get("usage"), Mapping) else {}
    trace_payload = {
        "provider_url": provider_url,
        "model": model,
        "request": {
            "messages": list(body.get("messages") or []),
            "tools": list(provider_tools),
            "tool_choice": body.get("tool_choice"),
            "temperature": body.get("temperature"),
            "max_tokens_field": str(max_tokens_field),
            "max_tokens": int(max_tokens),
        },
        "response": {
            "finish_reason": choices[0].get("finish_reason"),
            "message": dict(message),
            "usage": usage,
        },
        "selected_tool_calls": [
            {
                "tool_call_id": str(item.tool_call_id),
                "name": str(item.action.name),
                "arguments": dict(item.action.arguments),
            }
            for item in invocations
        ],
    }
    return MiproOpenEnvTurnResponse(
        assistant_message=assistant_message,
        tool_calls=invocations,
        usage=usage,
        provider_trace=trace_payload,
    )


def _estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total_chars = 0
    for message in messages:
        try:
            total_chars += len(json.dumps(message, default=str))
        except (TypeError, ValueError):
            total_chars += len(str(message))
    return total_chars // 4


def _resolve_archive_path(
    *,
    context: MiproOpenEnvProposerContext,
    config: MiproOpenEnvProposerConfig,
) -> Path:
    if config.archive_root is not None:
        base = Path(str(config.archive_root)).expanduser().resolve()
    else:
        ledger_path = str(context.workspace_locations.get("ledger_path") or "").strip()
        if ledger_path:
            base = Path(ledger_path).expanduser().resolve().parent
        else:
            workspace_root = _workspace_root_from_context(context)
            base = (
                workspace_root.parent
                if workspace_root is not None
                else Path.cwd().resolve()
            )
    lane_root = base / "lane_archives" / "proposer"
    lane_root.mkdir(parents=True, exist_ok=True)
    return lane_root / "previous_messages.json"


def _spill_messages_to_archive(
    *,
    archive_path: Path,
    spilled: list[dict[str, Any]],
) -> tuple[int, int]:
    existing: list[Any] = []
    if archive_path.exists():
        try:
            loaded = json.loads(archive_path.read_text())
            if isinstance(loaded, list):
                existing = loaded
        except (OSError, ValueError):
            existing = []
    existing.extend(spilled)
    archive_path.write_text(json.dumps(existing, indent=2, default=str))
    return len(spilled), len(existing)


def _trim_messages_if_needed(
    *,
    messages: list[dict[str, Any]],
    context: MiproOpenEnvProposerContext,
    config: MiproOpenEnvProposerConfig,
    archive_stats: dict[str, Any],
) -> None:
    threshold = int(config.context_budget_tokens)
    keep_tail = int(config.keep_tail_tokens)
    if _estimate_messages_tokens(messages) <= threshold:
        return
    if len(messages) <= 3:
        return
    drop_start = 2
    drop_end = drop_start
    tail_tokens = _estimate_messages_tokens(messages[drop_end:])
    n = len(messages)
    while drop_end < n - 1 and tail_tokens > keep_tail:
        drop_end += 1
        while drop_end < n and str(messages[drop_end].get("role") or "") == "tool":
            drop_end += 1
        tail_tokens = _estimate_messages_tokens(messages[drop_end:])
    if drop_end <= drop_start:
        return
    spilled = messages[drop_start:drop_end]
    archive_path = _resolve_archive_path(context=context, config=config)
    just_spilled, total_on_disk = _spill_messages_to_archive(
        archive_path=archive_path,
        spilled=spilled,
    )
    archive_stats["archive_path"] = str(archive_path)
    archive_stats["spill_count"] = int(archive_stats.get("spill_count") or 0) + 1
    archive_stats["archived_message_count"] = int(total_on_disk)
    note = {
        "role": "user",
        "content": json.dumps(
            {
                "archive_note": (
                    f"{total_on_disk} older proposer tool-call messages are archived on disk at "
                    f"{archive_path} (just spilled {just_spilled} more). "
                    "You have already gathered canonical state above. "
                    "Return your final decision now unless you truly need a detail "
                    "not visible in the remaining in-context messages. "
                    "Consult the archive file only if strictly necessary."
                )
            },
            sort_keys=True,
        ),
    }
    messages[drop_start:drop_end] = [note]


def _tool_followup_user_message(
    *,
    remaining_turns: int,
    failed_tool_results: list[tuple[str, str]],
    successful_patch_count: int,
) -> str | None:
    lines: list[str] = [
        "Canonical state changes must come from add_* tools or finish. Do not rely on free-form narration to mutate state.",
        "Prefer small coherent batches: related evidence reads together, or a targeted write followed by finish once the patch is submitted.",
    ]
    if failed_tool_results:
        lines.append("A recent tool call failed. Fix the arguments before retrying.")
        for name, error in failed_tool_results[-2:]:
            lines.append(f"- {name}: {error}")
    if remaining_turns <= 2:
        if successful_patch_count > 0:
            lines.append(
                "Turns are low. If writes are done, call finish now unless one more required tool call is truly necessary."
            )
        else:
            lines.append(
                "Turns are low. If you have enough evidence, submit one targeted patch and then call finish."
            )
    return "\n".join(lines).strip() or None


def _build_openenv_research_prompt(
    runtime_tools: list[dict[str, Any]],
    variant: MiproOpenEnvProposerVariant | Mapping[str, Any] | None = None,
) -> str:
    variant_model = _coerce_proposer_variant(variant)
    tool_names = [str(item.get("name") or "") for item in runtime_tools]
    prompt = (
        "You are an OpenEnv ReAct proposer for MIPRO search space expansion. "
        "Use native tool calling to choose actions. You may make multiple tool calls in one response when they form a coherent evidence-gathering batch or a write-plus-finish batch. "
        "Do not emit JSON wrappers. Do not narrate unless no action is possible. "
        f"Allowed actions: {tool_names}.\n"
        "HOW YOUR PROPOSALS ARE APPLIED:\n"
        "Each evaluated instruction bundle becomes the complete system prompt sent verbatim to the downstream model. "
        "You are building a reusable library of atomic transforms, and the engine composes compatible transforms into bundles up to the module's K limit.\n"
        "INSTRUCTION TYPOLOGY — diagnose which category is missing or wrong before proposing:\n"
        "  premise: a fact about the task the model needs to know\n"
        "  context: broader background\n"
        "  task_priority: what outcome to optimise for\n"
        "  core_task_description: what the task is\n"
        "  heuristics: strategy for a specific case\n"
        "  constraints: soft limit to minimise violations of\n"
        "  rules: hard constraint (must / must not)\n"
        "  input_description: what the input looks like\n"
        "  output_description: what the output should look like\n"
        "PREFER add_instruction_transform over add_instruction_candidate. "
        "Use follow to append a new rule/heuristic/premise. Use replace to fix an existing line that is wrong. "
        "Each transform must include instruction_type.\n"
        "EVIDENCE-FIRST LOOP:\n"
        "  1. list_component_options — read the exact current instruction bundle text, base option, and applied transform ids.\n"
        "  2. list_registered_instruction_transforms — inspect the reusable transform library before adding more edits.\n"
        "  3. query_candidates — identify the current best, baseline, and parent-linked candidates.\n"
        "  4. query_candidate_rollout_deltas and query_candidate_verdict_digest — read explicit improvement, regression, and verifier-motif summaries.\n"
        "  5. query_rollouts — inspect concrete rollout rows, especially score_components and verifier_verdict metadata, for one candidate and a comparison candidate.\n"
        "  6. query_evidence_files(kind='verifier_verdict') plus read_evidence_file — open full verifier_verdict.json payloads for representative wins and regressions.\n"
        "  7. get_rollout_trace — pair the verifier metadata with the actual prompt/response/ideal trace before you decide what to change.\n"
        "  8. list_sampled_train_rows(limit>=12) and list_recent_trial_rows(limit>=3) — use these compatibility views to understand task breadth and recent failures.\n"
        "You may perform several adjacent reads in one response when they belong to one diagnosis step.\n"
        "When verifier metadata exists, review at least one improved rollout and one regressed rollout before patching. "
        "Look at the reward, the score_components, and the full verifier_verdict payload together.\n"
        "DIAGNOSE: format errors usually want rules/constraints; knowledge errors want premises; task-group-specific mistakes want heuristics.\n"
        "PROPOSE: submit 1-2 targeted transforms grounded in the evidence. One rule, heuristic, or premise per transform is better than a broad rewrite. "
        "Think in terms of transform reuse and compatibility, not one-off prompt rewrites.\n"
        "GOOD TRANSFORMS:\n"
        "  - enforce the exact output format\n"
        "  - forbid a recurring unwanted output pattern\n"
        "  - add one missing domain premise\n"
        "  - add one task-group-specific heuristic\n"
        "BAD TRANSFORMS:\n"
        "  - rephrasing the baseline without new information\n"
        "  - generic advice like be careful or answer accurately\n"
        "  - task-specific rules that obviously break other task groups\n"
        "FINISH: call finish only after at least one successful add_* patch. If 3 consecutive patches are rejected, call list_registered_candidates and then finish immediately."
    )
    if variant_model.system_prompt is not None:
        prompt = variant_model.system_prompt
    if variant_model.system_prompt_append:
        prompt = f"{prompt}\n\n{variant_model.system_prompt_append}"
    return prompt


def _initial_live_messages(
    *,
    objective: str,
    runtime_tools: list[dict[str, Any]],
    proposer_state: Mapping[str, Any],
    transcript: list[dict[str, Any]],
    variant: MiproOpenEnvProposerVariant | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return _conversation_messages(
        system_prompt=_build_openenv_research_prompt(runtime_tools, variant=variant),
        initial_user_message=_orientation_user_message(
            objective=objective,
            proposer_state=proposer_state,
            transcript=transcript,
            variant=variant,
        ),
        transcript=[],
    )


async def _post_json_with_retries(
    *,
    provider_url: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    timeout_s: float,
    retry_attempts: int,
    retry_base_seconds: float,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    timeout = httpx.Timeout(float(timeout_s), connect=min(15.0, float(timeout_s)))
    last_exc: Exception | None = None
    response: httpx.Response | None = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max(1, int(retry_attempts))):
            try:
                response = await client.post(
                    provider_url,
                    headers=dict(headers),
                    json=dict(body),
                    params=dict(params or {}),
                )
                if response.status_code < 500 and response.status_code != 429:
                    break
                last_exc = httpx.HTTPStatusError(
                    f"retryable status {response.status_code}",
                    request=response.request,
                    response=response,
                )
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                response = None
            if attempt == max(1, int(retry_attempts)) - 1:
                break
            await asyncio.sleep(float(retry_base_seconds) * (2**attempt))
    if response is None:
        raise RuntimeError(
            f"mipro_proposer_provider_transport_error:{_truncate(str(last_exc or ''), limit=400)}"
        ) from last_exc
    if int(response.status_code) >= 400:
        raise RuntimeError(
            f"mipro_proposer_provider_http_{int(response.status_code)}:"
            f"{_truncate(str(response.text or ''), limit=400)}"
        )
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise RuntimeError("mipro_proposer_provider_non_object_response")
    return dict(payload)


def _openai_messages_to_gemini_contents(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert OpenAI-format messages to Gemini contents + systemInstruction text."""
    system_text = ""
    contents: list[dict[str, Any]] = []
    pending_fn_calls: dict[str, str] = {}  # tool_call_id -> function name

    for msg in messages:
        role = str(msg.get("role") or "")
        content = msg.get("content")

        if role == "system":
            system_text = str(content or "")
            continue

        if role == "user":
            contents.append({"role": "user", "parts": [{"text": str(content or "")}]})
            continue

        if role == "assistant":
            parts: list[dict[str, Any]] = []
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                fn_name = str(fn.get("name") or "")
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except Exception:
                    args = {}
                parts.append({"functionCall": {"name": fn_name, "args": args}})
                pending_fn_calls[str(tc.get("id") or fn_name)] = fn_name
            if not parts:
                parts = [{"text": str(content or "")}]
            contents.append({"role": "model", "parts": parts})
            continue

        if role == "tool":
            tool_call_id = str(msg.get("tool_call_id") or "")
            fn_name = str(msg.get("name") or pending_fn_calls.get(tool_call_id) or "")
            raw = msg.get("content") or ""
            try:
                result_obj = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                result_obj = {"text": str(raw)}
            contents.append({
                "role": "user",
                "parts": [{"functionResponse": {"name": fn_name, "response": result_obj}}],
            })
            pending_fn_calls.pop(tool_call_id, None)
            continue

    return system_text, contents


_GEMINI_SCHEMA_UNSUPPORTED = frozenset({
    "additionalProperties", "$schema", "$defs", "definitions",
    "if", "then", "else", "allOf", "anyOf", "oneOf", "not",
    "contentEncoding", "contentMediaType", "default", "examples",
})


def _clean_schema_for_gemini(schema: Any) -> Any:
    if isinstance(schema, Mapping):
        return {k: _clean_schema_for_gemini(v) for k, v in schema.items() if k not in _GEMINI_SCHEMA_UNSUPPORTED}
    if isinstance(schema, list):
        return [_clean_schema_for_gemini(item) for item in schema]
    return schema


def _runtime_tools_to_gemini_declarations(runtime_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    for item in runtime_tools:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        input_schema = item.get("input_schema")
        raw_schema = dict(input_schema) if isinstance(input_schema, Mapping) else {"type": "object"}
        if not raw_schema.get("type"):
            raw_schema["type"] = "object"
        schema = _clean_schema_for_gemini(raw_schema)
        declarations.append({
            "name": name,
            "description": str(item.get("description") or name),
            "parameters": schema,
        })
    return declarations


async def _next_action_via_gemini_native(
    *,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    runtime_tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    config: MiproOpenEnvProposerConfig,
) -> MiproOpenEnvTurnResponse:
    allowed_actions = {str(item.get("name") or "").strip() for item in runtime_tools}
    allowed_actions.discard("")
    declarations = _runtime_tools_to_gemini_declarations(runtime_tools)
    if not declarations:
        raise RuntimeError("mipro_proposer_empty_runtime_tool_catalog")
    _system_text, contents = _openai_messages_to_gemini_contents(messages)

    url = _GEMINI_GENERATE_URL.format(model=model)
    body: dict[str, Any] = {
        "contents": contents,
        "tools": [{"function_declarations": declarations}],
        "tool_config": {"function_calling_config": {"mode": "ANY"}},
        "system_instruction": {"parts": [{"text": _system_text}]},
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
        },
    }
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    payload = await _post_json_with_retries(
        provider_url=url,
        headers=headers,
        body=body,
        timeout_s=timeout_s,
        retry_attempts=int(config.chat_retry_attempts),
        retry_base_seconds=float(config.chat_retry_base_seconds),
        params=params,
    )
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("mipro_proposer_provider_empty_choices")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    invocations: list[MiproOpenEnvToolInvocation] = []
    text_parts: list[str] = []
    for idx, part in enumerate(parts, start=1):
        if "text" in part:
            text_parts.append(str(part.get("text") or ""))
        fn_call = part.get("functionCall")
        if isinstance(fn_call, Mapping):
            name = str(fn_call.get("name") or "").strip()
            args = dict(fn_call.get("args") or {})
            if name in allowed_actions:
                coerced = _coerce_action_from_payload({"name": name, "arguments": args}, allowed_actions=allowed_actions)
                if coerced is not None:
                    invocations.append(
                        MiproOpenEnvToolInvocation(
                            tool_call_id=f"gemini_call_{idx}",
                            action=coerced,
                        )
                    )
    text_content = " ".join(text_parts)
    if not invocations:
        finish_reason = str((candidates[0].get("finishReason") or "")).strip()
        debug_summary = json.dumps({
            "finishReason": finish_reason,
            "parts_keys": [list(p.keys()) for p in parts],
            "candidate_keys": list(candidates[0].keys()),
        })
        raise RuntimeError(
            f"mipro_proposer_provider_no_native_tool_call:{_truncate(text_content, limit=220)} [{debug_summary}]"
        )
    raw_usage = payload.get("usageMetadata")
    usage = (
        {
            "prompt_tokens": int(raw_usage.get("promptTokenCount") or 0),
            "completion_tokens": int(raw_usage.get("candidatesTokenCount") or 0),
            "total_tokens": int(raw_usage.get("totalTokenCount") or 0),
        }
        if isinstance(raw_usage, Mapping)
        else {}
    )
    trace_payload = {
        "provider_url": url,
        "model": model,
        "request": {
            "contents": contents,
            "tool_count": len(declarations),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        },
        "response": {
            "candidate_count": len(candidates),
            "parts": parts,
            "usage": usage,
        },
        "selected_tool_calls": [
            {
                "tool_call_id": str(item.tool_call_id),
                "name": str(item.action.name),
                "arguments": dict(item.action.arguments),
            }
            for item in invocations
        ],
    }
    return MiproOpenEnvTurnResponse(
        assistant_message=_assistant_message_from_invocations(
            invocations=invocations,
            content=text_content,
        ),
        tool_calls=tuple(invocations),
        usage=usage,
        provider_trace=trace_payload,
    )


@dataclass(slots=True)
class GeminiOpenEnvReactAgent:
    """Gemini-backed ReAct proposer using the native generateContent API with mode=ANY."""

    api_key: str
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_tokens: int = 800
    timeout_s: float = 120.0
    last_turn_debug: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.api_key = _require_non_empty(self.api_key, field_name="GeminiOpenEnvReactAgent.api_key")
        self.model = _require_non_empty(self.model, field_name="GeminiOpenEnvReactAgent.model")
        self.temperature = float(self.temperature)
        self.max_tokens = max(64, int(self.max_tokens))
        self.timeout_s = max(5.0, float(self.timeout_s))

    async def next_turn(
        self,
        *,
        objective: str,
        runtime_tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        proposer_state: Mapping[str, Any],
        config: MiproOpenEnvProposerConfig | None = None,
    ) -> MiproOpenEnvTurnResponse:
        _ = objective, proposer_state
        cfg = config or MiproOpenEnvProposerConfig()
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                turn = await _next_action_via_gemini_native(
                    api_key=self.api_key,
                    model=self.model,
                    temperature=min(self.temperature + 0.2 * attempt, 1.0),
                    max_tokens=self.max_tokens,
                    timeout_s=self.timeout_s,
                    runtime_tools=runtime_tools,
                    messages=messages,
                    config=cfg,
                )
                self.last_turn_debug = dict(turn.provider_trace)
                return turn
            except RuntimeError as exc:
                err = str(exc)
                if "no_native_tool_call" not in err and "provider_http_5" not in err:
                    raise
                last_exc = exc
        raise RuntimeError(f"GeminiOpenEnvReactAgent failed after 3 attempts: {last_exc}") from last_exc

    async def next_action(
        self,
        *,
        objective: str,
        runtime_tools: list[dict[str, Any]],
        transcript: list[dict[str, Any]],
        proposer_state: Mapping[str, Any],
    ) -> MiproOpenEnvAction:
        turn = await self.next_turn(
            objective=objective,
            runtime_tools=runtime_tools,
            messages=_conversation_messages(
                system_prompt=_build_openenv_research_prompt(runtime_tools),
                initial_user_message=_orientation_user_message(
                    objective=objective,
                    proposer_state=proposer_state,
                    transcript=transcript,
                ),
                transcript=transcript,
            ),
            proposer_state=proposer_state,
        )
        if turn.tool_calls:
            return turn.tool_calls[0].action
        return MiproOpenEnvAction(name="finish", arguments={"reason": "no_tool_calls"})


@dataclass(slots=True)
class OpenAIOpenEnvReactAgent:
    """OpenAI-backed ReAct proposer using native tool calls."""

    api_key: str
    model: str = "gpt-5.4-nano"
    temperature: float = 0.0
    max_tokens: int = 800
    timeout_s: float = 120.0
    last_turn_debug: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.api_key = _require_non_empty(self.api_key, field_name="OpenAIOpenEnvReactAgent.api_key")
        self.model = _require_non_empty(self.model, field_name="OpenAIOpenEnvReactAgent.model")
        self.temperature = float(self.temperature)
        self.max_tokens = max(64, int(self.max_tokens))
        self.timeout_s = max(5.0, float(self.timeout_s))

    async def next_turn(
        self,
        *,
        objective: str,
        runtime_tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        proposer_state: Mapping[str, Any],
        config: MiproOpenEnvProposerConfig | None = None,
    ) -> MiproOpenEnvTurnResponse:
        _ = objective, proposer_state
        turn = await _next_action_via_openai_compatible(
            provider_url=_OPENAI_CHAT_URL,
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_tokens_field="max_completion_tokens",
            timeout_s=self.timeout_s,
            runtime_tools=runtime_tools,
            messages=messages,
            config=config or MiproOpenEnvProposerConfig(),
        )
        self.last_turn_debug = dict(turn.provider_trace)
        return turn

    async def next_action(
        self,
        *,
        objective: str,
        runtime_tools: list[dict[str, Any]],
        transcript: list[dict[str, Any]],
        proposer_state: Mapping[str, Any],
    ) -> MiproOpenEnvAction:
        turn = await self.next_turn(
            objective=objective,
            runtime_tools=runtime_tools,
            messages=_conversation_messages(
                system_prompt=_build_openenv_research_prompt(runtime_tools),
                initial_user_message=_orientation_user_message(
                    objective=objective,
                    proposer_state=proposer_state,
                    transcript=transcript,
                ),
                transcript=transcript,
            ),
            proposer_state=proposer_state,
        )
        if turn.tool_calls:
            return turn.tool_calls[0].action
        return MiproOpenEnvAction(name="finish", arguments={"reason": "no_tool_calls"})


@dataclass(slots=True)
class GroqOpenEnvReactAgent:
    """Groq-backed ReAct agent using the OpenAI-compatible chat API."""

    api_key: str
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.0
    max_tokens: int = 800
    timeout_s: float = 120.0
    last_turn_debug: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.api_key = _require_non_empty(self.api_key, field_name="GroqOpenEnvReactAgent.api_key")
        self.model = _require_non_empty(self.model, field_name="GroqOpenEnvReactAgent.model")
        self.temperature = float(self.temperature)
        self.max_tokens = max(64, int(self.max_tokens))
        self.timeout_s = max(5.0, float(self.timeout_s))

    async def next_turn(
        self,
        *,
        objective: str,
        runtime_tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        proposer_state: Mapping[str, Any],
        config: MiproOpenEnvProposerConfig | None = None,
    ) -> MiproOpenEnvTurnResponse:
        _ = objective, proposer_state
        cfg = config or MiproOpenEnvProposerConfig()
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                turn = await _next_action_via_openai_compatible(
                    provider_url=_GROQ_CHAT_URL,
                    api_key=self.api_key,
                    model=self.model,
                    temperature=min(self.temperature + 0.3 * attempt, 1.0),
                    max_tokens=self.max_tokens,
                    max_tokens_field="max_tokens",
                    timeout_s=self.timeout_s,
                    runtime_tools=runtime_tools,
                    messages=messages,
                    config=cfg,
                )
                self.last_turn_debug = dict(turn.provider_trace)
                return turn
            except RuntimeError as exc:
                err = str(exc)
                if "no_native_tool_call" not in err and "tool_use_failed" not in err:
                    raise
                last_exc = exc
        raise RuntimeError(f"GroqOpenEnvReactAgent failed after 3 attempts: {last_exc}") from last_exc

    async def next_action(
        self,
        *,
        objective: str,
        runtime_tools: list[dict[str, Any]],
        transcript: list[dict[str, Any]],
        proposer_state: Mapping[str, Any],
    ) -> MiproOpenEnvAction:
        turn = await self.next_turn(
            objective=objective,
            runtime_tools=runtime_tools,
            messages=_conversation_messages(
                system_prompt=_build_openenv_research_prompt(runtime_tools),
                initial_user_message=_orientation_user_message(
                    objective=objective,
                    proposer_state=proposer_state,
                    transcript=transcript,
                ),
                transcript=transcript,
            ),
            proposer_state=proposer_state,
        )
        if turn.tool_calls:
            return turn.tool_calls[0].action
        return MiproOpenEnvAction(name="finish", arguments={"reason": "no_tool_calls"})


@dataclass(slots=True)
class HeuristicOpenEnvReactAgent:
    """Deterministic fallback proposer that adds one strict-format variant per module."""

    _pending: list[MiproOpenEnvAction] = field(default_factory=list)
    _initialized: bool = False
    last_turn_debug: dict[str, Any] | None = field(default=None, init=False, repr=False)

    async def next_turn(
        self,
        *,
        objective: str,
        runtime_tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        proposer_state: Mapping[str, Any],
        config: MiproOpenEnvProposerConfig | None = None,
    ) -> MiproOpenEnvTurnResponse:
        _ = runtime_tools, messages, config
        if not self._initialized:
            self._initialized = True
            self._pending.append(MiproOpenEnvAction(name="get_grounding_summary", arguments={}))
            self._pending.append(MiproOpenEnvAction(name="list_components", arguments={}))
            for component in proposer_state.get("components", []):
                if not isinstance(component, Mapping):
                    continue
                if str(component.get("kind") or "") != "instruction":
                    continue
                module_id = str(component.get("module_id") or "").strip()
                if not module_id:
                    continue
                instruction = (
                    "Follow task constraints exactly. "
                    "Keep reasoning concise, and output only the required final form. "
                    f"Optimization objective: {objective}"
                )
                self._pending.append(
                    MiproOpenEnvAction(
                        name="add_instruction_candidate",
                        arguments={"module_id": module_id, "instruction_text": instruction},
                    )
                )
            self._pending.append(MiproOpenEnvAction(name="finish", arguments={"reason": "heuristic_done"}))
        if self._pending:
            action = self._pending.pop(0)
        else:
            action = MiproOpenEnvAction(
                name="finish", arguments={"reason": "no_pending_actions"}
            )
        invocation = MiproOpenEnvToolInvocation(
            tool_call_id=f"heuristic_call_{len(self._pending) + 1}",
            action=action,
        )
        trace_payload = {
            "provider_url": "heuristic://local",
            "model": "heuristic",
            "request": {
                "messages": [],
                "tools": [dict(item) for item in runtime_tools],
                "objective": str(objective),
            },
            "response": {},
            "selected_tool_calls": [
                {
                    "tool_call_id": str(invocation.tool_call_id),
                    "name": str(action.name),
                    "arguments": dict(action.arguments),
                }
            ],
        }
        self.last_turn_debug = trace_payload
        return MiproOpenEnvTurnResponse(
            assistant_message=_assistant_message_from_invocations(
                invocations=[invocation]
            ),
            tool_calls=(invocation,),
            usage={},
            provider_trace=trace_payload,
        )

    async def next_action(
        self,
        *,
        objective: str,
        runtime_tools: list[dict[str, Any]],
        transcript: list[dict[str, Any]],
        proposer_state: Mapping[str, Any],
    ) -> MiproOpenEnvAction:
        turn = await self.next_turn(
            objective=objective,
            runtime_tools=runtime_tools,
            messages=[],
            proposer_state=proposer_state,
        )
        if turn.tool_calls:
            return turn.tool_calls[0].action
        return MiproOpenEnvAction(name="finish", arguments={"reason": "no_tool_calls"})


def _coerce_messages(raw: Any, *, config: MiproOpenEnvProposerConfig) -> tuple[DemoMessage, ...]:
    if not isinstance(raw, list):
        raise ValueError("messages must be a list")
    if not raw:
        raise ValueError("messages must contain at least one item")
    if len(raw) > int(config.max_messages_per_demo):
        raise ValueError("messages exceeds max_messages_per_demo")
    out: list[DemoMessage] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("messages entries must be objects")
        role = _require_non_empty(str(item.get("role") or ""), field_name="messages.role")
        content = _require_non_empty(str(item.get("content") or ""), field_name="messages.content")
        if len(content) > int(config.max_message_chars):
            raise ValueError("message content exceeds max_message_chars")
        out.append(DemoMessage(role=role, content=content))
    return tuple(out)


def _add_instruction_candidate(
    *,
    compiled_space: CompiledMiproSpace,
    module_id: str,
    instruction_text: str,
    config: MiproOpenEnvProposerConfig,
    parent_candidate_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[bool, str, str]:
    module_key = _require_non_empty(module_id, field_name="module_id")
    text = _require_non_empty(instruction_text, field_name="instruction_text")
    if len(text) > int(config.max_instruction_chars):
        raise ValueError("instruction_text exceeds max_instruction_chars")
    return register_instruction_candidate(
        compiled_space=compiled_space,
        module_id=module_key,
        instruction_text=text,
        parent_candidate_ids=tuple(
            [str(parent_candidate_id).strip()]
            if str(parent_candidate_id or "").strip()
            else []
        ),
        metadata=metadata,
    )


def apply_instruction_transforms(base_text: str, transforms: list[dict[str, Any]]) -> str:
    return compile_instruction_text_from_payloads(base_text, transforms)


def _add_instruction_transform(
    *,
    compiled_space: CompiledMiproSpace,
    module_id: str,
    base_option_id: str,
    transforms: list[dict[str, Any]],
    config: MiproOpenEnvProposerConfig,
    parent_candidate_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return register_instruction_transform(
        compiled_space=compiled_space,
        module_id=_require_non_empty(module_id, field_name="module_id"),
        target_option_id=_require_non_empty(
            base_option_id, field_name="base_option_id"
        ),
        transform_payloads=transforms,
        parent_candidate_ids=tuple(
            [str(parent_candidate_id).strip()]
            if str(parent_candidate_id or "").strip()
            else []
        ),
        metadata=metadata,
    )


def _add_demo_candidate(
    *,
    compiled_space: CompiledMiproSpace,
    module_id: str,
    slot_id: str,
    demo: MiproDemo,
    parent_candidate_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[bool, str, str]:
    module_key = _require_non_empty(module_id, field_name="module_id")
    slot_key = _require_non_empty(slot_id, field_name="slot_id")
    component_key = demo_component_key(module_key, slot_key)
    options = compiled_space.search_space.get(component_key)
    lookup = compiled_space.demo_lookup.get(component_key)
    if options is None or lookup is None:
        raise ValueError(
            f"unknown demo component for module_id='{module_key}', slot_id='{slot_key}'"
        )

    payload_key = _demo_payload_key(demo)
    for option_id, existing in lookup.items():
        if _demo_payload_key(existing) == payload_key:
            return False, option_id, component_key

    new_option_id = _next_option_id(lookup, prefix="d")
    lookup[new_option_id] = demo
    options.append(new_option_id)
    metadata_map = compiled_space.demo_metadata.setdefault(component_key, {})
    metadata_map[new_option_id] = {
        "module_id": module_key,
        "slot_id": slot_key,
        "component_key": component_key,
        "option_id": new_option_id,
        "kind": "demo",
        "origin": "openenv_demo_candidate",
        "parent_candidate_id": str(parent_candidate_id or "").strip() or None,
        **dict(metadata or {}),
    }
    return True, new_option_id, component_key


_PATCH_ACTIONS = {
    "add_instruction_candidate",
    "add_instruction_transform",
    "add_static_demo_candidate",
    "add_trajectory_demo_candidate",
}
_READ_ACTIONS = {
    "list_components",
    "list_component_options",
    "list_registered_instruction_transforms",
    "list_registered_candidates",
    "query_transform_compatibility",
    "get_grounding_summary",
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
_GROUNDING_READ_ACTIONS = {
    "get_grounding_summary",
    "list_recent_trial_rows",
    "get_recent_trial_row",
    "list_sampled_train_rows",
    "get_sampled_train_row",
    "query_candidates",
    "query_candidate_rollout_deltas",
    "query_candidate_verdict_digest",
    "list_registered_instruction_transforms",
    "query_transform_compatibility",
}
_ROW_EVIDENCE_READ_ACTIONS = {
    "list_recent_trial_rows",
    "get_recent_trial_row",
    "list_sampled_train_rows",
    "get_sampled_train_row",
    "query_rollouts",
    "query_evidence_files",
    "read_evidence_file",
    "query_candidate_rollout_deltas",
    "query_candidate_verdict_digest",
    "get_rollout_trace",
    "query_transform_compatibility",
}


def _context_rows(
    context: MiproOpenEnvProposerContext,
    payload_key: str,
) -> list[dict[str, Any]]:
    raw_rows = context.read_model_payload.get(payload_key)
    if raw_rows is None:
        raw_rows = context.grounding_payload.get(payload_key)
    if not isinstance(raw_rows, list):
        return []
    return [dict(item) for item in raw_rows if isinstance(item, Mapping)]


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    filters: Mapping[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        keep = True
        for key, expected in filters.items():
            if expected in (None, ""):
                continue
            actual = row.get(key)
            if str(actual or "").strip() != str(expected).strip():
                keep = False
                break
        if keep:
            filtered.append(row)
        if len(filtered) >= max(1, int(limit)):
            break
    return filtered


def _workspace_root_from_context(context: MiproOpenEnvProposerContext) -> Path | None:
    raw = str(context.workspace_locations.get("workspace_root") or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _read_context_evidence_file(
    context: MiproOpenEnvProposerContext,
    *,
    path: str,
    max_chars: int = 12000,
    offset: int = 0,
) -> dict[str, Any]:
    requested = Path(str(path)).expanduser()
    workspace_root = _workspace_root_from_context(context)
    if not requested.is_absolute():
        if workspace_root is None:
            raise ValueError("workspace_root_missing")
        requested = workspace_root / requested
    resolved = requested.resolve()
    if workspace_root is not None:
        try:
            resolved.relative_to(workspace_root)
        except ValueError as exc:
            raise ValueError("evidence file must live under workspace_root") from exc
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    text = resolved.read_text(encoding="utf-8", errors="replace")
    start = max(0, int(offset))
    limit = max(1, int(max_chars))
    end = start + limit
    return {
        "path": str(resolved),
        "offset": start,
        "truncated": len(text) > end,
        "content": text[start:end],
    }


def _execute_action(
    *,
    action: MiproOpenEnvAction,
    compiled_space: CompiledMiproSpace,
    context: MiproOpenEnvProposerContext,
    config: MiproOpenEnvProposerConfig,
    instruction_patches: list[MiproInstructionPatch],
    demo_patches: list[MiproDemoPatch],
) -> tuple[dict[str, Any], bool, bool]:
    name = action.name
    args = dict(action.arguments)
    if name == "finish":
        return (
            {"status": "ok", "action": name, "reason": str(args.get("reason") or "done")},
            True,
            False,
        )

    if name == "list_components":
        return (
            {"status": "ok", "action": name, "state": summarize_compiled_space(compiled_space)},
            False,
            False,
        )

    if name == "list_component_options":
        component_key = str(args.get("component_key") or "").strip()
        return (
            {
                "status": "ok",
                "action": name,
                "result": _list_component_options(compiled_space, component_key=component_key),
            },
            False,
            False,
        )

    if name == "list_registered_instruction_transforms":
        module_id = str(args.get("module_id") or "").strip()
        try:
            result = _list_registered_instruction_transforms(
                compiled_space, module_id=module_id
            )
        except ValueError as exc:
            return (
                {"status": "error", "action": name, "reason": str(exc)},
                False,
                False,
            )
        return (
            {"status": "ok", "action": name, "result": result},
            False,
            False,
        )

    if name == "query_transform_compatibility":
        module_id = str(args.get("module_id") or "").strip()
        base_option_id = str(args.get("base_option_id") or "").strip()
        raw_transform_ids = args.get("transform_ids")
        transform_ids = (
            [str(item) for item in raw_transform_ids]
            if isinstance(raw_transform_ids, list)
            else []
        )
        try:
            result = query_instruction_transform_compatibility(
                compiled_space,
                module_id=module_id,
                base_option_id=base_option_id,
                transform_ids=transform_ids,
            )
        except ValueError as exc:
            return (
                {"status": "error", "action": name, "reason": str(exc)},
                False,
                False,
            )
        return (
            {"status": "ok", "action": name, "result": result},
            False,
            False,
        )

    if name == "list_registered_candidates":
        component_key = str(args.get("component_key") or "").strip()
        lookup = compiled_space.instruction_lookup.get(component_key) or {}
        metadata_map = compiled_space.instruction_metadata.get(component_key) or {}
        candidates = [
            {
                "option_id": oid,
                "text": text,
                "base_instruction_option_id": str(
                    (metadata_map.get(oid) or {}).get("base_instruction_option_id") or ""
                )
                or None,
                "applied_transform_ids": list(
                    (metadata_map.get(oid) or {}).get("applied_transform_ids") or []
                ),
            }
            for oid, text in lookup.items()
        ]
        return (
            {
                "status": "ok",
                "action": name,
                "component_key": component_key,
                "registered_count": len(candidates),
                "candidates": candidates,
            },
            False,
            False,
        )

    if name == "get_grounding_summary":
        summary = context.grounding_payload.get("summary_stats")
        summary_map = dict(summary) if isinstance(summary, Mapping) else {}
        return (
            {
                "status": "ok",
                "action": name,
                "summary_stats": summary_map,
                "run_metadata": dict(context.run_metadata),
                "candidate_summary_counts": dict(context.candidate_summary_counts),
                "current_best_candidate_id": context.current_best_candidate_id,
                "baseline_candidate_id": context.baseline_candidate_id,
                "delta_digest_paths": dict(context.delta_digest_paths),
                "workspace_locations": dict(context.workspace_locations),
            },
            False,
            False,
        )

    if name == "list_recent_trial_rows":
        limit = int(args.get("limit") or 8)
        rows = _context_rows(context, "recent_trial_rows")
        payload = []
        for idx, row in enumerate(rows[: max(1, limit)]):
            details = row.get("details")
            details_map = dict(details) if isinstance(details, Mapping) else {}
            reward = row.get("reward")
            if reward is None:
                reward = details_map.get("reward")
            trace_value = row.get("trace")
            if trace_value is None:
                trace_value = details_map.get("trace")
            payload.append(
                {
                    "row_idx": idx,
                    "candidate_id": row.get("candidate_id"),
                    "score": row.get("score"),
                    "reward": reward,
                    "trace_preview": _truncate(str(trace_value or ""), limit=180),
                }
            )
        return (
            {"status": "ok", "action": name, "rows": payload},
            False,
            False,
        )

    if name == "get_recent_trial_row":
        row_idx = int(args.get("row_idx") or 0)
        rows = _context_rows(context, "recent_trial_rows")
        if row_idx < 0 or row_idx >= len(rows):
            return (
                {"status": "error", "action": name, "reason": "row_idx_out_of_range"},
                False,
                False,
            )
        return (
            {"status": "ok", "action": name, "row": rows[row_idx]},
            False,
            False,
        )

    if name == "list_sampled_train_rows":
        limit = int(args.get("limit") or 8)
        rows = _context_rows(context, "sampled_train_rows")
        payload = [
            {"row_idx": idx, "preview": _truncate(json.dumps(row, ensure_ascii=True), limit=220)}
            for idx, row in enumerate(rows[: max(1, limit)])
        ]
        return (
            {"status": "ok", "action": name, "rows": payload},
            False,
            False,
        )

    if name == "get_sampled_train_row":
        row_idx = int(args.get("row_idx") or 0)
        rows = _context_rows(context, "sampled_train_rows")
        if row_idx < 0 or row_idx >= len(rows):
            return (
                {"status": "error", "action": name, "reason": "row_idx_out_of_range"},
                False,
                False,
            )
        return (
            {"status": "ok", "action": name, "row": rows[row_idx]},
            False,
            False,
        )

    if name == "query_candidates":
        candidate_id = str(args.get("candidate_id") or "").strip() or None
        limit = int(args.get("limit") or 8)
        rows = _filter_rows(
            _context_rows(context, "candidates"),
            filters={"candidate_id": candidate_id},
            limit=limit,
        )
        return (
            {
                "status": "ok",
                "action": name,
                "rows": rows,
                "row_count": len(rows),
            },
            False,
            False,
        )

    if name == "query_rollouts":
        limit = int(args.get("limit") or 8)
        rows = _filter_rows(
            _context_rows(context, "rollouts"),
            filters={
                "candidate_id": str(args.get("candidate_id") or "").strip() or None,
                "rollout_id": str(args.get("rollout_id") or "").strip() or None,
                "split": str(args.get("split") or "").strip() or None,
            },
            limit=limit,
        )
        return (
            {
                "status": "ok",
                "action": name,
                "rows": rows,
                "row_count": len(rows),
            },
            False,
            False,
        )

    if name == "query_evidence_files":
        limit = int(args.get("limit") or 8)
        rows = _filter_rows(
            _context_rows(context, "evidence_files"),
            filters={
                "rollout_id": str(args.get("rollout_id") or "").strip() or None,
                "candidate_id": str(args.get("candidate_id") or "").strip() or None,
                "kind": str(args.get("kind") or "").strip() or None,
            },
            limit=limit,
        )
        return (
            {
                "status": "ok",
                "action": name,
                "rows": rows,
                "row_count": len(rows),
            },
            False,
            False,
        )

    if name == "read_evidence_file":
        try:
            payload = _read_context_evidence_file(
                context,
                path=str(args.get("path") or ""),
                max_chars=int(args.get("max_chars") or _MAX_TOOL_RESULT_CHARS),
                offset=int(args.get("offset") or 0),
            )
        except (FileNotFoundError, ValueError) as exc:
            return (
                {"status": "error", "action": name, "reason": str(exc)},
                False,
                False,
            )
        return (
            {"status": "ok", "action": name, **payload},
            False,
            False,
        )

    if name == "query_candidate_rollout_deltas":
        limit = int(args.get("limit") or 8)
        rows = _filter_rows(
            _context_rows(context, "candidate_deltas"),
            filters={
                "candidate_id": str(args.get("candidate_id") or "").strip() or None,
                "compare_to_candidate_id": str(
                    args.get("compare_to_candidate_id") or ""
                ).strip()
                or None,
                "split": str(args.get("split") or "").strip() or None,
            },
            limit=limit,
        )
        return (
            {
                "status": "ok",
                "action": name,
                "rows": rows,
                "row_count": len(rows),
            },
            False,
            False,
        )

    if name == "query_candidate_verdict_digest":
        limit = int(args.get("limit") or 8)
        rows = _filter_rows(
            _context_rows(context, "verdict_digests"),
            filters={
                "candidate_id": str(args.get("candidate_id") or "").strip() or None,
                "split": str(args.get("split") or "").strip() or None,
            },
            limit=limit,
        )
        return (
            {
                "status": "ok",
                "action": name,
                "rows": rows,
                "row_count": len(rows),
            },
            False,
            False,
        )

    if name == "get_rollout_trace":
        rollout_id = str(args.get("rollout_id") or "").strip()
        if not rollout_id:
            return (
                {"status": "error", "action": name, "reason": "missing_rollout_id"},
                False,
                False,
            )
        trace_files = [
            item
            for item in _context_rows(context, "evidence_files")
            if str(item.get("rollout_id") or "").strip() == rollout_id
            and str(item.get("kind") or "").strip()
            in {"rollout_trace", "trace_summary"}
        ]
        trace_payload = next(
            (item for item in trace_files if str(item.get("kind") or "") == "rollout_trace"),
            None,
        )
        trace_summary = next(
            (item for item in trace_files if str(item.get("kind") or "") == "trace_summary"),
            None,
        )
        if trace_payload is None:
            return (
                {
                    "status": "error",
                    "action": name,
                    "reason": f"no_rollout_trace_materialized:{rollout_id}",
                },
                False,
                False,
            )
        max_chars = int(args.get("max_chars") or _MAX_TOOL_RESULT_CHARS)
        offset = int(args.get("offset") or 0)
        trace_preview = _read_context_evidence_file(
            context,
            path=str(trace_payload.get("path") or ""),
            max_chars=max_chars,
            offset=offset,
        )
        trace_summary_preview = (
            _read_context_evidence_file(
                context,
                path=str(trace_summary.get("path") or ""),
                max_chars=max_chars,
                offset=offset,
            )
            if trace_summary is not None
            else None
        )
        return (
            {
                "status": "ok",
                "action": name,
                "rollout_id": rollout_id,
                "trace_file": trace_payload,
                "trace_preview": trace_preview,
                "trace_summary_file": trace_summary,
                "trace_summary_preview": trace_summary_preview,
            },
            False,
            False,
        )

    if name == "add_instruction_candidate":
        if len(instruction_patches) >= int(config.max_instruction_patches):
            return (
                {
                    "status": "ignored",
                    "action": name,
                    "reason": "max_instruction_patches_reached",
                },
                False,
                False,
            )
        added, option_id, component_key = _add_instruction_candidate(
            compiled_space=compiled_space,
            module_id=str(args.get("module_id") or ""),
            instruction_text=str(args.get("instruction_text") or ""),
            config=config,
            parent_candidate_id=context.current_best_candidate_id,
            metadata={
                "round_idx": int(context.round_idx),
                "baseline_candidate_id": context.baseline_candidate_id,
            },
        )
        if added:
            instruction_patches.append(
                MiproInstructionPatch(
                    module_id=str(args.get("module_id") or "").strip(),
                    component_key=component_key,
                    option_id=option_id,
                    instruction_text=str(args.get("instruction_text") or "").strip(),
                )
            )
        return (
            {
                "status": "ok",
                "action": name,
                "added": added,
                "component_key": component_key,
                "option_id": option_id,
            },
            False,
            bool(added),
        )

    if name == "add_instruction_transform":
        if len(instruction_patches) >= int(config.max_instruction_patches):
            return (
                {
                    "status": "ignored",
                    "action": name,
                    "reason": "max_instruction_patches_reached",
                },
                False,
                False,
            )
        raw_transforms = args.get("transforms")
        transforms = [
            dict(item)
            for item in raw_transforms
            if isinstance(item, Mapping)
        ] if isinstance(raw_transforms, list) else []
        try:
            transform_payload = _add_instruction_transform(
                compiled_space=compiled_space,
                module_id=str(args.get("module_id") or ""),
                base_option_id=str(args.get("base_option_id") or ""),
                transforms=transforms,
                config=config,
                parent_candidate_id=context.current_best_candidate_id,
                metadata={
                    "round_idx": int(context.round_idx),
                    "baseline_candidate_id": context.baseline_candidate_id,
                },
            )
        except (ValueError, KeyError) as exc:
            return {"status": "error", "action": name, "reason": str(exc)}, False, False
        added = bool(transform_payload.get("added"))
        option_id = str(transform_payload.get("primary_option_id") or "")
        component_key = str(transform_payload.get("component_key") or "")
        if added:
            materialized = str(
                compiled_space.instruction_lookup.get(component_key, {}).get(option_id, "")
            )
            instruction_patches.append(
                MiproInstructionPatch(
                    module_id=str(args.get("module_id") or "").strip(),
                    component_key=component_key,
                    option_id=option_id,
                    instruction_text=materialized,
                    base_option_id=str(
                        transform_payload.get("base_instruction_option_id") or ""
                    )
                    or None,
                    transform_id=str(
                        (transform_payload.get("transform_ids") or [""])[0]
                    ).strip()
                    or None,
                    bundle_option_ids=tuple(
                        str(item)
                        for item in (
                            transform_payload.get("created_bundle_option_ids") or []
                        )
                        if str(item).strip()
                    ),
                )
            )
        return (
            {
                "status": "ok",
                "action": name,
                "added": added,
                "component_key": component_key,
                "option_id": option_id,
                "transform_ids": list(transform_payload.get("transform_ids") or []),
                "created_bundle_option_ids": list(
                    transform_payload.get("created_bundle_option_ids") or []
                ),
                "base_instruction_option_id": transform_payload.get(
                    "base_instruction_option_id"
                ),
            },
            False,
            bool(
                added
                or bool(transform_payload.get("created_bundle_option_ids"))
                or bool(transform_payload.get("duplicate_transform_ids"))
            ),
        )

    if name == "add_static_demo_candidate":
        if len(demo_patches) >= int(config.max_demo_patches):
            return (
                {
                    "status": "ignored",
                    "action": name,
                    "reason": "max_demo_patches_reached",
                },
                False,
                False,
            )
        demo = StaticFewShotDemo(
            messages=_coerce_messages(args.get("messages"), config=config),
            demo_label=(str(args["demo_label"]).strip() if args.get("demo_label") is not None else None),
        )
        module_id = str(args.get("module_id") or "").strip()
        slot_id = str(args.get("slot_id") or "").strip()
        added, option_id, component_key = _add_demo_candidate(
            compiled_space=compiled_space,
            module_id=module_id,
            slot_id=slot_id,
            demo=demo,
            parent_candidate_id=context.current_best_candidate_id,
            metadata={
                "round_idx": int(context.round_idx),
                "baseline_candidate_id": context.baseline_candidate_id,
            },
        )
        if added:
            demo_patches.append(
                MiproDemoPatch(
                    module_id=module_id,
                    slot_id=slot_id,
                    component_key=component_key,
                    option_id=option_id,
                    demo=demo,
                )
            )
        return (
            {
                "status": "ok",
                "action": name,
                "added": added,
                "component_key": component_key,
                "option_id": option_id,
            },
            False,
            bool(added),
        )

    if name == "add_trajectory_demo_candidate":
        if len(demo_patches) >= int(config.max_demo_patches):
            return (
                {
                    "status": "ignored",
                    "action": name,
                    "reason": "max_demo_patches_reached",
                },
                False,
                False,
            )
        demo = TrajectorySnippetDemo(
            rollout_id=str(args.get("rollout_id") or ""),
            start_step=int(args.get("start_step") or 0),
            end_step=int(args.get("end_step") or 0),
            snippet_label=(
                str(args["snippet_label"]).strip()
                if args.get("snippet_label") is not None
                else None
            ),
        )
        module_id = str(args.get("module_id") or "").strip()
        slot_id = str(args.get("slot_id") or "").strip()
        added, option_id, component_key = _add_demo_candidate(
            compiled_space=compiled_space,
            module_id=module_id,
            slot_id=slot_id,
            demo=demo,
            parent_candidate_id=context.current_best_candidate_id,
            metadata={
                "round_idx": int(context.round_idx),
                "baseline_candidate_id": context.baseline_candidate_id,
            },
        )
        if added:
            demo_patches.append(
                MiproDemoPatch(
                    module_id=module_id,
                    slot_id=slot_id,
                    component_key=component_key,
                    option_id=option_id,
                    demo=demo,
                )
            )
        return (
            {
                "status": "ok",
                "action": name,
                "added": added,
                "component_key": component_key,
                "option_id": option_id,
            },
            False,
            bool(added),
        )

    return (
        {"status": "error", "action": name, "reason": "unknown_action"},
        False,
        False,
    )


async def run_openenv_react_proposer(
    *,
    compiled_space: CompiledMiproSpace,
    agent: MiproOpenEnvReactAgent,
    context: MiproOpenEnvProposerContext,
    config: MiproOpenEnvProposerConfig | None = None,
    variant: MiproOpenEnvProposerVariant | Mapping[str, Any] | None = None,
) -> MiproOpenEnvProposerOutcome:
    """Run one bounded OpenEnv-style proposer session and return an expanded space."""

    from synth_optimizers.miprov2.core.proposer_environment import MiproProposerEnvironment

    cfg = config or MiproOpenEnvProposerConfig()
    variant_model = _coerce_proposer_variant(variant)
    environment = MiproProposerEnvironment.in_memory(
        compiled_space=compiled_space,
        context=context,
        config=cfg,
        variant=variant_model,
    )
    working = environment.state.compiled_space
    catalog = environment.list_tools()
    runtime_tools = list(catalog.get("runtime_tools") or [])

    instruction_patches = environment.state.instruction_patches
    demo_patches = environment.state.demo_patches
    transcript: list[dict[str, Any]] = []
    action_counts: dict[str, int] = {}
    stop_reason = "max_turns_reached"
    noop_turns = 0
    read_action_count = 0
    patch_action_count = 0
    duplicate_patch_count = 0
    ignored_patch_count = 0
    policy_violation_count = 0
    grounding_read_action_count = 0
    evidence_read_action_count = 0
    read_tools_used: set[str] = set()
    consecutive_patch_actions = 0
    prompt_tokens = 0
    cached_prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    model_turn_count = 0
    tool_call_count = 0
    archive_stats: dict[str, Any] = {
        "spill_count": 0,
        "archived_message_count": 0,
        "archive_path": None,
    }
    live_messages: list[dict[str, Any]] = []

    for turn_idx in range(1, int(cfg.max_turns) + 1):
        turns_remaining_before_action = int(cfg.max_turns) - turn_idx + 1
        state = summarize_compiled_space(working)
        summary_stats = context.grounding_payload.get("summary_stats")
        summary_payload = dict(summary_stats) if isinstance(summary_stats, Mapping) else {}
        sampled_row_count = len(_context_rows(context, "sampled_train_rows"))
        trial_row_count = len(_context_rows(context, "recent_trial_rows"))
        state["round_idx"] = int(context.round_idx)
        state["instruction_patch_count"] = len(instruction_patches)
        state["demo_patch_count"] = len(demo_patches)
        state["turns_remaining_before_action"] = max(0, int(turns_remaining_before_action))
        state["recent_failures"] = list(context.recent_failures)
        state["recent_successes"] = list(context.recent_successes)
        state["grounding_summary"] = summary_payload
        state["grounding_counts"] = {
            "sampled_train_rows": sampled_row_count,
            "recent_trial_rows": trial_row_count,
        }
        state["run_metadata"] = dict(context.run_metadata)
        state["candidate_summary_counts"] = dict(context.candidate_summary_counts)
        state["current_best_candidate_id"] = context.current_best_candidate_id
        state["baseline_candidate_id"] = context.baseline_candidate_id
        state["delta_digest_paths"] = dict(context.delta_digest_paths)
        state["workspace_locations"] = dict(context.workspace_locations)
        if not live_messages:
            live_messages = _initial_live_messages(
                objective=context.objective,
                runtime_tools=runtime_tools,
                proposer_state=state,
                transcript=transcript,
                variant=variant_model,
            )
        _trim_messages_if_needed(
            messages=live_messages,
            context=context,
            config=cfg,
            archive_stats=archive_stats,
        )
        turns_remaining_after_action = max(0, int(cfg.max_turns) - turn_idx)
        turn_tool_results: list[dict[str, Any]] = []
        failed_tool_results: list[tuple[str, str]] = []
        assistant_message: dict[str, Any] = {}
        usage: dict[str, Any] = {}
        provider_trace: dict[str, Any] = {}
        stop_session = False
        try:
            turn_response = await agent.next_turn(
                objective=context.objective,
                runtime_tools=runtime_tools,
                messages=live_messages,
                proposer_state=state,
                config=cfg,
            )
            model_turn_count += 1
            assistant_message = dict(turn_response.assistant_message)
            usage = dict(turn_response.usage)
            provider_trace = dict(turn_response.provider_trace)
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            cached_prompt_tokens += _cached_prompt_tokens_from_usage(usage)
            completion_tokens += int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            total_tokens += int(usage.get("total_tokens") or 0)
            live_messages.append(assistant_message)
            if not turn_response.tool_calls:
                noop_turns += 1
                failed_tool_results.append(
                    ("model_transport", "no_tool_calls_returned")
                )
        except Exception as exc:
            noop_turns += 1
            provider_trace = {
                "provider_error": _truncate(str(exc), limit=800),
            }
            failed_tool_results.append(("model_transport", str(exc)))
            transcript.append(
                {
                    "turn": turn_idx,
                    "assistant_message": assistant_message,
                    "tool_results": [],
                    "usage": usage,
                    "provider_trace": provider_trace,
                }
            )
            followup_message = _tool_followup_user_message(
                remaining_turns=turns_remaining_after_action,
                failed_tool_results=failed_tool_results,
                successful_patch_count=len(instruction_patches) + len(demo_patches),
            )
            if followup_message:
                live_messages.append({"role": "user", "content": followup_message})
            if noop_turns >= int(cfg.max_noop_turns):
                stop_reason = "no_progress_budget_exhausted"
                break
            continue

        for invocation in turn_response.tool_calls:
            action = invocation.action
            action_name = str(action.name)
            tool_call_count += 1
            action_counts[action_name] = int(action_counts.get(action_name, 0)) + 1
            stop_after = False
            made_progress = False
            result: dict[str, Any]
            is_patch_action = action_name in _PATCH_ACTIONS
            is_read_action = action_name in _READ_ACTIONS
            policy_violation_reason: str | None = None
            successful_patch_count = len(instruction_patches) + len(demo_patches)
            if action_name == "finish" and successful_patch_count <= 0:
                policy_violation_reason = "finish_requires_successful_patch_submission"
            elif is_patch_action:
                if patch_action_count >= int(cfg.max_patch_actions_per_session):
                    policy_violation_reason = "max_patch_actions_per_session_reached"
                elif read_action_count < int(cfg.min_read_actions_before_patch):
                    policy_violation_reason = "min_read_actions_before_patch_not_met"
                elif grounding_read_action_count <= 0:
                    policy_violation_reason = "grounding_read_required_before_patch"
                elif (
                    bool(cfg.require_distinct_read_tools_before_patch)
                    and len(read_tools_used) < 2
                ):
                    policy_violation_reason = "distinct_read_tools_required_before_patch"
                elif patch_action_count >= 1 and evidence_read_action_count <= 0:
                    policy_violation_reason = "row_evidence_read_required_before_second_patch"
                elif (
                    int(cfg.max_consecutive_patch_actions) > 0
                    and consecutive_patch_actions >= int(cfg.max_consecutive_patch_actions)
                ):
                    policy_violation_reason = "max_consecutive_patch_actions_reached"

            if policy_violation_reason is not None:
                policy_violation_count += 1
                result = {
                    "status": "ignored",
                    "action": action_name,
                    "reason": policy_violation_reason,
                    "required_reads": int(cfg.min_read_actions_before_patch),
                    "observed_reads": read_action_count,
                    "grounding_reads": grounding_read_action_count,
                    "read_tools_used_count": len(read_tools_used),
                    "evidence_reads": evidence_read_action_count,
                    "patch_actions_seen": patch_action_count,
                    "consecutive_patch_actions": consecutive_patch_actions,
                }
                made_progress = False
            else:
                try:
                    tool_result = await environment.call_tool(
                        action.name,
                        dict(action.arguments),
                        actor_id="openenv",
                    )
                    result = dict(tool_result.get("result") or {})
                    stop_after = bool(tool_result.get("stop_session"))
                    made_progress = bool(tool_result.get("made_progress"))
                except Exception as exc:
                    result = {
                        "status": "error",
                        "action": action.name,
                        "reason": str(exc),
                    }
                    made_progress = False
            if (
                bool(cfg.count_successful_reads_as_progress)
                and is_read_action
                and str(result.get("status") or "") == "ok"
            ):
                made_progress = True

            if is_read_action:
                read_action_count += 1
                read_tools_used.add(action_name)
                if action_name in _GROUNDING_READ_ACTIONS:
                    grounding_read_action_count += 1
                if action_name in _ROW_EVIDENCE_READ_ACTIONS:
                    evidence_read_action_count += 1
                consecutive_patch_actions = 0
            if is_patch_action:
                patch_action_count += 1
                consecutive_patch_actions += 1
                if result.get("status") == "ignored":
                    ignored_patch_count += 1
                if result.get("status") == "ok" and result.get("added") is False:
                    duplicate_patch_count += 1
            elif not is_read_action:
                consecutive_patch_actions = 0

            result["turns_remaining_after_action"] = int(turns_remaining_after_action)
            tool_result_payload = {
                "tool_call_id": str(invocation.tool_call_id),
                "action": {"name": action.name, "arguments": dict(action.arguments)},
                "result": result,
                "made_progress": bool(made_progress),
                "read_action": bool(is_read_action),
                "patch_action": bool(is_patch_action),
                "turns_remaining_after_action": int(turns_remaining_after_action),
            }
            turn_tool_results.append(tool_result_payload)
            tool_message = {
                "role": "tool",
                "tool_call_id": str(invocation.tool_call_id),
                "content": _serialize_tool_result(result),
            }
            if action_name:
                tool_message["name"] = action_name
            live_messages.append(tool_message)

            if made_progress:
                noop_turns = 0
            else:
                noop_turns += 1
                failed_tool_results.append(
                    (
                        action_name,
                        str(result.get("reason") or result.get("status") or "no_progress"),
                    )
                )

            if stop_after:
                stop_reason = str(result.get("reason") or "finish")
                stop_session = True
            elif (
                len(instruction_patches) >= int(cfg.max_instruction_patches)
                and len(demo_patches) >= int(cfg.max_demo_patches)
            ):
                stop_reason = "patch_budget_exhausted"
                stop_session = True
            elif noop_turns >= int(cfg.max_noop_turns):
                stop_reason = "no_progress_budget_exhausted"
                stop_session = True
            if stop_session:
                break

        transcript.append(
            {
                "turn": turn_idx,
                "assistant_message": assistant_message,
                "tool_results": turn_tool_results,
                "usage": usage,
                "provider_trace": provider_trace,
            }
        )

        if stop_session:
            break

        followup_message = _tool_followup_user_message(
            remaining_turns=turns_remaining_after_action,
            failed_tool_results=failed_tool_results,
            successful_patch_count=len(instruction_patches) + len(demo_patches),
        )
        if followup_message:
            live_messages.append({"role": "user", "content": followup_message})

    return MiproOpenEnvProposerOutcome(
        compiled_space=environment.state.compiled_space,
        instruction_patches=instruction_patches,
        demo_patches=demo_patches,
        transcript=transcript,
        action_counts=action_counts,
        read_action_count=read_action_count,
        patch_action_count=patch_action_count,
        duplicate_patch_count=duplicate_patch_count,
        ignored_patch_count=ignored_patch_count,
        policy_violation_count=policy_violation_count,
        grounding_read_action_count=grounding_read_action_count,
        evidence_read_action_count=evidence_read_action_count,
        read_tools_used=tuple(sorted(read_tools_used)),
        stop_reason=stop_reason,
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model_turn_count=model_turn_count,
        tool_call_count=tool_call_count,
        archive_spill_count=int(archive_stats.get("spill_count") or 0),
        archived_message_count=int(archive_stats.get("archived_message_count") or 0),
        archive_path=(
            str(archive_stats.get("archive_path"))
            if archive_stats.get("archive_path") is not None
            else None
        ),
    )


async def sync_optimizer_search_space(
    *,
    optimizer: DiscreteMiproOptimizer,
    compiled_space: CompiledMiproSpace,
) -> None:
    """Push latest compiled-space options into DiscreteMiproOptimizer/TPE."""

    await optimizer.update_search_space(compiled_space.search_space)


def proposer_outcome_summary(outcome: MiproOpenEnvProposerOutcome) -> dict[str, Any]:
    """Stable, compact summary suitable for logs or orchestration metadata."""

    return {
        "stop_reason": outcome.stop_reason,
        "instruction_patches": len(outcome.instruction_patches),
        "demo_patches": len(outcome.demo_patches),
        "action_counts": dict(outcome.action_counts),
        "read_action_count": outcome.read_action_count,
        "patch_action_count": outcome.patch_action_count,
        "duplicate_patch_count": outcome.duplicate_patch_count,
        "ignored_patch_count": outcome.ignored_patch_count,
        "policy_violation_count": outcome.policy_violation_count,
        "grounding_read_action_count": outcome.grounding_read_action_count,
        "evidence_read_action_count": outcome.evidence_read_action_count,
        "read_tools_used": list(outcome.read_tools_used),
        "prompt_tokens": outcome.prompt_tokens,
        "cached_prompt_tokens": outcome.cached_prompt_tokens,
        "completion_tokens": outcome.completion_tokens,
        "total_tokens": outcome.total_tokens,
        "model_turn_count": outcome.model_turn_count,
        "tool_call_count": outcome.tool_call_count,
        "archive_spill_count": outcome.archive_spill_count,
        "archived_message_count": outcome.archived_message_count,
        "archive_path": outcome.archive_path,
        "component_counts": {
            key: len(values) for key, values in outcome.compiled_space.search_space.items()
        },
        "transcript_turns": len(outcome.transcript),
    }


__all__ = [
    "MiproOpenEnvAction",
    "MiproOpenEnvReactAgent",
    "MiproOpenEnvProposerContext",
    "MiproOpenEnvProposerConfig",
    "MiproInstructionPatch",
    "MiproDemoPatch",
    "MiproOpenEnvProposerOutcome",
    "GroqOpenEnvReactAgent",
    "OpenAIOpenEnvReactAgent",
    "HeuristicOpenEnvReactAgent",
    "apply_instruction_transforms",
    "build_openenv_tool_catalog",
    "clone_compiled_space",
    "summarize_compiled_space",
    "run_openenv_react_proposer",
    "sync_optimizer_search_space",
    "proposer_outcome_summary",
]
