"""Phase-1 program schema for MIPRO-style instruction/demo bundles.

This module models a typed program template and the materialized candidate payload
that comes back from decoding a discrete config.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

_CANDIDATE_PREFIX = "candidate_"


def _require_non_empty(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


@dataclass(slots=True, frozen=True)
class DemoMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "role", _require_non_empty(self.role, field_name="DemoMessage.role")
        )
        object.__setattr__(
            self,
            "content",
            _require_non_empty(self.content, field_name="DemoMessage.content"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DemoMessage:
        return cls(
            role=str(payload.get("role") or ""),
            content=str(payload.get("content") or ""),
        )


@dataclass(slots=True, frozen=True)
class StaticFewShotDemo:
    messages: tuple[DemoMessage, ...]
    demo_label: str | None = None
    demo_kind: Literal["static_few_shot"] = "static_few_shot"

    def __post_init__(self) -> None:
        msgs = tuple(self.messages)
        if not msgs:
            raise ValueError("StaticFewShotDemo.messages must contain at least one message")
        object.__setattr__(self, "messages", msgs)
        if self.demo_label is not None:
            object.__setattr__(
                self,
                "demo_label",
                _require_non_empty(self.demo_label, field_name="StaticFewShotDemo.demo_label"),
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "demo_kind": self.demo_kind,
            "messages": [item.to_dict() for item in self.messages],
        }
        if self.demo_label is not None:
            payload["demo_label"] = self.demo_label
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StaticFewShotDemo:
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("StaticFewShotDemo.messages must be a list")
        return cls(
            messages=tuple(DemoMessage.from_dict(item) for item in raw_messages if isinstance(item, Mapping)),
            demo_label=str(payload["demo_label"]).strip() if payload.get("demo_label") is not None else None,
        )


@dataclass(slots=True, frozen=True)
class TrajectorySnippetDemo:
    rollout_id: str
    start_step: int
    end_step: int
    snippet_label: str | None = None
    demo_kind: Literal["trajectory_snippet"] = "trajectory_snippet"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rollout_id",
            _require_non_empty(self.rollout_id, field_name="TrajectorySnippetDemo.rollout_id"),
        )
        start = int(self.start_step)
        end = int(self.end_step)
        if start < 0:
            raise ValueError("TrajectorySnippetDemo.start_step must be >= 0")
        if end < start:
            raise ValueError("TrajectorySnippetDemo.end_step must be >= start_step")
        object.__setattr__(self, "start_step", start)
        object.__setattr__(self, "end_step", end)
        if self.snippet_label is not None:
            object.__setattr__(
                self,
                "snippet_label",
                _require_non_empty(
                    self.snippet_label,
                    field_name="TrajectorySnippetDemo.snippet_label",
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "demo_kind": self.demo_kind,
            "rollout_id": self.rollout_id,
            "start_step": self.start_step,
            "end_step": self.end_step,
        }
        if self.snippet_label is not None:
            payload["snippet_label"] = self.snippet_label
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrajectorySnippetDemo:
        return cls(
            rollout_id=str(payload.get("rollout_id") or ""),
            start_step=int(payload.get("start_step") or 0),
            end_step=int(payload.get("end_step") or 0),
            snippet_label=str(payload["snippet_label"]).strip() if payload.get("snippet_label") is not None else None,
        )


MiproDemo: TypeAlias = StaticFewShotDemo | TrajectorySnippetDemo


def demo_from_dict(payload: Mapping[str, Any]) -> MiproDemo:
    demo_kind = str(payload.get("demo_kind") or "").strip().lower()
    if demo_kind == "static_few_shot":
        return StaticFewShotDemo.from_dict(payload)
    if demo_kind == "trajectory_snippet":
        return TrajectorySnippetDemo.from_dict(payload)
    raise ValueError(f"unsupported demo_kind: {demo_kind or '<empty>'}")


@dataclass(slots=True, frozen=True)
class DemoSlotTemplate:
    slot_id: str
    candidate_demos: tuple[MiproDemo, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "slot_id", _require_non_empty(self.slot_id, field_name="DemoSlotTemplate.slot_id")
        )
        demos = tuple(self.candidate_demos)
        if not demos:
            raise ValueError("DemoSlotTemplate.candidate_demos must contain at least one demo")
        object.__setattr__(self, "candidate_demos", demos)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "candidate_demos": [item.to_dict() for item in self.candidate_demos],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DemoSlotTemplate:
        raw = payload.get("candidate_demos")
        if not isinstance(raw, list):
            raise ValueError("DemoSlotTemplate.candidate_demos must be a list")
        demos = tuple(demo_from_dict(item) for item in raw if isinstance(item, Mapping))
        return cls(slot_id=str(payload.get("slot_id") or ""), candidate_demos=demos)


@dataclass(slots=True, frozen=True)
class MiproModuleTemplate:
    module_id: str
    instruction_candidates: tuple[str, ...]
    max_instruction_transforms_per_candidate: int = 10
    demo_slots: tuple[DemoSlotTemplate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "module_id",
            _require_non_empty(self.module_id, field_name="MiproModuleTemplate.module_id"),
        )
        instructions = tuple(
            _require_non_empty(item, field_name="MiproModuleTemplate.instruction_candidates")
            for item in self.instruction_candidates
        )
        if not instructions:
            raise ValueError("MiproModuleTemplate.instruction_candidates must be non-empty")
        if len(set(instructions)) != len(instructions):
            raise ValueError("MiproModuleTemplate.instruction_candidates must be unique")
        object.__setattr__(self, "instruction_candidates", instructions)
        max_transforms = int(self.max_instruction_transforms_per_candidate)
        if max_transforms < 0:
            raise ValueError(
                "MiproModuleTemplate.max_instruction_transforms_per_candidate must be >= 0"
            )
        object.__setattr__(
            self, "max_instruction_transforms_per_candidate", max_transforms
        )

        slots = tuple(self.demo_slots)
        slot_ids = [item.slot_id for item in slots]
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError(
                f"MiproModuleTemplate.demo_slots has duplicate slot_id values for module '{self.module_id}'"
            )
        object.__setattr__(self, "demo_slots", slots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "instruction_candidates": list(self.instruction_candidates),
            "max_instruction_transforms_per_candidate": int(
                self.max_instruction_transforms_per_candidate
            ),
            "demo_slots": [item.to_dict() for item in self.demo_slots],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MiproModuleTemplate:
        raw_slots = payload.get("demo_slots") or []
        if not isinstance(raw_slots, list):
            raise ValueError("MiproModuleTemplate.demo_slots must be a list")
        slots = tuple(
            DemoSlotTemplate.from_dict(item) for item in raw_slots if isinstance(item, Mapping)
        )
        raw_instructions = payload.get("instruction_candidates")
        if not isinstance(raw_instructions, list):
            raise ValueError("MiproModuleTemplate.instruction_candidates must be a list")
        instructions = tuple(str(item) for item in raw_instructions)
        return cls(
            module_id=str(payload.get("module_id") or ""),
            instruction_candidates=instructions,
            max_instruction_transforms_per_candidate=int(
                payload.get("max_instruction_transforms_per_candidate") or 10
            ),
            demo_slots=slots,
        )


@dataclass(slots=True, frozen=True)
class MiproProgramTemplate:
    modules: tuple[MiproModuleTemplate, ...]
    program_id: str = "mipro_program"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "program_id",
            _require_non_empty(self.program_id, field_name="MiproProgramTemplate.program_id"),
        )
        modules = tuple(self.modules)
        if not modules:
            raise ValueError("MiproProgramTemplate.modules must contain at least one module")
        module_ids = [item.module_id for item in modules]
        if len(set(module_ids)) != len(module_ids):
            raise ValueError("MiproProgramTemplate.modules has duplicate module_id values")
        object.__setattr__(self, "modules", modules)

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "modules": [item.to_dict() for item in self.modules],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MiproProgramTemplate:
        raw_modules = payload.get("modules")
        if not isinstance(raw_modules, list):
            raise ValueError("MiproProgramTemplate.modules must be a list")
        modules = tuple(
            MiproModuleTemplate.from_dict(item) for item in raw_modules if isinstance(item, Mapping)
        )
        return cls(program_id=str(payload.get("program_id") or "mipro_program"), modules=modules)


def materialized_lever_bundle_payload(
    *,
    program_id: str,
    selected_instructions: Mapping[str, str],
    selected_demos: Mapping[str, Mapping[str, MiproDemo]],
    selected_instruction_base_option_ids: Mapping[str, str] | None = None,
    selected_instruction_transform_ids: Mapping[str, tuple[str, ...]] | None = None,
    active_model_transform_id: str | None = None,
    active_finetune_ref: str | None = None,
    active_execution_mode: str | None = None,
) -> dict[str, Any]:
    module_ids = sorted(set(selected_instructions.keys()) | set(selected_demos.keys()))
    modules_payload: list[dict[str, Any]] = []
    base_option_ids = dict(selected_instruction_base_option_ids or {})
    transform_ids_map = dict(selected_instruction_transform_ids or {})
    for module_id in module_ids:
        instruction = _require_non_empty(
            selected_instructions[module_id], field_name=f"selected_instructions[{module_id}]"
        )
        raw_slot_payload = selected_demos.get(module_id, {})
        slot_payload = [
            {"slot_id": slot_id, "demo": raw_slot_payload[slot_id].to_dict()}
            for slot_id in sorted(raw_slot_payload.keys())
        ]
        modules_payload.append(
            {
                "module_id": _require_non_empty(
                    module_id, field_name="selected_instructions module_id"
                ),
                "instruction": instruction,
                "base_instruction_option_id": (
                    str(base_option_ids.get(module_id) or "").strip() or None
                ),
                "applied_transform_ids": list(transform_ids_map.get(module_id) or ()),
                "demos": slot_payload,
            }
        )
    return {
        "program_id": _require_non_empty(program_id, field_name="program_id"),
        "modules": modules_payload,
        "model_mutation": {
            "active_model_transform_id": (
                str(active_model_transform_id).strip() or None
                if active_model_transform_id is not None
                else None
            ),
            "active_finetune_ref": (
                str(active_finetune_ref).strip() or None
                if active_finetune_ref is not None
                else None
            ),
            "active_execution_mode": (
                str(active_execution_mode).strip() or None
                if active_execution_mode is not None
                else None
            ),
        },
    }


def materialized_lever_bundle_hash(
    *,
    program_id: str,
    selected_instructions: Mapping[str, str],
    selected_demos: Mapping[str, Mapping[str, MiproDemo]],
    selected_instruction_base_option_ids: Mapping[str, str] | None = None,
    selected_instruction_transform_ids: Mapping[str, tuple[str, ...]] | None = None,
    active_model_transform_id: str | None = None,
    active_finetune_ref: str | None = None,
    active_execution_mode: str | None = None,
) -> str:
    payload = materialized_lever_bundle_payload(
        program_id=program_id,
        selected_instructions=selected_instructions,
        selected_demos=selected_demos,
        selected_instruction_base_option_ids=selected_instruction_base_option_ids,
        selected_instruction_transform_ids=selected_instruction_transform_ids,
        active_model_transform_id=active_model_transform_id,
        active_finetune_ref=active_finetune_ref,
        active_execution_mode=active_execution_mode,
    )
    encoded = _canonical_json_text(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class MiproProgramCandidate:
    program_id: str
    selected_instructions: dict[str, str]
    selected_demos: dict[str, dict[str, MiproDemo]]
    selected_instruction_base_option_ids: dict[str, str] = field(default_factory=dict)
    selected_instruction_transform_ids: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    source_config: dict[str, str] = field(default_factory=dict)
    active_model_transform_id: str | None = None
    active_finetune_ref: str | None = None
    active_execution_mode: str | None = None
    parent_candidate_id: str | None = None
    parent_candidate_ids: tuple[str, ...] = ()
    candidate_id: str | None = None
    lever_bundle_hash: str | None = None

    def __post_init__(self) -> None:
        self.program_id = _require_non_empty(
            self.program_id, field_name="MiproProgramCandidate.program_id"
        )
        self.selected_instructions = {
            _require_non_empty(module_id, field_name="selected_instructions.module_id"): _require_non_empty(
                instruction,
                field_name="selected_instructions.instruction",
            )
            for module_id, instruction in self.selected_instructions.items()
        }
        if not self.selected_instructions:
            raise ValueError("MiproProgramCandidate.selected_instructions must be non-empty")

        normalized_demos: dict[str, dict[str, MiproDemo]] = {}
        for module_id, slot_map in self.selected_demos.items():
            mid = _require_non_empty(module_id, field_name="selected_demos.module_id")
            if mid not in self.selected_instructions:
                raise ValueError(
                    "MiproProgramCandidate.selected_demos contains unknown module_id "
                    f"'{mid}'"
                )
            normalized_slot_map: dict[str, MiproDemo] = {}
            for slot_id, demo in slot_map.items():
                sid = _require_non_empty(slot_id, field_name="selected_demos.slot_id")
                normalized_slot_map[sid] = demo
            normalized_demos[mid] = dict(sorted(normalized_slot_map.items(), key=lambda item: item[0]))
        self.selected_demos = dict(sorted(normalized_demos.items(), key=lambda item: item[0]))

        normalized_base_ids: dict[str, str] = {}
        for module_id, option_id in self.selected_instruction_base_option_ids.items():
            mid = _require_non_empty(
                module_id, field_name="selected_instruction_base_option_ids.module_id"
            )
            if mid not in self.selected_instructions:
                raise ValueError(
                    "MiproProgramCandidate.selected_instruction_base_option_ids contains "
                    f"unknown module_id '{mid}'"
                )
            normalized_base_ids[mid] = _require_non_empty(
                option_id,
                field_name="selected_instruction_base_option_ids.option_id",
            )
        self.selected_instruction_base_option_ids = dict(
            sorted(normalized_base_ids.items(), key=lambda item: item[0])
        )

        normalized_transform_ids: dict[str, tuple[str, ...]] = {}
        for module_id, transform_ids in self.selected_instruction_transform_ids.items():
            mid = _require_non_empty(
                module_id,
                field_name="selected_instruction_transform_ids.module_id",
            )
            if mid not in self.selected_instructions:
                raise ValueError(
                    "MiproProgramCandidate.selected_instruction_transform_ids contains "
                    f"unknown module_id '{mid}'"
                )
            normalized_transform_ids[mid] = tuple(
                _require_non_empty(
                    str(transform_id),
                    field_name="selected_instruction_transform_ids.transform_id",
                )
                for transform_id in transform_ids
                if str(transform_id).strip()
            )
        self.selected_instruction_transform_ids = dict(
            sorted(normalized_transform_ids.items(), key=lambda item: item[0])
        )

        self.source_config = dict(
            sorted(
                (
                    _require_non_empty(component, field_name="source_config.component"),
                    _require_non_empty(option, field_name="source_config.option"),
                )
                for component, option in self.source_config.items()
            )
        )

        self.active_model_transform_id = (
            _require_non_empty(
                self.active_model_transform_id,
                field_name="MiproProgramCandidate.active_model_transform_id",
            )
            if self.active_model_transform_id is not None
            else None
        )
        self.active_finetune_ref = (
            _require_non_empty(
                self.active_finetune_ref,
                field_name="MiproProgramCandidate.active_finetune_ref",
            )
            if self.active_finetune_ref is not None
            else None
        )
        self.active_execution_mode = (
            _require_non_empty(
                self.active_execution_mode,
                field_name="MiproProgramCandidate.active_execution_mode",
            )
            if self.active_execution_mode is not None
            else None
        )

        normalized_parent_ids = tuple(
            dict.fromkeys(
                _require_non_empty(
                    str(parent_id),
                    field_name="MiproProgramCandidate.parent_candidate_ids",
                )
                for parent_id in self.parent_candidate_ids
                if str(parent_id).strip()
            )
        )
        self.parent_candidate_ids = normalized_parent_ids
        if self.parent_candidate_id is None and normalized_parent_ids:
            self.parent_candidate_id = normalized_parent_ids[0]

        bundle_hash = self.lever_bundle_hash or materialized_lever_bundle_hash(
            program_id=self.program_id,
            selected_instructions=self.selected_instructions,
            selected_demos=self.selected_demos,
            selected_instruction_base_option_ids=self.selected_instruction_base_option_ids,
            selected_instruction_transform_ids=self.selected_instruction_transform_ids,
            active_model_transform_id=self.active_model_transform_id,
            active_finetune_ref=self.active_finetune_ref,
            active_execution_mode=self.active_execution_mode,
        )
        self.lever_bundle_hash = bundle_hash
        if self.candidate_id is None:
            self.candidate_id = f"{_CANDIDATE_PREFIX}{bundle_hash[:12]}"

    def materialized_payload(self) -> dict[str, Any]:
        return materialized_lever_bundle_payload(
            program_id=self.program_id,
            selected_instructions=self.selected_instructions,
            selected_demos=self.selected_demos,
            selected_instruction_base_option_ids=self.selected_instruction_base_option_ids,
            selected_instruction_transform_ids=self.selected_instruction_transform_ids,
            active_model_transform_id=self.active_model_transform_id,
            active_finetune_ref=self.active_finetune_ref,
            active_execution_mode=self.active_execution_mode,
        )

    def canonical_materialized_json(self) -> str:
        return _canonical_json_text(self.materialized_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "selected_instructions": dict(self.selected_instructions),
            "selected_demos": {
                module_id: {slot_id: demo.to_dict() for slot_id, demo in slot_map.items()}
                for module_id, slot_map in self.selected_demos.items()
            },
            "selected_instruction_base_option_ids": dict(
                self.selected_instruction_base_option_ids
            ),
            "selected_instruction_transform_ids": {
                module_id: list(transform_ids)
                for module_id, transform_ids in self.selected_instruction_transform_ids.items()
            },
            "source_config": dict(self.source_config),
            "active_model_transform_id": self.active_model_transform_id,
            "active_finetune_ref": self.active_finetune_ref,
            "active_execution_mode": self.active_execution_mode,
            "parent_candidate_id": self.parent_candidate_id,
            "parent_candidate_ids": list(self.parent_candidate_ids),
            "candidate_id": self.candidate_id,
            "lever_bundle_hash": self.lever_bundle_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MiproProgramCandidate:
        raw_instructions = payload.get("selected_instructions")
        if not isinstance(raw_instructions, Mapping):
            raise ValueError("MiproProgramCandidate.selected_instructions must be an object")

        raw_selected_demos = payload.get("selected_demos") or {}
        if not isinstance(raw_selected_demos, Mapping):
            raise ValueError("MiproProgramCandidate.selected_demos must be an object")

        selected_demos: dict[str, dict[str, MiproDemo]] = {}
        for module_id, raw_slot_map in raw_selected_demos.items():
            if not isinstance(raw_slot_map, Mapping):
                raise ValueError("MiproProgramCandidate.selected_demos module payload must be an object")
            selected_demos[str(module_id)] = {
                str(slot_id): demo_from_dict(demo_payload)
                for slot_id, demo_payload in raw_slot_map.items()
                if isinstance(demo_payload, Mapping)
            }

        raw_source_config = payload.get("source_config") or {}
        if not isinstance(raw_source_config, Mapping):
            raise ValueError("MiproProgramCandidate.source_config must be an object")

        raw_base_option_ids = payload.get("selected_instruction_base_option_ids") or {}
        if not isinstance(raw_base_option_ids, Mapping):
            raise ValueError(
                "MiproProgramCandidate.selected_instruction_base_option_ids must be an object"
            )

        raw_transform_ids = payload.get("selected_instruction_transform_ids") or {}
        if not isinstance(raw_transform_ids, Mapping):
            raise ValueError(
                "MiproProgramCandidate.selected_instruction_transform_ids must be an object"
            )

        return cls(
            program_id=str(payload.get("program_id") or ""),
            selected_instructions={str(k): str(v) for k, v in raw_instructions.items()},
            selected_demos=selected_demos,
            selected_instruction_base_option_ids={
                str(k): str(v) for k, v in raw_base_option_ids.items()
            },
            selected_instruction_transform_ids={
                str(module_id): tuple(
                    str(transform_id)
                    for transform_id in transform_ids
                    if str(transform_id).strip()
                )
                for module_id, transform_ids in raw_transform_ids.items()
                if isinstance(transform_ids, (list, tuple))
            },
            source_config={str(k): str(v) for k, v in raw_source_config.items()},
            active_model_transform_id=(
                str(payload["active_model_transform_id"]).strip()
                if payload.get("active_model_transform_id") is not None
                else None
            ),
            active_finetune_ref=(
                str(payload["active_finetune_ref"]).strip()
                if payload.get("active_finetune_ref") is not None
                else None
            ),
            active_execution_mode=(
                str(payload["active_execution_mode"]).strip()
                if payload.get("active_execution_mode") is not None
                else None
            ),
            parent_candidate_id=(
                str(payload["parent_candidate_id"]).strip()
                if payload.get("parent_candidate_id") is not None
                else None
            ),
            parent_candidate_ids=tuple(
                str(parent_id)
                for parent_id in (payload.get("parent_candidate_ids") or [])
                if str(parent_id).strip()
            ),
            candidate_id=str(payload["candidate_id"]).strip()
            if payload.get("candidate_id") is not None
            else None,
            lever_bundle_hash=(
                str(payload["lever_bundle_hash"]).strip()
                if payload.get("lever_bundle_hash") is not None
                else None
            ),
        )


__all__ = [
    "DemoMessage",
    "StaticFewShotDemo",
    "TrajectorySnippetDemo",
    "MiproDemo",
    "demo_from_dict",
    "DemoSlotTemplate",
    "MiproModuleTemplate",
    "MiproProgramTemplate",
    "MiproProgramCandidate",
    "materialized_lever_bundle_payload",
    "materialized_lever_bundle_hash",
]
