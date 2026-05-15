"""Compile typed MIPRO program templates into discrete TPE search spaces."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

from synth_optimizers.miprov2.core.instruction_transforms import (
    InstructionTransform,
    InstructionTransformError,
    canonical_transform_ids,
    compile_instruction_text,
    instruction_transform_signature_text,
)
from synth_optimizers.miprov2.core.optimizer import canonicalize_lever_bundle
from synth_optimizers.miprov2.core.program_model import (
    MiproDemo,
    MiproProgramCandidate,
    MiproProgramTemplate,
    MiproStageTemplate,
)


def instruction_component_key(module_id: str) -> str:
    return f"module:{module_id}:instruction"


def stage_component_key(stage_id: str) -> str:
    return f"stage:{stage_id}:instruction"


def demo_component_key(module_id: str, slot_id: str) -> str:
    return f"module:{module_id}:demo:{slot_id}"


def _bundle_signature_hash(
    *,
    module_id: str,
    base_option_id: str,
    transform_ids: tuple[str, ...],
) -> str:
    payload = json.dumps(
        {
            "module_id": str(module_id),
            "base_option_id": str(base_option_id),
            "transform_ids": list(transform_ids),
        },
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True, frozen=True)
class CompiledMiproSpace:
    program_template: MiproProgramTemplate
    search_space: dict[str, list[str]]
    instruction_lookup: dict[str, dict[str, str]]
    instruction_base_lookup: dict[str, dict[str, str]]
    instruction_transforms: dict[str, dict[str, InstructionTransform]]
    demo_lookup: dict[str, dict[str, MiproDemo]]
    component_order: tuple[str, ...]
    instruction_metadata: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )
    instruction_base_metadata: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )
    demo_metadata: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    # Stage-level lookup: stage_component_key → option_id → {module_id: instruction_text}
    # Populated by register_stage_candidate; used by the proposer and display layer.
    stage_instruction_lookup: dict[str, dict[str, dict[str, str]]] = field(
        default_factory=dict
    )

    def decode(self, config: Mapping[str, str]) -> MiproProgramCandidate:
        return decode_config(self, config)

    def stage_for_module(self, module_id: str) -> MiproStageTemplate | None:
        """Return the stage that owns this module_id, or None if not found."""
        for stage in self.program_template.stages:
            for m in stage.modules:
                if m.module_id == module_id:
                    return stage
        return None


def _ordered_modules(template: MiproProgramTemplate):
    return sorted(template.modules, key=lambda item: item.module_id)


def _module_template(
    compiled_space: CompiledMiproSpace,
    module_id: str,
):
    for module in compiled_space.program_template.modules:
        if module.module_id == str(module_id):
            return module
    raise ValueError(f"unknown module_id: {module_id}")


def _next_option_id(existing: Mapping[str, Any], *, prefix: str) -> str:
    max_seen = -1
    for option_id in existing.keys():
        text = str(option_id).strip()
        if not text.startswith(prefix):
            continue
        suffix = text[len(prefix) :]
        if suffix.isdigit():
            max_seen = max(max_seen, int(suffix))
    return f"{prefix}{max_seen + 1}"


def _sorted_parent_candidate_ids(*values: Any) -> tuple[str, ...]:
    raw: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            raw.extend(str(item).strip() for item in value if str(item).strip())
            continue
        text = str(value).strip()
        if text:
            raw.append(text)
    return tuple(dict.fromkeys(sorted(raw)))


def _bundle_metadata_payload(
    *,
    module_id: str,
    component_key: str,
    option_id: str,
    base_option_id: str,
    transform_ids: tuple[str, ...],
    compiled_text: str,
    origin: str,
    parent_candidate_ids: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bundle_hash = _bundle_signature_hash(
        module_id=module_id,
        base_option_id=base_option_id,
        transform_ids=transform_ids,
    )
    payload = {
        "module_id": module_id,
        "component_key": component_key,
        "option_id": option_id,
        "kind": "instruction_bundle",
        "origin": origin,
        "base_instruction_option_id": base_option_id,
        "applied_transform_ids": list(transform_ids),
        "transform_count": len(transform_ids),
        "transform_bundle_hash": bundle_hash,
        "compiled_prompt_text": compiled_text,
        "parent_candidate_ids": list(parent_candidate_ids),
        "parent_candidate_id": parent_candidate_ids[0] if parent_candidate_ids else None,
    }
    if metadata:
        payload.update(dict(metadata))
    return payload


def _register_bundle_option(
    *,
    compiled_space: CompiledMiproSpace,
    module_id: str,
    base_option_id: str,
    transform_ids: tuple[str, ...],
    compiled_text: str,
    origin: str,
    parent_candidate_ids: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
    preferred_prefix: str = "t",
) -> tuple[bool, str, str]:
    component_key = instruction_component_key(module_id)
    options = compiled_space.search_space.setdefault(component_key, [])
    lookup = compiled_space.instruction_lookup.setdefault(component_key, {})
    metadata_map = compiled_space.instruction_metadata.setdefault(component_key, {})
    canonical_ids = canonical_transform_ids(transform_ids)
    for option_id, payload in metadata_map.items():
        if (
            str(payload.get("base_instruction_option_id") or "") == str(base_option_id)
            and tuple(payload.get("applied_transform_ids") or ()) == canonical_ids
        ):
            return False, option_id, component_key
    option_id = _next_option_id(lookup, prefix=preferred_prefix)
    lookup[option_id] = str(compiled_text)
    options.append(option_id)
    metadata_map[option_id] = _bundle_metadata_payload(
        module_id=module_id,
        component_key=component_key,
        option_id=option_id,
        base_option_id=base_option_id,
        transform_ids=canonical_ids,
        compiled_text=compiled_text,
        origin=origin,
        parent_candidate_ids=parent_candidate_ids,
        metadata=metadata,
    )
    return True, option_id, component_key


def compile_search_space(template: MiproProgramTemplate) -> CompiledMiproSpace:
    search_space: dict[str, list[str]] = {}
    instruction_lookup: dict[str, dict[str, str]] = {}
    instruction_base_lookup: dict[str, dict[str, str]] = {}
    instruction_transforms: dict[str, dict[str, InstructionTransform]] = {}
    demo_lookup: dict[str, dict[str, MiproDemo]] = {}
    instruction_metadata: dict[str, dict[str, dict[str, Any]]] = {}
    instruction_base_metadata: dict[str, dict[str, dict[str, Any]]] = {}
    demo_metadata: dict[str, dict[str, dict[str, Any]]] = {}
    component_order: list[str] = []

    for module in _ordered_modules(template):
        instr_component = instruction_component_key(module.module_id)
        search_space[instr_component] = []
        instruction_lookup[instr_component] = {}
        instruction_base_lookup[instr_component] = {}
        instruction_transforms[instr_component] = {}
        instruction_metadata[instr_component] = {}
        instruction_base_metadata[instr_component] = {}
        component_order.append(instr_component)

        for idx, instruction in enumerate(module.instruction_candidates):
            option_id = f"i{idx}"
            instruction_base_lookup[instr_component][option_id] = instruction
            instruction_base_metadata[instr_component][option_id] = {
                "module_id": module.module_id,
                "component_key": instr_component,
                "option_id": option_id,
                "kind": "instruction_base",
                "origin": "template_seed",
                "max_instruction_transforms_per_candidate": int(
                    module.max_instruction_transforms_per_candidate
                ),
            }
            search_space[instr_component].append(option_id)
            instruction_lookup[instr_component][option_id] = instruction
            instruction_metadata[instr_component][option_id] = _bundle_metadata_payload(
                module_id=module.module_id,
                component_key=instr_component,
                option_id=option_id,
                base_option_id=option_id,
                transform_ids=(),
                compiled_text=instruction,
                origin="template_seed",
                metadata={
                    "max_instruction_transforms_per_candidate": int(
                        module.max_instruction_transforms_per_candidate
                    )
                },
            )

        for slot in sorted(module.demo_slots, key=lambda item: item.slot_id):
            demo_component = demo_component_key(module.module_id, slot.slot_id)
            demo_option_ids = [f"d{idx}" for idx, _ in enumerate(slot.candidate_demos)]
            search_space[demo_component] = demo_option_ids
            demo_lookup[demo_component] = {
                option_id: demo
                for option_id, demo in zip(
                    demo_option_ids, slot.candidate_demos, strict=False
                )
            }
            demo_metadata[demo_component] = {
                option_id: {
                    "module_id": module.module_id,
                    "slot_id": slot.slot_id,
                    "component_key": demo_component,
                    "option_id": option_id,
                    "kind": "demo",
                    "origin": "template_seed",
                }
                for option_id in demo_option_ids
            }
            component_order.append(demo_component)

    return CompiledMiproSpace(
        program_template=template,
        search_space=search_space,
        instruction_lookup=instruction_lookup,
        instruction_base_lookup=instruction_base_lookup,
        instruction_transforms=instruction_transforms,
        demo_lookup=demo_lookup,
        component_order=tuple(component_order),
        instruction_metadata=instruction_metadata,
        instruction_base_metadata=instruction_base_metadata,
        demo_metadata=demo_metadata,
    )


def register_instruction_candidate(
    *,
    compiled_space: CompiledMiproSpace,
    module_id: str,
    instruction_text: str,
    parent_candidate_ids: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> tuple[bool, str, str]:
    module = _module_template(compiled_space, module_id)
    component_key = instruction_component_key(module.module_id)
    base_lookup = compiled_space.instruction_base_lookup.setdefault(component_key, {})
    bundle_lookup = compiled_space.instruction_lookup.setdefault(component_key, {})
    for option_id, existing in base_lookup.items():
        if str(existing).strip() == str(instruction_text).strip():
            return False, option_id, component_key
    base_option_id = _next_option_id(base_lookup, prefix="i")
    base_lookup[base_option_id] = str(instruction_text)
    compiled_space.instruction_base_metadata.setdefault(component_key, {})[
        base_option_id
    ] = {
        "module_id": module.module_id,
        "component_key": component_key,
        "option_id": base_option_id,
        "kind": "instruction_base",
        "origin": "openenv_candidate",
        "max_instruction_transforms_per_candidate": int(
            module.max_instruction_transforms_per_candidate
        ),
        **dict(metadata or {}),
    }
    bundle_lookup[base_option_id] = str(instruction_text)
    compiled_space.search_space.setdefault(component_key, []).append(base_option_id)
    compiled_space.instruction_metadata.setdefault(component_key, {})[
        base_option_id
    ] = _bundle_metadata_payload(
        module_id=module.module_id,
        component_key=component_key,
        option_id=base_option_id,
        base_option_id=base_option_id,
        transform_ids=(),
        compiled_text=str(instruction_text),
        origin="openenv_candidate",
        parent_candidate_ids=parent_candidate_ids,
        metadata={
            "max_instruction_transforms_per_candidate": int(
                module.max_instruction_transforms_per_candidate
            ),
            **dict(metadata or {}),
        },
    )
    return True, base_option_id, component_key


def register_stage_candidate(
    *,
    compiled_space: CompiledMiproSpace,
    stage_id: str,
    module_instructions: Mapping[str, str],
) -> tuple[bool, str]:
    """Register a stage-level transform: a coordinated update to all modules in one stage.

    ``module_instructions`` maps module_id → instruction_text for every module in the stage.
    Returns (registered, option_id) where registered=False means an identical stage option
    already existed and option_id is the existing one.

    Internally, each per-module instruction is also registered in the per-module lookup via
    ``register_instruction_candidate`` so TPE can score modules individually if desired.
    All per-module registrations must succeed (or be deduped to the same option_id) for the
    stage option to be accepted.
    """
    stage_key = stage_component_key(stage_id)
    stage_lookup = compiled_space.stage_instruction_lookup.setdefault(stage_key, {})

    # Dedup: if an existing stage option has identical per-module texts, reuse it
    for existing_oid, existing_modules in stage_lookup.items():
        if all(
            str(existing_modules.get(mid, "")).strip() == str(text).strip()
            for mid, text in module_instructions.items()
        ) and set(existing_modules) == set(module_instructions):
            return False, existing_oid

    # Register each per-module instruction into the flat per-module search space
    per_module_option_ids: dict[str, str] = {}
    any_new = False
    for mid, text in module_instructions.items():
        registered, opt_id, _ = register_instruction_candidate(
            compiled_space=compiled_space,
            module_id=mid,
            instruction_text=text,
        )
        per_module_option_ids[mid] = opt_id
        if registered:
            any_new = True

    # Assign a new stage option_id
    new_stage_oid = _next_option_id(stage_lookup, prefix="s")
    stage_lookup[new_stage_oid] = {
        **{mid: text for mid, text in module_instructions.items()},
        "__per_module_option_ids__": per_module_option_ids,  # type: ignore[assignment]
    }
    return True, new_stage_oid


def register_instruction_transform(
    *,
    compiled_space: CompiledMiproSpace,
    module_id: str,
    target_option_id: str,
    transform_payloads: list[dict[str, Any]],
    parent_candidate_ids: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    module = _module_template(compiled_space, module_id)
    component_key = instruction_component_key(module.module_id)
    bundle_lookup = compiled_space.instruction_lookup.get(component_key) or {}
    bundle_metadata = compiled_space.instruction_metadata.get(component_key) or {}
    base_lookup = compiled_space.instruction_base_lookup.get(component_key) or {}
    if str(target_option_id) not in bundle_lookup:
        raise ValueError(
            f"base_option_id='{target_option_id}' not found in component '{component_key}'"
        )
    if not isinstance(transform_payloads, list) or not transform_payloads:
        raise ValueError("transforms must be a non-empty list")

    target_metadata = dict(bundle_metadata.get(str(target_option_id)) or {})
    base_option_id = str(
        target_metadata.get("base_instruction_option_id") or target_option_id
    ).strip()
    base_text = base_lookup.get(base_option_id)
    if base_text is None:
        raise ValueError(
            f"base instruction '{base_option_id}' not found for component '{component_key}'"
        )

    registry = compiled_space.instruction_transforms.setdefault(component_key, {})
    planned_registry: dict[str, InstructionTransform] = dict(registry)
    existing_target_transform_ids = canonical_transform_ids(
        target_metadata.get("applied_transform_ids") or ()
    )
    new_transform_ids: list[str] = []
    duplicate_transform_ids: list[str] = []
    pending_transforms: list[InstructionTransform] = []
    for idx, payload in enumerate(transform_payloads):
        localizer_type = str(payload.get("type") or "").strip().lower()
        if localizer_type not in {"replace", "follow"}:
            raise ValueError(
                f"unknown transform type: {localizer_type!r} (expected 'replace' or 'follow')"
            )
        localizer_type = cast(Any, localizer_type)
        target_key = "prev_line" if localizer_type == "replace" else "line_to_follow"
        candidate_transform = InstructionTransform(
            transform_id=_next_option_id(planned_registry, prefix="x"),
            module_id=module.module_id,
            localizer_type=localizer_type,
            target_text=str(payload.get(target_key) or ""),
            replacement_text=str(payload.get("replacement_line") or ""),
            base_instruction_anchor_id=base_option_id,
            priority=int(payload.get("priority") or idx),
            mergeable=bool(payload.get("mergeable", False)),
            metadata={
                "instruction_type": str(payload.get("instruction_type") or "").strip()
                or "other",
                **dict(metadata or {}),
            },
        )
        duplicate_id = None
        signature = instruction_transform_signature_text(candidate_transform)
        for transform_id, existing_transform in planned_registry.items():
            if instruction_transform_signature_text(existing_transform) == signature:
                duplicate_id = transform_id
                break
        if duplicate_id is not None:
            duplicate_transform_ids.append(duplicate_id)
            new_transform_ids.append(duplicate_id)
            continue
        planned_registry[candidate_transform.transform_id] = candidate_transform
        pending_transforms.append(candidate_transform)
        new_transform_ids.append(candidate_transform.transform_id)

    if not new_transform_ids:
        raise ValueError("no usable transforms were provided")

    planned_bundle_specs: list[dict[str, Any]] = []
    primary_option_id: str | None = None
    canonical_new_transform_ids = canonical_transform_ids(new_transform_ids)
    existing_bundle_items = list(
        (option_id, dict(payload))
        for option_id, payload in bundle_metadata.items()
        if str(payload.get("base_instruction_option_id") or "") == base_option_id
    )
    for option_id, payload in existing_bundle_items:
        current_transform_ids = canonical_transform_ids(
            payload.get("applied_transform_ids") or ()
        )
        if len(current_transform_ids) >= int(
            module.max_instruction_transforms_per_candidate
        ):
            continue
        if any(transform_id in current_transform_ids for transform_id in canonical_new_transform_ids):
            continue
        candidate_transform_ids = canonical_transform_ids(
            (*current_transform_ids, *canonical_new_transform_ids)
        )
        if len(candidate_transform_ids) > int(
            module.max_instruction_transforms_per_candidate
        ):
            continue
        try:
            compiled_text = compile_instruction_text(
                str(base_text),
                [planned_registry[transform_id] for transform_id in candidate_transform_ids],
                strict=True,
            )
        except InstructionTransformError:
            continue
        planned_bundle_specs.append(
            {
                "module_id": module.module_id,
                "base_option_id": base_option_id,
                "transform_ids": candidate_transform_ids,
                "compiled_text": compiled_text,
                "origin": "openenv_transform_bundle",
                "parent_candidate_ids": _sorted_parent_candidate_ids(
                    parent_candidate_ids,
                    payload.get("parent_candidate_ids"),
                ),
                "metadata": {
                    "max_instruction_transforms_per_candidate": int(
                        module.max_instruction_transforms_per_candidate
                    ),
                    "source_option_id": option_id,
                    **dict(metadata or {}),
                },
                "source_option_id": option_id,
            }
        )
        if option_id == str(target_option_id):
            primary_option_id = "__pending__"

    if primary_option_id is None or (
        len(existing_target_transform_ids)
        >= int(module.max_instruction_transforms_per_candidate)
        and not all(
            transform_id in existing_target_transform_ids
            for transform_id in canonical_new_transform_ids
        )
    ):
        raise ValueError("transform bundle could not be compiled against the target option")

    for transform in pending_transforms:
        registry[transform.transform_id] = transform

    created_bundle_option_ids: list[str] = []
    primary_created_option_id: str | None = None
    for spec in planned_bundle_specs:
        added, bundle_option_id, _component_key = _register_bundle_option(
            compiled_space=compiled_space,
            module_id=str(spec["module_id"]),
            base_option_id=str(spec["base_option_id"]),
            transform_ids=tuple(spec["transform_ids"]),
            compiled_text=str(spec["compiled_text"]),
            origin=str(spec["origin"]),
            parent_candidate_ids=tuple(spec["parent_candidate_ids"]),
            metadata=dict(spec["metadata"]),
        )
        if added:
            created_bundle_option_ids.append(bundle_option_id)
        if str(spec["source_option_id"]) == str(target_option_id):
            primary_created_option_id = bundle_option_id

    return {
        "added": len(created_bundle_option_ids) > 0,
        "component_key": component_key,
        "transform_ids": list(canonical_new_transform_ids),
        "duplicate_transform_ids": duplicate_transform_ids,
        "primary_option_id": primary_created_option_id,
        "created_bundle_option_ids": created_bundle_option_ids,
        "base_instruction_option_id": base_option_id,
    }


def list_registered_instruction_transforms(
    compiled_space: CompiledMiproSpace,
    *,
    module_id: str,
) -> list[dict[str, Any]]:
    component_key = instruction_component_key(module_id)
    registry = compiled_space.instruction_transforms.get(component_key) or {}
    return [transform.to_dict() for _, transform in sorted(registry.items())]


def query_instruction_transform_compatibility(
    compiled_space: CompiledMiproSpace,
    *,
    module_id: str,
    base_option_id: str,
    transform_ids: list[str],
) -> dict[str, Any]:
    component_key = instruction_component_key(module_id)
    base_lookup = compiled_space.instruction_base_lookup.get(component_key) or {}
    registry = compiled_space.instruction_transforms.get(component_key) or {}
    base_text = base_lookup.get(str(base_option_id))
    if base_text is None:
        raise ValueError(
            f"base instruction '{base_option_id}' not found for component '{component_key}'"
        )
    ordered_ids = canonical_transform_ids(transform_ids)
    transforms = []
    for transform_id in ordered_ids:
        transform = registry.get(transform_id)
        if transform is None:
            raise ValueError(f"unknown transform_id '{transform_id}'")
        transforms.append(transform)
    try:
        compiled_text = compile_instruction_text(base_text, transforms, strict=True)
    except InstructionTransformError as exc:
        return {
            "status": "incompatible",
            "component_key": component_key,
            "base_instruction_option_id": str(base_option_id),
            "transform_ids": list(ordered_ids),
            "reason": str(exc),
        }
    return {
        "status": "compatible",
        "component_key": component_key,
        "base_instruction_option_id": str(base_option_id),
        "transform_ids": list(ordered_ids),
        "compiled_prompt_preview": compiled_text[:240],
    }


def config_option_metadata(
    compiled_space: CompiledMiproSpace,
    config: Mapping[str, str],
) -> dict[str, Any]:
    normalized = canonicalize_lever_bundle(config)
    option_metadata: list[dict[str, Any]] = []
    parent_candidate_ids: set[str] = set()

    for component_key, option_id in normalized.items():
        metadata_map = compiled_space.instruction_metadata.get(component_key)
        if metadata_map is None:
            metadata_map = compiled_space.demo_metadata.get(component_key)
        if metadata_map is None:
            continue
        payload = metadata_map.get(str(option_id))
        if not isinstance(payload, Mapping):
            continue
        materialized = deepcopy(dict(payload))
        materialized["component_key"] = str(component_key)
        materialized["option_id"] = str(option_id)
        option_metadata.append(materialized)
        for parent_id in _sorted_parent_candidate_ids(
            payload.get("parent_candidate_ids"),
            payload.get("parent_candidate_id"),
        ):
            parent_candidate_ids.add(parent_id)

    sorted_parent_ids = sorted(parent_candidate_ids)
    return {
        "option_metadata": option_metadata,
        "parent_candidate_ids": sorted_parent_ids,
        "parent_candidate_id": (
            sorted_parent_ids[0] if len(sorted_parent_ids) == 1 else None
        ),
    }


def decode_config(
    compiled_space: CompiledMiproSpace,
    config: Mapping[str, str],
) -> MiproProgramCandidate:
    normalized = canonicalize_lever_bundle(config)
    resolved_metadata = config_option_metadata(compiled_space, normalized)
    selected_instructions: dict[str, str] = {}
    selected_instruction_base_option_ids: dict[str, str] = {}
    selected_instruction_transform_ids: dict[str, tuple[str, ...]] = {}
    selected_demos: dict[str, dict[str, MiproDemo]] = {}

    for module in _ordered_modules(compiled_space.program_template):
        instr_component = instruction_component_key(module.module_id)
        instr_option = normalized.get(instr_component)
        if instr_option is None:
            raise ValueError(f"missing config option for component '{instr_component}'")
        instruction = compiled_space.instruction_lookup.get(instr_component, {}).get(
            instr_option
        )
        if instruction is None:
            raise ValueError(
                f"invalid option '{instr_option}' for component '{instr_component}'"
            )
        selected_instructions[module.module_id] = instruction
        bundle_metadata = (
            compiled_space.instruction_metadata.get(instr_component, {}).get(instr_option)
            or {}
        )
        selected_instruction_base_option_ids[module.module_id] = str(
            bundle_metadata.get("base_instruction_option_id") or instr_option
        )
        selected_instruction_transform_ids[module.module_id] = tuple(
            str(transform_id)
            for transform_id in (bundle_metadata.get("applied_transform_ids") or [])
            if str(transform_id).strip()
        )

        module_demos: dict[str, MiproDemo] = {}
        for slot in sorted(module.demo_slots, key=lambda item: item.slot_id):
            demo_component = demo_component_key(module.module_id, slot.slot_id)
            demo_option = normalized.get(demo_component)
            if demo_option is None:
                raise ValueError(
                    f"missing config option for component '{demo_component}'"
                )
            demo_payload = compiled_space.demo_lookup.get(demo_component, {}).get(
                demo_option
            )
            if demo_payload is None:
                raise ValueError(
                    f"invalid option '{demo_option}' for component '{demo_component}'"
                )
            module_demos[slot.slot_id] = demo_payload
        if module_demos:
            selected_demos[module.module_id] = module_demos

    filtered_source = {
        component: option
        for component, option in normalized.items()
        if component in compiled_space.search_space
    }
    return MiproProgramCandidate(
        program_id=compiled_space.program_template.program_id,
        selected_instructions=selected_instructions,
        selected_demos=selected_demos,
        selected_instruction_base_option_ids=selected_instruction_base_option_ids,
        selected_instruction_transform_ids=selected_instruction_transform_ids,
        source_config=filtered_source,
        parent_candidate_id=cast(
            str | None, resolved_metadata.get("parent_candidate_id")
        ),
        parent_candidate_ids=tuple(
            str(parent_id)
            for parent_id in (resolved_metadata.get("parent_candidate_ids") or [])
        ),
    )


__all__ = [
    "CompiledMiproSpace",
    "compile_search_space",
    "config_option_metadata",
    "decode_config",
    "demo_component_key",
    "instruction_component_key",
    "list_registered_instruction_transforms",
    "query_instruction_transform_compatibility",
    "register_instruction_candidate",
    "register_instruction_transform",
]
