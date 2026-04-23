"""Shared instruction-transform models and compilation helpers for MIPROv2."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast


def _require_non_empty(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


class InstructionTransformError(ValueError):
    """Raised when a transform bundle cannot be compiled safely."""


@dataclass(slots=True, frozen=True)
class InstructionTransform:
    transform_id: str
    module_id: str
    localizer_type: Literal["replace", "follow"]
    target_text: str
    replacement_text: str
    base_instruction_anchor_id: str | None = None
    priority: int = 0
    mergeable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transform_id",
            _require_non_empty(
                self.transform_id, field_name="InstructionTransform.transform_id"
            ),
        )
        object.__setattr__(
            self,
            "module_id",
            _require_non_empty(
                self.module_id, field_name="InstructionTransform.module_id"
            ),
        )
        localizer_type = str(self.localizer_type or "").strip().lower()
        if localizer_type not in {"replace", "follow"}:
            raise ValueError(
                "InstructionTransform.localizer_type must be 'replace' or 'follow'"
            )
        object.__setattr__(self, "localizer_type", localizer_type)
        object.__setattr__(
            self,
            "target_text",
            _require_non_empty(
                self.target_text, field_name="InstructionTransform.target_text"
            ),
        )
        object.__setattr__(
            self,
            "replacement_text",
            _require_non_empty(
                self.replacement_text,
                field_name="InstructionTransform.replacement_text",
            ),
        )
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(self, "mergeable", bool(self.mergeable))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.base_instruction_anchor_id is not None:
            object.__setattr__(
                self,
                "base_instruction_anchor_id",
                _require_non_empty(
                    self.base_instruction_anchor_id,
                    field_name="InstructionTransform.base_instruction_anchor_id",
                ),
            )

    def anchor_text(self) -> str:
        return self.target_text.strip()

    def semantic_signature(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "localizer_type": self.localizer_type,
            "target_text": self.target_text.strip(),
            "replacement_text": self.replacement_text.strip(),
            "base_instruction_anchor_id": self.base_instruction_anchor_id,
            "priority": int(self.priority),
            "mergeable": bool(self.mergeable),
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "module_id": self.module_id,
            "localizer_type": self.localizer_type,
            "target_text": self.target_text,
            "replacement_text": self.replacement_text,
            "base_instruction_anchor_id": self.base_instruction_anchor_id,
            "priority": self.priority,
            "mergeable": self.mergeable,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InstructionTransform:
        return cls(
            transform_id=str(payload.get("transform_id") or ""),
            module_id=str(payload.get("module_id") or ""),
            localizer_type=cast_localizer_type(payload.get("localizer_type")),
            target_text=str(payload.get("target_text") or ""),
            replacement_text=str(payload.get("replacement_text") or ""),
            base_instruction_anchor_id=(
                str(payload.get("base_instruction_anchor_id") or "").strip() or None
            ),
            priority=int(payload.get("priority") or 0),
            mergeable=bool(payload.get("mergeable", False)),
            metadata=dict(payload.get("metadata") or {})
            if isinstance(payload.get("metadata"), Mapping)
            else {},
        )


def cast_localizer_type(value: Any) -> Literal["replace", "follow"]:
    text = str(value or "").strip().lower()
    if text not in {"replace", "follow"}:
        raise ValueError("localizer_type must be 'replace' or 'follow'")
    return cast(Literal["replace", "follow"], text)


def canonical_transform_ids(transform_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(item).strip() for item in transform_ids if str(item).strip()
        )
    )


def sort_instruction_transforms(
    transforms: Sequence[InstructionTransform],
) -> tuple[InstructionTransform, ...]:
    return tuple(
        sorted(
            transforms,
            key=lambda item: (
                int(item.priority),
                0 if item.localizer_type == "replace" else 1,
                str(item.transform_id),
            ),
        )
    )


def instruction_transform_signature_text(transform: InstructionTransform) -> str:
    return json.dumps(
        transform.semantic_signature(),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _normalize_anchor(value: str) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def _find_line_index(lines: Sequence[str], target: str) -> int:
    normalized_target = target.strip()
    for idx, line in enumerate(lines):
        if line.strip() == normalized_target:
            return idx
    for idx, line in enumerate(lines):
        if line.strip().endswith(normalized_target):
            return idx
    for idx, line in enumerate(lines):
        if normalized_target in line:
            return idx
    return -1


def _apply_replace_to_line(line: str, target: str, replacement: str) -> str:
    stripped_target = target.strip()
    if line.strip() == stripped_target:
        return replacement
    if line.strip().endswith(stripped_target):
        split_pos = line.rfind(stripped_target)
        return line[:split_pos] + replacement
    if stripped_target in line:
        return line.replace(stripped_target, replacement, 1)
    return line


def compile_instruction_text(
    base_text: str,
    transforms: Sequence[InstructionTransform],
    *,
    strict: bool = True,
) -> str:
    lines = list(str(base_text).splitlines())
    sorted_transforms = sort_instruction_transforms(transforms)
    seen_signatures: set[str] = set()
    occupied_anchors: dict[str, str] = {}

    for transform in sorted_transforms:
        signature = instruction_transform_signature_text(transform)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        anchor = _normalize_anchor(transform.anchor_text())
        prior_kind = occupied_anchors.get(anchor)
        if not transform.mergeable and prior_kind is not None:
            if prior_kind == "replace" or transform.localizer_type == "replace":
                raise InstructionTransformError(
                    f"incompatible overlapping transform anchor: {transform.anchor_text()!r}"
                )
        occupied_anchors[anchor] = transform.localizer_type

        if transform.localizer_type == "replace":
            idx = _find_line_index(lines, transform.target_text)
            if idx == -1:
                if strict:
                    raise InstructionTransformError(
                        f"replace anchor not found: {transform.target_text!r}"
                    )
                continue
            lines[idx] = _apply_replace_to_line(
                lines[idx], transform.target_text, transform.replacement_text
            )
            continue

        idx = _find_line_index(lines, transform.target_text)
        if idx == -1:
            if strict:
                raise InstructionTransformError(
                    f"follow anchor not found: {transform.target_text!r}"
                )
            continue
        insertion_line = transform.replacement_text.strip()
        if any(existing.strip() == insertion_line for existing in lines):
            continue
        lines.insert(idx + 1, transform.replacement_text)

    return "\n".join(lines)


def compile_instruction_text_from_payloads(
    base_text: str,
    payloads: Sequence[Mapping[str, Any]],
    *,
    module_id: str = "module",
    base_instruction_anchor_id: str | None = None,
    strict: bool = True,
) -> str:
    transforms: list[InstructionTransform] = []
    for idx, payload in enumerate(payloads):
        localizer_type = cast_localizer_type(payload.get("type"))
        target_key = "prev_line" if localizer_type == "replace" else "line_to_follow"
        transforms.append(
            InstructionTransform(
                transform_id=f"tmp_{idx}",
                module_id=module_id,
                localizer_type=localizer_type,
                target_text=str(payload.get(target_key) or ""),
                replacement_text=str(payload.get("replacement_line") or ""),
                base_instruction_anchor_id=base_instruction_anchor_id,
                priority=int(payload.get("priority") or idx),
                mergeable=bool(payload.get("mergeable", False)),
                metadata={
                    "instruction_type": str(payload.get("instruction_type") or "")
                },
            )
        )
    return compile_instruction_text(base_text, transforms, strict=strict)


__all__ = [
    "InstructionTransform",
    "InstructionTransformError",
    "canonical_transform_ids",
    "cast_localizer_type",
    "compile_instruction_text",
    "compile_instruction_text_from_payloads",
    "instruction_transform_signature_text",
    "sort_instruction_transforms",
]
