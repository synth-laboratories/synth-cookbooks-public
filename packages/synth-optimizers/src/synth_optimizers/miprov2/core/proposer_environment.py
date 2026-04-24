"""Interactive and autonomous MIPRO proposer tool environment."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from synth_optimizers.miprov2.core.checkpointing import (
    compiled_space_from_snapshot,
    compiled_space_to_snapshot,
    proposer_context_from_dict,
    proposer_context_to_dict,
)
from synth_optimizers.miprov2.core.program_model import demo_from_dict
from synth_optimizers.miprov2.core.proposer_openenv import (
    MiproDemoPatch,
    MiproInstructionPatch,
    MiproOpenEnvAction,
    MiproOpenEnvProposerConfig,
    MiproOpenEnvProposerContext,
    MiproOpenEnvProposerVariant,
    build_openenv_tool_catalog,
    clone_compiled_space,
    summarize_compiled_space,
)
from synth_optimizers.miprov2.core.proposer_sessions import (
    MiproProposerEvent,
    MiproProposerSession,
    MiproProposerSessionStore,
    new_event_id,
    new_session_id,
)
from synth_optimizers.miprov2.core.proposer_tools import (
    MiproProposerToolState,
    annotate_mipro_tool,
    execute_mipro_tool,
)


def _now() -> float:
    return float(time.time())


def _coerce_variant(
    variant: MiproOpenEnvProposerVariant | dict[str, Any] | None,
) -> MiproOpenEnvProposerVariant:
    if variant is None:
        return MiproOpenEnvProposerVariant()
    if isinstance(variant, MiproOpenEnvProposerVariant):
        return variant
    return MiproOpenEnvProposerVariant.from_dict(variant)


def _instruction_patch_to_dict(patch: MiproInstructionPatch) -> dict[str, Any]:
    return {
        "module_id": patch.module_id,
        "component_key": patch.component_key,
        "option_id": patch.option_id,
        "instruction_text": patch.instruction_text,
        "base_option_id": patch.base_option_id,
        "transform_id": patch.transform_id,
        "bundle_option_ids": list(patch.bundle_option_ids),
    }


def _instruction_patch_from_dict(payload: dict[str, Any]) -> MiproInstructionPatch:
    return MiproInstructionPatch(
        module_id=str(payload.get("module_id") or ""),
        component_key=str(payload.get("component_key") or ""),
        option_id=str(payload.get("option_id") or ""),
        instruction_text=str(payload.get("instruction_text") or ""),
        base_option_id=(
            str(payload["base_option_id"]) if payload.get("base_option_id") is not None else None
        ),
        transform_id=(
            str(payload["transform_id"]) if payload.get("transform_id") is not None else None
        ),
        bundle_option_ids=tuple(str(item) for item in list(payload.get("bundle_option_ids") or [])),
    )


def _demo_patch_to_dict(patch: MiproDemoPatch) -> dict[str, Any]:
    return {
        "module_id": patch.module_id,
        "slot_id": patch.slot_id,
        "component_key": patch.component_key,
        "option_id": patch.option_id,
        "demo": patch.demo.to_dict(),
    }


def _demo_patch_from_dict(payload: dict[str, Any]) -> MiproDemoPatch:
    return MiproDemoPatch(
        module_id=str(payload.get("module_id") or ""),
        slot_id=str(payload.get("slot_id") or ""),
        component_key=str(payload.get("component_key") or ""),
        option_id=str(payload.get("option_id") or ""),
        demo=demo_from_dict(dict(payload.get("demo") or {})),
    )


def tool_state_to_dict(state: MiproProposerToolState) -> dict[str, Any]:
    return {
        "compiled_space": compiled_space_to_snapshot(state.compiled_space),
        "proposer_context": proposer_context_to_dict(state.context),
        "config": asdict(state.config),
        "instruction_patches": [
            _instruction_patch_to_dict(patch) for patch in state.instruction_patches
        ],
        "demo_patches": [_demo_patch_to_dict(patch) for patch in state.demo_patches],
        "variant": (
            state.queue_state.get("variant")
            if isinstance(state.queue_state.get("variant"), dict)
            else MiproOpenEnvProposerVariant().to_dict()
        ),
        "memory_state": dict(state.memory_state),
        "queue_state": dict(state.queue_state),
    }


def tool_state_from_dict(payload: dict[str, Any]) -> MiproProposerToolState:
    config_payload = dict(payload.get("config") or {})
    config = MiproOpenEnvProposerConfig(**config_payload) if config_payload else MiproOpenEnvProposerConfig()
    queue_state = dict(payload.get("queue_state") or {})
    if "variant" not in queue_state:
        queue_state["variant"] = dict(payload.get("variant") or {})
    return MiproProposerToolState(
        compiled_space=compiled_space_from_snapshot(dict(payload.get("compiled_space") or {})),
        context=proposer_context_from_dict(dict(payload.get("proposer_context") or {})),
        config=config,
        instruction_patches=[
            _instruction_patch_from_dict(dict(item))
            for item in list(payload.get("instruction_patches") or [])
            if isinstance(item, dict)
        ],
        demo_patches=[
            _demo_patch_from_dict(dict(item))
            for item in list(payload.get("demo_patches") or [])
            if isinstance(item, dict)
        ],
        memory_state=dict(payload.get("memory_state") or {}),
        queue_state=queue_state,
    )


@dataclass(slots=True)
class MiproProposerEnvironment:
    session: MiproProposerSession
    state: MiproProposerToolState
    store: MiproProposerSessionStore | None = None
    variant: MiproOpenEnvProposerVariant | None = None

    @classmethod
    def in_memory(
        cls,
        *,
        compiled_space: Any,
        context: MiproOpenEnvProposerContext,
        config: MiproOpenEnvProposerConfig | None = None,
        variant: MiproOpenEnvProposerVariant | dict[str, Any] | None = None,
    ) -> "MiproProposerEnvironment":
        variant_model = _coerce_variant(variant)
        session = MiproProposerSession(
            session_id=new_session_id(),
            run_id=str(context.run_metadata.get("run_id") or "") or None,
            round_idx=int(context.round_idx),
            source_kind="live",
            variant=variant_model.to_dict(),
            workspace_root=str(context.workspace_locations.get("workspace_root") or "") or None,
        )
        state = MiproProposerToolState(
            compiled_space=clone_compiled_space(compiled_space),
            context=context,
            config=config or MiproOpenEnvProposerConfig(),
            queue_state={"variant": variant_model.to_dict()},
        )
        return cls(session=session, state=state, store=None, variant=variant_model)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: dict[str, Any],
        *,
        session_root: str | Path,
        source_ref: str | None = None,
        config: MiproOpenEnvProposerConfig | None = None,
        variant: MiproOpenEnvProposerVariant | dict[str, Any] | None = None,
        actor_id: str = "interactive",
    ) -> "MiproProposerEnvironment":
        variant_model = _coerce_variant(variant)
        context = proposer_context_from_dict(dict(checkpoint.get("proposer_context") or {}))
        state = MiproProposerToolState(
            compiled_space=compiled_space_from_snapshot(dict(checkpoint.get("compiled_space") or {})),
            context=context,
            config=config or MiproOpenEnvProposerConfig(),
            queue_state={"variant": variant_model.to_dict()},
        )
        store = MiproProposerSessionStore(session_root)
        session = MiproProposerSession(
            session_id=new_session_id(),
            run_id=str(checkpoint.get("run_id") or context.run_metadata.get("run_id") or "") or None,
            round_idx=int(checkpoint.get("round_idx") or context.round_idx or 0),
            source_kind="checkpoint",
            source_ref=source_ref or str(checkpoint.get("checkpoint_id") or ""),
            variant=variant_model.to_dict(),
            workspace_root=str(context.workspace_locations.get("workspace_root") or "") or None,
            metadata={"checkpoint_id": checkpoint.get("checkpoint_id")},
        )
        state_payload = tool_state_to_dict(state)
        store.create(
            session=session,
            pre_state=state_payload,
            current_state=state_payload,
            actor_id=actor_id,
        )
        return cls(session=session, state=state, store=store, variant=variant_model)

    @classmethod
    def load(
        cls,
        *,
        session_root: str | Path,
        session_id: str,
    ) -> "MiproProposerEnvironment":
        store = MiproProposerSessionStore(session_root)
        session = store.load_session(session_id)
        state = tool_state_from_dict(store.load_current_state(session))
        variant = _coerce_variant(session.variant or state.queue_state.get("variant") or {})
        return cls(session=session, state=state, store=store, variant=variant)

    def list_tools(self) -> dict[str, Any]:
        catalog = build_openenv_tool_catalog(self.state.compiled_space, variant=self.variant)
        runtime_tools = [annotate_mipro_tool(dict(tool)) for tool in list(catalog.get("runtime_tools") or [])]
        return {
            "runtime_tools": runtime_tools,
            "tool_count": len(runtime_tools),
            "session_id": self.session.session_id,
            "current_version": int(self.session.current_version),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session.session_id,
            "status": self.session.status,
            "run_id": self.session.run_id,
            "round_idx": int(self.session.round_idx),
            "tool_count": int(self.list_tools().get("tool_count") or 0),
            "event_count": int(self.session.event_count),
            "base_version": int(self.session.base_version),
            "current_version": int(self.session.current_version),
            "compiled_space_summary": summarize_compiled_space(self.state.compiled_space),
            "compiled_space_snapshot": compiled_space_to_snapshot(self.state.compiled_space),
            "proposer_context": proposer_context_to_dict(self.state.context),
            "instruction_patch_count": len(self.state.instruction_patches),
            "demo_patch_count": len(self.state.demo_patches),
            "instruction_patches": [
                _instruction_patch_to_dict(patch) for patch in self.state.instruction_patches
            ],
            "demo_patches": [_demo_patch_to_dict(patch) for patch in self.state.demo_patches],
            "metadata": dict(self.session.metadata),
        }

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        actor_id: str = "interactive",
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        if self.session.status != "open":
            raise RuntimeError(f"proposer session is not open: {self.session.status}")
        if expected_version is not None and int(expected_version) != int(self.session.current_version):
            raise RuntimeError("proposer session version mismatch")
        action = MiproOpenEnvAction(name=str(name), arguments=dict(arguments or {}))
        version_before = int(self.session.current_version)
        result = execute_mipro_tool(action=action, state=self.state)
        if result.state_mutated:
            self.session.current_version += 1
        self.session.updated_at = _now()
        payload = result.to_dict()
        payload["state_version"] = int(self.session.current_version)
        if self.store is not None:
            with self.store.lock(self.session.session_id):
                self.store.save_current_state(self.session, tool_state_to_dict(self.state))
                self.store.append_event(
                    self.session,
                    MiproProposerEvent(
                        event_id=new_event_id(),
                        session_id=self.session.session_id,
                        created_at=_now(),
                        event_type="tool_call",
                        actor_id=actor_id,
                        tool_name=str(name),
                        arguments=dict(arguments or {}),
                        result=payload,
                        state_version_before=version_before,
                        state_version_after=int(self.session.current_version),
                        mutation_summary=dict(result.mutation_summary),
                    ),
                )
        else:
            self.session.event_count += 1
        return payload

    def checkpoint(self, *, actor_id: str = "interactive") -> dict[str, Any]:
        payload = {
            "session": self.session.to_dict(),
            "state": tool_state_to_dict(self.state),
            "snapshot": self.snapshot(),
            "created_at": _now(),
        }
        if self.store is None:
            return payload
        with self.store.lock(self.session.session_id):
            path = self.store.write_checkpoint(self.session, payload)
            self.store.append_event(
                self.session,
                MiproProposerEvent(
                    event_id=new_event_id(),
                    session_id=self.session.session_id,
                    created_at=_now(),
                    event_type="checkpoint",
                    actor_id=actor_id,
                    state_version_before=int(self.session.current_version),
                    state_version_after=int(self.session.current_version),
                    metadata={"path": str(path)},
                ),
            )
        return {"status": "ok", "checkpoint_path": str(path), "session_id": self.session.session_id}

    def commit(self, *, actor_id: str = "interactive") -> dict[str, Any]:
        state_payload = tool_state_to_dict(self.state)
        commit_payload = {
            "session_id": self.session.session_id,
            "status": "committed",
            "committed_at": _now(),
            "pre_state_ref": self.session.pre_state_ref,
            "new_instruction_patches": [
                _instruction_patch_to_dict(patch) for patch in self.state.instruction_patches
            ],
            "new_demo_patches": [_demo_patch_to_dict(patch) for patch in self.state.demo_patches],
            "event_log_path": self.session.event_log_path,
            "metadata": dict(self.session.metadata),
        }
        self.session.status = "committed"
        if self.store is not None:
            with self.store.lock(self.session.session_id):
                committed_state_path, commit_path = self.store.write_commit(
                    self.session,
                    state_payload=state_payload,
                    commit_payload=commit_payload,
                )
                commit_payload["committed_state_ref"] = str(committed_state_path)
                commit_payload["commit_ref"] = str(commit_path)
                self.store.append_event(
                    self.session,
                    MiproProposerEvent(
                        event_id=new_event_id(),
                        session_id=self.session.session_id,
                        created_at=_now(),
                        event_type="commit",
                        actor_id=actor_id,
                        result=commit_payload,
                        state_version_before=int(self.session.current_version),
                        state_version_after=int(self.session.current_version),
                    ),
                )
        return commit_payload

    def discard(self, *, actor_id: str = "interactive") -> dict[str, Any]:
        self.session.status = "discarded"
        payload = {
            "session_id": self.session.session_id,
            "status": "discarded",
            "discarded_at": _now(),
            "event_log_path": self.session.event_log_path,
        }
        if self.store is not None:
            with self.store.lock(self.session.session_id):
                self.store.save_session(self.session)
                self.store.append_event(
                    self.session,
                    MiproProposerEvent(
                        event_id=new_event_id(),
                        session_id=self.session.session_id,
                        created_at=_now(),
                        event_type="discard",
                        actor_id=actor_id,
                        result=payload,
                        state_version_before=int(self.session.current_version),
                        state_version_after=int(self.session.current_version),
                    ),
                )
        return payload
