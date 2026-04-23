"""Runtime capability requirements for MIPROv2 optimizer modes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

from synth_containers.capabilities import (
    RouteHints,
    RuntimeCapabilitySurface,
    RuntimeMetadata,
    TokenEmissionCapabilities,
)
from synth_containers.compatibility import (
    CompatibilityReport,
    ConsumerRequirement,
    ConsumerTarget,
    evaluate_runtime_requirement,
)
from synth_containers.contracts import ContainerExecutionContract
from synth_containers.ontology import (
    CapabilityLevel,
    PrimitiveProtocol,
    StatefulnessTier,
)
from synth_containers.tool_runtime import ToolRuntimeCapabilities

from synth_optimizers.miprov2.core.contracts import MiproCandidateExecutionMode


RuntimeCapabilityInput: TypeAlias = (
    RuntimeMetadata
    | RuntimeCapabilitySurface
    | ContainerExecutionContract
    | Mapping[str, Any]
)


def _runtime_surface_from_mapping(payload: Mapping[str, Any]) -> RuntimeCapabilitySurface:
    data = dict(payload)
    if isinstance(data.get("tool_runtime"), Mapping):
        data["tool_runtime"] = ToolRuntimeCapabilities(**dict(data["tool_runtime"]))
    if isinstance(data.get("token_emission"), Mapping):
        data["token_emission"] = TokenEmissionCapabilities(**dict(data["token_emission"]))
    if isinstance(data.get("route_hints"), Mapping):
        data["route_hints"] = RouteHints(**dict(data["route_hints"]))
    return RuntimeCapabilitySurface(**data)


def runtime_capability_surface(
    value: RuntimeCapabilityInput,
) -> RuntimeCapabilitySurface:
    if isinstance(value, RuntimeMetadata):
        return value.capabilities
    if isinstance(value, RuntimeCapabilitySurface):
        return value
    if isinstance(value, ContainerExecutionContract):
        return _runtime_surface_from_mapping(value.capabilities)
    return _runtime_surface_from_mapping(value)


def mipro_runtime_requirement(
    *,
    execution_mode: MiproCandidateExecutionMode | str = MiproCandidateExecutionMode.PROMPT_ONLY,
    use_proposer: bool = False,
) -> ConsumerRequirement:
    normalized_mode = (
        execution_mode
        if isinstance(execution_mode, MiproCandidateExecutionMode)
        else MiproCandidateExecutionMode(str(execution_mode))
    )
    required_protocols: list[PrimitiveProtocol] = []
    required_any_protocol_groups: list[tuple[PrimitiveProtocol, ...]] = [
        (
            PrimitiveProtocol.ROLLOUT_RUNNABLE,
            PrimitiveProtocol.ASYNC_ROLLOUT_RUNNABLE,
        ),
        (
            PrimitiveProtocol.REWARD_EMITTING,
            PrimitiveProtocol.VERIFIER_BACKED,
        ),
    ]
    requires_trace = False
    requires_token_ids = False
    requires_logprobs = False
    requires_proxied_inference = False

    if use_proposer:
        required_protocols.append(PrimitiveProtocol.TRACE_EMITTING)
        requires_trace = True

    if normalized_mode is MiproCandidateExecutionMode.TINKER_SELF_DISTILL_SFT:
        required_protocols.extend(
            (
                PrimitiveProtocol.TRACE_EMITTING,
                PrimitiveProtocol.TOKEN_TRACE_EMITTING,
                PrimitiveProtocol.PROXIED_INFERENCE_BACKED,
            )
        )
        requires_trace = True
        requires_token_ids = True
        requires_logprobs = True
        requires_proxied_inference = True

    mode_text = normalized_mode.value
    if use_proposer:
        mode_text = f"{mode_text}+proposer"
    return ConsumerRequirement(
        target=ConsumerTarget.MIPROV2,
        summary=f"MIPROv2 {mode_text} runtime requirement.",
        required_protocols=tuple(dict.fromkeys(required_protocols)),
        required_any_protocol_groups=tuple(required_any_protocol_groups),
        minimum_statefulness=StatefulnessTier.STATELESS,
        minimum_level=CapabilityLevel.DERIVED,
        requires_trace=requires_trace,
        requires_token_ids=requires_token_ids,
        requires_logprobs=requires_logprobs,
        requires_proxied_inference=requires_proxied_inference,
    )


def evaluate_mipro_runtime_support(
    metadata: RuntimeCapabilityInput,
    *,
    execution_mode: MiproCandidateExecutionMode | str = MiproCandidateExecutionMode.PROMPT_ONLY,
    use_proposer: bool = False,
) -> CompatibilityReport:
    return evaluate_runtime_requirement(
        runtime_capability_surface(metadata),
        mipro_runtime_requirement(
            execution_mode=execution_mode,
            use_proposer=use_proposer,
        ),
    )


def assert_mipro_runtime_supported(
    metadata: RuntimeCapabilityInput,
    *,
    execution_mode: MiproCandidateExecutionMode | str = MiproCandidateExecutionMode.PROMPT_ONLY,
    use_proposer: bool = False,
) -> None:
    report = evaluate_mipro_runtime_support(
        metadata,
        execution_mode=execution_mode,
        use_proposer=use_proposer,
    )
    if report.supported:
        return
    chunks: list[str] = []
    if report.missing_protocols:
        chunks.append(f"missing_protocols={','.join(report.missing_protocols)}")
    if report.missing_protocol_groups:
        chunks.append(f"missing_protocol_groups={','.join(report.missing_protocol_groups)}")
    if report.missing_features:
        chunks.append(f"missing_features={','.join(report.missing_features)}")
    raise ValueError(f"MIPROv2 runtime unsupported ({'; '.join(chunks)})")


__all__ = [
    "RuntimeCapabilityInput",
    "assert_mipro_runtime_supported",
    "evaluate_mipro_runtime_support",
    "mipro_runtime_requirement",
    "runtime_capability_surface",
]
