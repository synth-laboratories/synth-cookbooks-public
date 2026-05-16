"""Public package facade for synth-lab MIPROv2.

This module exposes an explicit supported API surface. Legacy names remain available
via a temporary compatibility shim and emit deprecation warnings.
"""

from __future__ import annotations

import warnings
from typing import Any

from synth_optimizers.miprov2 import core as _core
from synth_optimizers.miprov2.artifacts import (
    MiproArtifactManifest,
    MiproArtifactPaths,
    MiproReportBenchResult,
    MiproRunArtifactSummary,
    default_miprov2_artifact_paths,
    write_miprov2_artifacts,
)
from synth_optimizers.miprov2.core import (
    CHECKPOINT_STAGE_BEFORE_PROPOSER,
    DiscreteMiproOptimizer,
    MiproCandidateExecutionMode,
    MiproCompatResult,
    MiproCompatRunConfig,
    MiproEvaluationBatch,
    MiproModuleTemplate,
    MiproOpenEnvProposerVariant,
    MiproPhase2Config,
    MiproProgramCandidate,
    MiproProgramTemplate,
    MiproSftConfig,
    compiled_space_from_snapshot,
    compiled_space_to_snapshot,
    export_candidate_train_scores_from_ledger,
    load_proposer_checkpoint,
    open_sqlite_run_ledger,
    proposed_candidates_from_outcome,
    proposer_replay_summary,
    run_proposer_from_checkpoint,
    run_train_loop,
    write_proposer_checkpoint,
)
from synth_optimizers.miprov2.container_adapter import (
    ContainerInterceptorAdapter,
    ContainerMiproAdapter,
    ContainerMiproInterceptorAdapter,
    ContainerMiproRolloutBinding,
    program_template_from_prompt_contract,
)
from synth_optimizers.miprov2.dspy_compat import (
    DspyBudgetSemantics,
    DspyMiproAdapter,
    DspyProgramCandidateModel,
    DspyProgramComponent,
    extract_dspy_candidate_model,
    optimize_dspy_program,
    resolve_dspy_budget,
)
from synth_optimizers.miprov2.mipro_compat import MiproCompatAdapter, async_optimize, optimize
from synth_optimizers.miprov2.local_interceptor import (
    InMemoryInterceptorTrialRegistry,
    InterceptorTrialRecord,
    apply_interceptor_deltas,
    create_local_interceptor_app,
    register_local_forward_app,
)
from synth_optimizers.miprov2.requirements import (
    assert_mipro_runtime_supported,
    evaluate_mipro_runtime_support,
    mipro_runtime_requirement,
    runtime_capability_surface,
)

__version__ = "0.1.1"

_SUPPORTED_PUBLIC_API = {
    "__version__": __version__,
    "DiscreteMiproOptimizer": DiscreteMiproOptimizer,
    "CHECKPOINT_STAGE_BEFORE_PROPOSER": CHECKPOINT_STAGE_BEFORE_PROPOSER,
    "MiproCandidateExecutionMode": MiproCandidateExecutionMode,
    "MiproCompatResult": MiproCompatResult,
    "MiproCompatRunConfig": MiproCompatRunConfig,
    "MiproEvaluationBatch": MiproEvaluationBatch,
    "MiproModuleTemplate": MiproModuleTemplate,
    "MiproOpenEnvProposerVariant": MiproOpenEnvProposerVariant,
    "MiproPhase2Config": MiproPhase2Config,
    "MiproProgramCandidate": MiproProgramCandidate,
    "MiproProgramTemplate": MiproProgramTemplate,
    "MiproSftConfig": MiproSftConfig,
    "compiled_space_from_snapshot": compiled_space_from_snapshot,
    "compiled_space_to_snapshot": compiled_space_to_snapshot,
    "export_candidate_train_scores_from_ledger": export_candidate_train_scores_from_ledger,
    "load_proposer_checkpoint": load_proposer_checkpoint,
    "proposed_candidates_from_outcome": proposed_candidates_from_outcome,
    "proposer_replay_summary": proposer_replay_summary,
    "run_proposer_from_checkpoint": run_proposer_from_checkpoint,
    "write_proposer_checkpoint": write_proposer_checkpoint,
    "ContainerInterceptorAdapter": ContainerInterceptorAdapter,
    "ContainerMiproAdapter": ContainerMiproAdapter,
    "ContainerMiproInterceptorAdapter": ContainerMiproInterceptorAdapter,
    "ContainerMiproRolloutBinding": ContainerMiproRolloutBinding,
    "program_template_from_prompt_contract": program_template_from_prompt_contract,
    "open_sqlite_run_ledger": open_sqlite_run_ledger,
    "run_train_loop": run_train_loop,
    "MiproArtifactManifest": MiproArtifactManifest,
    "MiproArtifactPaths": MiproArtifactPaths,
    "MiproRunArtifactSummary": MiproRunArtifactSummary,
    "MiproReportBenchResult": MiproReportBenchResult,
    "default_miprov2_artifact_paths": default_miprov2_artifact_paths,
    "write_miprov2_artifacts": write_miprov2_artifacts,
    "MiproCompatAdapter": MiproCompatAdapter,
    "async_optimize": async_optimize,
    "optimize": optimize,
    "InMemoryInterceptorTrialRegistry": InMemoryInterceptorTrialRegistry,
    "InterceptorTrialRecord": InterceptorTrialRecord,
    "apply_interceptor_deltas": apply_interceptor_deltas,
    "create_local_interceptor_app": create_local_interceptor_app,
    "register_local_forward_app": register_local_forward_app,
    "DspyBudgetSemantics": DspyBudgetSemantics,
    "DspyMiproAdapter": DspyMiproAdapter,
    "DspyProgramCandidateModel": DspyProgramCandidateModel,
    "DspyProgramComponent": DspyProgramComponent,
    "extract_dspy_candidate_model": extract_dspy_candidate_model,
    "optimize_dspy_program": optimize_dspy_program,
    "resolve_dspy_budget": resolve_dspy_budget,
    "assert_mipro_runtime_supported": assert_mipro_runtime_supported,
    "evaluate_mipro_runtime_support": evaluate_mipro_runtime_support,
    "mipro_runtime_requirement": mipro_runtime_requirement,
    "runtime_capability_surface": runtime_capability_surface,
}

_LEGACY_COMPAT_EXPORTS = {
    name
    for name in getattr(_core, "__all__", [])
    if name not in _SUPPORTED_PUBLIC_API
}

__all__ = sorted(_SUPPORTED_PUBLIC_API)


def __getattr__(name: str) -> Any:
    if name in _LEGACY_COMPAT_EXPORTS:
        warnings.warn(
            f"`synth_optimizers.miprov2.{name}` is deprecated and will be removed after one release window. "
            "Import from `synth_optimizers.miprov2.core` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_core, name)
    raise AttributeError(name)
