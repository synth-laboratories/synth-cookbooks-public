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
    DiscreteMiproOptimizer,
    MiproCandidateExecutionMode,
    MiproCompatResult,
    MiproCompatRunConfig,
    MiproEvaluationBatch,
    MiproModuleTemplate,
    MiproPhase2Config,
    MiproProgramCandidate,
    MiproProgramTemplate,
    MiproSftConfig,
    open_sqlite_run_ledger,
    run_train_loop,
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
from synth_optimizers.miprov2.gepa_ai_compat import GepaCompatAdapter, async_optimize, optimize
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
    "MiproCandidateExecutionMode": MiproCandidateExecutionMode,
    "MiproCompatResult": MiproCompatResult,
    "MiproCompatRunConfig": MiproCompatRunConfig,
    "MiproEvaluationBatch": MiproEvaluationBatch,
    "MiproModuleTemplate": MiproModuleTemplate,
    "MiproPhase2Config": MiproPhase2Config,
    "MiproProgramCandidate": MiproProgramCandidate,
    "MiproProgramTemplate": MiproProgramTemplate,
    "MiproSftConfig": MiproSftConfig,
    "open_sqlite_run_ledger": open_sqlite_run_ledger,
    "run_train_loop": run_train_loop,
    "MiproArtifactManifest": MiproArtifactManifest,
    "MiproArtifactPaths": MiproArtifactPaths,
    "MiproRunArtifactSummary": MiproRunArtifactSummary,
    "MiproReportBenchResult": MiproReportBenchResult,
    "default_miprov2_artifact_paths": default_miprov2_artifact_paths,
    "write_miprov2_artifacts": write_miprov2_artifacts,
    "GepaCompatAdapter": GepaCompatAdapter,
    "async_optimize": async_optimize,
    "optimize": optimize,
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
