"""The unit of a blog-post experiment.

A blog chart is a *view*; the durable unit underneath it is an `Experiment`: one
task container compared across a set of arms, under a single locked set of parity
conditions, producing evidence the charts consume.

This package formalizes that unit so the launch audit is *executable* instead of
hand-maintained in the README. Each `Experiment` declares its `ParityLock` and
then computes its own `Verdict` (FINISHED / REPLUMB / RERUN / MISSING) by reading
the real evidence files — the same checks we otherwise run by eye:

  - evidence present                 (benchmarks/<c>/summary.json exists)
  - wired into the live aggregate     (rows in evals/evidence/*.jsonl, not a backup)
  - budget floor                      (per-arm candidate and rollout counts pass the floor)
  - parity checks                     (recorded proposer/policy/route == the lock)

Run `uv run python -m experiment_unit` from the blog folder for the status table.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Modules here import `blog_paths` (one level up); make it importable regardless
# of how the package is invoked.
_BLOG_ROOT = Path(__file__).resolve().parent.parent
if str(_BLOG_ROOT) not in sys.path:
    sys.path.insert(0, str(_BLOG_ROOT))

from .model import (  # noqa: E402
    CheckResult,
    CheckStatus,
    Experiment,
    ExperimentKind,
    ParityLock,
    Verdict,
)
from .registry import REGISTRY, in_scope  # noqa: E402

__all__ = [
    "CheckResult",
    "CheckStatus",
    "Experiment",
    "ExperimentKind",
    "ParityLock",
    "Verdict",
    "REGISTRY",
    "in_scope",
]
