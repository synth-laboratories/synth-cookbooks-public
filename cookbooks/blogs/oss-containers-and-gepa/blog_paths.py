"""Shared paths for the OSS Containers and GEPA blog cookbook."""

from __future__ import annotations

import os
from pathlib import Path

BLOG_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = BLOG_ROOT / "experiment_records"
CHARTS_DIR = BLOG_ROOT / "charts"
REPO_ROOT = BLOG_ROOT.parents[2]
EVALS_DIR = REPO_ROOT / "cookbooks" / "optimizers" / "gepa" / "evals"
EVIDENCE_DIR = EVALS_DIR / "evidence"
WORKSPACE_ROOT = REPO_ROOT.parent
_FRONTEND_ROOT_OVERRIDE = os.environ.get("GEPA_BLOG_FRONTEND_ROOT")
FRONTEND_ROOT = (
    Path(_FRONTEND_ROOT_OVERRIDE).expanduser().resolve()
    if _FRONTEND_ROOT_OVERRIDE
    else WORKSPACE_ROOT / "frontend"
)
FRONTEND_DATA_DIR = (
    FRONTEND_ROOT
    / "src"
    / "components"
    / "blog"
    / "posts"
    / "introducing-gepa-platform"
    / "data"
)


def experiment_dir(container: str, chart: str, setup: str) -> Path:
    """Path to one runnable cell folder under experiment_records/."""
    chart_slug = chart.lower().removeprefix("chart ")
    if not chart_slug.startswith("chart_"):
        chart_slug = f"chart_{chart_slug}"
    setup_slug = setup.lower().replace(" ", "_").replace("/", "_")
    return EXPERIMENTS_DIR / f"{container}__{chart_slug}__{setup_slug}"


def chart_dir(name: str) -> Path:
    """Path to a chart producer folder under charts/."""
    return CHARTS_DIR / name
