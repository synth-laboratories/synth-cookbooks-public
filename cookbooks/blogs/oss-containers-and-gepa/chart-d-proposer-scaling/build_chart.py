#!/usr/bin/env python3
"""Build Chart D: proposer scaling figures from run manifests.

Scans runs/ for result_manifest.json files, groups by task × proposer model,
and emits:
  figures/proposer_scaling_data.json  - normalized data table
  frontend/.../data/proposer_scaling_data.json - blog data mirror
  figures/proposer_scaling.md         - markdown table
  figures/proposer_scaling.svg        - grouped bar chart (two panels, one per task)
  figures/source_evidence.json        - checksums for each public manifest snapshot used
  figures/manifest_snapshots/*.json   - trackable result_manifest snapshots

The launch build fails if any cell is missing core evidence.

Expected run directory layout:
  runs/<task>_<model_slug>/<run_id>/result_manifest.json

Model slugs: nano, mini, gpt54
Tasks:       healthbench, tau2_retail
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
WORKSPACE = Path("/Users/joshpurtell/Documents/GitHub")
EVIDENCE_SUMMARY_ROOT = (
    REPO_ROOT
    / "cookbooks"
    / "optimizers"
    / "gepa"
    / "evals"
    / "evidence"
    / "benchmarks"
)
FRONTEND_OUT = (
    WORKSPACE
    / "frontend"
    / "src"
    / "components"
    / "blog"
    / "posts"
    / "introducing-gepa-platform"
    / "data"
    / "proposer_scaling_data.json"
)
MANIFEST_SNAPSHOT_DIR = ROOT / "figures" / "manifest_snapshots"

# ---------------------------------------------------------------------------
# Experiment matrix (locked)
# ---------------------------------------------------------------------------

TASKS = ["healthbench", "tau2_retail"]
RUN_GROUP = "final_20260603"

# Display labels for SVG panel titles.
TASK_LABELS = {"healthbench": "HealthBench", "tau2_retail": "tau2 retail"}

PROPOSER_MODELS = [
    {"slug": "nano",  "label": "gpt-5.4-nano",  "model": "gpt-5.4-nano",  "reasoning_effort": "low"},
    {"slug": "mini",  "label": "gpt-5.4-mini",  "model": "gpt-5.4-mini",  "reasoning_effort": "medium"},
    {"slug": "gpt54", "label": "gpt-5.4",        "model": "gpt-5.4",        "reasoning_effort": "high"},
]

EXPECTED_POLICY_MODELS: dict[str, str] = {
    "healthbench": "google/gemini-2.5-flash-lite",
    "tau2_retail": "openrouter/google/gemini-3.1-flash-lite",
}

REQUIRED_CELL_FIELDS = (
    "seed_reward",
    "best_heldout_reward",
    "proposer_calls",
    "proposer_tokens",
    "total_cost_usd",
)

_SEED_REWARD_CACHE: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manifest_path(task: str, slug: str) -> Path:
    run_root = ROOT / "runs" / RUN_GROUP / f"{task}_{slug}"
    direct = run_root / "result_manifest.json"
    if direct.exists():
        return direct

    manifests = sorted(
        run_root.glob("*/result_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if manifests:
        return manifests[0]

    return direct


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _first_number(*values: Any) -> int | float | None:
    for value in values:
        if _is_number(value):
            return value
    return None


def _seed_reward(task: str) -> float:
    cached = _SEED_REWARD_CACHE.get(task)
    if cached is not None:
        return cached

    summary_path = EVIDENCE_SUMMARY_ROOT / task / "summary.json"
    try:
        value = _read_json(summary_path)["per_stack"]["synth_gepa"]["seed_heldout_score"]
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Chart D missing seed baseline for {task}: {summary_path}") from exc

    if not _is_number(value):
        raise SystemExit(f"Chart D seed baseline for {task} is not numeric: {value!r}")

    _SEED_REWARD_CACHE[task] = float(value)
    return float(value)


def _collect_model_values(value: Any) -> set[str]:
    models: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"agent_model", "policy_model"} and isinstance(child, str):
                models.add(child)
            elif key == "policy" and isinstance(child, dict) and isinstance(child.get("model"), str):
                models.add(child["model"])
            models.update(_collect_model_values(child))
    elif isinstance(value, list):
        for child in value:
            models.update(_collect_model_values(child))
    return models


def _policy_mismatch_note(task: str, manifest: dict[str, Any]) -> str | None:
    expected = EXPECTED_POLICY_MODELS.get(task)
    if expected is None:
        return None
    observed = _collect_model_values(manifest)
    if expected in observed:
        return None
    observed_label = ", ".join(sorted(observed)) if observed else "unknown"
    return f"stale policy model: expected {expected}; observed {observed_label}"


def _extract_cell(task: str, proposer: dict[str, str], manifest: dict[str, Any] | None) -> dict[str, Any]:
    slug = proposer["slug"]
    label = proposer["label"]
    auth_mode = "api_key" if slug == "nano" else "chatgpt"
    seed_reward = _seed_reward(task)

    if manifest is None:
        return {
            "task": task,
            "proposer_slug": slug,
            "proposer_label": label,
            "proposer_auth_mode": auth_mode,
            "status": "missing",
            "seed_reward": seed_reward,
            "best_heldout_reward": None,
            "proposer_calls": None,
            "proposer_tokens": None,
            "total_cost_usd": None,
            "notes": "run manifest missing",
        }

    mismatch_note = _policy_mismatch_note(task, manifest)
    if mismatch_note is not None:
        return {
            "task": task,
            "proposer_slug": slug,
            "proposer_label": label,
            "proposer_auth_mode": auth_mode,
            "status": "stale_policy_model",
            "seed_reward": seed_reward,
            "best_heldout_reward": None,
            "proposer_calls": None,
            "proposer_tokens": None,
            "total_cost_usd": None,
            "notes": mismatch_note,
        }

    best = manifest.get("best_candidate") or {}
    usage = manifest.get("usage") or {}
    run_meta = manifest.get("run") or {}

    best_heldout = best.get("heldout_reward")
    proposer_calls = usage.get("proposer_calls")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        total_tokens = prompt_tokens + completion_tokens
    total_cost_usd = _first_number(
        run_meta.get("total_cost_usd"),
        manifest.get("cost_usd"),
        usage.get("cost_usd"),
    )
    wall_clock_s = _first_number(run_meta.get("wall_clock_s"), manifest.get("wall_clock_s"))

    cell = {
        "task": task,
        "proposer_slug": slug,
        "proposer_label": label,
        "proposer_auth_mode": auth_mode,
        "status": "completed",
        "seed_reward": seed_reward,
        "best_heldout_reward": best_heldout,
        "proposer_calls": proposer_calls,
        "proposer_tokens": total_tokens,
        "total_cost_usd": total_cost_usd,
        "notes": "",
    }
    if wall_clock_s is not None:
        cell["wall_clock_s"] = wall_clock_s
    return cell


def _source_ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT)) if path.exists() else str(path),
        "sha256": _sha256(path) if path.exists() else None,
        "bytes": path.stat().st_size if path.exists() else None,
        "exists": path.exists(),
    }


def _manifest_snapshot_path(task: str, slug: str) -> Path:
    return MANIFEST_SNAPSHOT_DIR / f"{task}_{slug}.result_manifest.json"


def _write_manifest_snapshot(task: str, slug: str, source_path: Path) -> Path:
    snapshot_path = _manifest_snapshot_path(task, slug)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(source_path.read_bytes())
    return snapshot_path


def _assert_launch_ready(cells: list[dict[str, Any]]) -> None:
    failures: list[str] = []
    for cell in cells:
        label = f"{cell.get('task')} x {cell.get('proposer_label')}"
        if cell.get("status") != "completed":
            failures.append(f"{label}: status={cell.get('status')} notes={cell.get('notes')!r}")
        for field in REQUIRED_CELL_FIELDS:
            if not _is_number(cell.get(field)):
                failures.append(f"{label}: {field}={cell.get(field)!r}")

    if failures:
        joined = "\n- ".join(failures)
        raise SystemExit(f"Chart D launch data incomplete:\n- {joined}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_markdown(cells: list[dict[str, Any]]) -> str:
    by_key = {(c["task"], c["proposer_slug"]): c for c in cells}
    lines = [
        "| task | proposer | seed | best heldout | proposer calls | total tokens | notes |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for task in TASKS:
        for proposer in PROPOSER_MODELS:
            slug = proposer["slug"]
            label = proposer["label"]
            cell = by_key.get((task, slug), {})
            seed_r = cell.get("seed_reward")
            heldout_r = cell.get("best_heldout_reward")
            calls = cell.get("proposer_calls")
            tokens = cell.get("proposer_tokens")
            notes = cell.get("notes") or ""
            lines.append(
                "| "
                + " | ".join([
                    task,
                    label,
                    "—" if seed_r is None else f"{float(seed_r):.3f}",
                    "—" if heldout_r is None else f"{float(heldout_r):.3f}",
                    "—" if calls is None else str(int(calls)),
                    "—" if tokens is None else str(int(tokens)),
                    notes,
                ])
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def render_svg(cells: list[dict[str, Any]]) -> str:
    """Two side-by-side grouped bar charts, one per task."""
    by_key = {(c["task"], c["proposer_slug"]): c for c in cells}
    proposer_labels = [p["label"] for p in PROPOSER_MODELS]
    proposer_slugs = [p["slug"] for p in PROPOSER_MODELS]

    # Chart geometry
    panel_w = 340
    panel_gap = 40
    total_w = len(TASKS) * panel_w + (len(TASKS) - 1) * panel_gap + 120
    height = 360
    left_margin = 60
    top = 56
    chart_h = 210
    bar_w = 56
    bar_gap = 14
    group_w = len(PROPOSER_MODELS) * (bar_w + bar_gap) - bar_gap

    colors = ["#f97316", "#f4c542", "#38bdf8"]  # nano, mini, gpt-5.4

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{height}" viewBox="0 0 {total_w} {height}">',
        '<rect width="100%" height="100%" fill="#151210"/>',
        f'<text x="24" y="32" fill="#f7efe8" font-family="Inter, sans-serif" font-size="16" font-weight="600">Chart D — Proposer Scaling: heldout reward vs proposer model</text>',
    ]

    # Y-axis tick lines (shared across panels)
    for panel_idx, task in enumerate(TASKS):
        panel_x = left_margin + panel_idx * (panel_w + panel_gap)
        svg.append(
            f'<text x="{panel_x + group_w // 2}" y="{top - 10}" fill="#d8c9bd" '
            f'font-family="Inter, sans-serif" font-size="13" text-anchor="middle">{TASK_LABELS.get(task, task)}</text>'
        )
        for tick_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = top + chart_h - tick_val * chart_h
            svg.append(
                f'<line x1="{panel_x - 6}" x2="{panel_x + group_w + 20}" '
                f'y1="{y:.1f}" y2="{y:.1f}" stroke="#302821" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{panel_x - 10}" y="{y + 4:.1f}" fill="#b9aaa0" '
                f'font-family="Inter, sans-serif" font-size="10" text-anchor="end">{tick_val:.2f}</text>'
            )

        # Bars
        for j, (slug, label, color) in enumerate(zip(proposer_slugs, proposer_labels, colors)):
            cell = by_key.get((task, slug), {})
            reward = cell.get("best_heldout_reward")
            x = panel_x + j * (bar_w + bar_gap)

            if reward is not None:
                bar_h = float(reward) * chart_h
                y = top + chart_h - bar_h
                svg.append(
                    f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" '
                    f'fill="{color}" rx="3"/>'
                )
                svg.append(
                    f'<text x="{x + bar_w // 2}" y="{y - 5:.1f}" fill="#f7efe8" '
                    f'font-family="Inter, sans-serif" font-size="10" text-anchor="middle">'
                    f'{float(reward):.3f}</text>'
                )
            else:
                # Missing evidence: draw a hatched unavailable cell. Launch builds reject this.
                y_ph = top + chart_h - 20
                svg.append(
                    f'<rect x="{x}" y="{y_ph:.1f}" width="{bar_w}" height="20" '
                    f'fill="none" stroke="{color}" stroke-width="1" stroke-dasharray="4 3" rx="3"/>'
                )
                svg.append(
                    f'<text x="{x + bar_w // 2}" y="{y_ph + 13:.1f}" fill="{color}" '
                    f'font-family="Inter, sans-serif" font-size="9" text-anchor="middle">n/a</text>'
                )

            # X label (model short name)
            short = label.replace("gpt-5.4-", "").replace("gpt-", "")
            svg.append(
                f'<text x="{x + bar_w // 2}" y="{top + chart_h + 16}" fill="#b9aaa0" '
                f'font-family="Inter, sans-serif" font-size="10" text-anchor="middle">{short}</text>'
            )

    # Legend
    legend_y = height - 22
    legend_x = left_margin
    for j, (label, color) in enumerate(zip(proposer_labels, colors)):
        lx = legend_x + j * 140
        svg.append(f'<rect x="{lx}" y="{legend_y}" width="10" height="10" fill="{color}" rx="2"/>')
        svg.append(
            f'<text x="{lx + 16}" y="{legend_y + 9}" fill="#d8c9bd" '
            f'font-family="Inter, sans-serif" font-size="11">{label}</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    cells: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []

    for task in TASKS:
        for proposer in PROPOSER_MODELS:
            slug = proposer["slug"]
            path = _manifest_path(task, slug)
            manifest = _read_manifest(path)
            cell = _extract_cell(task, proposer, manifest)
            snapshot_path = (
                _write_manifest_snapshot(task, slug, path)
                if path.exists()
                else _manifest_snapshot_path(task, slug)
            )
            cells.append(cell)
            evidence.append({
                "task": task,
                "proposer_slug": slug,
                "proposer_label": proposer["label"],
                "original_manifest_path": str(path.relative_to(REPO_ROOT)),
                "manifest_snapshot_path": str(snapshot_path.relative_to(REPO_ROOT)),
                "source_ref": _source_ref(snapshot_path),
                "cell": cell,
            })

    _assert_launch_ready(cells)

    data = {
        "chart": "proposer_scaling",
        "generated_from": str(ROOT.relative_to(REPO_ROOT)),
        "description": (
            "Synth GEPA heldout reward vs proposer model (size ladder) for HealthBench "
            "and tau2-bench retail. Policy model held constant per task. "
            "3 proposer models (gpt-5.4-nano/mini/gpt-5.4) × 2 tasks = 6 cells."
        ),
        "proposer_models": [p["label"] for p in PROPOSER_MODELS],
        "tasks": TASKS,
        "cells": cells,
    }

    out_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)

    for path in (out_dir / "proposer_scaling_data.json", FRONTEND_OUT):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")

    (out_dir / "source_evidence.json").write_text(json.dumps({
        "chart": data["chart"],
        "generated_from": data["generated_from"],
        "evidence": evidence,
    }, indent=2) + "\n")
    (out_dir / "proposer_scaling.md").write_text(render_markdown(cells))
    (out_dir / "proposer_scaling.svg").write_text(render_svg(cells) + "\n")

    print(json.dumps(data, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
