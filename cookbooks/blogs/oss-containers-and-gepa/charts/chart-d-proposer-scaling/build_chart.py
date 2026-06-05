#!/usr/bin/env python3
"""Build Chart D: proposer scaling figures from run manifests.

Scans runs/ for result_manifest.json files, groups by task × proposer model,
and emits:
  figures/proposer_scaling_data.json  - normalized data table
  frontend/.../data/proposer_scaling_data.json - blog data mirror
  figures/proposer_scaling.md         - markdown table
  figures/proposer_scaling.svg        - grouped bar chart (two panels, one per task)
  figures/source_evidence.json        - checksums for each public manifest snapshot used
  figures/manifest_snapshots/*.json   - compact public chart-audit summaries

The draft build fails if any cell is missing core evidence.

Expected run directory layout:
  runs/<task>_<model_slug>/<run_id>/result_manifest.json

Model slugs: nano, mini, gpt54
Tasks:       healthbench, tau2_retail
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BLOG_ROOT = ROOT.parents[1]
sys.path.insert(0, str(BLOG_ROOT))
from blog_paths import EVIDENCE_DIR, FRONTEND_DATA_DIR, REPO_ROOT  # noqa: E402

EVIDENCE_SUMMARY_ROOT = EVIDENCE_DIR / "benchmarks"
FRONTEND_OUT = FRONTEND_DATA_DIR / "proposer_scaling_data.json"
MANIFEST_SNAPSHOT_DIR = ROOT / "figures" / "manifest_snapshots"
HOME_PREFIX = str(Path.home())
WORKSPACE_PREFIX = str(Path.home() / "Documents" / "GitHub")

# ---------------------------------------------------------------------------
# Experiment matrix (locked)
# ---------------------------------------------------------------------------

TASKS = ["healthbench", "tau2_retail"]
RUN_GROUP = "final_20260603"

# Display labels for SVG panel titles.
TASK_LABELS = {
    "healthbench": "HealthBench Pro",
    "tau2_retail": "tau2-bench retail",
}

PROPOSER_MODELS = [
    {"slug": "nano",  "label": "gpt-5.4-nano",  "model": "gpt-5.4-nano",  "reasoning_effort": "low"},
    {"slug": "mini",  "label": "gpt-5.4-mini",  "model": "gpt-5.4-mini",  "reasoning_effort": "medium"},
    {"slug": "gpt54", "label": "gpt-5.4",        "model": "gpt-5.4",        "reasoning_effort": "high"},
]

EXPECTED_POLICY_MODELS: dict[str, str] = {
    "healthbench": "google/gemini-2.5-flash-lite",
    "tau2_retail": "gemini/gemini-3.1-flash-lite",
}

REQUIRED_CELL_FIELDS = (
    "initial_observed_reward",
    "best_observed_reward",
    "proposer_calls",
    "proposer_tokens",
    "total_cost_usd",
)

_COMPARISON_HELDOUT_SEED_REWARD_CACHE: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manifest_path(task: str, slug: str) -> Path:
    run_root = ROOT / "runs" / RUN_GROUP / f"{task}_{slug}"
    direct = run_root / "result_manifest.json"
    if direct.exists():
        return direct

    candidate_roots = [run_root]
    candidate_roots.extend((ROOT / "runs" / RUN_GROUP).glob(f"{task}_{slug}_*"))
    manifests = sorted(
        (
            manifest
            for candidate_root in candidate_roots
            for manifest in candidate_root.glob("*/result_manifest.json")
        ),
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


def _comparison_heldout_seed_reward(task: str) -> float:
    cached = _COMPARISON_HELDOUT_SEED_REWARD_CACHE.get(task)
    if cached is not None:
        return cached

    summary_path = EVIDENCE_SUMMARY_ROOT / task / "summary.json"
    try:
        value = _read_json(summary_path)["per_stack"]["synth_gepa"]["seed_heldout_score"]
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Chart D missing heldout seed baseline context for {task}: {summary_path}") from exc

    if not _is_number(value):
        raise SystemExit(f"Chart D heldout seed baseline context for {task} is not numeric: {value!r}")

    _COMPARISON_HELDOUT_SEED_REWARD_CACHE[task] = float(value)
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


def _public_path(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(REPO_ROOT.resolve(strict=False)).as_posix()
    except ValueError:
        return "<external-workspace-path>"


def _public_string(value: str) -> str:
    if HOME_PREFIX not in value:
        return value
    if value.startswith(f"{WORKSPACE_PREFIX}/"):
        return _public_path(Path(value))
    if value.startswith(f"{HOME_PREFIX}/.codex"):
        return "${CODEX_HOME}"
    if value.startswith(HOME_PREFIX):
        return "<home>"
    return value.replace(WORKSPACE_PREFIX, "<workspace>").replace(f"{HOME_PREFIX}/.codex", "${CODEX_HOME}").replace(
        HOME_PREFIX,
        "<home>",
    )


def _public_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _public_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_public_value(child) for child in value]
    if isinstance(value, str):
        return _public_string(value)
    return value


def _split_details(details: dict[str, Any]) -> dict[str, Any]:
    allowed: dict[str, Any] = {}
    for key, value in details.items():
        if _is_number(value) or isinstance(value, bool):
            allowed[key] = value
        elif isinstance(value, str) and key in {
            "heldout_split",
            "policy_model",
            "policy_provider",
            "proposer_model",
            "run_id",
            "train_split",
        }:
            allowed[key] = value
        elif isinstance(value, list) and key in {"heldout_ids", "train_ids"}:
            allowed[key.removesuffix("_ids") + "_count"] = len(value)
    return _public_value(allowed)


def _compact_manifest_snapshot(
    task: str,
    proposer: dict[str, str],
    manifest: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    best = manifest.get("best_candidate") or {}
    first_state = next(
        (entry for entry in manifest.get("state_history") or [] if isinstance(entry, dict)),
        {},
    )
    first_details = first_state.get("details") if isinstance(first_state, dict) else {}
    if not isinstance(first_details, dict):
        first_details = {}

    return {
        "schema": "chart_d_manifest_snapshot.v3",
        "task": task,
        "proposer_slug": proposer["slug"],
        "proposer_label": proposer["label"],
        "run_id": first_details.get("run_id") or source_path.parent.name,
        "source_manifest": {
            "sha256": _sha256(source_path),
            "bytes": source_path.stat().st_size,
        },
        "policy_models_observed": sorted(_collect_model_values(manifest)),
        "split": _split_details(first_details),
        "usage": _public_value(manifest.get("usage") or {}),
        "cost_usd": manifest.get("cost_usd"),
        "stopped_by": _public_value(manifest.get("stopped_by") or {}),
        "best_candidate": {
            "candidate_id": best.get("candidate_id"),
            "parent_id": best.get("parent_id"),
            "source": best.get("source"),
            "status": best.get("status"),
            "train_reward": best.get("train_reward"),
            "heldout_reward": best.get("heldout_reward"),
            "acceptance_score": best.get("acceptance_score"),
            "minibatch_reward": best.get("minibatch_reward"),
        },
    }


def _extract_curve(task: str, slug: str) -> dict[str, Any] | None:
    """Best observed score so far by candidate index from candidate_registry.json."""
    path = _manifest_path(task, slug)
    if not path.exists():
        return None
    registry_path = path.parent / "candidate_registry.json"
    if not registry_path.exists():
        return None

    registry = _read_json(registry_path)
    if not isinstance(registry, list):
        return None

    x_candidate: list[int] = []
    y_best: list[float] = []
    reward_sources: list[str] = []
    best_so_far = float("-inf")
    for entry in registry:
        heldout_reward = entry.get("heldout_reward")
        train_reward = entry.get("train_reward")
        if _is_number(heldout_reward):
            observed = heldout_reward
            reward_source = "heldout_reward"
        elif _is_number(train_reward):
            observed = train_reward
            reward_source = "train_reward"
        else:
            continue
        score = float(observed)
        best_so_far = max(best_so_far, score)
        x_candidate.append(len(x_candidate))
        y_best.append(best_so_far)
        reward_sources.append(reward_source)

    if not x_candidate:
        return None

    return {
        "x": {"candidate": x_candidate},
        "y": {"best_observed_score": y_best},
        "initial": y_best[0],
        "final": y_best[-1],
        "reward_source": reward_sources[0] if len(set(reward_sources)) == 1 else "mixed",
    }


def _extract_cell(task: str, proposer: dict[str, str], manifest: dict[str, Any] | None) -> dict[str, Any]:
    slug = proposer["slug"]
    label = proposer["label"]
    auth_mode = "api_key" if slug == "nano" else "chatgpt"
    comparison_heldout_seed_reward = _comparison_heldout_seed_reward(task)

    if manifest is None:
        return {
            "task": task,
            "proposer_slug": slug,
            "proposer_label": label,
            "proposer_auth_mode": auth_mode,
            "status": "pending",
            "comparison_heldout_seed_reward": comparison_heldout_seed_reward,
            "initial_observed_reward": None,
            "best_observed_reward": None,
            "proposer_calls": None,
            "proposer_tokens": None,
            "total_cost_usd": None,
            "pending_note": "run manifest missing",
        }

    mismatch_note = _policy_mismatch_note(task, manifest)
    if mismatch_note is not None:
        return {
            "task": task,
            "proposer_slug": slug,
            "proposer_label": label,
            "proposer_auth_mode": auth_mode,
            "status": "pending",
            "comparison_heldout_seed_reward": comparison_heldout_seed_reward,
            "initial_observed_reward": None,
            "best_observed_reward": None,
            "proposer_calls": None,
            "proposer_tokens": None,
            "total_cost_usd": None,
            "pending_note": mismatch_note,
        }

    best = manifest.get("best_candidate") or {}
    usage = manifest.get("usage") or {}
    run_meta = manifest.get("run") or {}

    if _is_number(best.get("heldout_reward")):
        best_observed = best.get("heldout_reward")
        best_observed_source = "heldout_reward"
    elif _is_number(best.get("train_reward")):
        best_observed = best.get("train_reward")
        best_observed_source = "train_reward"
    else:
        best_observed = None
        best_observed_source = "missing"
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
        "status": "available",
        "comparison_heldout_seed_reward": comparison_heldout_seed_reward,
        "initial_observed_reward": None,
        "best_observed_reward": best_observed,
        "best_observed_reward_source": best_observed_source,
        "proposer_calls": proposer_calls,
        "proposer_tokens": total_tokens,
        "total_cost_usd": total_cost_usd,
        "notes": (
            "posthoc heldout reward shown"
            if best_observed_source == "heldout_reward"
            else "heldout skipped; observed train optimization reward shown"
        ),
    }
    if wall_clock_s is not None:
        cell["wall_clock_s"] = wall_clock_s

    curve = _extract_curve(task, slug)
    if curve is not None:
        cell["curve"] = curve
        cell["initial_observed_reward"] = curve["initial"]
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
    proposer = next((item for item in PROPOSER_MODELS if item["slug"] == slug), None)
    if proposer is None:
        raise SystemExit(f"Unknown proposer slug for Chart D snapshot: {slug}")
    snapshot_path = _manifest_snapshot_path(task, slug)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(source_path)
    snapshot = _compact_manifest_snapshot(task, proposer, manifest, source_path)
    snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    return snapshot_path


def _assert_launch_ready(cells: list[dict[str, Any]]) -> None:
    failures: list[str] = []
    for cell in cells:
        label = f"{cell.get('task')} x {cell.get('proposer_label')}"
        if cell.get("status") != "available":
            failures.append(f"{label}: status={cell.get('status')} notes={cell.get('notes')!r}")
        if "seed_reward" in cell:
            failures.append(f"{label}: ambiguous seed_reward field must not be emitted for Chart D")
        for field in REQUIRED_CELL_FIELDS:
            if not _is_number(cell.get(field)):
                failures.append(f"{label}: {field}={cell.get(field)!r}")
        curve = cell.get("curve")
        if not isinstance(curve, dict):
            failures.append(f"{label}: curve missing")
        elif not _is_number(curve.get("initial")):
            failures.append(f"{label}: curve.initial={curve.get('initial')!r}")
        elif abs(float(curve["initial"]) - float(cell.get("initial_observed_reward", float("nan")))) > 1e-12:
            failures.append(
                f"{label}: initial_observed_reward does not match curve.initial "
                f"({cell.get('initial_observed_reward')!r} vs {curve.get('initial')!r})"
            )

    if failures:
        joined = "\n- ".join(failures)
        raise SystemExit(f"Chart D launch data incomplete:\n- {joined}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_markdown(cells: list[dict[str, Any]]) -> str:
    by_key = {(c["task"], c["proposer_slug"]): c for c in cells}
    lines = [
        "| task | proposer | initial observed reward | best observed reward | A/C heldout seed context | reward source | proposer calls | total tokens | notes |",
        "|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for task in TASKS:
        for proposer in PROPOSER_MODELS:
            slug = proposer["slug"]
            label = proposer["label"]
            cell = by_key.get((task, slug), {})
            initial_r = cell.get("initial_observed_reward")
            observed_r = cell.get("best_observed_reward")
            heldout_seed_r = cell.get("comparison_heldout_seed_reward")
            reward_source = cell.get("best_observed_reward_source") or "—"
            calls = cell.get("proposer_calls")
            tokens = cell.get("proposer_tokens")
            notes = cell.get("notes") or ""
            lines.append(
                "| "
                + " | ".join([
                    task,
                    label,
                    "—" if initial_r is None else f"{float(initial_r):.3f}",
                    "—" if observed_r is None else f"{float(observed_r):.3f}",
                    "—" if heldout_seed_r is None else f"{float(heldout_seed_r):.3f}",
                    str(reward_source),
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

    colors = ["#ea7a29", "#c89b1f", "#2d9bb0"]  # nano, mini, gpt-5.4

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{height}" viewBox="0 0 {total_w} {height}">',
        '<rect width="100%" height="100%" fill="#fffaf4"/>',
        f'<text x="24" y="32" fill="#2b2118" font-family="Inter, sans-serif" font-size="16" font-weight="600">Chart D — Proposer Scaling: observed reward vs proposer model</text>',
    ]

    # Y-axis tick lines (shared across panels)
    for panel_idx, task in enumerate(TASKS):
        panel_x = left_margin + panel_idx * (panel_w + panel_gap)
        svg.append(
            f'<text x="{panel_x + group_w // 2}" y="{top - 10}" fill="#4b3a2d" '
            f'font-family="Inter, sans-serif" font-size="13" text-anchor="middle">{TASK_LABELS.get(task, task)}</text>'
        )
        for tick_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = top + chart_h - tick_val * chart_h
            svg.append(
                f'<line x1="{panel_x - 6}" x2="{panel_x + group_w + 20}" '
                f'y1="{y:.1f}" y2="{y:.1f}" stroke="#eadfd3" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{panel_x - 10}" y="{y + 4:.1f}" fill="#7c6f65" '
                f'font-family="Inter, sans-serif" font-size="10" text-anchor="end">{tick_val:.2f}</text>'
            )

        # Bars
        for j, (slug, label, color) in enumerate(zip(proposer_slugs, proposer_labels, colors)):
            cell = by_key.get((task, slug), {})
            reward = cell.get("best_observed_reward")
            x = panel_x + j * (bar_w + bar_gap)

            if reward is not None:
                bar_h = float(reward) * chart_h
                y = top + chart_h - bar_h
                svg.append(
                    f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" '
                    f'fill="{color}" rx="3"/>'
                )
                svg.append(
                    f'<text x="{x + bar_w // 2}" y="{y - 5:.1f}" fill="#2b2118" '
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
                f'<text x="{x + bar_w // 2}" y="{top + chart_h + 16}" fill="#6d625b" '
                f'font-family="Inter, sans-serif" font-size="10" text-anchor="middle">{short}</text>'
            )

    # Legend
    legend_y = height - 22
    legend_x = left_margin
    for j, (label, color) in enumerate(zip(proposer_labels, colors)):
        lx = legend_x + j * 140
        svg.append(f'<rect x="{lx}" y="{legend_y}" width="10" height="10" fill="{color}" rx="2"/>')
        svg.append(
            f'<text x="{lx + 16}" y="{legend_y + 9}" fill="#4b3a2d" '
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
            if path.exists():
                cell["run_id"] = path.parent.name
            cell["manifest_snapshot_path"] = _public_path(snapshot_path)
            cell["source_ref"] = _source_ref(snapshot_path)
            cells.append(cell)
            evidence.append({
                "task": task,
                "proposer_slug": slug,
                "proposer_label": proposer["label"],
                "manifest_snapshot_path": _public_path(snapshot_path),
                "source_ref": _source_ref(snapshot_path),
                "chart_cell": {
                    key: cell.get(key)
                    for key in (
                        "task",
                        "proposer_slug",
                        "proposer_label",
                        "status",
                        "run_id",
                        "initial_observed_reward",
                        "comparison_heldout_seed_reward",
                        "best_observed_reward",
                        "best_observed_reward_source",
                        "proposer_calls",
                        "proposer_tokens",
                        "total_cost_usd",
                        "curve",
                    )
                },
            })

    _assert_launch_ready(cells)

    data = {
        "chart": "proposer_scaling",
        "generated_from": str(ROOT.relative_to(REPO_ROOT)),
        "source_evidence_path": str((ROOT / "figures" / "source_evidence.json").relative_to(REPO_ROOT)),
        "description": (
            "Synth GEPA best observed reward vs candidate index per proposer model. "
            "HealthBench Pro and tau2-bench retail cells are backed by compact public manifest summaries. "
            "Chart D reports observed optimization reward; heldout scoring was skipped for this sweep. "
            "3 proposer models × 2 tasks."
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
