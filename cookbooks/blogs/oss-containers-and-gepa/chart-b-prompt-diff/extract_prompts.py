#!/usr/bin/env python3
"""Extract best-candidate prompts for Chart B from Chart A source runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
CHART_A = ROOT.parent / "chart-a-head-to-head"

TASKS = {
    "banking77": {
        "field": "stage2_system",
        "synth": CHART_A / "runs" / "synth_gepa" / "banking77_parity_synth_gepa" / "result_manifest.json",
        "gepa_ai": CHART_A / "runs" / "gepa_ai_via_container" / "banking77_20260521_011836" / "summary.json",
    },
    "tblite": {
        "field": "starting_prompt",
        "synth": CHART_A / "runs" / "synth_gepa" / "tblite_parity_synth_gepa" / "result_manifest.json",
        "gepa_ai": CHART_A / "runs" / "gepa_ai_via_container" / "tblite_20260522_014223" / "summary.json",
    },
    "crafter": {
        "field": "react_system_prompt",
        "synth": CHART_A / "runs" / "synth_gepa" / "crafter_parity_synth_gepa" / "result_manifest.json",
        "gepa_ai": CHART_A / "runs" / "gepa_ai_via_container" / "crafter_20260522_014303" / "summary.json",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def synth_prompt(path: Path, field: str) -> str:
    best = read_json(path).get("best_candidate") or {}
    payload = best.get("payload") or {}
    lever_values = ((best.get("lever_bundle") or {}).get("values") or {})
    return str(payload.get(field) or lever_values.get(field) or "")


def gepa_ai_prompt(path: Path, field: str) -> str:
    best = read_json(path).get("best_candidate") or {}
    return str(best.get(field) or best.get("system_prompt") or "")


def main() -> int:
    lines = ["# Chart B Source Runs", ""]
    for task, cfg in TASKS.items():
        out_dir = ROOT / "prompts" / task
        out_dir.mkdir(parents=True, exist_ok=True)
        synth = synth_prompt(cfg["synth"], cfg["field"])
        gepa_ai = gepa_ai_prompt(cfg["gepa_ai"], cfg["field"])
        (out_dir / "synth_gepa_best.txt").write_text(synth + "\n")
        (out_dir / "gepa_ai_best.txt").write_text(gepa_ai + "\n")
        (out_dir / "diff.md").write_text(render_diff_note(task, synth, gepa_ai))
        lines.extend([
            f"## {task}",
            "",
            f"- Synth GEPA: `{cfg['synth'].relative_to(REPO_ROOT)}`",
            f"- gepa-ai: `{cfg['gepa_ai'].relative_to(REPO_ROOT)}`",
            "",
        ])
    (ROOT / "source_runs.md").write_text("\n".join(lines))
    return 0


def render_diff_note(task: str, synth: str, gepa_ai: str) -> str:
    if task == "banking77":
        summary = (
            "Synth GEPA tightens the Banking77 instruction with exact-label and specificity constraints. "
            "gepa-ai kept the seed prompt in the selected best run."
        )
    elif task == "tblite":
        summary = (
            "The selected gepa-ai TBLite prompt stayed at the seed because the heldout split was already perfect. "
            "The fresh Synth GEPA same-container run also selected the seed prompt."
        )
    else:
        summary = (
            "Both fresh same-container Crafter runs improve over the seed prompt. "
            "gepa-ai expands into an explicit ordered strategy; Synth GEPA adds a compact priority loop and survival rules."
        )
    return (
        f"# {task} Prompt Diff\n\n"
        f"{summary}\n\n"
        "## Synth GEPA\n\n"
        f"```text\n{synth}\n```\n\n"
        "## gepa-ai\n\n"
        f"```text\n{gepa_ai}\n```\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
