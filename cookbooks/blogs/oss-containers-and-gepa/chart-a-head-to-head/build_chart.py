#!/usr/bin/env python3
"""Build Chart A head-to-head data from final GEPA evidence summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
WORKSPACE = Path("/Users/joshpurtell/Documents/GitHub")
EVALS = REPO_ROOT / "cookbooks" / "optimizers" / "gepa" / "evals"
EVID = EVALS / "evidence"
FRONTEND_OUT = (
    WORKSPACE
    / "frontend"
    / "src"
    / "components"
    / "blog"
    / "posts"
    / "introducing-gepa-platform"
    / "data"
    / "head_to_head_data.json"
)

TASKS = [
    ("healthbench", "HealthBench Pro"),
    ("harvey_lab", "Harvey Lab Tax"),
    ("tau2_retail", "tau2-bench retail"),
    ("dungeongrid", "DungeonGrid"),
]
STACK_LABELS = {
    "gepa_ai": "gepa-ai",
    "synth_gepa": "Synth GEPA",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def source_ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def summary_path(task: str) -> Path:
    return EVID / "benchmarks" / task / "summary.json"


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def require_number(value: Any, label: str) -> int | float:
    if not is_number(value):
        raise SystemExit(f"Chart A missing numeric {label}: {value!r}")
    return value


def row_for_stack(task: str, label: str, stack: str, summary: dict[str, Any]) -> dict[str, Any]:
    stack_summary = (summary.get("per_stack") or {}).get(stack)
    if not stack_summary:
        raise SystemExit(f"Chart A missing stack summary for {task}/{stack}")
    return {
        "task": task,
        "task_label": label,
        "stack": STACK_LABELS[stack],
        "stack_id": stack,
        "heldout_reward": require_number(stack_summary.get("best_heldout_score"), f"{task}/{stack}.best_heldout_score"),
        "seed_heldout_reward": require_number(stack_summary.get("seed_heldout_score"), f"{task}/{stack}.seed_heldout_score"),
        "lift_over_seed": require_number(stack_summary.get("lift_over_seed"), f"{task}/{stack}.lift_over_seed"),
        "best_candidate_id": stack_summary.get("best_candidate_id"),
        "candidate_count": require_number(stack_summary.get("num_candidates"), f"{task}/{stack}.num_candidates"),
        "rollout_calls": require_number(stack_summary.get("total_rollouts"), f"{task}/{stack}.total_rollouts"),
        "total_cost_usd": require_number(stack_summary.get("cost_metered_usd"), f"{task}/{stack}.cost_metered_usd"),
        "codex_subscription_cost_usd": require_number(stack_summary.get("cost_with_codex_subscription_usd"), f"{task}/{stack}.cost_with_codex_subscription_usd"),
        "wall_clock_s": require_number(stack_summary.get("total_elapsed_seconds"), f"{task}/{stack}.total_elapsed_seconds"),
        "source": str(summary_path(task).relative_to(REPO_ROOT)),
    }


def load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for task, label in TASKS:
        path = summary_path(task)
        if not path.exists():
            raise SystemExit(f"Chart A missing summary for {task}: {path}")
        summary = read_json(path)
        for stack in ("gepa_ai", "synth_gepa"):
            rows.append(row_for_stack(task, label, stack, summary))
        evidence.append({
            "task": task,
            "summary": source_ref(path),
            "run_ids": {
                stack: (summary.get("per_stack") or {}).get(stack, {}).get("run_ids")
                for stack in ("gepa_ai", "synth_gepa")
            },
        })
    return rows, evidence


def render_markdown(rows: list[dict[str, Any]]) -> str:
    by_key = {(row["task"], row["stack_id"]): row for row in rows}
    lines = [
        "| Task | gepa-ai seed | gepa-ai best | Synth seed | Synth best | Synth - gepa | Winner |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for task, label in TASKS:
        gepa_row = by_key.get((task, "gepa_ai"), {})
        synth_row = by_key.get((task, "synth_gepa"), {})
        gepa_seed = gepa_row.get("seed_heldout_reward")
        gepa = gepa_row.get("heldout_reward")
        synth_seed = synth_row.get("seed_heldout_reward")
        synth = synth_row.get("heldout_reward")
        if not (is_number(gepa_seed) and is_number(gepa) and is_number(synth_seed) and is_number(synth)):
            raise SystemExit(f"Chart A incomplete markdown row for {task}")
        delta = synth - gepa
        if synth > gepa:
            winner = "Synth GEPA"
        elif gepa > synth:
            winner = "gepa-ai"
        else:
            winner = "tie"
        lines.append(
            "| "
            + " | ".join([
                label,
                f"{gepa_seed:.3f}",
                f"{gepa:.3f}",
                f"{synth_seed:.3f}",
                f"{synth:.3f}",
                f"{delta:+.3f}",
                winner,
            ])
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_svg(rows: list[dict[str, Any]]) -> str:
    by_key = {(row["task"], row["stack_id"]): row for row in rows}
    width = 860
    height = 430
    row_h = 70
    labels = []
    for i, (task, label) in enumerate(TASKS):
        y = 112 + i * row_h
        gepa_row = by_key.get((task, "gepa_ai"), {})
        synth_row = by_key.get((task, "synth_gepa"), {})
        gepa_seed = gepa_row.get("seed_heldout_reward")
        gepa = gepa_row.get("heldout_reward")
        synth_seed = synth_row.get("seed_heldout_reward")
        synth = synth_row.get("heldout_reward")
        if not (is_number(gepa_seed) and is_number(gepa) and is_number(synth_seed) and is_number(synth)):
            raise SystemExit(f"Chart A incomplete SVG row for {task}")
        labels.append(
            f'<text x="60" y="{y}" font-family="monospace" font-size="17" fill="#221f1b">{label}</text>'
        )
        for x, value, color in (
            (290, gepa_seed, "#8b8175"),
            (410, gepa, "#c8ad45"),
            (530, synth_seed, "#8b8175"),
            (650, synth, "#d88437"),
        ):
            txt = f"{float(value):.3f}"
            labels.append(
                f'<text x="{x}" y="{y}" font-family="monospace" font-size="17" fill="{color}" font-weight="700">{txt}</text>'
            )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#faf9f5"/>
<text x="60" y="54" font-family="Arial, sans-serif" font-size="31" font-weight="700" fill="#221f1b">Final GEPA head-to-head</text>
<text x="60" y="82" font-family="monospace" font-size="14" fill="#766d63">same containers · matched policy model · posthoc heldout evidence</text>
<text x="290" y="112" font-family="monospace" font-size="12" fill="#8b8175">gepa seed</text>
<text x="410" y="112" font-family="monospace" font-size="12" fill="#8b8175">gepa best</text>
<text x="530" y="112" font-family="monospace" font-size="12" fill="#8b8175">Synth seed</text>
<text x="650" y="112" font-family="monospace" font-size="12" fill="#8b8175">Synth best</text>
{''.join(labels)}
</svg>
"""


def main() -> int:
    rows, evidence = load_rows()
    data = {
        "chart": "final_gepa_head_to_head",
        "generated_from": str(ROOT.relative_to(REPO_ROOT)),
        "definition": "best posthoc heldout reward by stack for the final four GEPA blog benchmarks",
        "tasks": [{"key": key, "label": label} for key, label in TASKS],
        "rows": rows,
    }
    out_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    for path in (out_dir / "head_to_head_data.json", FRONTEND_OUT):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
    (out_dir / "source_evidence.json").write_text(json.dumps({
        "chart": data["chart"],
        "generated_from": data["generated_from"],
        "evidence": evidence,
    }, indent=2) + "\n")
    (out_dir / "head_to_head.md").write_text(render_markdown(rows))
    (out_dir / "head_to_head.svg").write_text(render_svg(rows))
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
