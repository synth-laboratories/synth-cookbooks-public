#!/usr/bin/env python3
"""Build Chart A release evidence from checked-in run artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]

TASKS = {
    "banking77": {
        "seed": 0.88,
        "synth_manifest": ROOT / "runs" / "synth_gepa" / "banking77_parity_synth_gepa" / "result_manifest.json",
        "gepa_ai_summary": ROOT / "runs" / "gepa_ai_via_container" / "banking77_20260521_011836" / "summary.json",
        "notes": "True same-container comparison on 24 train / 200 heldout rows with 2400-call budget.",
    },
    "tblite": {
        "seed": 1.0,
        "synth_manifest": ROOT / "runs" / "synth_gepa" / "tblite_parity_synth_gepa" / "result_manifest.json",
        "gepa_ai_summary": ROOT / "runs" / "gepa_ai_via_container" / "tblite_20260522_014223" / "summary.json",
        "notes": "Fresh same-container comparison on 3 train / 2 heldout rows; seed is already perfect on heldout.",
    },
    "crafter": {
        "seed": 1.0,
        "synth_manifest": ROOT / "runs" / "synth_gepa" / "crafter_parity_synth_gepa" / "result_manifest.json",
        "gepa_ai_summary": ROOT / "runs" / "gepa_ai_via_container" / "crafter_20260522_014303" / "summary.json",
        "notes": "Fresh same-container comparison on 2 train / 1 heldout row at the small public budget.",
    },
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


def repo_relative_path(raw_path: str | None) -> str | None:
    if not raw_path:
        return raw_path
    path = Path(raw_path)
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            marker = "cookbooks/"
            if marker in raw_path:
                return raw_path[raw_path.index(marker):]
    return str(path)


def synth_row(task: str, path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    best = manifest.get("best_candidate") or {}
    usage = manifest.get("usage") or {}
    return {
        "task": task,
        "stack": "Synth GEPA",
        "heldout_reward": best.get("heldout_reward"),
        "minibatch_reward": best.get("minibatch_reward"),
        "best_candidate_id": best.get("candidate_id"),
        "rollout_calls": usage.get("rollout_calls"),
        "proposer_calls": usage.get("proposer_calls"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "source": str(path.relative_to(REPO_ROOT)),
    }


def synth_evidence(task: str, path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    best = manifest.get("best_candidate") or {}
    usage = manifest.get("usage") or {}
    return {
        "task": task,
        "stack": "Synth GEPA",
        "source": source_ref(path),
        "best_candidate": {
            "candidate_id": best.get("candidate_id"),
            "source": best.get("source"),
            "train_reward": best.get("train_reward"),
            "heldout_reward": best.get("heldout_reward"),
            "minibatch_reward": best.get("minibatch_reward"),
            "status": best.get("status"),
            "payload": best.get("payload"),
        },
        "usage": {
            "rollout_calls": usage.get("rollout_calls"),
            "proposer_calls": usage.get("proposer_calls"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "artifact_refs": [
            {
                "kind": ref.get("kind"),
                "path": repo_relative_path(ref.get("path")),
                "sha256": ref.get("sha256"),
                "bytes": ref.get("bytes"),
            }
            for ref in (manifest.get("artifact_refs") or [])
        ],
    }


def gepa_ai_row(task: str, path: Path) -> dict[str, Any]:
    summary = read_json(path)
    return {
        "task": task,
        "stack": "gepa-ai",
        "heldout_reward": summary.get("best_val_score"),
        "seed_heldout_reward": summary.get("seed_val_score"),
        "best_candidate_id": summary.get("best_idx"),
        "rollout_calls": summary.get("rollout_calls"),
        "proposer_calls": summary.get("reflection_calls"),
        "prompt_tokens": (summary.get("rollout_prompt_tokens") or 0) + (summary.get("reflection_prompt_tokens") or 0),
        "completion_tokens": (summary.get("rollout_completion_tokens") or 0) + (summary.get("reflection_completion_tokens") or 0),
        "total_usd": summary.get("total_usd"),
        "wall_clock_s": summary.get("wall_clock_s"),
        "source": str(path.relative_to(REPO_ROOT)),
    }


def gepa_ai_evidence(task: str, path: Path) -> dict[str, Any]:
    summary = read_json(path)
    best = summary.get("best_candidate") or {}
    return {
        "task": task,
        "stack": "gepa-ai",
        "source": source_ref(path),
        "seed_val_score": summary.get("seed_val_score"),
        "best_val_score": summary.get("best_val_score"),
        "best_idx": summary.get("best_idx"),
        "rollout_calls": summary.get("rollout_calls"),
        "reflection_calls": summary.get("reflection_calls"),
        "wall_clock_s": summary.get("wall_clock_s"),
        "total_usd": summary.get("total_usd"),
        "tokens": {
            "rollout_prompt_tokens": summary.get("rollout_prompt_tokens"),
            "rollout_completion_tokens": summary.get("rollout_completion_tokens"),
            "reflection_prompt_tokens": summary.get("reflection_prompt_tokens"),
            "reflection_completion_tokens": summary.get("reflection_completion_tokens"),
        },
        "best_candidate": best,
    }


def main() -> int:
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for task, cfg in TASKS.items():
        rows.append({
            "task": task,
            "stack": "Seed candidate",
            "heldout_reward": cfg["seed"],
            "source": cfg["notes"],
        })
        rows.append(gepa_ai_row(task, cfg["gepa_ai_summary"]))
        rows.append(synth_row(task, cfg["synth_manifest"]))
        evidence.append(gepa_ai_evidence(task, cfg["gepa_ai_summary"]))
        evidence.append(synth_evidence(task, cfg["synth_manifest"]))

    data = {
        "chart": "compute_parity_head_to_head",
        "generated_from": str(ROOT.relative_to(REPO_ROOT)),
        "caveat": (
            "All three public cookbook rows use the same HTTP container boundary for gepa-ai "
            "and Synth GEPA. Banking77 is the broadest comparison; TBLite and Crafter are "
            "small public smoke-scale parity splits."
        ),
        "rows": rows,
    }

    out_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "head_to_head_data.json").write_text(json.dumps(data, indent=2))
    (out_dir / "source_evidence.json").write_text(json.dumps({
        "chart": data["chart"],
        "generated_from": data["generated_from"],
        "caveat": data["caveat"],
        "source_note": "Compact tracked snapshot of ignored raw runs used to build Chart A.",
        "evidence": evidence,
    }, indent=2))
    (out_dir / "head_to_head.md").write_text(render_markdown(rows))
    (out_dir / "head_to_head.svg").write_text(render_svg(rows))
    print(json.dumps(data, indent=2))
    return 0


def render_markdown(rows: list[dict[str, Any]]) -> str:
    tasks = ["banking77", "tblite", "crafter"]
    stacks = ["Seed candidate", "gepa-ai", "Synth GEPA"]
    by_key = {(row["task"], row["stack"]): row for row in rows}
    lines = [
        "| Stack | Banking77 | TBLite | Crafter |",
        "|---|---:|---:|---:|",
    ]
    for stack in stacks:
        cells = [stack]
        for task in tasks:
            value = by_key[(task, stack)].get("heldout_reward")
            cells.append("—" if value is None else f"{float(value):.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_svg(rows: list[dict[str, Any]]) -> str:
    tasks = ["banking77", "tblite", "crafter"]
    stacks = ["Seed candidate", "gepa-ai", "Synth GEPA"]
    colors = {"Seed candidate": "#8a8a8a", "gepa-ai": "#f4c542", "Synth GEPA": "#f97316"}
    by_key = {(row["task"], row["stack"]): row for row in rows}
    width = 760
    height = 340
    left = 120
    top = 48
    chart_h = 220
    group_w = 190
    bar_w = 34
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#151210"/>',
        '<text x="24" y="30" fill="#f7efe8" font-family="Inter, sans-serif" font-size="18">Compute-parity head-to-head</text>',
    ]
    for tick in [0, 0.5, 1.0, 1.5, 2.0]:
        y = top + chart_h - tick / 2.0 * chart_h
        svg.append(f'<line x1="{left}" x2="{width - 28}" y1="{y:.1f}" y2="{y:.1f}" stroke="#302821" stroke-width="1"/>')
        svg.append(f'<text x="72" y="{y + 4:.1f}" fill="#b9aaa0" font-family="Inter, sans-serif" font-size="11">{tick:.1f}</text>')
    for i, task in enumerate(tasks):
        gx = left + i * group_w
        svg.append(f'<text x="{gx + 22}" y="{height - 36}" fill="#d8c9bd" font-family="Inter, sans-serif" font-size="12">{task}</text>')
        for j, stack in enumerate(stacks):
            row = by_key[(task, stack)]
            value = float(row.get("heldout_reward") or 0)
            bar_h = value / 2.0 * chart_h
            x = gx + j * (bar_w + 8)
            y = top + chart_h - bar_h
            svg.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{colors[stack]}" rx="2"/>')
            svg.append(f'<text x="{x + 2}" y="{y - 5:.1f}" fill="#f7efe8" font-family="Inter, sans-serif" font-size="10">{value:.2f}</text>')
    for j, stack in enumerate(stacks):
        x = left + j * 150
        svg.append(f'<rect x="{x}" y="{height - 18}" width="10" height="10" fill="{colors[stack]}"/>')
        svg.append(f'<text x="{x + 16}" y="{height - 9}" fill="#d8c9bd" font-family="Inter, sans-serif" font-size="11">{stack}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


if __name__ == "__main__":
    raise SystemExit(main())
