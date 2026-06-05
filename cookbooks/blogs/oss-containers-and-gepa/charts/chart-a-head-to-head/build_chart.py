#!/usr/bin/env python3
"""Build Chart A head-to-head data from current GEPA evidence summaries."""

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

EVID = EVIDENCE_DIR
FRONTEND_OUT = FRONTEND_DATA_DIR / "core_head_to_head_data.json"

TASKS_READY = [
    ("healthbench", "HealthBench Pro"),
    ("tau2_retail", "tau2-bench retail"),
    ("banking77", "Banking77"),
    ("hotpotqa", "HotpotQA"),
]
PENDING_TASKS = []
ALL_TASKS = TASKS_READY + PENDING_TASKS
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seed_candidate_ids(benchmark: str) -> dict[str, str]:
    ids: dict[str, str] = {}
    for row in load_jsonl(EVID / "candidate_timeline.jsonl"):
        if row.get("benchmark") != benchmark:
            continue
        if row.get("candidate_index") == 0:
            ids[str(row["stack"])] = str(row["candidate_id"])
    return ids


_SEED_CANDIDATE_IDS: dict[str, dict[str, str]] = {}


def seed_heldout_cumulative_solved(
    benchmark: str,
    stack: str,
    threshold: float,
) -> int:
    if benchmark not in _SEED_CANDIDATE_IDS:
        _SEED_CANDIDATE_IDS[benchmark] = seed_candidate_ids(benchmark)
    candidate_id = _SEED_CANDIDATE_IDS[benchmark].get(stack)
    if not candidate_id:
        return 0
    solved: set[int] = set()
    for row in load_jsonl(EVID / "heldout_evaluations.jsonl"):
        if row.get("benchmark") != benchmark or row.get("stack") != stack:
            continue
        if row.get("candidate_id") != candidate_id:
            continue
        if float(row.get("reward") or 0) >= threshold:
            solved.add(int(row["heldout_seed"]))
    return len(solved)


def pct_heldout_over_seed(lift: float, seed_score: float) -> float:
    if seed_score > 1e-9:
        return 100.0 * lift / seed_score
    return 0.0 if abs(lift) < 1e-9 else 100.0


def pct_coverage_over_seed_baseline(cumulative: float, seed_cumulative: float) -> float:
    if seed_cumulative > 0:
        return 100.0 * (cumulative - seed_cumulative) / seed_cumulative
    return 0.0


def row_for_stack(task: str, label: str, stack: str, summary: dict[str, Any]) -> dict[str, Any]:
    stack_summary = (summary.get("per_stack") or {}).get(stack)
    if not stack_summary:
        raise SystemExit(f"Chart A missing stack summary for {task}/{stack}")
    summary_ref = source_ref(summary_path(task))
    heldout_total = require_number(
        stack_summary.get("heldout_seeds_total"),
        f"{task}/{stack}.heldout_seeds_total",
    )
    train_total = require_number(
        stack_summary.get("train_seeds_total"),
        f"{task}/{stack}.train_seeds_total",
    )
    heldout_cumulative = require_number(
        stack_summary.get("heldout_cumulative_solved"),
        f"{task}/{stack}.heldout_cumulative_solved",
    )
    seed_score = require_number(
        stack_summary.get("seed_heldout_score"),
        f"{task}/{stack}.seed_heldout_score",
    )
    lift = require_number(stack_summary.get("lift_over_seed"), f"{task}/{stack}.lift_over_seed")
    threshold = float(
        (summary.get("seed_coverage") or {}).get("heldout", {}).get("coverage_reward_threshold", 1.0)
    )
    seed_cumulative = float(seed_heldout_cumulative_solved(task, stack, threshold))
    return {
        "task": task,
        "task_label": label,
        "stack": STACK_LABELS[stack],
        "stack_id": stack,
        "heldout_reward": require_number(stack_summary.get("best_heldout_score"), f"{task}/{stack}.best_heldout_score"),
        "seed_heldout_reward": seed_score,
        "lift_over_seed": lift,
        "heldout_pct_over_seed": round(pct_heldout_over_seed(lift, seed_score), 4),
        "seed_heldout_cumulative_solved": seed_cumulative,
        "heldout_cumulative_solved": heldout_cumulative,
        "coverage_pct_over_seed_baseline": round(
            pct_coverage_over_seed_baseline(heldout_cumulative, seed_cumulative),
            4,
        ),
        "heldout_seeds_total": heldout_total,
        "train_cumulative_solved": require_number(
            stack_summary.get("train_cumulative_solved"),
            f"{task}/{stack}.train_cumulative_solved",
        ),
        "train_seeds_total": train_total,
        "joint_pareto_points": require_number(
            stack_summary.get("joint_pareto_points"),
            f"{task}/{stack}.joint_pareto_points",
        ),
        "joint_pareto_share": require_number(
            stack_summary.get("joint_pareto_share"),
            f"{task}/{stack}.joint_pareto_share",
        ),
        "best_candidate_id": stack_summary.get("best_candidate_id"),
        "candidate_count": require_number(stack_summary.get("num_candidates"), f"{task}/{stack}.num_candidates"),
        "rollout_calls": require_number(stack_summary.get("total_rollouts"), f"{task}/{stack}.total_rollouts"),
        "total_cost_usd": require_number(stack_summary.get("cost_metered_usd"), f"{task}/{stack}.cost_metered_usd"),
        "codex_subscription_cost_usd": require_number(stack_summary.get("cost_with_codex_subscription_usd"), f"{task}/{stack}.cost_with_codex_subscription_usd"),
        "wall_clock_s": require_number(stack_summary.get("total_elapsed_seconds"), f"{task}/{stack}.total_elapsed_seconds"),
        "source": str(summary_path(task).relative_to(REPO_ROOT)),
        "source_ref": summary_ref,
        "run_ids": stack_summary.get("run_ids") or [],
        "status": "available",
    }


def pending_row(task: str, label: str, stack: str) -> dict[str, Any]:
    return {
        "task": task,
        "task_label": label,
        "stack": STACK_LABELS[stack],
        "stack_id": stack,
        "status": "pending",
        "heldout_reward": 0,
        "seed_heldout_reward": 0,
        "lift_over_seed": 0,
        "heldout_pct_over_seed": 0,
        "seed_heldout_cumulative_solved": 0,
        "coverage_pct_over_seed_baseline": 0,
        "heldout_cumulative_solved": 0,
        "heldout_seeds_total": 0,
        "train_cumulative_solved": 0,
        "train_seeds_total": 0,
        "joint_pareto_points": 0,
        "joint_pareto_share": 0,
        "candidate_count": 0,
        "rollout_calls": 0,
        "total_cost_usd": 0,
        "codex_subscription_cost_usd": 0,
        "wall_clock_s": 0,
        "source": None,
        "pending_note": "Synth vs gepa-ai parity run not landed yet.",
    }


def load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for task, label in TASKS_READY:
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
    for task, label in PENDING_TASKS:
        for stack in ("gepa_ai", "synth_gepa"):
            rows.append(pending_row(task, label, stack))
    return rows, evidence


def render_markdown(rows: list[dict[str, Any]]) -> str:
    by_key = {(row["task"], row["stack_id"]): row for row in rows}
    lines = [
        "| Task | gepa-ai seed | gepa-ai best | Synth seed | Synth best | Synth - gepa | Winner |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for task, label in ALL_TASKS:
        gepa_row = by_key.get((task, "gepa_ai"), {})
        synth_row = by_key.get((task, "synth_gepa"), {})
        if gepa_row.get("status") == "pending" or synth_row.get("status") == "pending":
            lines.append(f"| {label} | — | — | — | — | — | pending |")
            continue
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
    for i, (task, label) in enumerate(ALL_TASKS):
        y = 112 + i * row_h
        gepa_row = by_key.get((task, "gepa_ai"), {})
        synth_row = by_key.get((task, "synth_gepa"), {})
        if gepa_row.get("status") == "pending" or synth_row.get("status") == "pending":
            labels.append(
                f'<text x="60" y="{y}" font-family="monospace" font-size="17" fill="#766d63">{label} (pending)</text>'
            )
            continue
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
<text x="60" y="54" font-family="Arial, sans-serif" font-size="31" font-weight="700" fill="#221f1b">GEPA head-to-head</text>
<text x="60" y="82" font-family="monospace" font-size="14" fill="#766d63">same containers · pinned policy model · posthoc heldout evidence</text>
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
        "chart": "core_gepa_head_to_head",
        "generated_from": str(ROOT.relative_to(REPO_ROOT)),
        "source_evidence_path": str((ROOT / "figures" / "source_evidence.json").relative_to(REPO_ROOT)),
        "definition": "Same-container Synth GEPA vs gepa-ai: best heldout, cumulative heldout coverage, available train metadata, and joint-Pareto frontier size",
        "tasks": [
            {"key": key, "label": label}
            for key, label in ALL_TASKS
        ],
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
