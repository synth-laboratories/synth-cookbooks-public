"""Render comparison figures from the evidence files.

Reads evidence/curve_points.jsonl and evidence/candidate_timeline.jsonl and writes
PNGs into figures/<benchmark>/:

  best_heldout_vs_index.png        best-so-far heldout by candidate index
  best_heldout_vs_rollouts.png     best-so-far heldout by cumulative rollouts
  best_heldout_vs_time.png         best-so-far heldout by wall-clock seconds
  best_heldout_vs_cost.png         best-so-far heldout by cumulative $ (rollout+proposer)
  pareto_coverage_vs_index.png     fraction of final Pareto frontier discovered
  scatter_heldout_vs_cost.png      every candidate: heldout vs cumulative $
  scatter_heldout_vs_time.png      every candidate: heldout vs elapsed seconds
  comparison_grid.png              all of the above on one canvas

Usage (from evals/):
    python scripts/plot_curves.py --benchmark banking77
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = EVALS_DIR / "evidence"
FIGURES_DIR = EVALS_DIR / "figures"

STACK_STYLE = {
    "synth_gepa": {"color": "#2563eb", "label": "Synth GEPA", "marker": "o"},
    "gepa_ai": {"color": "#dc2626", "label": "gepa-ai", "marker": "s"},
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _stacks(rows: list[dict]) -> list[str]:
    return sorted({r.get("stack", "") for r in rows if r.get("stack")})


def _curve(curves: list[dict], stack: str, metric: str, x_name: str) -> tuple[list, list]:
    pts = sorted(
        [p for p in curves if p.get("stack") == stack and p.get("metric") == metric and p.get("x_name") == x_name],
        key=lambda p: p["x"],
    )
    return [p["x"] for p in pts], [p["y"] for p in pts]


def _pareto_front(points: list[tuple[float, float, str]]) -> set[str]:
    """Max y (heldout), min x (cost/time). Returns ids on the frontier."""
    dominated = set()
    for i, (xi, yi, _ci) in enumerate(points):
        for j, (xj, yj, _cj) in enumerate(points):
            if i == j:
                continue
            if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                dominated.add(i)
                break
    return {points[i][2] for i in range(len(points)) if i not in dominated}


def _step_plot(ax, curves, stacks, metric, x_name, xlabel, ylabel, title):
    for s in stacks:
        xs, ys = _curve(curves, s, metric, x_name)
        if not xs:
            continue
        st = STACK_STYLE.get(s, {"color": None, "label": s, "marker": "o"})
        ax.step(xs, ys, where="post", color=st["color"], label=st["label"],
                marker=st["marker"], markersize=4, linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def _scatter_plot(ax, timeline, stacks, x_field, xlabel, title):
    for s in stacks:
        cands = [c for c in timeline if c.get("stack") == s and c.get(x_field) is not None
                 and c.get("heldout_score") is not None]
        if not cands:
            continue
        st = STACK_STYLE.get(s, {"color": None, "label": s, "marker": "o"})
        xs = [c[x_field] for c in cands]
        ys = [c["heldout_score"] for c in cands]
        ax.scatter(xs, ys, color=st["color"], label=st["label"], marker=st["marker"],
                   s=36, alpha=0.7, edgecolors="white", linewidths=0.5)
        # Pareto front (min x, max heldout) across this stack's candidates.
        pts = [(c[x_field], c["heldout_score"], c["candidate_id"]) for c in cands]
        front_ids = _pareto_front(pts)
        front = sorted([p for p in pts if p[2] in front_ids], key=lambda p: p[0])
        if len(front) > 1:
            ax.plot([p[0] for p in front], [p[1] for p in front],
                    color=st["color"], linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("heldout score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def main() -> int:
    p = argparse.ArgumentParser(description="Render comparison figures.")
    p.add_argument("--benchmark", default="banking77")
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib not installed. Run: uv add matplotlib")

    curves = [
        r for r in load_jsonl(EVIDENCE_DIR / "curve_points.jsonl")
        if r.get("benchmark") == args.benchmark
    ]
    timeline = [
        r for r in load_jsonl(EVIDENCE_DIR / "candidate_timeline.jsonl")
        if r.get("benchmark") == args.benchmark
    ]
    if not timeline:
        raise SystemExit("candidate_timeline.jsonl is empty for the requested benchmark — run evaluate_heldout.py first")

    stacks = _stacks(timeline)
    out_dir = FIGURES_DIR / args.benchmark
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        ("best_heldout_so_far", "candidate_index", "candidate index", "best heldout so far",
         "Best heldout vs candidate index", "best_heldout_vs_index.png", "step"),
        ("best_heldout_so_far", "cumulative_rollout_count", "cumulative policy rollouts", "best heldout so far",
         "Best heldout vs rollouts (equal-budget axis)", "best_heldout_vs_rollouts.png", "step"),
        ("best_heldout_so_far", "elapsed_seconds", "wall-clock seconds", "best heldout so far",
         "Best heldout vs wall-clock time", "best_heldout_vs_time.png", "step"),
        ("best_heldout_so_far", "cumulative_cost_usd", "cumulative cost (USD)", "best heldout so far",
         "Best heldout vs cost (rollout + proposer)", "best_heldout_vs_cost.png", "step"),
        ("pareto_coverage", "candidate_index", "candidate index", "Pareto coverage",
         "Pareto frontier coverage vs index", "pareto_coverage_vs_index.png", "step"),
    ]
    scatters = [
        ("cumulative_cost_usd", "cumulative cost (USD)", "Heldout vs cost (per candidate)",
         "scatter_heldout_vs_cost.png"),
        ("elapsed_seconds", "wall-clock seconds", "Heldout vs time (per candidate)",
         "scatter_heldout_vs_time.png"),
    ]

    written = []
    # Individual figures.
    for metric, x_name, xlabel, ylabel, title, fname, _kind in panels:
        fig, ax = plt.subplots(figsize=(6, 4))
        _step_plot(ax, curves, stacks, metric, x_name, xlabel, ylabel, title)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130)
        plt.close(fig)
        written.append(fname)

    for x_field, xlabel, title, fname in scatters:
        fig, ax = plt.subplots(figsize=(6, 4))
        _scatter_plot(ax, timeline, stacks, x_field, xlabel, title)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130)
        plt.close(fig)
        written.append(fname)

    # Combined grid.
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    flat = axes.flatten()
    i = 0
    for metric, x_name, xlabel, ylabel, title, _fname, _kind in panels:
        _step_plot(flat[i], curves, stacks, metric, x_name, xlabel, ylabel, title)
        i += 1
    for x_field, xlabel, title, _fname in scatters:
        _scatter_plot(flat[i], timeline, stacks, x_field, xlabel, title)
        i += 1
    for j in range(i, len(flat)):
        flat[j].axis("off")
    fig.suptitle(f"Synth GEPA vs gepa-ai — {args.benchmark} (equal search-rollout budget)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_dir / "comparison_grid.png", dpi=130)
    plt.close(fig)
    written.append("comparison_grid.png")

    print(f"Wrote {len(written)} figures → {out_dir}", flush=True)
    for w in written:
        print(f"  {w}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
