"""Render the comparison curves as overlaid ASCII line charts (terminal-friendly).

Reads evidence/curve_points.jsonl and prints best-heldout-so-far against candidate
index, cumulative rollouts, wall-clock time, and cumulative cost, plus Pareto
coverage — both stacks overlaid on each chart. Also writes the same text to
evidence/ascii_charts.txt.

  legend:  ☆ = gepa-ai    ★ = synth_gepa    ✦ = both

Usage (from evals/):
    python scripts/plot_ascii.py --benchmark banking77
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = EVALS_DIR / "evidence"

MARK = {"gepa_ai": "☆", "synth_gepa": "★"}
OVERLAP = "✦"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def benchmark_evidence_dir(benchmark: str) -> Path:
    path = EVIDENCE_DIR / "benchmarks" / benchmark
    path.mkdir(parents=True, exist_ok=True)
    return path


def series(curves: list[dict], stack: str, metric: str, x_name: str) -> list[tuple[float, float]]:
    pts = [(p["x"], p["y"]) for p in curves
           if p.get("stack") == stack and p.get("metric") == metric and p.get("x_name") == x_name]
    return sorted(pts, key=lambda t: t[0])


def _step_value(pts: list[tuple[float, float]], x: float) -> float | None:
    """Carry-forward (step) value at x: last y whose x' <= x."""
    val = None
    for px, py in pts:
        if px <= x + 1e-9:
            val = py
        else:
            break
    return val


def render(series_by_stack: dict[str, list[tuple[float, float]]], title: str,
           xlabel: str, ylabel: str, width: int = 64, height: int = 16,
           step: bool = True, x_int: bool = False) -> str:
    allpts = [p for pts in series_by_stack.values() for p in pts]
    if not allpts:
        return f"{title}\n  (no data)\n"
    xs = [p[0] for p in allpts]
    ys = [p[1] for p in allpts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymax == ymin:
        ymax = ymin + 1e-6
    if xmax == xmin:
        xmax = xmin + 1e-6
    # Pad y a touch.
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad

    grid = [[" "] * width for _ in range(height)]

    def col_x(c: int) -> float:
        return xmin + (xmax - xmin) * c / (width - 1)

    def row_of(y: float) -> int:
        r = int(round((ymax - y) / (ymax - ymin) * (height - 1)))
        return max(0, min(height - 1, r))

    def col_of(x: float) -> int:
        c = int(round((x - xmin) / (xmax - xmin) * (width - 1)))
        return max(0, min(width - 1, c))

    def place(r: int, c: int, m: str) -> None:
        cur = grid[r][c]
        grid[r][c] = OVERLAP if (cur != " " and cur != m) else m

    for stack, pts in series_by_stack.items():
        if not pts:
            continue
        m = MARK.get(stack, "?")
        if step:
            # Staircase: carry-forward value across every column at/after first x.
            for c in range(width):
                x = col_x(c)
                if x < pts[0][0]:
                    continue
                y = _step_value(pts, x)
                if y is None:
                    continue
                place(row_of(y), c, m)
        else:
            # Line/scatter: place each data point, connect consecutive points.
            prev = None
            for px, py in pts:
                c, r = col_of(px), row_of(py)
                if prev is not None:
                    pc, pr = prev
                    steps = max(abs(c - pc), abs(r - pr)) or 1
                    for k in range(1, steps + 1):
                        ic = int(round(pc + (c - pc) * k / steps))
                        ir = int(round(pr + (r - pr) * k / steps))
                        place(ir, ic, m)
                place(r, c, m)
                prev = (c, r)

    # Assemble with y-axis labels.
    lines = [f"{title}"]
    for r in range(height):
        yval = ymax - (ymax - ymin) * r / (height - 1)
        axis = f"{yval:6.3f} |"
        lines.append(axis + "".join(grid[r]))
    lines.append(" " * 7 + "+" + "-" * width)
    # x ticks at start / mid / end
    def fmt_x(v: float) -> str:
        return f"{v:.0f}" if x_int else (f"{v:.2f}" if v < 100 else f"{v:.0f}")
    left = fmt_x(xmin)
    mid = fmt_x((xmin + xmax) / 2)
    right = fmt_x(xmax)
    tick = " " * 8 + left
    tick += " " * max(1, (width // 2) - len(tick) + 8) + mid
    tick += " " * max(1, (width + 8) - len(tick) - len(right)) + right
    lines.append(tick)
    lines.append(" " * 8 + xlabel + f"   (y = {ylabel})")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="ASCII comparison charts.")
    p.add_argument("--benchmark", default="banking77")
    args = p.parse_args()

    curves = [
        r for r in load_jsonl(EVIDENCE_DIR / "curve_points.jsonl")
        if r.get("benchmark") == args.benchmark
    ]
    if not curves:
        raise SystemExit("curve_points.jsonl is empty for the requested benchmark — run build_curves.py first")
    stacks = sorted({c["stack"] for c in curves})

    charts = [
        ("best_heldout_so_far", "candidate_index", "candidate index", "best heldout", True, True),
        ("best_heldout_so_far", "cumulative_rollout_count", "cumulative rollouts (equal budget)", "best heldout", True, True),
        ("best_heldout_so_far", "elapsed_seconds", "wall-clock seconds", "best heldout", True, False),
        ("best_heldout_so_far", "cumulative_cost_usd", "cumulative cost USD (metered)", "best heldout", True, False),
        ("pareto_coverage", "candidate_index", "candidate index", "Pareto coverage", True, True),
        ("heldout_rolling_avg", "candidate_index", "candidate index", "rolling-avg heldout (w=5)", False, True),
    ]

    out_lines = [
        f"Synth GEPA (★) vs gepa-ai (☆) — {args.benchmark}  [✦ = overlap]",
        "=" * 72,
        "",
    ]
    for metric, x_name, xlabel, ylabel, step, x_int in charts:
        sbs = {s: series(curves, s, metric, x_name) for s in stacks}
        title = f"▸ {ylabel}  vs  {xlabel}"
        out_lines.append(render(sbs, title, xlabel, ylabel, step=step, x_int=x_int))
        out_lines.append("")

    charts_text = "\n".join(out_lines)
    print(charts_text)
    benchmark_dir = benchmark_evidence_dir(args.benchmark)
    (EVIDENCE_DIR / "ascii_charts.txt").write_text(charts_text)
    (benchmark_dir / "ascii_charts.txt").write_text(charts_text)

    # Combined report: summary table (from build_evidence) + charts, one file.
    table_path = EVIDENCE_DIR / "summary_table.txt"
    report_parts = []
    if table_path.exists():
        report_parts.append(table_path.read_text().rstrip())
        report_parts.append("\n" + "=" * 72 + "\n")
    report_parts.append(charts_text)
    report_path = EVIDENCE_DIR / "comparison_report.txt"
    report_path.write_text("\n".join(report_parts) + "\n")
    (benchmark_dir / "comparison_report.txt").write_text("\n".join(report_parts) + "\n")

    print(f"\n(written: {EVIDENCE_DIR / 'ascii_charts.txt'})", flush=True)
    if table_path.exists():
        print(f"(written: {report_path}  — table + charts)", flush=True)
    else:
        print("(run build_evidence.py first to include the summary table in "
              "comparison_report.txt)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
