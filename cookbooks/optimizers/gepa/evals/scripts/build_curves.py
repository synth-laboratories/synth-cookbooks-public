"""Build comparison curves from candidate_timeline.jsonl.

Generates:
  - rolling heldout average by candidate index (window=5)
  - cumulative heldout average by candidate index
  - best heldout so far by candidate index
  - best heldout so far by elapsed time
  - best heldout so far by cumulative cost
  - Pareto coverage by candidate index (fraction of final Pareto frontier discovered)
  - Pareto coverage by elapsed time
  - Pareto coverage by cumulative cost

Writes evidence/curve_points.jsonl — one row per (stack, run_id, metric, x_name, x, y).

Usage (from evals/):
    python scripts/build_curves.py
    python scripts/build_curves.py --candidate-timeline evidence/candidate_timeline.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = EVALS_DIR / "evidence"
ROLLING_WINDOW = 5


def load_timeline(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def benchmark_evidence_dir(benchmark: str) -> Path:
    path = EVIDENCE_DIR / "benchmarks" / benchmark
    path.mkdir(parents=True, exist_ok=True)
    return path


def _group_by_run(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r.get("stack", ""), r.get("benchmark", ""), r.get("run_id", ""))
        groups.setdefault(key, []).append(r)
    return groups


def _sort_by_index(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: int(r.get("candidate_index") or 0))


def _pareto_2d(points: list[tuple[float, float]]) -> set[int]:
    """Return indices of Pareto-optimal points maximizing dim0, minimizing dim1."""
    dominated = set()
    for i, (a0, a1) in enumerate(points):
        for j, (b0, b1) in enumerate(points):
            if i == j:
                continue
            if b0 >= a0 and b1 <= a1 and (b0 > a0 or b1 < a1):
                dominated.add(i)
                break
    return {i for i in range(len(points)) if i not in dominated}


def compute_pareto_frontier(rows: list[dict]) -> set[str]:
    """Return candidate_ids on the 3-objective Pareto frontier (maximize heldout, minimize cost, minimize time)."""
    valid = [r for r in rows if r.get("heldout_score") is not None]
    if not valid:
        return set()

    has_cost = any(r.get("cumulative_cost_usd") is not None for r in valid)
    has_time = any(r.get("elapsed_seconds") is not None for r in valid)

    # Reduce to 2D if only heldout is available, then 3D Pareto via repeated 2D.
    frontier_ids: set[str] = {r["candidate_id"] for r in valid}

    if has_cost:
        pts = [(r.get("heldout_score", 0), r.get("cumulative_cost_usd", 0)) for r in valid]
        indices = _pareto_2d(pts)
        cost_frontier = {valid[i]["candidate_id"] for i in indices}
        frontier_ids &= cost_frontier

    if has_time:
        pts = [(r.get("heldout_score", 0), r.get("elapsed_seconds") or 0) for r in valid]
        indices = _pareto_2d(pts)
        time_frontier = {valid[i]["candidate_id"] for i in indices}
        frontier_ids &= time_frontier

    # Fallback: if intersection is empty (can happen with 3D reduction), just maximize heldout.
    if not frontier_ids:
        best = max(valid, key=lambda r: r.get("heldout_score") or 0)
        return {best["candidate_id"]}

    return frontier_ids


def rolling_average(scores: list[float], window: int) -> list[float]:
    result = []
    for i in range(len(scores)):
        start = max(0, i - window + 1)
        chunk = scores[start:i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def cumulative_average(scores: list[float]) -> list[float]:
    result = []
    total = 0.0
    for i, s in enumerate(scores):
        total += s
        result.append(total / (i + 1))
    return result


def best_so_far(scores: list[float]) -> list[float]:
    result = []
    best = float("-inf")
    for s in scores:
        best = max(best, s)
        result.append(best)
    return result


def build_curves_for_run(rows: list[dict], final_frontier: set[str]) -> list[dict]:
    rows = _sort_by_index(rows)
    stack = rows[0]["stack"]
    benchmark = rows[0].get("benchmark", "")
    run_id = rows[0]["run_id"]

    scores = [float(r.get("heldout_score") or 0.0) for r in rows]
    indices = list(range(len(rows)))

    out: list[dict] = []

    def _row(metric: str, x_name: str, x_vals: list, y_vals: list) -> None:
        for x, y in zip(x_vals, y_vals):
            out.append({
                "stack": stack,
                "benchmark": benchmark,
                "run_id": run_id,
                "metric": metric,
                "x_name": x_name,
                "x": x,
                "y": y,
            })

    # Heldout curves by candidate index.
    _row("heldout_rolling_avg", "candidate_index", indices, rolling_average(scores, ROLLING_WINDOW))
    _row("heldout_cumulative_avg", "candidate_index", indices, cumulative_average(scores))
    _row("best_heldout_so_far", "candidate_index", indices, best_so_far(scores))

    # Best heldout by elapsed time.
    elapsed_vals = [r.get("elapsed_seconds") for r in rows]
    if any(v is not None for v in elapsed_vals):
        # Sort by elapsed and recompute.
        paired = sorted(
            [(e or 0.0, s) for e, s in zip(elapsed_vals, scores)],
            key=lambda x: x[0],
        )
        t_vals = [p[0] for p in paired]
        s_vals = [p[1] for p in paired]
        _row("best_heldout_so_far", "elapsed_seconds", t_vals, best_so_far(s_vals))

    # Best heldout by cumulative cost.
    cost_vals = [r.get("cumulative_cost_usd") for r in rows]
    if any(v is not None for v in cost_vals):
        paired = sorted(
            [(c or 0.0, s) for c, s in zip(cost_vals, scores)],
            key=lambda x: x[0],
        )
        c_vals = [p[0] for p in paired]
        s_vals = [p[1] for p in paired]
        _row("best_heldout_so_far", "cumulative_cost_usd", c_vals, best_so_far(s_vals))

    # Best heldout by cumulative rollout count (the apples-to-apples compute axis).
    rollout_vals = [r.get("cumulative_rollout_count") for r in rows]
    if any(v is not None for v in rollout_vals):
        paired = sorted(
            [(n or 0, s) for n, s in zip(rollout_vals, scores)],
            key=lambda x: x[0],
        )
        n_vals = [p[0] for p in paired]
        s_vals = [p[1] for p in paired]
        _row("best_heldout_so_far", "cumulative_rollout_count", n_vals, best_so_far(s_vals))

    # Pareto coverage by candidate index.
    if final_frontier:
        n = len(final_frontier)
        covered_ids: set[str] = set()
        coverage_by_idx = []
        for r in rows:
            if r["candidate_id"] in final_frontier:
                covered_ids.add(r["candidate_id"])
            coverage_by_idx.append(len(covered_ids) / n)
        _row("pareto_coverage", "candidate_index", indices, coverage_by_idx)

        # Pareto coverage by elapsed time.
        if any(v is not None for v in elapsed_vals):
            paired = sorted(
                [(e or 0.0, r["candidate_id"]) for e, r in zip(elapsed_vals, rows)],
                key=lambda x: x[0],
            )
            covered_ids = set()
            t_vals = []
            cov_vals = []
            for t, cid in paired:
                if cid in final_frontier:
                    covered_ids.add(cid)
                t_vals.append(t)
                cov_vals.append(len(covered_ids) / n)
            _row("pareto_coverage", "elapsed_seconds", t_vals, cov_vals)

        # Pareto coverage by cumulative cost.
        if any(v is not None for v in cost_vals):
            paired = sorted(
                [(c or 0.0, r["candidate_id"]) for c, r in zip(cost_vals, rows)],
                key=lambda x: x[0],
            )
            covered_ids = set()
            c_vals = []
            cov_vals = []
            for c, cid in paired:
                if cid in final_frontier:
                    covered_ids.add(cid)
                c_vals.append(c)
                cov_vals.append(len(covered_ids) / n)
            _row("pareto_coverage", "cumulative_cost_usd", c_vals, cov_vals)

        # Pareto coverage by cumulative rollout count.
        if any(v is not None for v in rollout_vals):
            paired = sorted(
                [(v or 0, r["candidate_id"]) for v, r in zip(rollout_vals, rows)],
                key=lambda x: x[0],
            )
            covered_ids = set()
            n_vals = []
            cov_vals = []
            for v, cid in paired:
                if cid in final_frontier:
                    covered_ids.add(cid)
                n_vals.append(v)
                cov_vals.append(len(covered_ids) / n)
            _row("pareto_coverage", "cumulative_rollout_count", n_vals, cov_vals)

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Build comparison curves.")
    p.add_argument(
        "--candidate-timeline",
        default=str(EVIDENCE_DIR / "candidate_timeline.jsonl"),
    )
    p.add_argument("--output", default=str(EVIDENCE_DIR / "curve_points.jsonl"))
    p.add_argument("--benchmark", default=None)
    args = p.parse_args()

    timeline_path = Path(args.candidate_timeline)
    if not timeline_path.exists():
        raise SystemExit(f"candidate_timeline not found: {timeline_path}")

    rows = load_timeline(timeline_path)
    if args.benchmark:
        rows = [r for r in rows if r.get("benchmark") == args.benchmark]
    if not rows:
        raise SystemExit("candidate_timeline.jsonl is empty for the requested benchmark")

    groups = _group_by_run(rows)
    print(f"Loaded {len(rows)} candidates across {len(groups)} run(s)", flush=True)

    # Compute the global Pareto frontier across all runs together, per stack.
    per_stack: dict[str, list[dict]] = {}
    for r in rows:
        per_stack.setdefault(r.get("stack", ""), []).append(r)

    all_curves: list[dict] = []
    for (stack, bench, run_id), run_rows in groups.items():
        # Final frontier = union across all runs of same stack.
        stack_rows = per_stack.get(stack, [])
        frontier = compute_pareto_frontier(stack_rows)
        curves = build_curves_for_run(run_rows, frontier)
        all_curves.extend(curves)
        best_score = max((r.get("heldout_score") or 0.0) for r in run_rows)
        print(
            f"  {stack} / {run_id[:30]}: {len(run_rows)} candidates, "
            f"best_heldout={best_score:.3f}, pareto_frontier={len(frontier)}",
            flush=True,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in all_curves:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(all_curves)} curve points → {out_path}", flush=True)
    if args.benchmark:
        scoped_path = benchmark_evidence_dir(args.benchmark) / "curve_points.jsonl"
        with open(scoped_path, "w") as f:
            for row in all_curves:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote benchmark-scoped curve points → {scoped_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
