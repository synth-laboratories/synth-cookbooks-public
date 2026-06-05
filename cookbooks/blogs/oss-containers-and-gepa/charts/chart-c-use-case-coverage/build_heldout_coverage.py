#!/usr/bin/env python3
"""Chart C — heldout coverage producer (Synth GEPA vs gepa-ai).

HARD RULE: all blog data is produced by code in this folder. This reads the
heldout evidence emitted by the shared GEPA eval harness
(../../../optimizers/gepa/evals/scripts/{evaluate_heldout,build_evidence}.py;
the harness's own docs say "the blog should consume the evidence") and emits the
cumulative curves the frontend `pareto-coverage-chart.tsx` renders.

Y-axis series (one active at a time in the UI):
  - heldout_coverage: distinct heldout seeds with reward >= threshold (union up to K)
  - best_heldout_score: max posthoc heldout_score among candidates 1..K
  - train_pareto_reward_sum: sum over pareto-train seeds of best train reward up to K
    (pareto train seed = train seed where final joint-Pareto candidates tie global best)
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BLOG_ROOT = ROOT.parents[1]
sys.path.insert(0, str(BLOG_ROOT))
from blog_paths import EVIDENCE_DIR, FRONTEND_DATA_DIR, REPO_ROOT  # noqa: E402

EVID = EVIDENCE_DIR
OUT = ROOT / "figures" / "use_case_heldout_coverage_data.json"
FRONTEND_OUT = FRONTEND_DATA_DIR / "use_case_heldout_coverage_data.json"
BENCHES = ["healthbench", "tau2_retail", "banking77", "hotpotqa"]


STACKS = ["synth_gepa", "gepa_ai"]


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def source_ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def summary(bench: str) -> dict:
    return json.loads((EVID / "benchmarks" / bench / "summary.json").read_text())


def _pareto_2d(points: list[tuple[float, float]]) -> set[int]:
    dominated = set()
    for i, (a0, a1) in enumerate(points):
        for j, (b0, b1) in enumerate(points):
            if i == j:
                continue
            if b0 >= a0 and b1 <= a1 and (b0 > a0 or b1 < a1):
                dominated.add(i)
                break
    return {i for i in range(len(points)) if i not in dominated}


def compute_pareto_frontier(timeline_rows: list[dict]) -> set[str]:
    """Joint Pareto candidate IDs: max heldout, min cost, min time (same as build_curves)."""
    valid = [r for r in timeline_rows if r.get("heldout_score") is not None]
    if not valid:
        return set()

    has_cost = any(r.get("cumulative_cost_usd") is not None for r in valid)
    has_time = any(r.get("elapsed_seconds") is not None for r in valid)
    frontier_ids: set[str] = {r["candidate_id"] for r in valid}

    if has_cost:
        pts = [(r.get("heldout_score", 0), r.get("cumulative_cost_usd", 0)) for r in valid]
        indices = _pareto_2d(pts)
        frontier_ids &= {valid[i]["candidate_id"] for i in indices}

    if has_time:
        pts = [(r.get("heldout_score", 0), r.get("elapsed_seconds") or 0) for r in valid]
        indices = _pareto_2d(pts)
        frontier_ids &= {valid[i]["candidate_id"] for i in indices}

    if not frontier_ids:
        best = max(valid, key=lambda r: r.get("heldout_score") or 0)
        return {best["candidate_id"]}
    return frontier_ids


def pareto_train_seeds(train_recs: list[dict], frontier_ids: set[str]) -> set[int]:
    """Train seeds where final joint-Pareto candidates achieve the global-best train reward."""
    by_seed: dict[int, list[tuple[str, float]]] = {}
    for r in train_recs:
        seed = int(r["heldout_seed"])
        by_seed.setdefault(seed, []).append((r["candidate_id"], float(r.get("reward") or 0)))

    out: set[int] = set()
    for seed, pairs in by_seed.items():
        global_best = max(p[1] for p in pairs)
        pareto_best = max((p[1] for p in pairs if p[0] in frontier_ids), default=-1.0)
        if pareto_best >= global_best - 1e-9:
            out.add(seed)
    return out


def candidate_order(records: list[dict]) -> tuple[list[str], dict[str, list[dict]]]:
    order: list[str] = []
    by_cand: dict[str, list[dict]] = {}
    for r in records:
        cid = r["candidate_id"]
        if cid not in by_cand:
            by_cand[cid] = []
            order.append(cid)
        by_cand[cid].append(r)
    return order, by_cand


def build_series_curves(
    heldout_recs: list[dict],
    train_recs: list[dict],
    threshold: float,
    timeline_by_cand: dict[str, dict],
    timeline_rows: list[dict],
) -> tuple[dict[str, list], list[int], list[float], list[float]]:
    order, heldout_by_cand = candidate_order(heldout_recs)
    _, train_by_cand = candidate_order(train_recs)

    frontier_ids = compute_pareto_frontier(timeline_rows)
    pareto_seeds = pareto_train_seeds(train_recs, frontier_ids)

    heldout_coverage: list[int] = []
    best_heldout: list[float] = []
    train_pareto_sum: list[float] = []

    x_candidate: list[int] = []
    x_time: list[float] = []
    x_cost: list[float] = []

    solved: set[int] = set()
    best_heldout_so_far = float("-inf")
    best_train_by_seed: dict[int, float] = {}

    for i, cid in enumerate(order):
        for r in heldout_by_cand[cid]:
            if (r.get("reward") or 0) >= threshold:
                solved.add(int(r["heldout_seed"]))
        heldout_coverage.append(len(solved))

        tl = timeline_by_cand.get(cid) or {}
        score = float(tl.get("heldout_score") or 0.0)
        best_heldout_so_far = max(best_heldout_so_far, score)
        best_heldout.append(round(best_heldout_so_far, 6))

        for r in train_by_cand.get(cid, []):
            seed = int(r["heldout_seed"])
            prev = best_train_by_seed.get(seed, 0.0)
            best_train_by_seed[seed] = max(prev, float(r.get("reward") or 0))
        train_pareto_sum.append(
            round(sum(best_train_by_seed.get(s, 0.0) for s in pareto_seeds), 6)
        )

        x_candidate.append(int(tl.get("candidate_index", i)))
        x_time.append(round(float(tl.get("elapsed_seconds") or 0.0), 3))
        x_cost.append(round(float(tl.get("cumulative_cost_usd") or 0.0), 6))

    y_series = {
        "heldout_coverage": heldout_coverage,
        "best_heldout_score": best_heldout,
        "train_pareto_reward_sum": train_pareto_sum,
    }
    meta = {
        "pareto_train_seed_count": len(pareto_seeds),
        "pareto_train_seeds": sorted(pareto_seeds),
        "joint_pareto_candidate_count": len(frontier_ids),
    }
    return y_series, meta, x_candidate, x_time, x_cost


def main() -> int:
    heldout_records = load_jsonl(EVID / "heldout_evaluations.jsonl")
    train_records = load_jsonl(EVID / "train_evaluations.jsonl")
    timeline_rows = load_jsonl(EVID / "candidate_timeline.jsonl")
    out = {
        "chart": "use_case_heldout_coverage",
        "generated_from": str(ROOT.relative_to(REPO_ROOT)),
        "source_evidence_path": str((ROOT / "figures" / "source_evidence.json").relative_to(REPO_ROOT)),
        "x_axes": {
            "candidate": "candidate index K (evaluated candidates, 0-based)",
            "time": "optimizer wall time at candidate evaluation (elapsed_seconds)",
            "cost": "cumulative metered cost at candidate evaluation (USD)",
        },
        "y_axes": {
            "heldout_coverage": "distinct heldout seeds solved (reward >= threshold) by union up to K",
            "best_heldout_score": "best posthoc heldout_score among candidates evaluated up to K",
            "train_pareto_reward_sum": (
                "sum of per-seed best train rewards on pareto-train seeds up to K "
                "(seed is pareto-train if final joint-Pareto candidates tie global best on that seed)"
            ),
        },
        "stacks": STACKS,
        "source": (
            "cookbooks/optimizers/gepa/evals/evidence/"
            "{heldout_evaluations,train_evaluations,candidate_timeline}.jsonl + summary.json"
        ),
        "benchmarks": [],
    }
    evidence = [
        {
            "source": "heldout_evaluations",
            "ref": source_ref(EVID / "heldout_evaluations.jsonl"),
        },
        {
            "source": "train_evaluations",
            "ref": source_ref(EVID / "train_evaluations.jsonl"),
        },
        {
            "source": "candidate_timeline",
            "ref": source_ref(EVID / "candidate_timeline.jsonl"),
        },
    ]
    for bench in BENCHES:
        sm = summary(bench)
        sm_path = EVID / "benchmarks" / bench / "summary.json"
        evidence.append({
            "benchmark": bench,
            "summary": source_ref(sm_path),
            "run_ids": {
                stack: (sm.get("per_stack") or {}).get(stack, {}).get("run_ids")
                for stack in STACKS
            },
        })
        threshold = sm["seed_coverage"]["heldout"]["coverage_reward_threshold"]
        total = sm["seed_coverage"]["heldout"]["seeds_total"]
        train_total = sm["seed_coverage"]["train"]["seeds_total"]
        entry = {
            "key": bench,
            "total_heldout_rows": total,
            "total_train_rows": train_total,
            "threshold": threshold,
            "series": {},
        }
        for stack in STACKS:
            heldout_recs = [
                r for r in heldout_records if r["benchmark"] == bench and r["stack"] == stack
            ]
            if not heldout_recs:
                continue
            train_recs = [
                r for r in train_records if r["benchmark"] == bench and r["stack"] == stack
            ]
            tl_for_stack = {
                r["candidate_id"]: r
                for r in timeline_rows
                if r["benchmark"] == bench and r["stack"] == stack
            }
            tl_rows = [
                r for r in timeline_rows if r["benchmark"] == bench and r["stack"] == stack
            ]
            y_series, meta, x_cand, x_time, x_cost = build_series_curves(
                heldout_recs,
                train_recs,
                threshold,
                tl_for_stack,
                tl_rows,
            )
            coverage = y_series["heldout_coverage"]
            expected = sm["per_stack"][stack]["heldout_cumulative_solved"]
            if coverage[-1] != expected:
                raise SystemExit(
                    f"VALIDATION FAILED {bench}/{stack}: coverage final {coverage[-1]} != summary {expected}"
                )
            entry["series"][stack] = {
                "run_id": sm["per_stack"][stack]["run_ids"][0],
                "covered": coverage,
                "y": y_series,
                "x": {
                    "candidate": x_cand,
                    "time": x_time,
                    "cost": x_cost,
                },
                "meta": meta,
                "seed": coverage[0],
                "final": coverage[-1],
                "beyond_seed": coverage[-1] - coverage[0],
                "best_heldout_score": sm["per_stack"][stack]["best_heldout_score"],
            }
            print(
                f"{bench:12} {stack:10} coverage={coverage[-1]}/{total} "
                f"best_heldout={y_series['best_heldout_score'][-1]:.4f} "
                f"train_pareto_sum={y_series['train_pareto_reward_sum'][-1]:.2f} "
                f"(pareto_train_seeds={meta['pareto_train_seed_count']}) ✓"
            )
        out["benchmarks"].append(entry)

    for path in (OUT, FRONTEND_OUT):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {path}")
    (ROOT / "figures" / "source_evidence.json").write_text(json.dumps({
        "chart": out["chart"],
        "generated_from": out["generated_from"],
        "evidence": evidence,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
