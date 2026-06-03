#!/usr/bin/env python3
"""Chart C — heldout coverage producer (Synth GEPA vs gepa-ai).

HARD RULE: all blog data is produced by code in this folder. This reads the
heldout evidence emitted by the shared GEPA eval harness
(../../../optimizers/gepa/evals/scripts/{evaluate_heldout,build_evidence}.py;
the harness's own docs say "the blog should consume the evidence") and emits the
cumulative heldout-coverage curves the frontend `pareto-coverage-chart.tsx`
renders. No hand-authored numbers.

Metric: cumulative heldout coverage = count of distinct heldout seeds with
reward >= the per-benchmark threshold by any candidate up to index K, over each
task's final heldout split — for each stack (synth_gepa, gepa_ai).

Self-check: each curve's final value is asserted against the harness's published
`heldout_cumulative_solved` in summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

EVALS = Path(__file__).resolve().parents[3] / "optimizers/gepa/evals"
EVID = EVALS / "evidence"
OUT = Path(__file__).resolve().parent / "figures" / "use_case_heldout_coverage_data.json"
WORKSPACE = Path("/Users/joshpurtell/Documents/GitHub")
FRONTEND_OUT = (
    WORKSPACE
    / "frontend"
    / "src"
    / "components"
    / "blog"
    / "posts"
    / "introducing-gepa-platform"
    / "data"
    / "use_case_heldout_coverage_data.json"
)
BENCHES = ["healthbench", "harvey_lab", "tau2_retail", "dungeongrid"]
STACKS = ["synth_gepa", "gepa_ai"]


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def summary(bench: str) -> dict:
    return json.loads((EVID / "benchmarks" / bench / "summary.json").read_text())


def coverage_curve(records: list[dict], threshold: float) -> tuple[list[int], int]:
    """records: heldout eval rows for one (benchmark, stack). Returns
    (cumulative distinct solved seeds per candidate index, total seeds)."""
    # candidate order = order of first appearance (harness writes in gen order)
    order: list[str] = []
    by_cand: dict[str, list[dict]] = {}
    for r in records:
        cid = r["candidate_id"]
        if cid not in by_cand:
            by_cand[cid] = []
            order.append(cid)
        by_cand[cid].append(r)
    seeds = {r["heldout_seed"] for r in records}
    solved: set[int] = set()
    curve: list[int] = []
    for cid in order:
        for r in by_cand[cid]:
            if (r.get("reward") or 0) >= threshold:
                solved.add(r["heldout_seed"])
        curve.append(len(solved))
    return curve, len(seeds)


def main() -> int:
    all_records = load_jsonl(EVID / "heldout_evaluations.jsonl")
    out = {
        "chart": "use_case_heldout_coverage",
        "metric": "cumulative_heldout_coverage",
        "definition": "distinct heldout seeds solved by any candidate up to K (per-benchmark reward threshold)",
        "stacks": STACKS,
        "source": "cookbooks/optimizers/gepa/evals/evidence/heldout_evaluations.jsonl (+ benchmark summary.json)",
        "benchmarks": [],
    }
    for bench in BENCHES:
        sm = summary(bench)
        threshold = sm["seed_coverage"]["heldout"]["coverage_reward_threshold"]
        total = sm["seed_coverage"]["heldout"]["seeds_total"]
        entry = {"key": bench, "total_heldout_rows": total, "threshold": threshold, "series": {}}
        for stack in STACKS:
            recs = [r for r in all_records if r["benchmark"] == bench and r["stack"] == stack]
            if not recs:
                continue
            curve, n_seeds = coverage_curve(recs, threshold)
            expected = sm["per_stack"][stack]["heldout_cumulative_solved"]
            if curve[-1] != expected:
                raise SystemExit(
                    f"VALIDATION FAILED {bench}/{stack}: curve final {curve[-1]} != summary {expected}"
                )
            entry["series"][stack] = {
                "run_id": sm["per_stack"][stack]["run_ids"][0],
                "covered": curve,
                "seed": curve[0],
                "final": curve[-1],
                "beyond_seed": curve[-1] - curve[0],
                "best_heldout_score": sm["per_stack"][stack]["best_heldout_score"],
            }
            print(f"{bench:12} {stack:10} /{total}: final={curve[-1]} (+{curve[-1]-curve[0]}) ✓ matches summary")
        out["benchmarks"].append(entry)

    for path in (OUT, FRONTEND_OUT):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
