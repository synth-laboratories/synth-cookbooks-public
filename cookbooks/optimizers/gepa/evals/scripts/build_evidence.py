"""Assemble all evidence files into a compact, blog-ready package.

Reads the evidence/ directory and writes:
  evidence/run_manifest.json     — stacks, run IDs, versions, parity controls
  evidence/parity_controls.json  — extracted parity settings for both stacks
  evidence/source_checksums.json — SHA-256 of all tracked evidence files
  evidence/summary.json          — compact summary suitable for blog figures

Usage (from evals/):
    python scripts/build_evidence.py --benchmark banking77
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = EVALS_DIR / "evidence"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


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


def load_config(benchmark: str) -> dict:
    cfg_path = EVALS_DIR / "configs" / f"{benchmark}.toml"
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)


def benchmark_evidence_dir(benchmark: str) -> Path:
    path = EVIDENCE_DIR / "benchmarks" / benchmark
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(name: str, data: dict, *, benchmark_dir: Path) -> None:
    text = json.dumps(data, indent=2)
    (EVIDENCE_DIR / name).write_text(text)
    (benchmark_dir / name).write_text(text)


def write_text(name: str, text: str, *, benchmark_dir: Path) -> None:
    (EVIDENCE_DIR / name).write_text(text)
    (benchmark_dir / name).write_text(text)


def _run_summary(stack: str, candidates: list[dict]) -> dict:
    stack_cands = [c for c in candidates if c.get("stack") == stack]
    if not stack_cands:
        return {}
    run_ids = list({c["run_id"] for c in stack_cands})
    with_heldout = [c for c in stack_cands if c.get("heldout_score") is not None]
    scores = [c["heldout_score"] for c in with_heldout]
    seed_cand = next((c for c in stack_cands if c.get("candidate_index") == 0), None)
    best_cand = max(with_heldout, key=lambda c: c["heldout_score"]) if with_heldout else None

    def _max(field: str):
        vals = [c.get(field) for c in stack_cands if c.get(field) is not None]
        return max(vals) if vals else None

    return {
        "run_ids": run_ids,
        "num_candidates": len(stack_cands),
        "seed_heldout_score": seed_cand.get("heldout_score") if seed_cand else None,
        "best_heldout_score": max(scores) if scores else None,
        "mean_heldout_score": sum(scores) / len(scores) if scores else None,
        "lift_over_seed": (
            max(scores) - seed_cand.get("heldout_score", 0)
            if scores and seed_cand and seed_cand.get("heldout_score") is not None
            else None
        ),
        "candidates_with_heldout": len(scores),
        # Total compute spent during search (apples-to-apples axes).
        "total_rollouts": _max("cumulative_rollout_count"),
        "total_cost_usd": _max("cumulative_cost_usd"),
        "total_rollout_cost_usd": _max("cumulative_rollout_cost_usd"),
        "total_proposer_cost_usd": _max("cumulative_proposer_cost_usd"),
        "total_elapsed_seconds": _max("elapsed_seconds"),
        # Two cost scenarios for the proposer/reflection LLM:
        #   metered     = proposer billed per-token via the API (= total_cost_usd)
        #   with_codex_subscription = proposer covered by a flat ChatGPT/codex
        #     subscription, so its marginal token cost is $0 (rollout cost only).
        #   For gepa-ai both are equal (its reflection LM is always metered API).
        "cost_metered_usd": _max("cumulative_cost_usd"),
        "cost_with_codex_subscription_usd": _max("cumulative_rollout_cost_usd"),
        # Cost/time at the point the best candidate was found.
        "best_candidate_id": best_cand.get("candidate_id") if best_cand else None,
        "best_at_rollouts": best_cand.get("cumulative_rollout_count") if best_cand else None,
        "best_at_cost_usd": best_cand.get("cumulative_cost_usd") if best_cand else None,
        "best_at_elapsed_seconds": best_cand.get("elapsed_seconds") if best_cand else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Assemble compact evidence package.")
    p.add_argument("--benchmark", default="banking77")
    args = p.parse_args()

    cfg = load_config(args.benchmark)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_dir = benchmark_evidence_dir(args.benchmark)

    # Load candidate timeline.
    candidates = [
        c for c in load_jsonl(EVIDENCE_DIR / "candidate_timeline.jsonl")
        if c.get("benchmark") == args.benchmark
    ]
    stacks = sorted({c.get("stack", "") for c in candidates if c.get("stack")})

    # parity_controls.json — extracted controls from the benchmark config.
    parity = {
        "benchmark": args.benchmark,
        "config_path": str(EVALS_DIR / "configs" / f"{args.benchmark}.toml"),
        "train_seeds": cfg["dataset"]["train_seeds"],
        "heldout_seeds": cfg["dataset"]["heldout_seeds"],
        "train_shuffle_seed": cfg["benchmark"]["train_shuffle_seed"],
        "test_shuffle_seed": cfg["benchmark"]["test_shuffle_seed"],
        "coverage_reward_threshold": float(cfg["benchmark"].get("coverage_reward_threshold", 1.0)),
        "seed_candidate": cfg["seed_candidate"],
        "search_rollout_budget": cfg["limits"]["search_rollout_budget"],
        "pricing_usd_per_1m": cfg["limits"]["cost"],
        **cfg["parity_controls"],
    }
    write_json("parity_controls.json", parity, benchmark_dir=benchmark_dir)
    print(f"Wrote parity_controls.json", flush=True)

    # run_manifest.json — what ran, when, under which controls.
    commands = load_jsonl(EVIDENCE_DIR / "commands.jsonl")
    run_commands = [
        c for c in commands
        if c.get("phase") == "run" and c.get("benchmark") == args.benchmark
    ]
    runs_by_stack: dict[str, list[dict]] = {}
    for cmd in run_commands:
        runs_by_stack.setdefault(cmd.get("stack", ""), []).append(cmd)

    manifest = {
        "benchmark": args.benchmark,
        "stacks": stacks,
        "run_commands": run_commands,
        "num_candidates_total": len(candidates),
        "per_stack": {
            s: {
                "run_commands": runs_by_stack.get(s, []),
                **_run_summary(s, candidates),
            }
            for s in stacks
        },
        "parity_controls_path": str(EVIDENCE_DIR / "parity_controls.json"),
    }
    write_json("run_manifest.json", manifest, benchmark_dir=benchmark_dir)
    print(f"Wrote run_manifest.json", flush=True)

    # summary.json — compact, blog-ready.
    curve_points = [
        p for p in load_jsonl(EVIDENCE_DIR / "curve_points.jsonl")
        if p.get("benchmark") == args.benchmark
    ]
    reviews = [
        r for r in load_jsonl(EVIDENCE_DIR / "candidate_reviews.jsonl")
        if r.get("benchmark") == args.benchmark
    ]

    # Mean slope of the moving-average heldout curve (overall trend per candidate
    # index): total rise of the rolling-average curve over its index span.
    def _mean_rolling_slope(stack: str):
        pts = sorted(
            [p for p in curve_points if p.get("stack") == stack
             and p.get("metric") == "heldout_rolling_avg" and p.get("x_name") == "candidate_index"],
            key=lambda p: p["x"],
        )
        if len(pts) < 2 or pts[-1]["x"] == pts[0]["x"]:
            return None
        return (pts[-1]["y"] - pts[0]["y"]) / (pts[-1]["x"] - pts[0]["x"])

    # Joint Pareto frontier across BOTH stacks pooled, over
    # (maximize heldout, minimize metered cost, minimize elapsed time). Each
    # stack's share of the frontier shows who sits on the joint efficiency curve.
    joint_pts = [
        (c["stack"], c["candidate_id"], c["heldout_score"], c["cumulative_cost_usd"], c["elapsed_seconds"])
        for c in candidates
        if c.get("heldout_score") is not None and c.get("cumulative_cost_usd") is not None
        and c.get("elapsed_seconds") is not None
    ]
    joint_frontier: list[tuple[str, str]] = []
    for i, (si, ci, hi, coi, ti) in enumerate(joint_pts):
        dominated = False
        for j, (sj, cj, hj, coj, tj) in enumerate(joint_pts):
            if i == j:
                continue
            if hj >= hi and coj <= coi and tj <= ti and (hj > hi or coj < coi or tj < ti):
                dominated = True
                break
        if not dominated:
            joint_frontier.append((si, ci))
    joint_size = len(joint_frontier)

    def _joint_pareto_count(stack: str) -> int:
        return sum(1 for s, _c in joint_frontier if s == stack)

    coverage_reward_threshold = float(cfg["benchmark"].get("coverage_reward_threshold", 1.0))

    # Cumulative per-seed coverage: of a split's seeds, how many reached the
    # benchmark's configured coverage threshold by AT LEAST ONE candidate. For
    # exact-match tasks this is reward >= 1.0; for rubric-score tasks it can be
    # positive rubric credit.
    def _coverage_for_split(filename: str) -> tuple[dict[str, dict[str, set]], set, set]:
        solved: dict[str, dict[str, set]] = {}
        seeds: set = set()
        for r in load_jsonl(EVIDENCE_DIR / filename):
            if r.get("benchmark") != args.benchmark:
                continue
            seeds.add(r["heldout_seed"])
            if float(r.get("reward", 0) or 0) >= coverage_reward_threshold:
                solved.setdefault(r["stack"], {}).setdefault(r["candidate_id"], set()).add(r["heldout_seed"])
        joint: set = set()
        for byc in solved.values():
            for ss in byc.values():
                joint |= ss
        return solved, seeds, joint

    split_files = {"heldout": "heldout_evaluations.jsonl", "train": "train_evaluations.jsonl"}
    cov = {label: _coverage_for_split(fn) for label, fn in split_files.items()}

    def _coverage(stack: str) -> dict:
        out: dict = {}
        for label, (solved, seeds, _joint) in cov.items():
            by_cand = solved.get(stack, {})
            union: set = set()
            for ss in by_cand.values():
                union |= ss
            best_single = max((len(ss) for ss in by_cand.values()), default=0)
            n = len(seeds)
            out[f"{label}_seeds_total"] = n
            out[f"{label}_best_single_solved"] = best_single
            out[f"{label}_cumulative_solved"] = len(union)
            out[f"{label}_cumulative_coverage"] = (len(union) / n) if n else None
            out[f"{label}_breadth_gain"] = len(union) - best_single
        return out

    # Back-compat handles for the print/summary below (heldout split).
    _h_solved, _h_seeds, joint_union = cov["heldout"]
    n_heldout = len(_h_seeds)

    extra = {
        s: {
            "num_candidates_created": _run_summary(s, candidates).get("num_candidates"),
            "mean_rolling_heldout_slope": _mean_rolling_slope(s),
            "joint_pareto_points": _joint_pareto_count(s),
            "joint_pareto_share": (_joint_pareto_count(s) / joint_size) if joint_size else None,
            **_coverage(s),
        }
        for s in stacks
    }

    def _best_so_far_curve(stack: str) -> list[dict]:
        pts = [p for p in curve_points if p.get("stack") == stack and p.get("metric") == "best_heldout_so_far" and p.get("x_name") == "candidate_index"]
        return [{"x": p["x"], "y": p["y"]} for p in pts]

    def _pareto_coverage_curve(stack: str) -> list[dict]:
        pts = [p for p in curve_points if p.get("stack") == stack and p.get("metric") == "pareto_coverage" and p.get("x_name") == "candidate_index"]
        return [{"x": p["x"], "y": p["y"]} for p in pts]

    def _curve(stack: str, metric: str, x_name: str) -> list[dict]:
        pts = [p for p in curve_points if p.get("stack") == stack and p.get("metric") == metric and p.get("x_name") == x_name]
        return [{"x": p["x"], "y": p["y"]} for p in pts]

    summary = {
        "benchmark": args.benchmark,
        "generated_at": _now_iso(),
        "joint_pareto_frontier_size": joint_size,
        "joint_pareto_objectives": "maximize heldout_score, minimize cumulative_cost_usd (metered), minimize elapsed_seconds",
        "seed_coverage": {
            label: {
                "seeds_total": len(seeds),
                "coverage_reward_threshold": coverage_reward_threshold,
                "joint_cumulative_solved": len(joint),
                "never_solved": len(seeds) - len(joint),
            }
            for label, (_solved, seeds, joint) in cov.items()
        },
        "per_stack": {
            s: {
                **_run_summary(s, candidates),
                **extra[s],
                "best_so_far_curve": _best_so_far_curve(s),
                "best_so_far_by_rollouts": _curve(s, "best_heldout_so_far", "cumulative_rollout_count"),
                "best_so_far_by_cost": _curve(s, "best_heldout_so_far", "cumulative_cost_usd"),
                "best_so_far_by_time": _curve(s, "best_heldout_so_far", "elapsed_seconds"),
                "pareto_coverage_curve": _pareto_coverage_curve(s),
            }
            for s in stacks
        },
        "parity_controls": parity,
        "num_reviews": len(reviews),
        "caveats": [
            "Heldout scores are re-evaluated for every candidate of both stacks on "
            "the same dataset.heldout_seeds, independent of each stack's internal scoring.",
            "Cost is computed from recorded token usage via the shared [limits.cost] "
            "price table, applied identically to both stacks. Both proposer/reflection "
            "paths use gpt-5.4-mini through Codex app-server ChatGPT auth in this run; "
            "metered cost reports the token-accounted counterfactual, while "
            "with_codex_subscription treats those proposer/reflection tokens as "
            "marginal $0.",
            "Search-rollout budget is held equal across stacks "
            f"(search_rollout_budget={parity.get('search_rollout_budget')}); synth_gepa's "
            "internal heldout is pinned to the minimum (max_heldout_rollouts=1) and ignored.",
            "num_candidates counts candidates evaluated within budget. synth_gepa "
            "proposed additional candidates in late generations that the budget never "
            "let it evaluate; those are excluded from the comparison.",
            "joint_pareto_share uses metered cost; under a Codex subscription "
            "both stacks' cost-axis positions improve by removing the marginal "
            "proposer/reflection token charge.",
        ],
    }
    write_json("summary.json", summary, benchmark_dir=benchmark_dir)
    print(f"Wrote summary.json", flush=True)

    # source_checksums.json — checksums of all tracked evidence files.
    evidence_files = [
        "run_manifest.json",
        "parity_controls.json",
        "commands.jsonl",
        "proposal_timeline.jsonl",
        "candidate_timeline.jsonl",
        "heldout_evaluations.jsonl",
        "candidate_reviews.jsonl",
        "curve_points.jsonl",
        "summary.json",
    ]
    checksums = {}
    for fname in evidence_files:
        fpath = EVIDENCE_DIR / fname
        if fpath.exists():
            checksums[fname] = sha256_file(fpath)
    write_json("source_checksums.json", checksums, benchmark_dir=benchmark_dir)
    print(f"Wrote source_checksums.json ({len(checksums)} files checksummed)", flush=True)

    # Build comparison table (printed AND saved to summary_table.txt).
    def _fmt(v, spec: str) -> str:
        return format(v, spec) if isinstance(v, (int, float)) else "N/A"

    tbl: list[str] = []
    tbl.append(f"Synth GEPA vs gepa-ai — {args.benchmark}  (equal search-rollout budget = "
               f"{parity.get('search_rollout_budget')})")
    tbl.append("=" * 72)
    tbl.append("── Headline ──────────────────────────────────────────────────────")
    tbl.append(f"  joint Pareto frontier size (both stacks pooled): {joint_size}")
    for s in stacks:
        sr = _run_summary(s, candidates)
        n = sr.get("num_candidates", 0)
        share = extra[s]["joint_pareto_share"]
        share_str = f"{extra[s]['joint_pareto_points']}/{joint_size} ({share:.0%})" if share is not None else "N/A"
        tbl.append(
            f"  {s:20s} candidates={n:3d}  "
            f"seed={_fmt(sr.get('seed_heldout_score'), '.3f')}  "
            f"best={_fmt(sr.get('best_heldout_score'), '.3f')}  "
            f"lift={_fmt(sr.get('lift_over_seed'), '+.3f')}  "
            f"mov.avg slope={_fmt(extra[s]['mean_rolling_heldout_slope'], '+.4f')}/cand  "
            f"rollouts={_fmt(sr.get('total_rollouts'), 'd')}  "
            f"cost(metered)=${_fmt(sr.get('cost_metered_usd'), '.4f')}  "
            f"cost(codex-sub)=${_fmt(sr.get('cost_with_codex_subscription_usd'), '.4f')}  "
            f"time={_fmt(sr.get('total_elapsed_seconds'), '.0f')}s  "
            f"pareto={share_str}"
        )
    tbl.append("")
    tbl.append(
        "── Cumulative seed coverage "
        f"(reward >= {coverage_reward_threshold:g} by >=1 candidate) ──"
    )
    for s in stacks:
        e = extra[s]
        tbl.append(
            f"  {s:20s} "
            f"train={e['train_cumulative_solved']}/{e['train_seeds_total']} "
            f"(best 1 cand {e['train_best_single_solved']})   "
            f"heldout={e['heldout_cumulative_solved']}/{e['heldout_seeds_total']} "
            f"(best 1 cand {e['heldout_best_single_solved']})"
        )
    for label, (_solved, seeds, joint) in cov.items():
        tbl.append(
            f"  joint {label:7s}: {len(joint)}/{len(seeds)} ever solved  |  "
            f"never solved by anyone: {len(seeds) - len(joint)}"
        )

    table_text = "\n".join(tbl)
    print("\n" + table_text, flush=True)
    write_text("summary_table.txt", table_text + "\n", benchmark_dir=benchmark_dir)
    print(f"\nWrote summary_table.txt", flush=True)

    return 0


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


if __name__ == "__main__":
    sys.exit(main())
