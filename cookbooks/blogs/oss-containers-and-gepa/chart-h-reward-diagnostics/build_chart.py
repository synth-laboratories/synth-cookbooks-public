#!/usr/bin/env python3
"""Build reward diagnostics for the GEPA platform blog from final evidence rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parent
COOKBOOK_REPO = ROOT.parents[3]
WORKSPACE = Path("/Users/joshpurtell/Documents/GitHub")
EVALS = COOKBOOK_REPO / "cookbooks" / "optimizers" / "gepa" / "evals"
EVIDENCE = EVALS / "evidence"
HELDOUT_PATH = EVIDENCE / "heldout_evaluations.jsonl"
HARVEY_SUMMARY_PATH = EVIDENCE / "benchmarks" / "harvey_lab" / "summary.json"
DUNGEONGRID_SUMMARY_PATH = EVIDENCE / "benchmarks" / "dungeongrid" / "summary.json"
DEFAULT_FRONTEND_OUTPUT = (
    WORKSPACE
    / "frontend"
    / "src"
    / "components"
    / "blog"
    / "posts"
    / "introducing-gepa-platform"
    / "data"
    / "reward_diagnostics_data.json"
)

STACKS = [
    ("synth_gepa", "Synth GEPA"),
    ("gepa_ai", "gepa-ai"),
]

HARVEY_BINS = [
    ("zero", "0", 0.0, 0.0),
    ("lt_0_05", "0-0.05", 0.0, 0.05),
    ("lt_0_15", "0.05-0.15", 0.05, 0.15),
    ("lt_0_30", "0.15-0.30", 0.15, 0.30),
    ("gte_0_30", ">=0.30", 0.30, None),
]

DUNGEONGRID_SIGNALS = [
    ("objective.item_recovered", "Objective recovered"),
    ("treasure.first_treasure_collected", "Treasure collected"),
    ("treasure.first_chest_opened", "Chest opened"),
    ("hazards.trap_disarmed", "Trap disarmed"),
    ("doors.first_door_opened", "Door opened"),
    ("exploration.first_room_revealed", "Room revealed"),
]


def repo_path(path: Path) -> str:
    resolved = path.resolve()
    for root in (COOKBOOK_REPO, WORKSPACE):
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            continue
    return str(resolved)


def source_ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": repo_path(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def number(value: Any) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"expected numeric reward, got {value!r}")
    return float(value)


def best_candidate(summary: dict[str, Any], stack_id: str) -> str:
    try:
        candidate_id = summary["per_stack"][stack_id]["best_candidate_id"]
    except KeyError as exc:
        raise KeyError(f"summary missing best candidate for {stack_id}") from exc
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError(f"invalid best candidate id for {stack_id}: {candidate_id!r}")
    return candidate_id


def rows_for(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    stack_id: str,
    candidate_id: str,
) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if row.get("benchmark") == benchmark
        and row.get("stack") == stack_id
        and row.get("candidate_id") == candidate_id
    ]
    filtered.sort(key=lambda row: int(row.get("heldout_seed") or 0))
    return filtered


def summarize_rewards(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [number(row.get("reward")) for row in rows]
    if not rewards:
        raise ValueError("cannot summarize empty reward rows")
    positive = sum(1 for reward in rewards if reward > 0)
    return {
        "n": len(rewards),
        "mean_reward": mean(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "positive_rows": positive,
        "zero_rows": len(rewards) - positive,
    }


def harvey_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rewards = [number(row.get("reward")) for row in rows]
    if not rewards:
        raise ValueError("cannot bin empty Harvey reward rows")
    bins: list[dict[str, Any]] = []
    for key, label, lower, upper in HARVEY_BINS:
        if key == "zero":
            count = sum(1 for reward in rewards if reward == 0)
        elif upper is None:
            count = sum(1 for reward in rewards if reward >= lower)
        else:
            count = sum(1 for reward in rewards if lower < reward < upper)
        bins.append({
            "key": key,
            "label": label,
            "count": count,
            "n": len(rewards),
            "rate": count / len(rewards),
        })
    return bins


def build_harvey(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    stacks = []
    rows_by_seed: dict[int, dict[str, float]] = {}
    for stack_id, stack_label in STACKS:
        candidate_id = best_candidate(summary, stack_id)
        stack_rows = rows_for(
            rows,
            benchmark="harvey_lab",
            stack_id=stack_id,
            candidate_id=candidate_id,
        )
        if not stack_rows:
            raise ValueError(f"no Harvey heldout rows for {stack_id} {candidate_id}")
        for row in stack_rows:
            seed = int(row["heldout_seed"])
            rows_by_seed.setdefault(seed, {})[stack_id] = number(row.get("reward"))
        stack_summary = dict(summarize_rewards(stack_rows))
        stack_summary.update({
            "stack_id": stack_id,
            "stack_label": stack_label,
            "candidate_id": candidate_id,
            "summary_best_heldout": summary["per_stack"][stack_id]["best_heldout_score"],
            "summary_seed_heldout": summary["per_stack"][stack_id]["seed_heldout_score"],
            "lift_over_seed": summary["per_stack"][stack_id]["lift_over_seed"],
            "bins": harvey_bins(stack_rows),
            "rows": [
                {
                    "heldout_seed": int(row["heldout_seed"]),
                    "reward": number(row.get("reward")),
                    "prompt_tokens": int(row.get("prompt_tokens") or 0),
                    "completion_tokens": int(row.get("completion_tokens") or 0),
                }
                for row in stack_rows
            ],
        })
        stacks.append(stack_summary)

    seed_deltas = []
    for seed in sorted(rows_by_seed):
        entry = rows_by_seed[seed]
        synth_reward = entry.get("synth_gepa")
        gepa_reward = entry.get("gepa_ai")
        if synth_reward is None or gepa_reward is None:
            raise ValueError(f"unpaired Harvey reward row for heldout seed {seed}")
        seed_deltas.append({
            "heldout_seed": seed,
            "synth_gepa": synth_reward,
            "gepa_ai": gepa_reward,
            "delta": synth_reward - gepa_reward,
        })

    return {
        "benchmark": "harvey_lab",
        "task_label": "Harvey Lab Tax",
        "metric_label": "Fractional row reward",
        "coverage_threshold": summary["parity_controls"]["coverage_reward_threshold"],
        "stacks": stacks,
        "seed_deltas": seed_deltas,
    }


def achievement_set(row: dict[str, Any]) -> set[str]:
    details = row.get("reward_details") or {}
    achievements = details.get("achievements") or []
    return {str(value) for value in achievements}


def signal_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("cannot compute signal rates from empty rows")
    rates = []
    for signal, label in DUNGEONGRID_SIGNALS:
        hits = sum(1 for row in rows if signal in achievement_set(row))
        rates.append({
            "signal": signal,
            "label": label,
            "count": hits,
            "n": len(rows),
            "rate": hits / len(rows),
        })
    return rates


def build_dungeongrid(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    stacks = []
    signals_by_stack: dict[str, dict[str, dict[str, Any]]] = {}
    for stack_id, stack_label in STACKS:
        candidate_id = best_candidate(summary, stack_id)
        stack_rows = rows_for(
            rows,
            benchmark="dungeongrid",
            stack_id=stack_id,
            candidate_id=candidate_id,
        )
        if not stack_rows:
            raise ValueError(f"no DungeonGrid heldout rows for {stack_id} {candidate_id}")
        signals = signal_rates(stack_rows)
        signals_by_stack[stack_id] = {signal["signal"]: signal for signal in signals}
        stack_summary = dict(summarize_rewards(stack_rows))
        stack_summary.update({
            "stack_id": stack_id,
            "stack_label": stack_label,
            "candidate_id": candidate_id,
            "summary_best_heldout": summary["per_stack"][stack_id]["best_heldout_score"],
            "summary_seed_heldout": summary["per_stack"][stack_id]["seed_heldout_score"],
            "lift_over_seed": summary["per_stack"][stack_id]["lift_over_seed"],
            "signals": signals,
            "rows": [
                {
                    "heldout_seed": int(row["heldout_seed"]),
                    "example_id": (row.get("reward_details") or {}).get("example_id"),
                    "reward": number(row.get("reward")),
                    "n_steps": (row.get("reward_details") or {}).get("n_steps"),
                    "policy_calls": (row.get("reward_details") or {}).get("policy_calls"),
                    "achievements": sorted(achievement_set(row)),
                }
                for row in stack_rows
            ],
        })
        stacks.append(stack_summary)

    signal_deltas = []
    for signal, label in DUNGEONGRID_SIGNALS:
        synth_signal = signals_by_stack["synth_gepa"][signal]
        gepa_signal = signals_by_stack["gepa_ai"][signal]
        synth_rate = synth_signal["rate"]
        gepa_rate = gepa_signal["rate"]
        signal_deltas.append({
            "signal": signal,
            "label": label,
            "synth_gepa_rate": synth_rate,
            "gepa_ai_rate": gepa_rate,
            "delta": synth_rate - gepa_rate,
            "synth_gepa_count": synth_signal["count"],
            "gepa_ai_count": gepa_signal["count"],
            "n": synth_signal["n"],
        })

    return {
        "benchmark": "dungeongrid",
        "task_label": "DungeonGrid",
        "metric_label": "Heldout reward and achievement rates",
        "coverage_threshold": summary["parity_controls"]["coverage_reward_threshold"],
        "stacks": stacks,
        "signal_deltas": signal_deltas,
    }


def build() -> dict[str, Any]:
    missing = [
        repo_path(path)
        for path in (HELDOUT_PATH, HARVEY_SUMMARY_PATH, DUNGEONGRID_SUMMARY_PATH)
        if not path.exists()
    ]
    if missing:
        raise SystemExit(f"Chart H missing launch artifacts: {', '.join(missing)}")

    heldout_rows = read_jsonl(HELDOUT_PATH)
    harvey_summary = read_json(HARVEY_SUMMARY_PATH)
    dungeongrid_summary = read_json(DUNGEONGRID_SUMMARY_PATH)
    return {
        "status": "ready",
        "chart": "reward_diagnostics",
        "generated_from": repo_path(ROOT / "build_chart.py"),
        "sources": {
            "producer": source_ref(ROOT / "build_chart.py"),
            "heldout_evaluations": source_ref(HELDOUT_PATH),
            "harvey_summary": source_ref(HARVEY_SUMMARY_PATH),
            "dungeongrid_summary": source_ref(DUNGEONGRID_SUMMARY_PATH),
        },
        "stacks": [{"stack_id": stack_id, "stack_label": label} for stack_id, label in STACKS],
        "harvey_lab": build_harvey(heldout_rows, harvey_summary),
        "dungeongrid": build_dungeongrid(heldout_rows, dungeongrid_summary),
        "notes": [
            "Harvey Lab rows use the final best candidate per stack and show fractional rubric reward, not only positive-row coverage.",
            "DungeonGrid rows use the final best candidate per stack and count reward_details.achievements on heldout episodes.",
        ],
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "reward_diagnostics_data.json")
    parser.add_argument("--frontend-output", type=Path, default=DEFAULT_FRONTEND_OUTPUT)
    args = parser.parse_args()

    data = build()
    write_json(args.output, data)
    write_json(args.frontend_output, data)
    dungeongrid = data.get("dungeongrid", {})
    harvey = data.get("harvey_lab", {})
    print(json.dumps({
        "status": data["status"],
        "wrote": [repo_path(args.output), repo_path(args.frontend_output)],
        "harvey_rows": {
            stack["stack_id"]: stack["n"]
            for stack in harvey.get("stacks", [])
        },
        "dungeongrid_objective_recovered": {
            stack["stack_id"]: next(
                signal["count"]
                for signal in stack.get("signals", [])
                if signal["signal"] == "objective.item_recovered"
            )
            for stack in dungeongrid.get("stacks", [])
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
