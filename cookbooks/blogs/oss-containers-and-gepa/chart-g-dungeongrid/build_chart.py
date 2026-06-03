#!/usr/bin/env python3
"""Build DungeonGrid blog data from final GEPA evidence rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
COOKBOOK_REPO = ROOT.parents[3]
WORKSPACE = Path("/Users/joshpurtell/Documents/GitHub")
EVALS = COOKBOOK_REPO / "cookbooks" / "optimizers" / "gepa" / "evals"
EVID = EVALS / "evidence"
DEFAULT_FRONTEND_OUTPUT = (
    WORKSPACE
    / "frontend"
    / "src"
    / "components"
    / "blog"
    / "posts"
    / "introducing-gepa-platform"
    / "data"
    / "dungeongrid_data.json"
)

BENCHMARK = "dungeongrid"
STACK = "synth_gepa"
PROGRESS_SIGNALS = [
    "objective.item_recovered",
    "treasure.first_treasure_collected",
    "exploration.ten_floor_tiles",
    "exploration.first_room_revealed",
    "doors.first_door_opened",
    "hazards.trap_disarmed",
    "treasure.first_chest_opened",
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def require_number(value: Any, label: str) -> int | float:
    if not is_number(value):
        raise SystemExit(f"Chart G missing numeric {label}: {value!r}")
    return value


def candidate_label(index: int) -> str:
    return "seed" if index == 0 else str(index)


def build() -> dict[str, Any]:
    candidate_path = EVID / "candidate_timeline.jsonl"
    heldout_path = EVID / "heldout_evaluations.jsonl"
    summary_path = EVID / "benchmarks" / BENCHMARK / "summary.json"
    missing = [
        repo_path(path)
        for path in (candidate_path, heldout_path, summary_path)
        if not path.exists()
    ]
    if missing:
        raise SystemExit(f"Chart G missing launch artifacts: {', '.join(missing)}")

    candidates = [
        row for row in read_jsonl(candidate_path)
        if row.get("benchmark") == BENCHMARK and row.get("stack") == STACK
    ]
    candidates.sort(key=lambda row: int(row.get("candidate_index") or 0))
    heldout = [
        row for row in read_jsonl(heldout_path)
        if row.get("benchmark") == BENCHMARK and row.get("stack") == STACK
    ]
    if not candidates or not heldout:
        raise SystemExit("Chart G found no DungeonGrid Synth GEPA candidate or heldout rows")

    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in heldout:
        by_candidate.setdefault(row["candidate_id"], []).append(row)

    running_best: float | None = None
    hillclimb_points: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for index, row in enumerate(candidates):
        score = float(require_number(row.get("heldout_score"), f"candidate {row.get('candidate_id')}.heldout_score"))
        running_best = score if running_best is None else max(running_best, score)
        label = candidate_label(index)
        cid = row["candidate_id"]
        n = len(by_candidate.get(cid, []))
        if n == 0:
            raise SystemExit(f"Chart G candidate has no heldout rows: {cid}")
        hillclimb_points.append({
            "k": index,
            "label": label,
            "candidate_id": cid,
            "score": score,
            "best_so_far": running_best,
            "status": "evaluated",
            "n": n,
        })
        candidate_rows.append({
            "candidate_id": cid,
            "label": label,
            "score": score,
            "n": n,
        })

    achievements = []
    for signal in PROGRESS_SIGNALS:
        points = []
        for index, row in enumerate(candidates):
            rows = by_candidate.get(row["candidate_id"], [])
            denom = len(rows)
            if denom == 0:
                raise SystemExit(f"Chart G signal has zero heldout rows for {row['candidate_id']}/{signal}")
            hits = 0
            for ev in rows:
                details = ev.get("reward_details") or {}
                if signal in (details.get("achievements") or []):
                    hits += 1
            points.append({
                "candidate_id": row["candidate_id"],
                "label": candidate_label(index),
                "value": (hits / denom) if denom else None,
                "n": denom,
            })
        achievements.append({
            "signal": signal,
            "points": points,
            "max": max(p["value"] for p in points),
        })

    best = max(candidates, key=lambda row: row.get("heldout_score") or float("-inf"))
    data = {
        "status": "ready",
        "chart": "dungeongrid_gepa_public",
        "generated_from": repo_path(ROOT / "build_chart.py"),
        "run": {
            "task": "DungeonGrid public GEPA prompt run",
            "agent_count": 2,
            "agent_rule": "DungeonGrid container runs two heroes per rollout",
            "source_task": "cookbooks/optimizers/gepa/evals/configs/dungeongrid.toml",
            "source_candidate_timeline": source_ref(candidate_path),
            "source_heldout_evaluations": source_ref(heldout_path),
            "source_summary": source_ref(summary_path) if summary_path.exists() else None,
        },
        "best_candidate": {
            "candidate_id": best.get("candidate_id"),
            "score": best.get("heldout_score"),
            "n": best.get("heldout_n"),
        },
        "candidate_count": len(candidates),
        "candidates": candidate_rows,
        "progress_signals": PROGRESS_SIGNALS,
        "hillclimb": {
            "y_label": "heldout reward",
            "points": hillclimb_points,
        },
        "achievements": achievements,
        "notes": [
            "Achievement frequencies are the fraction of heldout rollout rows whose reward_details.achievements contains the signal.",
            "The hillclimb curve uses Synth GEPA candidate_timeline heldout scores after the final evidence scope filter.",
        ],
    }
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "dungeongrid_data.json")
    parser.add_argument("--frontend-output", type=Path, default=DEFAULT_FRONTEND_OUTPUT)
    args = parser.parse_args()

    data = build()
    write_json(args.output, data)
    write_json(args.frontend_output, data)
    print(json.dumps({
        "status": data["status"],
        "wrote": [repo_path(args.output), repo_path(args.frontend_output)],
        "candidate_count": data.get("candidate_count", 0),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
