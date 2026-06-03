"""Filter global evidence JSONL files to the final blog run scope.

The eval harness appends evidence across many exploratory runs. Blog producers
must not read stale rows, so this script rewrites the evidence files to only the
run IDs listed in configs/blog_final_scope.json. Run it after all final posthoc
evaluations have completed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


EVALS_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = EVALS_DIR / "evidence"
DEFAULT_SCOPE = EVALS_DIR / "configs" / "blog_final_scope.json"

JSONL_FILES = [
    "proposal_timeline.jsonl",
    "candidate_timeline.jsonl",
    "heldout_evaluations.jsonl",
    "train_evaluations.jsonl",
    "candidate_reviews.jsonl",
    "curve_points.jsonl",
]


def load_scope(path: Path) -> set[tuple[str, str, str]]:
    scope = json.loads(path.read_text())
    allowed: set[tuple[str, str, str]] = set()
    for benchmark, stacks in (scope.get("benchmarks") or {}).items():
        for stack, run_id in stacks.items():
            allowed.add((benchmark, stack, run_id))
    return allowed


def keep_row(row: dict, allowed: set[tuple[str, str, str]]) -> bool:
    return (row.get("benchmark"), row.get("stack"), row.get("run_id")) in allowed


def filter_jsonl(path: Path, allowed: set[tuple[str, str, str]], backup_dir: Path | None) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / path.name)

    kept: list[str] = []
    removed = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            removed += 1
            continue
        if keep_row(row, allowed):
            kept.append(json.dumps(row))
        else:
            removed += 1
    path.write_text(("\n".join(kept) + "\n") if kept else "")
    return len(kept), removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Scope GEPA evidence to final blog run IDs.")
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    allowed = load_scope(args.scope)
    backup_dir = None if args.no_backup else EVIDENCE_DIR / "backups" / "pre_blog_final_scope"
    for name in JSONL_FILES:
        kept, removed = filter_jsonl(EVIDENCE_DIR / name, allowed, backup_dir)
        print(f"{name}: kept {kept}, removed {removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
