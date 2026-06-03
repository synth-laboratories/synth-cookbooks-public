#!/usr/bin/env python3
"""Prepare the public Harvey LAB (Tax) dataset bundle.

Clones Harvey AI's open-source Legal Agent Benchmark (MIT,
github.com/harveyai/harvey-labs) and normalizes the **Tax** practice area into a
single JSONL the container reads. Documents are read as text (the public
container is text-in/text-out: the agent sees document text rather than
navigating a sandboxed file system — a disclosed simplification of the full LAB
agentic task).

  python prepare_dataset.py            # clones + writes data/harvey_lab_tax_tasks.jsonl

Each record:
  {task_id, title, instructions, documents_text, criteria:[{id,title,match_criteria}], split}
Split is deterministic per task_id (bottom 25% of a stable hash = heldout).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

SOURCE_REPO_URL = "https://github.com/harveyai/harvey-labs"
PRACTICE_AREA = "tax"
HOLDOUT_FRACTION = 0.25
HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / ".harvey-labs-src"
OUT = HERE / "data" / "harvey_lab_tax_tasks.jsonl"
# Per-document text budget so a single rollout's context stays bounded.
MAX_DOC_CHARS = 12000
MAX_DOCS = 12


def _ensure_source(source_dir: Path) -> Path:
    if (source_dir / "tasks").is_dir():
        return source_dir
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", SOURCE_REPO_URL, str(source_dir)],
        check=True,
    )
    return source_dir


def _split_for(task_id: str) -> str:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "heldout" if bucket < HOLDOUT_FRACTION else "train"


def _read_text(path: Path) -> str:
    """Best-effort plain-text read. Non-text/binary docs are skipped."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:MAX_DOC_CHARS]
    except Exception:
        return ""


def normalize_task(task_dir: Path, source_root: Path) -> dict[str, Any]:
    config = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    task_id = task_dir.relative_to(source_root / "tasks").as_posix()
    docs = []
    for p in sorted((task_dir / "documents").glob("*")):
        if not p.is_file():
            continue
        text = _read_text(p)
        if text.strip():
            docs.append({"name": p.name, "text": text})
        if len(docs) >= MAX_DOCS:
            break
    criteria = [
        {"id": c["id"], "title": c.get("title", ""), "match_criteria": c["match_criteria"]}
        for c in config.get("criteria", [])
        if c.get("match_criteria")
    ]
    return {
        "task_id": task_id,
        "title": config.get("title", ""),
        "instructions": config.get("instructions", ""),
        "documents": docs,
        "criteria": criteria,
        "criterion_count": len(criteria),
        "split": _split_for(task_id),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = ap.parse_args()

    source_root = _ensure_source(args.source)
    area_dir = source_root / "tasks" / PRACTICE_AREA
    if not area_dir.is_dir():
        raise SystemExit(f"Tax practice area not found: {area_dir}")

    records = [
        normalize_task(cfg.parent, source_root)
        for cfg in sorted(area_dir.rglob("task.json"))
    ]
    records = [r for r in records if r["criteria"]]
    records.sort(key=lambda r: r["task_id"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=True) + "\n")

    n_train = sum(1 for r in records if r["split"] == "train")
    n_held = sum(1 for r in records if r["split"] == "heldout")
    print(
        f"wrote {len(records)} Tax tasks ({n_train} train / {n_held} heldout, "
        f"{sum(r['criterion_count'] for r in records)} criteria) -> {OUT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
