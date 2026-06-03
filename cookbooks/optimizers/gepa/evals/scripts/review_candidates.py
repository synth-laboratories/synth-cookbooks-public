"""Generate candidate proposal reviews and diffs.

For each candidate in candidate_timeline.jsonl:
  - Compute unified text diff against its parent candidate.
  - Classify edit intent using the OpenAI API (optional, requires --llm-review).
  - Write per-candidate diff JSON files.
  - Write evidence/candidate_reviews.jsonl.
  - Write evidence/candidate_diffs/banking77/best_synth_vs_gepa_ai.json.

Usage (from evals/):
    python scripts/review_candidates.py --benchmark banking77
    python scripts/review_candidates.py --benchmark banking77 --llm-review
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = EVALS_DIR / "evidence"
SYNTH_AI_ENV = EVALS_DIR.parents[3].parent / "synth-ai" / ".env"

EDIT_LABELS = [
    "specificity",
    "format_control",
    "decomposition",
    "evidence_use",
    "rubric_alignment",
    "safety_constraint_tightening",
    "redundant_dominated_edit",
    "other",
]

REVIEW_SYSTEM = """\
You are an expert at analyzing prompt optimization. Given a parent prompt and a
child prompt (proposed by an optimizer), identify the primary edit intent.

Return a JSON object with:
  "edit_labels": list of applicable labels from [specificity, format_control,
    decomposition, evidence_use, rubric_alignment, safety_constraint_tightening,
    redundant_dominated_edit, other]
  "edit_summary": one sentence describing the key change
  "rationale": one or two sentences explaining why this edit might improve performance
  "predicted_direction": "positive", "negative", or "neutral" based on the edit
"""


def load_env() -> None:
    if SYNTH_AI_ENV.is_file():
        for line in SYNTH_AI_ENV.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def text_diff(parent_text: str, child_text: str) -> str:
    parent_lines = parent_text.splitlines(keepends=True)
    child_lines = child_text.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        parent_lines, child_lines,
        fromfile="parent", tofile="child",
        lineterm="",
    ))
    return "\n".join(diff_lines)


def classify_edit_llm(parent_text: str, child_text: str, diff_text: str) -> dict:
    from openai import OpenAI
    client = OpenAI()
    user_msg = (
        f"PARENT:\n{parent_text}\n\n"
        f"CHILD:\n{child_text}\n\n"
        f"DIFF:\n{diff_text}"
    )
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"edit_labels": ["other"], "edit_summary": content[:200], "rationale": "", "predicted_direction": "neutral"}


def main() -> int:
    load_env()
    p = argparse.ArgumentParser(description="Generate candidate reviews and diffs.")
    p.add_argument("--benchmark", default="banking77")
    p.add_argument("--llm-review", action="store_true", help="Use LLM to classify edit intent")
    p.add_argument("--candidate-timeline", default=str(EVIDENCE_DIR / "candidate_timeline.jsonl"))
    p.add_argument("--proposal-timeline", default=str(EVIDENCE_DIR / "proposal_timeline.jsonl"))
    args = p.parse_args()

    ct_path = Path(args.candidate_timeline)
    pt_path = Path(args.proposal_timeline)

    if not ct_path.exists():
        raise SystemExit(f"candidate_timeline not found: {ct_path}")
    if not pt_path.exists():
        raise SystemExit(f"proposal_timeline not found: {pt_path}")

    candidates = [c for c in load_jsonl(ct_path) if c.get("benchmark") == args.benchmark]
    proposals = [p_row for p_row in load_jsonl(pt_path) if p_row.get("benchmark") == args.benchmark]

    # Build lookup: (stack, run_id, candidate_id) → proposal row.
    proposal_map: dict[str, dict] = {}
    for p_row in proposals:
        k = f"{p_row.get('stack')}:{p_row.get('run_id')}:{p_row.get('candidate_id')}"
        proposal_map[k] = p_row

    # Build lookup: candidate_id → candidate row (for parent lookups).
    cand_by_key: dict[str, dict] = {
        f"{c['stack']}:{c['run_id']}:{c['candidate_id']}": c for c in candidates
    }
    proposal_by_id: dict[str, dict] = {
        f"{p_row['stack']}:{p_row['run_id']}:{p_row['candidate_id']}": p_row
        for p_row in proposals
    }

    diff_base = EVIDENCE_DIR / "candidate_diffs" / args.benchmark
    reviews_path = EVIDENCE_DIR / "candidate_reviews.jsonl"

    # Load already-reviewed candidate IDs.
    done_ids: set[str] = set()
    if reviews_path.exists():
        for line in reviews_path.read_text().splitlines():
            try:
                row = json.loads(line)
                if row.get("benchmark") == args.benchmark:
                    done_ids.add(f"{row.get('stack')}:{row.get('run_id')}:{row.get('candidate_id')}")
            except json.JSONDecodeError:
                pass

    new_reviews = 0
    with open(reviews_path, "a") as review_f:
        for cand in candidates:
            cid = cand["candidate_id"]
            stack = cand["stack"]
            run_id = cand["run_id"]
            key = f"{stack}:{run_id}:{cid}"

            if key in done_ids:
                continue

            payload = json.loads(cand.get("candidate_payload_json") or "{}")
            cand_text = list(payload.values())[0] if payload else ""

            parent_cid = cand.get("parent_candidate_id")
            parent_text = ""
            if parent_cid:
                parent_key = f"{stack}:{run_id}:{parent_cid}"
                p_cand = cand_by_key.get(parent_key) or proposal_by_id.get(parent_key)
                if p_cand:
                    p_payload = json.loads(p_cand.get("candidate_payload_json") or "{}")
                    parent_text = list(p_payload.values())[0] if p_payload else ""

            diff_text = text_diff(parent_text, cand_text) if parent_text else ""

            llm_result: dict = {}
            if args.llm_review and parent_text and cand_text and cand_text != parent_text:
                try:
                    llm_result = classify_edit_llm(parent_text, cand_text, diff_text)
                except Exception as exc:
                    llm_result = {"error": str(exc)}

            review_row = {
                "stack": stack,
                "benchmark": args.benchmark,
                "run_id": run_id,
                "candidate_id": cid,
                "candidate_index": cand.get("candidate_index"),
                "parent_candidate_id": parent_cid,
                "heldout_score": cand.get("heldout_score"),
                "train_score": cand.get("train_score"),
                "cumulative_cost_usd": cand.get("cumulative_cost_usd"),
                "elapsed_seconds": cand.get("elapsed_seconds"),
                "proposal_source": cand.get("proposal_source"),
                "edit_labels": llm_result.get("edit_labels"),
                "edit_summary": llm_result.get("edit_summary") or cand.get("edit_summary"),
                "rationale": llm_result.get("rationale") or cand.get("rationale"),
                "predicted_direction": llm_result.get("predicted_direction"),
                "diff_text": diff_text or None,
            }
            review_f.write(json.dumps(review_row) + "\n")
            new_reviews += 1

            # Write per-candidate diff JSON.
            stack_dir = diff_base / stack.replace("_", "-").replace("synth-gepa", "synth_gepa").replace("gepa-ai", "gepa_ai")
            stack_dir = diff_base / stack
            stack_dir.mkdir(parents=True, exist_ok=True)
            diff_json_path = stack_dir / f"{cid}.json"
            diff_json = {
                "candidate_id": cid,
                "stack": stack,
                "run_id": run_id,
                "candidate_index": cand.get("candidate_index"),
                "parent_candidate_id": parent_cid,
                "candidate_text": cand_text,
                "parent_text": parent_text,
                "diff": diff_text,
                "heldout_score": cand.get("heldout_score"),
                "edit_labels": llm_result.get("edit_labels"),
                "edit_summary": llm_result.get("edit_summary"),
            }
            diff_json_path.write_text(json.dumps(diff_json, indent=2))

    print(f"Wrote {new_reviews} new reviews → {reviews_path}", flush=True)

    # Build best_synth_vs_gepa_ai.json comparing best candidates from each stack.
    _write_best_comparison(candidates, args.benchmark, diff_base)

    return 0


def _write_best_comparison(candidates: list[dict], benchmark: str, diff_base: Path) -> None:
    by_stack: dict[str, list[dict]] = {}
    for c in candidates:
        by_stack.setdefault(c["stack"], []).append(c)

    best_per_stack: dict[str, dict] = {}
    for stack, rows in by_stack.items():
        valid = [r for r in rows if r.get("heldout_score") is not None]
        if valid:
            best_per_stack[stack] = max(valid, key=lambda r: r["heldout_score"])

    if len(best_per_stack) < 2:
        return

    stacks = list(best_per_stack.keys())
    texts: dict[str, str] = {}
    for s in stacks:
        payload = json.loads(best_per_stack[s].get("candidate_payload_json") or "{}")
        texts[s] = list(payload.values())[0] if payload else ""

    if len(stacks) >= 2:
        s1, s2 = stacks[0], stacks[1]
        diff_text = text_diff(texts.get(s1, ""), texts.get(s2, ""))
    else:
        diff_text = ""

    out = {
        "benchmark": benchmark,
        "stacks": {
            s: {
                "candidate_id": best_per_stack[s]["candidate_id"],
                "run_id": best_per_stack[s]["run_id"],
                "heldout_score": best_per_stack[s].get("heldout_score"),
                "candidate_text": texts.get(s, ""),
            }
            for s in stacks
        },
        "diff": diff_text,
    }
    diff_base.mkdir(parents=True, exist_ok=True)
    path = diff_base / "best_synth_vs_gepa_ai.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote best comparison → {path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
