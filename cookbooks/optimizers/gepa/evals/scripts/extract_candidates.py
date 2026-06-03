"""Extract and normalize candidates from a run into evidence/proposal_timeline.jsonl.

For Synth GEPA: reads candidate_registry.json + events.jsonl from the run dir,
and proposer token usage from proposer_workspaces/<gen>/.agent_artifacts/.
For gepa-ai:   reads gepa_ai_run.json from the run dir.

Each candidate row is enriched with cumulative cost/time/rollout figures computed
on a SHARED basis (lib_compare) so the two stacks are apples-to-apples:

    elapsed_seconds              wall-clock from run start to candidate evaluation
    cumulative_rollout_count     policy rollouts spent so far
    cumulative_rollout_cost_usd  policy-model $ so far (shared price table)
    cumulative_proposer_cost_usd proposer/reflection $ so far (shared price table)
    cumulative_cost_usd          rollout + proposer $ so far

Appends rows to evidence/proposal_timeline.jsonl (idempotent on
stack:run_id:candidate_id).

Usage (from evals/):
    python scripts/extract_candidates.py --benchmark banking77 --stack synth_gepa --run-dir runs/synth_gepa/banking77/<run_id>/
    python scripts/extract_candidates.py --benchmark banking77 --stack gepa_ai   --run-dir runs/gepa_ai/banking77/<ts>/
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import lib_compare

EVALS_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = EVALS_DIR / "evidence"


def load_config(benchmark: str) -> dict:
    cfg_path = EVALS_DIR / "configs" / f"{benchmark}.toml"
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)


def proposal_text(payload: dict, cfg: dict | None = None) -> str:
    if not payload:
        return ""
    mutable = ((cfg or {}).get("benchmark") or {}).get("mutable_field")
    if mutable and payload.get(mutable):
        return str(payload[mutable])
    return str(payload.get("stage2_system") or payload.get("stage1_system") or list(payload.values())[0])


# ── synth_gepa event parsing ────────────────────────────────────────────────

def _parse_events(events_path: Path) -> dict:
    """Return {candidate_id: {registered_at, evaluated_at, heldout_at, rejected_at}}."""
    timing: dict[str, dict] = {}
    if not events_path.exists():
        return timing
    for line in events_path.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = e.get("type", "")
        ts = e.get("ts")
        fields = e.get("fields") or {}
        cid = fields.get("candidate_id")
        if not cid:
            continue
        timing.setdefault(cid, {})
        if t == "candidate.registered" and ts:
            timing[cid]["registered_at"] = ts
        elif t in ("candidate.evaluated", "candidate.minibatch_evaluated") and ts:
            timing[cid].setdefault("evaluated_at", ts)
        elif t == "heldout.completed" and ts:
            timing[cid]["heldout_at"] = ts
        elif t == "candidate.rejected" and ts:
            timing[cid]["rejected_at"] = ts
    return timing


def _build_synth_ledger(events_path: Path, run_dir: Path, pricing: lib_compare.Pricing) -> tuple[lib_compare.CostLedger, float | None]:
    """Build a time-ordered cost ledger and return (ledger, run_start_epoch).

    Policy rollout cost/count come from runtime.job.completed candidate_usage.
    Proposer cost comes from the codex token artifacts, attributed at each
    generation's proposer.completed timestamp.
    """
    ledger = lib_compare.CostLedger()
    run_start_epoch: float | None = None
    proposer_completed_ts: dict[int, str] = {}

    if events_path.exists():
        for line in events_path.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = e.get("type", "")
            ts = e.get("ts")
            fields = e.get("fields") or {}
            if t == "gepa.run.started" and ts:
                run_start_epoch = lib_compare.iso_to_epoch(ts)
            elif t == "proposer.completed" and ts:
                gen = fields.get("generation")
                if gen is not None:
                    proposer_completed_ts[int(gen)] = ts
            elif t == "runtime.job.completed":
                usage = fields.get("candidate_usage") or {}
                if not usage:
                    continue  # proposer run (no rollout tokens reported here)
                pt = ct = calls = 0
                for _cid, u in usage.items():
                    if not isinstance(u, dict):
                        continue
                    pt += int(u.get("prompt_tokens", 0) or 0)
                    ct += int(u.get("completion_tokens", 0) or 0)
                    calls += int(u.get("calls", 0) or 0)
                ledger.add(
                    lib_compare.iso_to_epoch(ts),
                    rollout_cost=pricing.rollout_cost(pt, ct),
                    rollout_count=calls,
                )

    # Proposer token cost, attributed at each generation's completion time.
    for usage in lib_compare.parse_synth_proposer_usage(run_dir):
        gen = usage["generation"]
        ts = proposer_completed_ts.get(gen)
        cost = pricing.proposer_cost(
            usage["input_tokens"], usage["cached_input_tokens"], usage["output_tokens"]
        )
        ledger.add(lib_compare.iso_to_epoch(ts), proposer_cost=cost)

    ledger.finalize()
    return ledger, run_start_epoch


def extract_synth_gepa(run_dir: Path, benchmark: str, pricing: lib_compare.Pricing, cfg: dict) -> list[dict]:
    reg_path = run_dir / "candidate_registry.json"
    events_path = run_dir / "events.jsonl"

    if not reg_path.exists():
        raise SystemExit(f"candidate_registry.json not found in {run_dir}")

    registry = json.loads(reg_path.read_text())
    timing = _parse_events(events_path)
    ledger, run_start_epoch = _build_synth_ledger(events_path, run_dir, pricing)

    manifest_path = run_dir / "result_manifest.json"
    run_id = run_dir.name
    if manifest_path.exists():
        try:
            run_id = json.loads(manifest_path.read_text()).get("run_id") or run_id
        except (json.JSONDecodeError, OSError):
            pass

    rows = []
    skipped = 0
    for idx, entry in enumerate(registry):
        cid = entry.get("candidate_id", f"cand_{idx}")
        payload = entry.get("payload") or (entry.get("lever_bundle") or {}).get("values") or {}
        parent_id = entry.get("parent_id")
        source = (entry.get("lever_bundle") or {}).get("source", "proposed" if parent_id else "seed")
        t = timing.get(cid, {})
        created_at = t.get("registered_at") or t.get("evaluated_at") or t.get("rejected_at")
        evaluated_at = t.get("evaluated_at") or t.get("rejected_at") or created_at

        # Skip candidates that were proposed but never evaluated (budget ran out
        # before they entered minibatch eval). They consumed no search rollouts
        # and have no train measurement, so they are not part of the
        # within-budget comparison — gepa-ai's equivalent un-evaluated proposals
        # are not recorded either.
        if evaluated_at is None:
            skipped += 1
            continue

        rows.append({
            "stack": "synth_gepa",
            "benchmark": benchmark,
            "run_id": run_id,
            "candidate_id": cid,
            "candidate_index": idx,
            "generation": _infer_generation_synth(entry),
            "parent_candidate_id": parent_id,
            "proposer_model": None,
            "proposal_source": source,
            "proposal_text": proposal_text(payload, cfg),
            "candidate_payload_json": json.dumps(payload),
            "rationale": None,
            "edit_summary": None,
            "created_at": created_at,
            "evaluated_at": evaluated_at,
            "heldout_at": t.get("heldout_at"),
            "train_score": _get_train_score(entry),
            "train_heldout_score": entry.get("heldout_reward"),
            "acceptance_score": entry.get("acceptance_score"),
        })

    if skipped:
        print(f"  (skipped {skipped} never-evaluated synth proposals — beyond budget)", flush=True)
    _attribute(rows, ledger, run_start_epoch)
    return rows


def _infer_generation_synth(entry: dict) -> int | None:
    lb = entry.get("lever_bundle") or {}
    parent_ids = lb.get("parent_ids") or []
    if not parent_ids and entry.get("parent_id") is None:
        return 0
    return None


def _get_train_score(entry: dict) -> float | None:
    frames = entry.get("sensor_frames") or []
    for frame in frames:
        reward = (frame.get("aggregate") or {}).get("reward")
        if reward is not None:
            return float(reward)
    mb = entry.get("minibatch_reward")
    if mb is not None:
        return float(mb)
    return None


# ── gepa_ai parsing ─────────────────────────────────────────────────────────

def _build_gepa_ai_ledger(run_data: dict, pricing: lib_compare.Pricing) -> tuple[lib_compare.CostLedger, float | None]:
    ledger = lib_compare.CostLedger()
    run_start_epoch = lib_compare.iso_to_epoch(run_data.get("started_at"))

    for ec in run_data.get("eval_calls") or []:
        ledger.add(
            lib_compare.iso_to_epoch(ec.get("finished_at")),
            rollout_cost=pricing.rollout_cost(
                int(ec.get("prompt_tokens", 0) or 0),
                int(ec.get("completion_tokens", 0) or 0),
            ),
            rollout_count=int(ec.get("rollout_count", ec.get("batch_size", 0)) or 0),
        )

    for rc in run_data.get("reflection_call_log") or []:
        ledger.add(
            lib_compare.iso_to_epoch(rc.get("finished_at")),
            proposer_cost=pricing.proposer_cost(
                int(rc.get("prompt_tokens", 0) or 0),
                int(rc.get("cached_prompt_tokens", 0) or 0),
                int(rc.get("completion_tokens", 0) or 0),
            ),
        )

    ledger.finalize()
    return ledger, run_start_epoch


def extract_gepa_ai(run_dir: Path, benchmark: str, pricing: lib_compare.Pricing, cfg: dict) -> list[dict]:
    run_data_path = run_dir / "gepa_ai_run.json"
    if not run_data_path.exists():
        raise SystemExit(f"gepa_ai_run.json not found in {run_dir}")

    run_data = json.loads(run_data_path.read_text())
    run_id = run_data.get("run_id", run_dir.name)
    eval_calls = run_data.get("eval_calls") or []
    ledger, run_start_epoch = _build_gepa_ai_ledger(run_data, pricing)

    # Match candidate payloads to their (first) eval call for timing.
    payload_to_eval: dict[str, dict] = {}
    for ec in eval_calls:
        key = json.dumps(ec.get("candidate_payload") or {}, sort_keys=True)
        payload_to_eval.setdefault(key, ec)

    rows = []
    for entry in run_data.get("candidates") or []:
        idx = entry.get("idx", 0)
        payload = entry.get("candidate_payload") or {}
        val_score = entry.get("val_aggregate_score")
        parent_idxs = entry.get("parent_idxs") or []
        parent_id = f"gepa_ai_cand_{parent_idxs[0]}" if parent_idxs and parent_idxs[0] is not None else None
        cid = f"gepa_ai_cand_{idx}"
        source = "seed" if idx == 0 else "proposed"

        ec = payload_to_eval.get(json.dumps(payload, sort_keys=True), {})

        rows.append({
            "stack": "gepa_ai",
            "benchmark": benchmark,
            "run_id": run_id,
            "candidate_id": cid,
            "candidate_index": idx,
            "generation": 0 if idx == 0 else None,
            "parent_candidate_id": parent_id,
            "proposer_model": run_data.get("reflection_model"),
            "proposal_source": source,
            "proposal_text": proposal_text(payload, cfg),
            "candidate_payload_json": json.dumps(payload),
            "rationale": None,
            "edit_summary": None,
            "created_at": ec.get("started_at"),
            "evaluated_at": ec.get("finished_at"),
            "heldout_at": None,
            "train_score": ec.get("mean_score"),
            "train_heldout_score": val_score,
            "acceptance_score": None,
        })

    _attribute(rows, ledger, run_start_epoch)
    return rows


# ── shared cumulative attribution ───────────────────────────────────────────

def _attribute(rows: list[dict], ledger: lib_compare.CostLedger, run_start_epoch: float | None) -> None:
    """Attach elapsed_seconds + cumulative cost/rollout figures to each row.

    Anchored at the candidate's evaluation-completion time, so the figures
    reflect everything spent by the moment the candidate's score is known.
    """
    if run_start_epoch is None:
        starts = [lib_compare.iso_to_epoch(r.get("created_at")) for r in rows]
        starts = [s for s in starts if s is not None]
        run_start_epoch = min(starts) if starts else None

    for r in rows:
        anchor = lib_compare.iso_to_epoch(r.get("evaluated_at")) or lib_compare.iso_to_epoch(r.get("created_at"))
        if anchor is not None and run_start_epoch is not None:
            r["elapsed_seconds"] = round(max(0.0, anchor - run_start_epoch), 3)
        else:
            r["elapsed_seconds"] = None
        cum = ledger.cumulative_at(anchor)
        r["cumulative_rollout_count"] = cum["rollout_count"]
        r["cumulative_rollout_cost_usd"] = cum["rollout_cost_usd"]
        r["cumulative_proposer_cost_usd"] = cum["proposer_cost_usd"]
        r["cumulative_cost_usd"] = cum["total_cost_usd"]


def main() -> int:
    p = argparse.ArgumentParser(description="Extract candidates into proposal_timeline.jsonl")
    p.add_argument("--benchmark", default="banking77")
    p.add_argument("--stack", required=True, choices=["synth_gepa", "gepa_ai"])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", default=None, help="Override output path")
    args = p.parse_args()

    cfg = load_config(args.benchmark)
    pricing = lib_compare.load_pricing(cfg)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = EVALS_DIR / run_dir
    if not run_dir.exists():
        raise SystemExit(f"run-dir not found: {run_dir}")

    if args.stack == "synth_gepa":
        rows = extract_synth_gepa(run_dir, args.benchmark, pricing, cfg)
    else:
        rows = extract_gepa_ai(run_dir, args.benchmark, pricing, cfg)

    out_path = Path(args.output) if args.output else EVIDENCE_DIR / "proposal_timeline.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_keys: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                row = json.loads(line)
                existing_keys.add(f"{row.get('stack')}:{row.get('run_id')}:{row.get('candidate_id')}")
            except json.JSONDecodeError:
                pass

    added = 0
    with open(out_path, "a") as f:
        for row in rows:
            key = f"{row['stack']}:{row['run_id']}:{row['candidate_id']}"
            if key not in existing_keys:
                f.write(json.dumps(row) + "\n")
                added += 1

    print(f"Extracted {len(rows)} candidates ({added} new) → {out_path}", flush=True)
    if rows:
        last = rows[-1]
        print(
            f"  last candidate: rollouts={last['cumulative_rollout_count']} "
            f"cost=${last['cumulative_cost_usd']:.4f} "
            f"(rollout=${last['cumulative_rollout_cost_usd']:.4f} "
            f"proposer=${last['cumulative_proposer_cost_usd']:.4f}) "
            f"elapsed={last['elapsed_seconds']}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
