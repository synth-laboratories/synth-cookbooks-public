#!/usr/bin/env python3
"""
gepa-ai head-to-head baseline on Banking77 single-stage.

Runs `gepa.optimize(...)` against the SAME 24-train + 12-test fixture
that the Synth GEPA Banking77 container exposes, so the two stacks
can be compared apples-to-apples on identical inputs/labels.

Both stacks under parity conditions:
  - Same data:           cookbooks/optimizers/gepa/banking77_container/synth_service_app.py
  - Same seed candidate: stage2_system prompt below
  - Same task model:     gpt-4.1-nano (configurable via TASK_LM env)
  - Same reflection LM:  gpt-4.1-nano (configurable via REFLECTION_LM env)
  - Same budget:         GEPA_AI_MAX_METRIC_CALLS env (default 40)

Env:
  OPENAI_API_KEY          required (litellm reads it)
  TASK_LM                 model id for task (default openai/gpt-4.1-nano)
  REFLECTION_LM           model id for reflection (default same as TASK_LM)
  GEPA_AI_MAX_METRIC_CALLS  total rollout budget (default 40)
  GEPA_AI_OUT_DIR         output dir (default ./runs/gepa_ai/banking77_<ts>)

Run from the synth-cookbooks-public repo root:

  source /Users/joshpurtell/Documents/GitHub/synth-ai/.env
  python3 cookbooks/blogs/oss-containers-and-gepa/chart-a-head-to-head/configs/gepa_ai/banking77_run.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

# Import Banking77 fixture rows directly from the Synth container so both
# stacks evaluate the same inputs.
REPO_ROOT = Path(__file__).resolve().parents[6]  # synth-cookbooks-public/
sys.path.insert(0, str(REPO_ROOT / "cookbooks" / "optimizers" / "gepa" / "banking77_container"))
from synth_service_app import ROWS  # noqa: E402  type: ignore

import gepa  # noqa: E402


SEED_SYSTEM = (
    "Classify the customer banking query into exactly one Banking77 intent. "
    "Return the answer only with the best single label."
)


def to_default_data_inst(row: dict) -> dict:
    """Map a container fixture row → gepa-ai DefaultDataInst shape."""
    return {
        "input": row["text"],
        "additional_context": {},
        # ContainsAnswerEvaluator checks `data["answer"] in response` — label is unique snake_case.
        "answer": row["label"],
    }


def main() -> int:
    if "OPENAI_API_KEY" not in os.environ:
        sys.stderr.write("ERROR: OPENAI_API_KEY not set.\n")
        return 2

    task_lm = os.environ.get("TASK_LM", "openai/gpt-4.1-nano")
    reflection_lm = os.environ.get("REFLECTION_LM", task_lm)
    max_metric_calls = int(os.environ.get("GEPA_AI_MAX_METRIC_CALLS", "40"))

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        os.environ.get(
            "GEPA_AI_OUT_DIR",
            str(Path(__file__).resolve().parent.parent.parent / "runs" / "gepa_ai" / f"banking77_{ts}"),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [r for r in ROWS if r["split"] == "train"]
    test_rows = [r for r in ROWS if r["split"] == "test"]

    trainset = [to_default_data_inst(r) for r in train_rows]
    valset = [to_default_data_inst(r) for r in test_rows]

    print(
        f"[gepa-ai banking77] task_lm={task_lm} reflection_lm={reflection_lm} "
        f"max_metric_calls={max_metric_calls} train={len(trainset)} val={len(valset)}",
        flush=True,
    )

    seed_candidate = {"system_prompt": SEED_SYSTEM}

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        seed=0,
        run_dir=str(out_dir / "gepa_run"),
        display_progress_bar=False,
        raise_on_exception=True,
    )

    # GEPAResult exposes per-candidate aggregate subscores via val_aggregate_subscores.
    val_aggs = list(result.val_aggregate_subscores or [])
    best_idx = result.best_idx
    best_val_score = val_aggs[best_idx] if val_aggs and best_idx is not None else None
    seed_val_score = val_aggs[0] if val_aggs else None
    best_candidate = result.best_candidate

    summary = {
        "stack": "gepa_ai",
        "task": "banking77",
        "task_lm": task_lm,
        "reflection_lm": reflection_lm,
        "max_metric_calls": max_metric_calls,
        "train_n": len(trainset),
        "val_n": len(valset),
        "seed_val_score": seed_val_score,
        "best_val_score": best_val_score,
        "best_candidate": best_candidate,
        "total_metric_calls": result.total_metric_calls,
        "num_candidates": result.num_candidates,
        "num_full_val_evals": result.num_full_val_evals,
        "val_aggregate_subscores": val_aggs,
        "best_idx": best_idx,
        "run_dir": str(result.run_dir) if result.run_dir else None,
        "out_dir": str(out_dir),
        "timestamp": ts,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
