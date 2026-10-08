#!/usr/bin/env python3
"""
Head-to-head: run gepa-ai's optimizer against the SAME live Banking77
container that Synth GEPA uses, so both stacks exchange proposals over
identical /rollout codepaths.

The adapter implements gepa.GEPAAdapter.evaluate() by POSTing each data
instance to the container's /rollout endpoint with the proposed
candidate. The container performs the live OpenAI call + label scoring
and returns a real reward. This isolates the optimizer-side difference
from the scoring-side difference.

Run from the synth-cookbooks-public repo root, after starting the
Banking77 container separately:

  # terminal 1
  cd cookbooks/optimizers/gepa
  source /Users/joshpurtell/Documents/GitHub/synth-ai/.env
  uv run --project banking77_container python banking77_container/synth_service_app.py \
    --host 127.0.0.1 --port 8810

  # terminal 2
  source /Users/joshpurtell/Documents/GitHub/synth-ai/.env
  CONTAINER_URL=http://127.0.0.1:8810 GEPA_AI_MAX_METRIC_CALLS=40 \
    python3 cookbooks/blogs/oss-containers-and-gepa/chart-a-head-to-head/configs/gepa_ai/banking77_via_container.py

Env:
  OPENAI_API_KEY            required (passed through to container env too)
  CONTAINER_URL             default http://127.0.0.1:8810
  REFLECTION_LM             default openai/gpt-4.1-nano
  GEPA_AI_MAX_METRIC_CALLS  default 40
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter


# Import the canonical 24-train + 12-test fixture rows from the container.
REPO_ROOT = Path(__file__).resolve().parents[6]  # synth-cookbooks-public/
sys.path.insert(0, str(REPO_ROOT / "cookbooks" / "optimizers" / "gepa" / "banking77_container"))
from synth_service_app import ROWS  # type: ignore  # noqa: E402


CONTAINER_URL = os.environ.get("CONTAINER_URL", "http://127.0.0.1:8810").rstrip("/")


def container_rollout(seed: int, split: str, candidate: dict[str, str]) -> dict[str, Any]:
    body = json.dumps({"seed": seed, "split": split, "candidate": candidate}).encode()
    req = urllib.request.Request(
        f"{CONTAINER_URL}/rollout",
        data=body,
        method="POST",
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


# gpt-4.1-nano price (per OpenAI pricing as of 2026Q1). Override via env for
# other models without touching this file.
USD_PER_INPUT_TOKEN = float(os.environ.get("PRICE_PER_INPUT_TOKEN", "1e-7"))   # $0.10 / 1M
USD_PER_OUTPUT_TOKEN = float(os.environ.get("PRICE_PER_OUTPUT_TOKEN", "4e-7")) # $0.40 / 1M


class Banking77ContainerAdapter(GEPAAdapter):
    """gepa-ai adapter that proxies /rollout to the Banking77 live container."""

    def __init__(self) -> None:
        # Track cumulative container-side token usage (i.e. policy model spend).
        self.rollout_prompt_tokens = 0
        self.rollout_completion_tokens = 0
        self.rollout_calls = 0

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        from concurrent.futures import ThreadPoolExecutor

        max_workers = int(os.environ.get("GEPA_AI_ROLLOUT_CONCURRENCY", "16"))
        outputs: list[dict[str, Any] | None] = [None] * len(batch)
        scores: list[float | None] = [None] * len(batch)
        trajectories: list[dict[str, Any] | None] | None = (
            [None] * len(batch) if capture_traces else None
        )

        def _one(i: int, data: dict[str, Any]) -> tuple[int, dict[str, Any], float, dict[str, Any] | None, dict[str, int]]:
            seed = int(data["seed"])
            split = str(data["split"])
            usage_in = 0
            usage_out = 0
            try:
                roll = container_rollout(seed=seed, split=split, candidate=candidate)
            except Exception as exc:
                output = {"full_assistant_response": f"<rollout error: {exc!r}>"}
                traj = {
                    "data": data,
                    "full_assistant_response": f"<rollout error: {exc!r}>",
                    "feedback": "Container /rollout failed; treated as score=0.",
                } if capture_traces else None
                return i, output, 0.0, traj, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

            ri = roll.get("reward_info") or {}
            details = ri.get("details") or {}
            reward = float(ri.get("outcome_reward") or 0.0)
            prediction = str(details.get("prediction") or "")
            expected = str(details.get("expected") or data.get("label") or "")
            usage = roll.get("usage") or {}
            usage_in = int(usage.get("prompt_tokens") or 0)
            usage_out = int(usage.get("completion_tokens") or 0)

            output = {"full_assistant_response": prediction}
            traj = None
            if capture_traces:
                if reward >= 1.0:
                    feedback = f"Correct. Predicted '{prediction}' matches expected '{expected}'."
                else:
                    feedback = (
                        f"Incorrect. Predicted '{prediction}'; expected '{expected}'. "
                        "Ensure the response is exactly one Banking77 label, lowercase snake_case, "
                        "with no extra words."
                    )
                traj = {
                    "data": data,
                    "full_assistant_response": prediction,
                    "feedback": feedback,
                }
            return i, output, reward, traj, {
                "prompt_tokens": usage_in,
                "completion_tokens": usage_out,
                "calls": 1,
            }

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_one, i, data) for i, data in enumerate(batch)]
            for fut in futures:
                i, output, reward, traj, u = fut.result()
                outputs[i] = output
                scores[i] = reward
                if trajectories is not None:
                    trajectories[i] = traj
                self.rollout_prompt_tokens += u["prompt_tokens"]
                self.rollout_completion_tokens += u["completion_tokens"]
                self.rollout_calls += u["calls"]

        return EvaluationBatch(
            outputs=outputs,  # type: ignore[arg-type]
            scores=scores,  # type: ignore[arg-type]
            trajectories=trajectories,  # type: ignore[arg-type]
            objective_scores=None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        assert len(components_to_update) == 1
        comp = components_to_update[0]
        trajs = eval_batch.trajectories or []
        items: list[dict[str, Any]] = []
        for traj in trajs:
            items.append({
                "Inputs": traj["data"].get("text") or traj["data"].get("input") or "",
                "Generated Outputs": traj.get("full_assistant_response") or "",
                "Feedback": traj.get("feedback") or "",
            })
        if not items:
            raise Exception("No valid predictions found for any module.")
        return {comp: items}


SEED_SYSTEM = (
    "Classify the customer banking query into exactly one Banking77 intent. "
    "Return the answer only with the best single label."
)


class OpenAIReflectionLM:
    """Reflection LM that calls openai SDK directly (no litellm). Tracks usage."""

    def __init__(self, model: str) -> None:
        from openai import OpenAI
        self.model = model
        self.client = OpenAI()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def __call__(self, prompt) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = list(prompt)
        resp = self.client.chat.completions.create(model=self.model, messages=messages)
        text = (resp.choices[0].message.content or "").strip()
        try:
            self.prompt_tokens += int(getattr(resp.usage, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(resp.usage, "completion_tokens", 0) or 0)
            self.calls += 1
        except Exception:
            pass
        return text


def main() -> int:
    import time as _time
    if "OPENAI_API_KEY" not in os.environ:
        sys.stderr.write("ERROR: OPENAI_API_KEY not set.\n")
        return 2

    reflection_model = os.environ.get("REFLECTION_MODEL", "gpt-4.1-nano")
    max_metric_calls = int(os.environ.get("GEPA_AI_MAX_METRIC_CALLS", "40"))

    reflection_lm = OpenAIReflectionLM(reflection_model)
    t0 = _time.time()

    train_rows = [{"seed": r["seed"], "split": r["split"], "text": r["text"], "label": r["label"]}
                  for r in ROWS if r["split"] == "train"]
    test_rows = [{"seed": r["seed"], "split": r["split"], "text": r["text"], "label": r["label"]}
                 for r in ROWS if r["split"] == "test"]

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent.parent.parent / "runs" / "gepa_ai_via_container" / f"banking77_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[head-to-head] container={CONTAINER_URL} reflection_model={reflection_model} "
        f"max_metric_calls={max_metric_calls} train={len(train_rows)} val={len(test_rows)}",
        flush=True,
    )

    adapter = Banking77ContainerAdapter()
    seed_candidate = {"system_prompt": SEED_SYSTEM}

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train_rows,
        valset=test_rows,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        seed=0,
        run_dir=str(out_dir / "gepa_run"),
        display_progress_bar=False,
        raise_on_exception=True,
    )

    # Pull per-candidate aggregate val subscores from the saved state file
    # (gepa-ai's GEPAResult doesn't surface them directly).
    import pickle
    state_path = out_dir / "gepa_run" / "gepa_state.bin"
    if state_path.exists():
        state = pickle.load(open(state_path, "rb"))
        subs = state["prog_candidate_val_subscores"]
        aggs = [sum(s.values()) / len(s) for s in subs]
    else:
        aggs = []

    elapsed_s = round(_time.time() - t0, 2)
    rollout_in = adapter.rollout_prompt_tokens
    rollout_out = adapter.rollout_completion_tokens
    refl_in = reflection_lm.prompt_tokens
    refl_out = reflection_lm.completion_tokens
    rollout_usd = rollout_in * USD_PER_INPUT_TOKEN + rollout_out * USD_PER_OUTPUT_TOKEN
    refl_usd = refl_in * USD_PER_INPUT_TOKEN + refl_out * USD_PER_OUTPUT_TOKEN
    total_usd = rollout_usd + refl_usd

    summary = {
        "stack": "gepa_ai_via_container",
        "task": "banking77",
        "container_url": CONTAINER_URL,
        "reflection_model": reflection_model,
        "max_metric_calls": max_metric_calls,
        "train_n": len(train_rows),
        "val_n": len(test_rows),
        "seed_val_score": aggs[0] if aggs else None,
        "best_val_score": max(aggs) if aggs else None,
        "best_idx": result.best_idx,
        "best_candidate": result.best_candidate,
        "total_metric_calls": result.total_metric_calls,
        "num_candidates": result.num_candidates,
        "val_aggregate_subscores": aggs,
        "rollout_calls": adapter.rollout_calls,
        "rollout_prompt_tokens": rollout_in,
        "rollout_completion_tokens": rollout_out,
        "reflection_calls": reflection_lm.calls,
        "reflection_prompt_tokens": refl_in,
        "reflection_completion_tokens": refl_out,
        "rollout_usd": round(rollout_usd, 6),
        "reflection_usd": round(refl_usd, 6),
        "total_usd": round(total_usd, 6),
        "wall_clock_s": elapsed_s,
        "out_dir": str(out_dir),
        "timestamp": ts,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
