#!/usr/bin/env python3
"""Run gepa-ai against a Synth GEPA container over the public /rollout API."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import sys
import time
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter


TASKS: dict[str, dict[str, Any]] = {
    "tblite": {
        "candidate_field": "starting_prompt",
        "seed_candidate": {
            "starting_prompt": (
                "You are a Python coding agent. Implement the requested function so that "
                "all of its hidden tests pass. Output ONLY the function source code, no "
                "markdown fences, no surrounding prose, no example usage. Match the "
                "function signature exactly. Handle edge cases. Use the standard library."
            )
        },
        "train_seeds": [0, 1, 2],
        "val_seeds": [100, 101],
        "default_url": "http://127.0.0.1:8811",
    },
    "crafter": {
        "candidate_field": "react_system_prompt",
        "seed_candidate": {
            "react_system_prompt": (
                "You are controlling a Crafter survival agent. Each turn you see a compact "
                "text observation (player stats, inventory, local map). Respond ONLY with a "
                "single <tool_call> block of the form: "
                "<tool_call>{\"name\":\"crafter_interact\",\"arguments\":{\"actions_list\":[\"move_right\",\"do\"]}}</tool_call>. "
                "Use 1-5 valid actions per call. Prioritize collecting wood, placing a table, "
                "crafting tools, then collecting stone/coal/iron. Avoid lava."
            )
        },
        "train_seeds": [11, 13],
        "val_seeds": [101],
        "default_url": "http://127.0.0.1:8812",
    },
}

USD_PER_INPUT_TOKEN = float(os.environ.get("PRICE_PER_INPUT_TOKEN", "1e-7"))
USD_PER_OUTPUT_TOKEN = float(os.environ.get("PRICE_PER_OUTPUT_TOKEN", "4e-7"))


def request_json(url: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}{path}",
        data=data,
        method="GET" if body is None else "POST",
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


class ContainerAdapter(GEPAAdapter):
    def __init__(self, *, container_url: str, candidate_field: str) -> None:
        self.container_url = container_url.rstrip("/")
        self.candidate_field = candidate_field
        self.rollout_prompt_tokens = 0
        self.rollout_completion_tokens = 0
        self.rollout_calls = 0

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        max_workers = int(os.environ.get("GEPA_AI_ROLLOUT_CONCURRENCY", "4"))
        outputs: list[dict[str, Any] | None] = [None] * len(batch)
        scores: list[float | None] = [None] * len(batch)
        trajectories: list[dict[str, Any] | None] | None = [None] * len(batch) if capture_traces else None

        def one(i: int, row: dict[str, Any]) -> tuple[int, dict[str, Any], float, dict[str, Any] | None, dict[str, int]]:
            try:
                rollout = request_json(
                    self.container_url,
                    "/rollout",
                    {"dataset_row": row, "seed": row["seed"], "split": row["split"], "candidate": candidate},
                )
            except Exception as exc:
                output = {"full_assistant_response": f"<rollout error: {exc!r}>"}
                trace = {
                    "data": row,
                    "full_assistant_response": output["full_assistant_response"],
                    "feedback": "Container /rollout failed; score=0.",
                } if capture_traces else None
                return i, output, 0.0, trace, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

            reward_info = rollout.get("reward_info") or {}
            summary = rollout.get("summary") or {}
            details = reward_info.get("details") or {}
            reward = float(reward_info.get("outcome_reward") or 0.0)
            usage = rollout.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            response = json.dumps(summary, sort_keys=True)
            feedback = "Reward %.4f. Details: %s" % (reward, json.dumps(details, sort_keys=True))
            trace = {"data": row, "full_assistant_response": response, "feedback": feedback} if capture_traces else None
            return i, {"full_assistant_response": response}, reward, trace, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "calls": 1,
            }

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(one, i, row) for i, row in enumerate(batch)]
            for fut in futures:
                i, output, reward, trace, usage = fut.result()
                outputs[i] = output
                scores[i] = reward
                if trajectories is not None:
                    trajectories[i] = trace
                self.rollout_prompt_tokens += usage["prompt_tokens"]
                self.rollout_completion_tokens += usage["completion_tokens"]
                self.rollout_calls += usage["calls"]

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
        component = components_to_update[0]
        items: list[dict[str, Any]] = []
        for trajectory in eval_batch.trajectories or []:
            items.append({
                "Inputs": json.dumps(trajectory.get("data", {}), sort_keys=True),
                "Generated Outputs": trajectory.get("full_assistant_response") or "",
                "Feedback": trajectory.get("feedback") or "",
            })
        if not items:
            raise RuntimeError("No reflective trajectories captured.")
        return {component: items}


class OpenAIReflectionLM:
    def __init__(self, model: str) -> None:
        from openai import OpenAI
        self.model = model
        self.client = OpenAI()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def __call__(self, prompt: Any) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        response = self.client.chat.completions.create(model=self.model, messages=messages)
        text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            self.calls += 1
        return text


def rows_for(container_url: str, split: str, seeds: list[int]) -> list[dict[str, Any]]:
    return list(request_json(container_url, "/dataset/rows", {"split": split, "seeds": seeds}).get("rows") or [])


def aggregate_scores(state_path: Path) -> list[float]:
    if not state_path.exists():
        return []
    state = pickle.load(open(state_path, "rb"))
    subscores = state.get("prog_candidate_val_subscores") or []
    return [sum(scores.values()) / len(scores) for scores in subscores if scores]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--container-url")
    parser.add_argument("--max-metric-calls", type=int, default=40)
    parser.add_argument("--reflection-model", default=os.environ.get("REFLECTION_MODEL", "gpt-4.1-nano"))
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        sys.stderr.write("ERROR: OPENAI_API_KEY not set.\n")
        return 2

    task = TASKS[args.task]
    container_url = (args.container_url or task["default_url"]).rstrip("/")
    train_rows = rows_for(container_url, "train", task["train_seeds"])
    val_rows = rows_for(container_url, "test", task["val_seeds"])
    out_dir = Path(__file__).resolve().parent.parent.parent / "runs" / "gepa_ai_via_container" / f"{args.task}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = ContainerAdapter(container_url=container_url, candidate_field=task["candidate_field"])
    reflection_lm = OpenAIReflectionLM(args.reflection_model)
    t0 = time.time()
    result = gepa.optimize(
        seed_candidate=task["seed_candidate"],
        trainset=train_rows,
        valset=val_rows,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
        seed=0,
        run_dir=str(out_dir / "gepa_run"),
        display_progress_bar=False,
        raise_on_exception=True,
    )

    scores = aggregate_scores(out_dir / "gepa_run" / "gepa_state.bin")
    rollout_usd = adapter.rollout_prompt_tokens * USD_PER_INPUT_TOKEN + adapter.rollout_completion_tokens * USD_PER_OUTPUT_TOKEN
    reflection_usd = reflection_lm.prompt_tokens * USD_PER_INPUT_TOKEN + reflection_lm.completion_tokens * USD_PER_OUTPUT_TOKEN
    summary = {
        "stack": "gepa_ai_via_container",
        "task": args.task,
        "container_url": container_url,
        "reflection_model": args.reflection_model,
        "max_metric_calls": args.max_metric_calls,
        "train_n": len(train_rows),
        "val_n": len(val_rows),
        "seed_val_score": scores[0] if scores else None,
        "best_val_score": max(scores) if scores else None,
        "best_idx": result.best_idx,
        "best_candidate": result.best_candidate,
        "total_metric_calls": result.total_metric_calls,
        "num_candidates": result.num_candidates,
        "val_aggregate_subscores": scores,
        "rollout_calls": adapter.rollout_calls,
        "rollout_prompt_tokens": adapter.rollout_prompt_tokens,
        "rollout_completion_tokens": adapter.rollout_completion_tokens,
        "reflection_calls": reflection_lm.calls,
        "reflection_prompt_tokens": reflection_lm.prompt_tokens,
        "reflection_completion_tokens": reflection_lm.completion_tokens,
        "rollout_usd": round(rollout_usd, 6),
        "reflection_usd": round(reflection_usd, 6),
        "total_usd": round(rollout_usd + reflection_usd, 6),
        "wall_clock_s": round(time.time() - t0, 2),
        "out_dir": str(out_dir),
        "timestamp": out_dir.name.rsplit("_", 1)[-1],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
