from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import sys
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COOKBOOK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = COOKBOOK_ROOT.parents[2]
DEFAULT_ARTIFACTS_DIR = COOKBOOK_ROOT / "run_artifacts" / "dry_run"
CONFIG_PATH = COOKBOOK_ROOT / "miprov2" / "banking77_openai_split_confusable_30x300_100heldout.json"
OPTIMIZERS_SRC = REPO_ROOT / "packages" / "synth-optimizers" / "src"
CONTAINERS_SRC = REPO_ROOT / "packages" / "synth-containers" / "src"
BANKING77_CONTAINER_SRC = COOKBOOK_ROOT / "banking77_container"
LOCAL_SYNTH_ENV = REPO_ROOT.parent / "synth-ai" / ".env"


KNOWN_EVIDENCE = {
    "dataset": "banking77",
    "slice": "confusable_20_label",
    "train_seeds": 30,
    "heldout_seeds": 100,
    "target_train_candidates": 10,
    "target_total_train_rollouts": 300,
    "baseline_train": 0.7666666667,
    "best_train": 0.8333333333,
    "heldout_baseline": 0.7900,
    "heldout_best": 0.8200,
    "heldout_lift": 0.0300,
    "result_status": "positive",
    "evidence_status": "archived_private_draft_result_needs_public_rerun",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_summary(path: Path, plan: dict[str, Any]) -> None:
    evidence = plan["known_evidence"]
    lines = [
        "# Banking77 MIPROv2 Dry Run",
        "",
        f"- generated_at: `{plan['generated_at']}`",
        f"- mode: `{plan['mode']}`",
        f"- config: `{plan['config']}`",
        f"- package_import: `{plan['optimizer_package']['target_import']}`",
        f"- baseline_train: `{evidence['baseline_train']}`",
        f"- best_train: `{evidence['best_train']}`",
        f"- heldout_baseline: `{evidence['heldout_baseline']}`",
        f"- heldout_best: `{evidence['heldout_best']}`",
        f"- heldout_lift: `{evidence['heldout_lift']}`",
        f"- result_status: `{evidence['result_status']}`",
        "",
        "This artifact preserves the known result while marking it as needing a",
        "public rerun before release claims are made.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _relative_to(path: str | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    config = _load_config(config_path)
    return {
        "cookbook": "optimizers/miprov2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "config": _relative_to(str(config_path), COOKBOOK_ROOT) or str(config_path),
        "config_summary": {
            "run_id": config.get("run_id"),
            "policy_model": config.get("policy_model"),
            "proposer_model": config.get("proposer_model"),
            "verifier_model": config.get("verifier_model"),
            "target_train_candidates": config.get("target_train_candidates"),
            "target_total_train_rollouts": config.get("target_total_train_rollouts"),
            "train_seed_count": len(config.get("train_seeds") or []),
            "heldout_seed_count": len(config.get("heldout_seeds") or []),
            "label_id_count": len(config.get("label_ids") or []),
        },
        "container": {
            "path": "banking77_container",
            "public_entrypoint": "banking77_container/synth_service_app.py",
            "source_files": [
                "synth_service_app.py",
                "container.py",
                "container_spec.json",
                "task_contract.json",
                "RUBRIC.json",
            ],
        },
        "optimizer_package": {
            "distribution": "synth-optimizers",
            "target_import": "synth_optimizers.miprov2",
            "current_status": "miprov2 package migrated; cookbook execute path uses native phase-3 OpenEnv proposer",
        },
        "known_evidence": KNOWN_EVIDENCE,
        "expected_artifacts": [
            "plan.json",
            "summary.md",
            "known_evidence.json",
            "live_run/summary.json",
        ],
    }


def _ensure_local_imports() -> None:
    for path in (str(OPTIMIZERS_SRC), str(CONTAINERS_SRC), str(BANKING77_CONTAINER_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_local_api_keys() -> list[str]:
    loaded: list[str] = []
    if not LOCAL_SYNTH_ENV.exists():
        return loaded
    for raw_line in LOCAL_SYNTH_ENV.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key not in {"OPENAI_API_KEY", "SYNTH_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"}:
            continue
        if os.environ.get(key):
            continue
        os.environ[key] = value.strip().strip("'").strip('"')
        loaded.append(key)
    return loaded


def _seed_candidate() -> dict[str, str]:
    from synth_service_app import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT

    return {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "user_prompt": DEFAULT_USER_PROMPT,
    }


def _load_rows(config: dict[str, Any], *, smoke: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset

    from synth_service_app import DATASET_NAME

    label_ids = [int(label_id) for label_id in (config.get("label_ids") or [])]
    label_id_set = set(label_ids)

    def rows_for_split(split: str, seeds: list[int]) -> list[dict[str, Any]]:
        dataset = load_dataset(DATASET_NAME, split=split)
        features = getattr(dataset, "features", {}) or {}
        label_names = list(getattr(features.get("label"), "names", None) or [])
        filtered_indices: list[int] = []
        indices_by_label: dict[int, list[int]] = {label_id: [] for label_id in label_ids}
        for index, row in enumerate(dataset):
            label_idx = int(row.get("label", 0))
            if not label_id_set or label_idx in label_id_set:
                filtered_indices.append(index)
            if label_idx in indices_by_label:
                indices_by_label[label_idx].append(index)
        rows: list[dict[str, Any]] = []
        for offset, seed in enumerate(seeds):
            if label_ids:
                target_label = label_ids[offset % len(label_ids)]
                label_indices = indices_by_label[target_label]
                if not label_indices:
                    continue
                index = label_indices[int(seed) % len(label_indices)]
            else:
                index = filtered_indices[int(seed) % len(filtered_indices)]
            row = dataset[index]
            label_idx = int(row.get("label", 0))
            label = label_names[label_idx] if 0 <= label_idx < len(label_names) else f"label_{label_idx}"
            rows.append(
                {
                    "seed": int(seed),
                    "index": int(index),
                    "split": split,
                    "text": str(row.get("text", "")),
                    "label_idx": label_idx,
                    "label": label,
                }
            )
        return rows

    train_seeds = [int(seed) for seed in (config.get("train_seeds") or [])]
    heldout_seeds = [int(seed) for seed in (config.get("heldout_seeds") or [])]
    if smoke:
        train_seeds = train_seeds[:3]
        heldout_seeds = heldout_seeds[:5]
    return rows_for_split("train", train_seeds), rows_for_split("test", heldout_seeds)


def _candidate_variants() -> dict[str, list[str]]:
    from synth_service_app import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT

    return {
        "system_prompt": [
            DEFAULT_SYSTEM_PROMPT,
            (
                "Classify the customer banking query into exactly one Banking77 intent. "
                "Prefer the most specific available label and respond only with a banking77_classify tool call."
            ),
            (
                "You are a precise Banking77 intent classifier. Distinguish close banking-support intents carefully, "
                "then call banking77_classify with exactly one label from the provided intent list."
            ),
            (
                "Map each customer banking query to the single best Banking77 intent. Do not invent labels, "
                "do not explain, and use only the banking77_classify tool."
            ),
        ],
        "user_prompt": [
            DEFAULT_USER_PROMPT,
            (
                "Customer query:\n{query}\n\n"
                "Choose exactly one intent from this list:\n{available_intents}\n\n"
                "Call banking77_classify with the exact intent string."
            ),
            (
                "Query:\n{query}\n\n"
                "Valid Banking77 intents:\n{available_intents}\n\n"
                "Return the best matching intent with banking77_classify."
            ),
        ],
    }


class Banking77MiproAdapter:
    def __init__(self, policy_cfg: dict[str, Any], *, max_concurrency: int) -> None:
        self.policy_cfg = dict(policy_cfg)
        self.max_concurrency = max(1, int(max_concurrency))

    async def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> dict[str, Any]:
        from synth_service_app import (
            DEFAULT_SYSTEM_PROMPT,
            DEFAULT_USER_PROMPT,
            Banking77Dataset,
            call_chat_completion,
            extract_prediction,
            _normalize_label,
        )

        dataset = Banking77Dataset()
        label_names = dataset.label_names
        available_intents = "\n".join(f"{idx + 1}. {label}" for idx, label in enumerate(label_names))
        system_prompt = str(candidate.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
        user_template = str(candidate.get("user_prompt") or DEFAULT_USER_PROMPT)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def evaluate_row(row: dict[str, Any]) -> tuple[dict[str, Any], float, dict[str, Any] | None]:
            user_prompt = user_template.format(
                query=row["text"],
                available_intents=available_intents,
            )
            raw_response = ""
            response_json: dict[str, Any] = {}
            tool_calls: list[dict[str, Any]] = []
            retry_attempts = 4
            for attempt in range(1, retry_attempts + 1):
                try:
                    async with semaphore:
                        raw_response, response_json, tool_calls = await call_chat_completion(
                            policy_cfg=self.policy_cfg,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                        )
                    break
                except Exception as exc:
                    status_code = getattr(getattr(exc, "response", None), "status_code", None)
                    is_retryable = status_code in {408, 409, 429, 500, 502, 503, 504}
                    if attempt >= retry_attempts or not is_retryable:
                        raise
                    await asyncio.sleep(min(20.0, 1.5 * (2 ** (attempt - 1))))
            prediction = extract_prediction(
                raw_text=raw_response,
                tool_calls=tool_calls,
                label_names=label_names,
            )
            expected = str(row["label"])
            score = 1.0 if _normalize_label(prediction) == _normalize_label(expected) else 0.0
            output = {
                "seed": row["seed"],
                "index": row["index"],
                "split": row["split"],
                "prediction": prediction,
                "expected": expected,
                "correct": score >= 1.0,
            }
            usage = _usage_from_response(response_json)
            trace = None
            if capture_traces:
                trace = {
                    **output,
                    "query": row["text"],
                    "raw_response": raw_response,
                    "response_id": response_json.get("id") if isinstance(response_json, dict) else None,
                    "usage": usage,
                }
            return output, score, trace

        results = await asyncio.gather(*(evaluate_row(dict(row)) for row in batch))
        outputs = [item[0] for item in results]
        scores = [float(item[1]) for item in results]
        traces = [item[2] for item in results if item[2] is not None]
        usage_totals = _empty_usage_totals()
        for trace in traces:
            if isinstance(trace, dict):
                _add_usage_totals(usage_totals, dict(trace.get("usage") or {}))
        return {
            "outputs": outputs,
            "scores": scores,
            "traces": traces,
            "metadata": {
                "candidate": dict(candidate),
                "capture_traces": bool(capture_traces),
                "usage": usage_totals,
            },
        }

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> dict[str, list[str]]:
        _ = candidate, eval_batch, components_to_update
        return _candidate_variants()


def _policy_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": config.get("provider") or "openai",
        "inference_url": config.get("inference_url") or "https://api.openai.com/v1/chat/completions",
        "model": config.get("policy_model") or "gpt-4.1-nano",
        "temperature": float(config.get("policy_temperature") or 0.0),
        "max_completion_tokens": int(config.get("policy_max_completion_tokens") or 120),
    }


def _candidate_to_prompt_map(candidate: Any) -> dict[str, str]:
    selected = dict(getattr(candidate, "selected_instructions", {}) or {})
    return {
        "system_prompt": str(selected.get("system_prompt") or ""),
        "user_prompt": str(selected.get("user_prompt") or ""),
    }


def _best_train_scores(observations: list[Any]) -> list[float]:
    return [float(item.score) for item in observations]


def _empty_usage_totals() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _cached_prompt_tokens_from_usage(usage: dict[str, Any]) -> int:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        return int(details.get("cached_tokens") or 0)
    details = usage.get("input_tokens_details")
    if isinstance(details, dict):
        return int(details.get("cached_tokens") or 0)
    return int(usage.get("cached_prompt_tokens") or usage.get("cached_input_tokens") or 0)


def _usage_from_response(response_json: dict[str, Any]) -> dict[str, int]:
    usage = response_json.get("usage") if isinstance(response_json, dict) else None
    if not isinstance(usage, dict):
        return _empty_usage_totals()
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    cached_prompt_tokens = _cached_prompt_tokens_from_usage(usage)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _add_usage_totals(left: dict[str, int], right: dict[str, Any]) -> dict[str, int]:
    right_with_cache = dict(right)
    if "cached_prompt_tokens" not in right_with_cache:
        right_with_cache["cached_prompt_tokens"] = _cached_prompt_tokens_from_usage(right_with_cache)
    for key in ("prompt_tokens", "cached_prompt_tokens", "completion_tokens", "total_tokens"):
        left[key] = int(left.get(key) or 0) + int(right_with_cache.get(key) or 0)
    return left


def _cost_estimate_from_config(
    *,
    config: dict[str, Any],
    policy_usage: dict[str, int],
    proposer_usage: dict[str, int],
) -> tuple[float | None, str]:
    pricing = config.get("pricing_usd_per_1m_tokens")
    if not isinstance(pricing, dict):
        return None, "missing_pricing_usd_per_1m_tokens"

    def cost_for(prefix: str, usage: dict[str, int]) -> float:
        input_rate = float(pricing.get(f"{prefix}_input") or 0.0)
        cached_input_rate = float(pricing.get(f"{prefix}_cached_input") or input_rate)
        output_rate = float(pricing.get(f"{prefix}_output") or 0.0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        cached_prompt_tokens = min(prompt_tokens, int(usage.get("cached_prompt_tokens") or 0))
        uncached_prompt_tokens = max(0, prompt_tokens - cached_prompt_tokens)
        return (
            (uncached_prompt_tokens / 1_000_000.0) * input_rate
            + (cached_prompt_tokens / 1_000_000.0) * cached_input_rate
            + (int(usage.get("completion_tokens") or 0) / 1_000_000.0) * output_rate
        )

    return round(cost_for("policy", policy_usage) + cost_for("proposer", proposer_usage), 8), "estimated"


class OpenAIReflectionCallable:
    def __init__(self, config: dict[str, Any]) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "").strip() or None)
        self.model = str(config.get("proposer_model") or "gpt-5.4-mini")
        self.max_completion_tokens = int(config.get("proposer_max_completion_tokens") or 1200)
        self.usage_totals = _empty_usage_totals()

    def __call__(self, prompt: str | list[dict[str, str]]) -> str:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_completion_tokens,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage_payload = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
            _add_usage_totals(self.usage_totals, _usage_from_response({"usage": usage_payload}))
        message = response.choices[0].message if response.choices else None
        content = getattr(message, "content", "") if message is not None else ""
        if isinstance(content, list):
            return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
        return str(content or "")


def _heldout_seed_rewards(
    *,
    baseline_details: dict[str, Any] | None,
    best_details: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    baseline_outputs = list((baseline_details or {}).get("outputs") or [])
    baseline_scores = list((baseline_details or {}).get("scores") or [])
    best_outputs = list((best_details or {}).get("outputs") or [])
    best_scores = list((best_details or {}).get("scores") or [])
    by_row: dict[str, dict[str, Any]] = {}
    for ordinal, (output, score) in enumerate(zip(baseline_outputs, baseline_scores, strict=False)):
        if not isinstance(output, dict):
            continue
        seed = int(output.get("seed") or 0)
        row_key = f"{output.get('split') or ''}:{output.get('index')}:{ordinal}"
        by_row.setdefault(row_key, {"row_id": row_key, "ordinal": ordinal, "seed": seed})
        by_row[row_key].update(
            {
                "index": output.get("index"),
                "split": output.get("split"),
                "expected": output.get("expected"),
                "baseline_prediction": output.get("prediction"),
                "baseline_reward": float(score),
                "baseline_correct": bool(output.get("correct")),
            }
        )
    for ordinal, (output, score) in enumerate(zip(best_outputs, best_scores, strict=False)):
        if not isinstance(output, dict):
            continue
        seed = int(output.get("seed") or 0)
        row_key = f"{output.get('split') or ''}:{output.get('index')}:{ordinal}"
        by_row.setdefault(row_key, {"row_id": row_key, "ordinal": ordinal, "seed": seed})
        by_row[row_key].update(
            {
                "index": output.get("index", by_row[row_key].get("index")),
                "split": output.get("split", by_row[row_key].get("split")),
                "expected": output.get("expected", by_row[row_key].get("expected")),
                "best_prediction": output.get("prediction"),
                "best_reward": float(score),
                "best_correct": bool(output.get("correct")),
            }
        )
    rows = []
    for _row_key, row_value in sorted(by_row.items(), key=lambda item: int(item[1].get("ordinal") or 0)):
        row = dict(row_value)
        if row.get("baseline_reward") is not None and row.get("best_reward") is not None:
            row["reward_delta"] = float(row["best_reward"]) - float(row["baseline_reward"])
        rows.append(row)
    return rows


def _summarize_recent_trials(observations: list[Any], limit: int) -> dict[str, Any]:
    rows = []
    for item in observations[-max(1, int(limit)):]:
        details = dict(getattr(item, "details", {}) or {})
        rows.append(
            {
                "candidate_id": getattr(item, "candidate_id", None),
                "score": float(getattr(item, "score", 0.0)),
                "split": details.get("split"),
                "scores": details.get("scores"),
                "outputs": details.get("outputs"),
            }
        )
    scores = _best_train_scores(observations)
    return {
        "count": len(observations),
        "mean_score": (sum(scores) / len(scores)) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "recent_trials": rows,
    }


def _outputs_scores_from_eval_batch(eval_batch: Any) -> tuple[list[Any], list[float]]:
    outputs = list(getattr(eval_batch, "outputs", []) or [])
    scores = [float(item) for item in list(getattr(eval_batch, "scores", []) or [])]
    return outputs, scores


def _heldout_details_from_batch(
    *,
    candidate_id: str,
    candidate: dict[str, str],
    eval_batch: Any,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    outputs, scores = _outputs_scores_from_eval_batch(eval_batch)
    normalized_outputs: list[dict[str, Any]] = []
    for row, output, score in zip(rows, outputs, scores, strict=False):
        output_dict = dict(output) if isinstance(output, dict) else {"prediction": output}
        summary = output_dict.get("summary") if isinstance(output_dict.get("summary"), dict) else {}
        prediction = (
            output_dict.get("prediction")
            or output_dict.get("predicted_intent")
            or summary.get("prediction")
            or summary.get("predicted_intent")
            or summary.get("output")
        )
        normalized_outputs.append(
            {
                "seed": row.get("seed"),
                "index": row.get("index"),
                "split": row.get("split"),
                "expected": row.get("label"),
                "prediction": prediction,
                "correct": float(score) >= 1.0,
            }
        )
    return {
        "candidate_id": candidate_id,
        "split": "heldout",
        "scores": scores,
        "outputs": normalized_outputs,
        "selected_instructions": dict(candidate),
    }


def _write_comparison_artifacts(
    *,
    live_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    mode: str,
    optimizer_path: str,
    run_id: str,
    best_candidate: dict[str, Any],
    baseline_score: float | None,
    best_score: float | None,
    lift: float | None,
    total_metric_calls: int,
    policy_usage: dict[str, int] | None = None,
    proposer_usage: dict[str, int] | None = None,
    baseline_heldout_details: dict[str, Any] | None = None,
    best_heldout_details: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    policy_usage_totals = dict(policy_usage or _empty_usage_totals())
    proposer_usage_totals = dict(proposer_usage or _empty_usage_totals())
    if policy_usage is None and proposer_usage is None:
        estimated_cost_usd, cost_estimate_status = None, "missing_token_usage"
    else:
        estimated_cost_usd, cost_estimate_status = _cost_estimate_from_config(
            config=config,
            policy_usage=policy_usage_totals,
            proposer_usage=proposer_usage_totals,
        )
    heldout_seed_rewards = _heldout_seed_rewards(
        baseline_details=baseline_heldout_details,
        best_details=best_heldout_details,
    )
    heldout_rows = max(
        len((baseline_heldout_details or {}).get("outputs") or []),
        len((best_heldout_details or {}).get("outputs") or []),
    )
    payload = {
        "run_id": run_id,
        "mode": mode,
        "best_candidate": best_candidate,
        "baseline_score": baseline_score,
        "best_score": best_score,
        "lift": lift,
        "heldout_baseline_score": baseline_score,
        "heldout_best_score": best_score,
        "heldout_lift": lift,
        "heldout_rows": heldout_rows,
        "total_metric_calls": int(total_metric_calls),
        "policy_usage": policy_usage_totals,
        "proposer_usage": proposer_usage_totals,
        "estimated_cost_usd": estimated_cost_usd,
        "cost_estimate_status": cost_estimate_status,
        "heldout_seed_rewards": heldout_seed_rewards,
        "metadata": dict(metadata or {}),
    }
    _write_json(live_dir / "result.json", payload)
    _write_json(output_dir / "artifacts" / "best_candidate.json", {"candidate": best_candidate, "score": best_score})
    _write_json(
        output_dir / "artifacts" / "heldout_eval.json",
        {"baseline_score": baseline_score, "best_score": best_score, "lift": lift, "seed_rewards": heldout_seed_rewards},
    )
    _write_json(
        output_dir / "artifacts" / "heldout_seed_rewards.json",
        {
            "run_id": run_id,
            "task_id": "banking77",
            "baseline_candidate_id": (baseline_heldout_details or {}).get("candidate_id"),
            "best_candidate_id": (best_heldout_details or {}).get("candidate_id"),
            "rows": heldout_seed_rewards,
        },
    )
    _write_json(
        output_dir / "artifacts" / "miprov2_run_summary.json",
        {
            "run_id": run_id,
            "task_id": "banking77",
            "mode": mode,
            "best_candidate_id": (best_heldout_details or {}).get("candidate_id"),
            "baseline_score": baseline_score,
            "best_score": best_score,
            "heldout_score": best_score,
            "lift": lift,
            "total_metric_calls": int(total_metric_calls),
            "heldout_rows": heldout_rows,
            "policy_usage": policy_usage_totals,
            "proposer_usage": proposer_usage_totals,
            "estimated_cost_usd": estimated_cost_usd,
            "cost_estimate_status": cost_estimate_status,
        },
    )
    _write_json(
        output_dir / "artifacts" / "result_manifest.json",
        {
            "run_id": run_id,
            "task_id": "banking77",
            "mode": mode,
            "optimizer_path": optimizer_path,
            "policy_model": config.get("policy_model"),
            "proposer_model": config.get("proposer_model"),
            "total_metric_calls": int(total_metric_calls),
            "policy_usage": policy_usage_totals,
            "proposer_usage": proposer_usage_totals,
            "estimated_cost_usd": estimated_cost_usd,
            "cost_estimate_status": cost_estimate_status,
            "metadata": dict(metadata or {}),
        },
    )
    _write_json(
        live_dir / "command_result.json",
        {
            "returncode": 0,
            "mode": mode,
            "optimizer_path": optimizer_path,
            "policy_model": config.get("policy_model"),
            "proposer_model": config.get("proposer_model"),
            "total_metric_calls": int(total_metric_calls),
            "heldout_baseline_score": baseline_score,
            "heldout_best_score": best_score,
            "heldout_lift": lift,
            "heldout_rows": heldout_rows,
            "policy_usage": policy_usage_totals,
            "proposer_usage": proposer_usage_totals,
            "estimated_cost_usd": estimated_cost_usd,
            "cost_estimate_status": cost_estimate_status,
        },
    )


def execute_live(artifacts_dir: Path, *, config_path: Path, smoke: bool) -> int:
    _ensure_local_imports()
    loaded_key_names = _load_local_api_keys()
    from synth_optimizers.miprov2.core import (
        DiscreteMiproOptimizer,
        MiproGroundingHooks,
        MiproModuleTemplate,
        MiproOpenEnvProposerConfig,
        MiproPhase3Config,
        MiproProgramTemplate,
        OpenAIOpenEnvReactAgent,
        TpeConfig,
        compile_search_space,
        export_candidate_train_scores_from_ledger,
        run_phase3_loop,
    )
    from synth_optimizers.miprov2.core.run_ledger import SQLiteMiproRunLedger

    config = _load_config(config_path)
    train_rows, heldout_rows = _load_rows(config, smoke=smoke)
    live_dir = artifacts_dir / ("live_native_smoke" if smoke else "live_native_run")
    output_dir = live_dir / "miprov2_artifacts"
    ledger_path = live_dir / "ledger.sqlite"
    max_concurrency = min(int(config.get("concurrency") or 4), 4 if smoke else 20)
    program_template = MiproProgramTemplate(
        program_id="banking77_instruction_miprov2",
        modules=(
            MiproModuleTemplate(
                module_id="system_prompt",
                instruction_candidates=(_seed_candidate()["system_prompt"],),
            ),
            MiproModuleTemplate(
                module_id="user_prompt",
                instruction_candidates=(_seed_candidate()["user_prompt"],),
            ),
        ),
    )
    compiled = compile_search_space(program_template)
    optimizer = asyncio.run(
        DiscreteMiproOptimizer.from_search_space(
            search_space=compiled.search_space,
            tpe_config=TpeConfig(),
            rng_seed=int(config.get("rng_seed") or 41),
        )
    )
    adapter = Banking77MiproAdapter(_policy_config(config), max_concurrency=max_concurrency)
    metric_call_count = 0
    policy_usage_totals = _empty_usage_totals()
    heldout_details_by_candidate_id: dict[str, dict[str, Any]] = {}
    heldout_details_order: list[dict[str, Any]] = []

    async def evaluate_rows(rows: list[dict[str, Any]], candidate: Any, *, split: str) -> tuple[float, dict[str, Any]]:
        nonlocal metric_call_count
        batch = await adapter.evaluate(rows, _candidate_to_prompt_map(candidate), capture_traces=True)
        scores = [float(score) for score in batch["scores"]]
        metric_call_count += len(rows)
        _add_usage_totals(policy_usage_totals, dict((batch.get("metadata") or {}).get("usage") or {}))
        details = {
            "candidate_id": str(getattr(candidate, "candidate_id", "") or ""),
            "split": split,
            "scores": scores,
            "outputs": batch["outputs"],
            "traces": batch["traces"],
            "selected_instructions": dict(getattr(candidate, "selected_instructions", {}) or {}),
        }
        if split == "heldout":
            candidate_id = str(getattr(candidate, "candidate_id", "") or "")
            if candidate_id:
                heldout_details_by_candidate_id[candidate_id] = dict(details)
            heldout_details_order.append(dict(details))
        return (sum(scores) / len(scores) if scores else 0.0, details)

    async def evaluate_train(candidate: Any) -> tuple[float, dict[str, Any]]:
        return await evaluate_rows(train_rows, candidate, split="train")

    async def evaluate_heldout(candidate: Any) -> tuple[float, dict[str, Any]]:
        return await evaluate_rows(heldout_rows, candidate, split="heldout")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for native MIPRO OpenEnv proposer execution.")
    proposer_agent = OpenAIOpenEnvReactAgent(
        api_key=api_key,
        model=str(config.get("proposer_model") or "gpt-5.4-mini"),
        temperature=float(config.get("proposer_temperature") or 1.0),
        max_tokens=int(config.get("proposer_max_completion_tokens") or 1200),
        timeout_s=float(config.get("proposer_timeout_s") or 120),
    )
    proposer_rounds = 1 if smoke else int(config.get("max_proposer_sessions") or 6)
    train_rounds_per_proposer_round = 1 if smoke else max(
        1,
        int(config.get("target_train_candidates") or 10) // max(1, proposer_rounds),
    )
    outcome = asyncio.run(
        run_phase3_loop(
            compiled_space=compiled,
            optimizer=optimizer,
            agent=proposer_agent,
            evaluate_train=evaluate_train,
            evaluate_heldout=evaluate_heldout,
            grounding_hooks=MiproGroundingHooks(
                sample_train_rows=lambda _round_idx, limit: train_rows[: max(1, int(limit))],
                summarize_recent_trials=lambda observations, limit: _summarize_recent_trials(observations, limit),
            ),
            config=MiproPhase3Config(
                proposer_rounds=proposer_rounds,
                train_rounds_per_proposer_round=train_rounds_per_proposer_round,
                bootstrap_train_rounds=0,
                top_k=1 if smoke else min(4, max_concurrency),
                max_concurrency=max_concurrency,
                seed_with_baseline=True,
                heldout_interval=None,
                compute_final_heldout=True,
                proposer_trace_dir=str(output_dir / "artifacts" / "proposer_traces"),
                checkpoint_policy=str(config.get("checkpoint_policy") or "none"),
                checkpoint_dir=str(output_dir / "checkpoints"),
                proposer_config=MiproOpenEnvProposerConfig(
                    max_turns=8 if smoke else int(config.get("proposer_max_turns") or 32),
                    max_noop_turns=4 if smoke else int(config.get("proposer_max_noop_turns") or 12),
                    max_patch_actions_per_session=2 if smoke else int(config.get("proposer_max_patch_actions") or 8),
                    max_instruction_patches=2 if smoke else int(config.get("proposer_max_patch_actions") or 8),
                    max_demo_patches=0,
                    archive_root=str(output_dir / "artifacts" / "proposer_archives"),
                ),
            ),
            run_id=f"{config.get('run_id')}_native{'_smoke' if smoke else ''}",
            ledger_path=str(ledger_path),
            resume=False,
        )
    )

    run_read_model: dict[str, Any] = {}
    candidate_train_scores: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    if outcome.run_id and outcome.ledger_path:
        ledger = SQLiteMiproRunLedger(
            run_id=str(outcome.run_id),
            ledger_path=str(outcome.ledger_path),
            program_id=program_template.program_id,
            mode="phase3",
            resume=True,
        )
        try:
            run_read_model = ledger.build_run_read_model()
            candidate_train_scores = export_candidate_train_scores_from_ledger(
                ledger,
                task_id="banking77",
                output_path=output_dir / "artifacts" / "candidate_train_scores.json",
            )
            events = list(reversed(ledger.query_events(limit=1000)))
        finally:
            ledger.close()

    best_candidate = outcome.best_train_candidate.to_dict() if outcome.best_train_candidate is not None else None
    proposer_trace_paths = [
        _relative_to(path, live_dir) for path in outcome.proposer_trace_paths
    ]
    ledger_path_relative = _relative_to(outcome.ledger_path, live_dir)
    best_candidate_id = str((best_candidate or {}).get("candidate_id") or "")
    baseline_heldout_details = heldout_details_order[0] if heldout_details_order else None
    best_heldout_details = (
        heldout_details_by_candidate_id.get(best_candidate_id)
        or (heldout_details_order[-1] if heldout_details_order else None)
    )
    heldout_seed_rewards = _heldout_seed_rewards(
        baseline_details=baseline_heldout_details,
        best_details=best_heldout_details,
    )
    proposer_usage_totals = _empty_usage_totals()
    for session in outcome.proposer_sessions:
        summary = dict(session.get("proposer_summary") or {}) if isinstance(session, dict) else {}
        _add_usage_totals(proposer_usage_totals, summary)
    estimated_cost_usd, cost_estimate_status = _cost_estimate_from_config(
        config=config,
        policy_usage=policy_usage_totals,
        proposer_usage=proposer_usage_totals,
    )
    payload = {
        "run_id": outcome.run_id,
        "ledger_path": ledger_path_relative,
        "mode": "native_phase3_openenv",
        "best_candidate": best_candidate,
        "baseline_train_score": outcome.baseline_train_score,
        "best_train_score": outcome.best_train_score,
        "heldout_baseline_score": outcome.heldout_baseline_score,
        "heldout_best_score": outcome.heldout_best_score,
        "heldout_lift": outcome.heldout_lift,
        "total_metric_calls": metric_call_count,
        "proposer_sessions": outcome.proposer_sessions,
        "proposer_round_diagnostics": outcome.proposer_round_diagnostics,
        "proposer_diagnostics_aggregate": outcome.proposer_diagnostics_aggregate,
        "proposer_trace_paths": proposer_trace_paths,
        "policy_usage": dict(policy_usage_totals),
        "proposer_usage": dict(proposer_usage_totals),
        "estimated_cost_usd": estimated_cost_usd,
        "cost_estimate_status": cost_estimate_status,
        "heldout_seed_rewards": heldout_seed_rewards,
        "train_observation_count": len(outcome.train_observations),
        "heldout_snapshots": [asdict(snapshot) for snapshot in outcome.heldout_snapshots],
        "run_read_model": run_read_model,
        "candidate_train_scores": candidate_train_scores,
    }
    _write_json(live_dir / "result.json", payload)
    _write_json(
        output_dir / "artifacts" / "best_candidate.json",
        {
            "candidate": best_candidate,
            "score": outcome.heldout_best_score,
            "train_score": outcome.best_train_score,
        },
    )
    _write_json(
        output_dir / "artifacts" / "heldout_eval.json",
        {
            "baseline_score": outcome.heldout_baseline_score,
            "best_score": outcome.heldout_best_score,
            "lift": outcome.heldout_lift,
            "snapshots": [asdict(snapshot) for snapshot in outcome.heldout_snapshots],
            "seed_rewards": heldout_seed_rewards,
        },
    )
    _write_json(
        output_dir / "artifacts" / "heldout_seed_rewards.json",
        {
            "run_id": outcome.run_id,
            "task_id": "banking77",
            "baseline_candidate_id": (heldout_details_order[0] or {}).get("candidate_id")
            if heldout_details_order
            else None,
            "best_candidate_id": best_candidate_id,
            "rows": heldout_seed_rewards,
        },
    )
    _write_json(
        output_dir / "artifacts" / "miprov2_run_summary.json",
        {
            "run_id": outcome.run_id,
            "task_id": "banking77",
            "mode": "native_phase3_openenv",
            "best_candidate_id": (best_candidate or {}).get("candidate_id"),
            "baseline_score": outcome.heldout_baseline_score,
            "best_score": outcome.heldout_best_score,
            "heldout_score": outcome.heldout_best_score,
            "train_score": outcome.best_train_score,
            "lift": outcome.heldout_lift,
            "total_metric_calls": metric_call_count,
            "policy_usage": dict(policy_usage_totals),
            "proposer_usage": dict(proposer_usage_totals),
            "estimated_cost_usd": estimated_cost_usd,
            "cost_estimate_status": cost_estimate_status,
            "artifacts": {
                "best_candidate": "artifacts/best_candidate.json",
                "heldout_eval": "artifacts/heldout_eval.json",
                "heldout_seed_rewards": "artifacts/heldout_seed_rewards.json",
                "run_summary": "artifacts/miprov2_run_summary.json",
                "result_manifest": "artifacts/result_manifest.json",
                "run_read_model": "artifacts/run_read_model.json",
                "run_events": "artifacts/run_events.jsonl",
                "candidate_train_scores": "artifacts/candidate_train_scores.json",
                "checkpoints": "checkpoints",
                "proposer_traces": "artifacts/proposer_traces",
            },
        },
    )
    _write_json(
        output_dir / "artifacts" / "result_manifest.json",
        {
            "run_id": outcome.run_id,
            "task_id": "banking77",
            "mode": "native_phase3_openenv",
            "ledger_path": ledger_path_relative,
            "proposer_model": config.get("proposer_model"),
            "policy_model": config.get("policy_model"),
            "train_rows": len(train_rows),
            "heldout_rows": len(heldout_rows),
            "total_metric_calls": metric_call_count,
            "policy_usage": dict(policy_usage_totals),
            "proposer_usage": dict(proposer_usage_totals),
            "estimated_cost_usd": estimated_cost_usd,
            "cost_estimate_status": cost_estimate_status,
            "local_key_fallback_used": bool(loaded_key_names),
            "loaded_local_key_count": len(loaded_key_names),
        },
    )
    if run_read_model:
        _write_json(output_dir / "artifacts" / "run_read_model.json", run_read_model)
    if events:
        event_path = output_dir / "artifacts" / "run_events.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.write_text(
            "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n",
            encoding="utf-8",
        )
    _write_json(
        live_dir / "command_result.json",
        {
            "returncode": 0,
            "mode": "live_native_smoke" if smoke else "live_native_run",
            "optimizer_path": "native_phase3_openenv",
            "proposer_model": config.get("proposer_model"),
            "policy_model": config.get("policy_model"),
            "train_rows": len(train_rows),
            "heldout_rows": len(heldout_rows),
            "total_metric_calls": metric_call_count,
            "baseline_train_score": outcome.baseline_train_score,
            "best_train_score": outcome.best_train_score,
            "heldout_baseline_score": outcome.heldout_baseline_score,
            "heldout_best_score": outcome.heldout_best_score,
            "heldout_lift": outcome.heldout_lift,
            "policy_usage": dict(policy_usage_totals),
            "proposer_usage": dict(proposer_usage_totals),
            "estimated_cost_usd": estimated_cost_usd,
            "cost_estimate_status": cost_estimate_status,
            "proposer_round_count": len(outcome.proposer_sessions),
            "proposer_trace_paths": proposer_trace_paths,
            "local_key_fallback_used": bool(loaded_key_names),
            "loaded_local_key_count": len(loaded_key_names),
        },
    )
    print(f"wrote native MIPROv2 OpenEnv artifacts to {live_dir}")
    return 0


def execute_gepa(artifacts_dir: Path, *, config_path: Path, smoke: bool) -> int:
    _ensure_local_imports()
    _load_local_api_keys()
    from gepa import optimize
    from gepa.core.adapter import EvaluationBatch
    from synth_containers.http_client import HTTPContainerClient
    from synth_optimizers.miprov2 import ContainerGepaAdapter, ContainerGepaRolloutBinding
    from synth_service_app import app

    config = _load_config(config_path)
    train_rows, heldout_rows = _load_rows(config, smoke=smoke)
    live_dir = artifacts_dir / ("live_gepa_smoke" if smoke else "live_gepa_run")
    output_dir = live_dir / "miprov2_artifacts"
    live_dir.mkdir(parents=True, exist_ok=True)
    policy_config = _policy_config(config)
    client = HTTPContainerClient.from_app(app)
    adapter = ContainerGepaAdapter(
        client=client,
        binding=ContainerGepaRolloutBinding(
            task_id="banking77.intent_classification",
            extra_request={"policy": {"config": policy_config}},
        ),
        component_candidates=_candidate_variants(),
        max_concurrency=min(int(config.get("concurrency") or 4), 4 if smoke else 20),
    )

    class SyncContainerAdapter:
        propose_new_texts = None

        def evaluate(self, batch: list[dict[str, Any]], candidate: dict[str, str], capture_traces: bool = False) -> Any:
            result = asyncio.run(adapter.evaluate(batch, candidate, capture_traces=capture_traces))
            return EvaluationBatch(
                outputs=list(result.outputs),
                scores=list(result.scores),
                trajectories=list(result.traces) if result.traces else None,
                objective_scores=None,
            )

        def make_reflective_dataset(
            self,
            candidate: dict[str, str],
            eval_batch: Any,
            components_to_update: list[str],
        ) -> dict[str, list[dict[str, Any]]]:
            from synth_optimizers.miprov2.core import MiproEvaluationBatch

            batch = MiproEvaluationBatch(
                outputs=list(getattr(eval_batch, "outputs", []) or []),
                scores=list(getattr(eval_batch, "scores", []) or []),
                traces=list(getattr(eval_batch, "trajectories", []) or []),
            )
            return adapter.make_reflective_dataset(candidate, batch, components_to_update)

    sync_adapter = SyncContainerAdapter()
    seed_candidate = _seed_candidate()
    reflection_lm = OpenAIReflectionCallable(config)
    configured_metric_cap = config.get("gepa_max_metric_calls")
    metric_cap = (
        int(configured_metric_cap)
        if configured_metric_cap is not None
        else len(train_rows) + len(heldout_rows) + (len(train_rows) if smoke else len(train_rows) * 2)
    )
    run_id = f"{config.get('run_id')}_gepa{'_smoke' if smoke else ''}"
    run_dir = live_dir / "public_gepa_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_box: list[Any] = []
    error_box: list[BaseException] = []

    def run_optimize() -> None:
        try:
            result_box.append(
                optimize(
                    seed_candidate=seed_candidate,
                    trainset=train_rows,
                    valset=heldout_rows,
                    adapter=sync_adapter,
                    reflection_lm=reflection_lm,
                    max_metric_calls=max(1, int(metric_cap)),
                    run_dir=str(run_dir),
                    seed=int(config.get("rng_seed") or 41),
                    display_progress_bar=False,
                    track_best_outputs=False,
                    raise_on_exception=False,
                )
            )
        except BaseException as exc:  # noqa: BLE001
            error_box.append(exc)

    thread = threading.Thread(target=run_optimize, daemon=True)
    thread.start()
    thread.join(timeout=90 if smoke else 900)
    if thread.is_alive():
        raise RuntimeError("public_gepa_timeout")
    if error_box:
        raise error_box[0]
    gepa_result = result_box[0]
    best_candidate = dict(getattr(gepa_result, "best_candidate", seed_candidate) or seed_candidate)
    baseline_eval = sync_adapter.evaluate(heldout_rows, seed_candidate, capture_traces=True)
    best_eval = sync_adapter.evaluate(heldout_rows, best_candidate, capture_traces=True)
    baseline_outputs, baseline_scores = _outputs_scores_from_eval_batch(baseline_eval)
    best_outputs, best_scores = _outputs_scores_from_eval_batch(best_eval)
    baseline_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    best_score = sum(best_scores) / len(best_scores) if best_scores else 0.0
    baseline_details = _heldout_details_from_batch(
        candidate_id="baseline",
        candidate=seed_candidate,
        eval_batch=baseline_eval,
        rows=heldout_rows,
    )
    best_details = _heldout_details_from_batch(
        candidate_id="best",
        candidate=best_candidate,
        eval_batch=best_eval,
        rows=heldout_rows,
    )
    try:
        gepa_version = importlib.metadata.version("gepa")
    except Exception:
        gepa_version = None
    _write_comparison_artifacts(
        live_dir=live_dir,
        output_dir=output_dir,
        config=config,
        mode="public_gepa",
        optimizer_path="gepa-ai",
        run_id=run_id,
        best_candidate=best_candidate,
        baseline_score=float(baseline_score),
        best_score=float(best_score),
        lift=float(best_score - baseline_score),
        total_metric_calls=int(adapter.metric_call_count),
        policy_usage=dict(adapter.usage_totals),
        proposer_usage=dict(reflection_lm.usage_totals),
        baseline_heldout_details=baseline_details,
        best_heldout_details=best_details,
        metadata={
            "gepa_version": gepa_version,
            "gepa_run_dir": str(run_dir),
            "gepa_max_metric_calls": int(metric_cap),
            "gepa_num_candidates": getattr(gepa_result, "num_candidates", None),
            "gepa_best_idx": getattr(gepa_result, "best_idx", None),
        },
    )
    asyncio.run(client.aclose())
    print(f"wrote public GEPA artifacts to {live_dir}")
    return 0


def _dspy_model_id(model: str) -> str:
    text = str(model or "").strip()
    return text if "/" in text else f"openai/{text}"


def _normalize_label_text(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _dspy_examples(rows: list[dict[str, Any]]) -> list[Any]:
    import dspy

    return [
        dspy.Example(text=str(row.get("text") or ""), gold_label=str(row.get("label") or "")).with_inputs("text")
        for row in rows
    ]


def _dspy_metric(example: Any, pred: Any, trace: Any = None, *_: Any, **__: Any) -> float:
    del trace
    expected = _normalize_label_text(str(getattr(example, "gold_label", "") or ""))
    prediction = _normalize_label_text(str(getattr(pred, "intent", pred) or ""))
    return 1.0 if prediction == expected else 0.0


def _build_dspy_program(seed_prompt: str) -> Any:
    import dspy
    from synth_service_app import Banking77Dataset

    labels = ", ".join(Banking77Dataset().label_names)

    class Banking77Signature(dspy.Signature):  # type: ignore[misc, valid-type]
        """Predict the Banking77 intent label for the input customer text."""

        text: str = dspy.InputField(desc="Customer message")
        intent: str = dspy.OutputField(desc=f"One label from: {labels}")

    class Banking77Program(dspy.Module):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.classify = dspy.Predict(Banking77Signature)

        def forward(self, text: str) -> Any:
            prediction = self.classify(text=text)
            return dspy.Prediction(intent=str(getattr(prediction, "intent", "")))

    program = Banking77Program()
    program.classify.signature = program.classify.signature.with_instructions(str(seed_prompt))
    return program


def _evaluate_dspy(program: Any, examples: list[Any], *, num_threads: int) -> tuple[float, list[Any], list[float]]:
    import dspy

    evaluator = dspy.Evaluate(
        devset=list(examples),
        metric=_dspy_metric,
        num_threads=max(1, int(num_threads)),
        return_all_scores=True,
        failure_score=0.0,
        display_table=False,
        display_progress=False,
        max_errors=max(100, len(examples) * 50),
    )
    result = evaluator(program)
    rows = list(getattr(result, "results", []) or [])
    outputs = [item[1] for item in rows if isinstance(item, tuple) and len(item) >= 2]
    scores = [float(item[2]) for item in rows if isinstance(item, tuple) and len(item) >= 3]
    return (sum(scores) / len(scores) if scores else 0.0), outputs, scores


def _dspy_candidate_map(program: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_name, predictor in list(program.named_predictors()):
        signature = getattr(predictor, "signature", None)
        out[str(raw_name)] = str(getattr(signature, "instructions", "") or "")
    return dict(sorted(out.items(), key=lambda item: item[0]))


def _usage_totals_from_dspy_tracker(usage_by_lm: dict[str, Any], model_id: str) -> dict[str, int]:
    totals = _empty_usage_totals()
    raw_model = str(model_id or "")
    suffix = raw_model.split("/", 1)[1] if "/" in raw_model else raw_model
    for lm_name, usage in dict(usage_by_lm or {}).items():
        name = str(lm_name or "")
        if name not in {raw_model, suffix} and not name.endswith(f"/{suffix}"):
            continue
        if isinstance(usage, dict):
            _add_usage_totals(totals, usage)
    return totals


def execute_dspy_mipro(artifacts_dir: Path, *, config_path: Path, smoke: bool) -> int:
    _ensure_local_imports()
    _load_local_api_keys()
    import dspy

    config = _load_config(config_path)
    train_rows, heldout_rows = _load_rows(config, smoke=smoke)
    live_dir = artifacts_dir / ("live_dspy_mipro_smoke" if smoke else "live_dspy_mipro_run")
    output_dir = live_dir / "miprov2_artifacts"
    live_dir.mkdir(parents=True, exist_ok=True)
    train_examples = _dspy_examples(train_rows)
    heldout_examples = _dspy_examples(heldout_rows)
    task_lm = dspy.LM(_dspy_model_id(str(config.get("policy_model") or "gpt-4.1-nano")), cache=False)
    prompt_lm = dspy.LM(_dspy_model_id(str(config.get("proposer_model") or "gpt-5.4-mini")), cache=False)
    dspy.configure(lm=task_lm)
    num_threads = 1 if smoke else min(int(config.get("concurrency") or 4), 20)
    baseline_program = _build_dspy_program(_seed_candidate()["system_prompt"])
    with dspy.track_usage() as usage_tracker:
        baseline_score, baseline_outputs, baseline_scores = _evaluate_dspy(
            baseline_program,
            heldout_examples,
            num_threads=num_threads,
        )
        teleprompter = dspy.MIPROv2(
            metric=_dspy_metric,
            prompt_model=prompt_lm,
            task_model=task_lm,
            auto=None,
            num_candidates=2 if smoke else min(16, int(config.get("target_train_candidates") or 8)),
            num_threads=num_threads,
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
            seed=int(config.get("rng_seed") or 41),
            verbose=False,
        )
        optimized_program = teleprompter.compile(
            student=baseline_program,
            trainset=train_examples,
            valset=heldout_examples,
            num_trials=1 if smoke else max(1, int(config.get("target_train_candidates") or 8) - 1),
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
            minibatch=False,
            seed=int(config.get("rng_seed") or 41),
        )
        best_score, best_outputs, best_scores = _evaluate_dspy(
            optimized_program,
            heldout_examples,
            num_threads=num_threads,
        )
    dspy_usage_by_lm = usage_tracker.get_total_tokens()
    policy_usage_totals = _usage_totals_from_dspy_tracker(dspy_usage_by_lm, str(task_lm.model))
    proposer_usage_totals = _usage_totals_from_dspy_tracker(dspy_usage_by_lm, str(prompt_lm.model))
    best_candidate = _dspy_candidate_map(optimized_program)
    baseline_details = {
        "candidate_id": "baseline",
        "scores": baseline_scores,
        "outputs": [
            {
                "seed": row.get("seed"),
                "index": row.get("index"),
                "split": row.get("split"),
                "expected": row.get("label"),
                "prediction": str(getattr(output, "intent", output) or ""),
                "correct": float(score) >= 1.0,
            }
            for row, output, score in zip(heldout_rows, baseline_outputs, baseline_scores, strict=False)
        ],
    }
    best_details = {
        "candidate_id": "best",
        "scores": best_scores,
        "outputs": [
            {
                "seed": row.get("seed"),
                "index": row.get("index"),
                "split": row.get("split"),
                "expected": row.get("label"),
                "prediction": str(getattr(output, "intent", output) or ""),
                "correct": float(score) >= 1.0,
            }
            for row, output, score in zip(heldout_rows, best_outputs, best_scores, strict=False)
        ],
    }
    try:
        dspy_version = importlib.metadata.version("dspy")
    except Exception:
        dspy_version = None
    total_metric_calls = len(heldout_examples) * 2 + len(train_examples) * (2 if smoke else int(config.get("target_train_candidates") or 8))
    _write_comparison_artifacts(
        live_dir=live_dir,
        output_dir=output_dir,
        config=config,
        mode="dspy_miprov2",
        optimizer_path="dspy-miprov2",
        run_id=f"{config.get('run_id')}_dspy_mipro{'_smoke' if smoke else ''}",
        best_candidate=best_candidate,
        baseline_score=float(baseline_score),
        best_score=float(best_score),
        lift=float(best_score - baseline_score),
        total_metric_calls=int(total_metric_calls),
        policy_usage=policy_usage_totals,
        proposer_usage=proposer_usage_totals,
        baseline_heldout_details=baseline_details,
        best_heldout_details=best_details,
        metadata={"dspy_version": dspy_version, "dspy_usage_by_lm": dspy_usage_by_lm},
    )
    print(f"wrote DSPy MIPROv2 artifacts to {live_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or run the Banking77 MIPROv2 cookbook.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--optimizer-mode",
        choices=("synth-miprov2", "gepa-ai", "dspy-miprov2"),
        default="synth-miprov2",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a small live execution before the full config budget.")
    args = parser.parse_args(argv)

    artifacts_dir = Path(args.artifacts_dir)
    plan = build_plan(args)
    _write_json(artifacts_dir / "plan.json", plan)
    _write_json(artifacts_dir / "known_evidence.json", KNOWN_EVIDENCE)
    _write_summary(artifacts_dir / "summary.md", plan)
    if not args.execute:
        print(f"wrote dry-run artifacts to {artifacts_dir}")
        return 0
    if args.optimizer_mode == "gepa-ai":
        return execute_gepa(artifacts_dir, config_path=Path(args.config), smoke=bool(args.smoke))
    if args.optimizer_mode == "dspy-miprov2":
        return execute_dspy_mipro(artifacts_dir, config_path=Path(args.config), smoke=bool(args.smoke))
    return execute_live(artifacts_dir, config_path=Path(args.config), smoke=bool(args.smoke))


if __name__ == "__main__":
    raise SystemExit(main())
