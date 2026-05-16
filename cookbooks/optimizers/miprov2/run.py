from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
import random
import sys
import textwrap
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── terminal colors (auto-disabled when stdout is not a tty) ──────────────────
_USE_COLOR = sys.stdout.isatty()

def _ansi(text: str, *codes: int) -> str:
    return f"\033[{';'.join(str(c) for c in codes)}m{text}\033[0m" if _USE_COLOR else str(text)

def _dim(t: str) -> str: return _ansi(t, 2)
def _bold(t: str) -> str: return _ansi(t, 1)
def _red(t: str) -> str: return _ansi(t, 31)
def _green(t: str) -> str: return _ansi(t, 32)
def _yellow(t: str) -> str: return _ansi(t, 33)
def _cyan(t: str) -> str: return _ansi(t, 36)
def _magenta(t: str) -> str: return _ansi(t, 35)
def _bgreen(t: str) -> str: return _ansi(t, 1, 32)
def _byellow(t: str) -> str: return _ansi(t, 1, 33)


COOKBOOK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = COOKBOOK_ROOT.parents[2]
DEFAULT_ARTIFACTS_DIR = COOKBOOK_ROOT / "run_artifacts" / "dry_run"
CONFIG_PATH = COOKBOOK_ROOT / "miprov2" / "banking77_openai_split_confusable_30x300_100heldout.json"
OPTIMIZERS_SRC = REPO_ROOT / "packages" / "synth-optimizers" / "src"
CONTAINERS_SRC = REPO_ROOT / "packages" / "synth-containers" / "src"
BANKING77_CONTAINER_SRC = COOKBOOK_ROOT / "banking77_container"
LOCAL_SYNTH_ENV = REPO_ROOT / ".env"
_LOCAL_SYNTH_ENV_FALLBACK = REPO_ROOT.parent / "synth-ai" / ".env"


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
    if os.environ.get("MIPRO_SKIP_LOCAL_API_KEY_LOAD"):
        return loaded
    env_path = LOCAL_SYNTH_ENV if LOCAL_SYNTH_ENV.exists() else _LOCAL_SYNTH_ENV_FALLBACK
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
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
    from synth_service_app import (
        DEFAULT_STAGE1_SYSTEM_PROMPT,
        DEFAULT_STAGE1_USER_PROMPT,
        DEFAULT_STAGE2_SYSTEM_PROMPT,
        DEFAULT_STAGE2_USER_PROMPT,
    )

    return {
        "stage1_system": DEFAULT_STAGE1_SYSTEM_PROMPT,
        "stage1_user": DEFAULT_STAGE1_USER_PROMPT,
        "stage2_system": DEFAULT_STAGE2_SYSTEM_PROMPT,
        "stage2_user": DEFAULT_STAGE2_USER_PROMPT,
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


def _miprov2_contract_routes(metadata_payload: dict[str, Any]) -> dict[str, str]:
    metadata = metadata_payload.get("metadata") if isinstance(metadata_payload.get("metadata"), dict) else {}
    contracts = metadata.get("optimizer_contracts") if isinstance(metadata.get("optimizer_contracts"), dict) else {}
    mipro = contracts.get("miprov2") if isinstance(contracts.get("miprov2"), dict) else {}
    return {str(k): str(v) for k, v in mipro.items() if isinstance(v, str)}


def _candidate_variants_from_program(program_payload: dict[str, Any]) -> dict[str, list[str]]:
    search_space = program_payload.get("search_space") if isinstance(program_payload.get("search_space"), dict) else {}
    raw = search_space.get("initial_candidates") if isinstance(search_space.get("initial_candidates"), dict) else {}
    variants: dict[str, list[str]] = {}
    for module_id, values in raw.items():
        if isinstance(values, list):
            clean = [str(value) for value in values if str(value).strip()]
        elif isinstance(values, str):
            clean = [values]
        else:
            clean = []
        if clean:
            variants[str(module_id)] = clean
    return variants


async def _load_container_rows(client: Any, config: dict[str, Any], *, smoke: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filters: dict[str, Any] = {}
    if config.get("label_ids"):
        filters["label_ids"] = [int(label_id) for label_id in (config.get("label_ids") or [])]
    train_seeds = [int(seed) for seed in (config.get("train_seeds") or [])]
    heldout_seeds = [int(seed) for seed in (config.get("heldout_seeds") or [])]
    if smoke:
        train_seeds = train_seeds[:3]
        heldout_seeds = heldout_seeds[:5]

    async def rows_for_split(split: str, seeds: list[int]) -> list[dict[str, Any]]:
        payload = {
            "split": split,
            "seeds": seeds,
            "filters": filters,
        }
        response = await client.dataset_rows(payload)
        raw_rows = response.get("rows") if isinstance(response.get("rows"), list) else []
        rows: list[dict[str, Any]] = []
        for item in raw_rows:
            if not isinstance(item, dict):
                continue
            example = item.get("example") if isinstance(item.get("example"), dict) else item
            row = dict(example)
            row.setdefault("seed", item.get("seed"))
            row.setdefault("split", split)
            rows.append(row)
        return rows

    return await rows_for_split("train", train_seeds), await rows_for_split("test", heldout_seeds)


def _candidate_variants() -> dict[str, list[str]]:
    from synth_service_app import (
        DEFAULT_STAGE1_SYSTEM_PROMPT,
        DEFAULT_STAGE1_USER_PROMPT,
        DEFAULT_STAGE2_SYSTEM_PROMPT,
        DEFAULT_STAGE2_USER_PROMPT,
    )

    return {
        "stage1_system": [
            DEFAULT_STAGE1_SYSTEM_PROMPT,
            (
                "Group the customer banking query into one category. "
                "Call predict_category with the best matching group name."
            ),
            (
                "Identify the broad category of this banking issue and call predict_category with it. "
                "Choose the single most relevant category."
            ),
        ],
        "stage1_user": [
            DEFAULT_STAGE1_USER_PROMPT,
            (
                "Customer query:\n{query}\n\n"
                "Category groups:\n{available_categories}\n\n"
                "Call predict_category with the best matching category."
            ),
            (
                "Query:\n{query}\n\n"
                "Valid categories:\n{available_categories}\n\n"
                "Return the best matching category with predict_category."
            ),
        ],
        "stage2_system": [
            DEFAULT_STAGE2_SYSTEM_PROMPT,
            (
                "Classify the customer banking query into exactly one Banking77 intent. "
                "Prefer the most specific available label and respond only with a banking77_classify tool call."
            ),
            (
                "You are a precise Banking77 intent classifier. Distinguish close banking-support intents carefully, "
                "then call banking77_classify with exactly one label from the provided intent list."
            ),
        ],
        "stage2_user": [
            DEFAULT_STAGE2_USER_PROMPT,
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


class Banking77StageSpec:
    """Specification for one stage in the Banking77 evaluation pipeline."""

    def __init__(
        self,
        *,
        module_prefix: str,
        tool_name: str,
        tool_description: str,
        tool_result_field: str,
        default_system: str,
        default_user: str,
        get_template_vars: Any,
        tool_enum: list[str] | None = None,
    ) -> None:
        self.module_prefix = module_prefix
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_result_field = tool_result_field
        self.default_system = default_system
        self.default_user = default_user
        self.get_template_vars = get_template_vars  # (row, prev_stage_outputs) -> dict[str, str]
        self.tool_enum = tool_enum


class Banking77MiproAdapter:
    def __init__(
        self,
        policy_cfg: dict[str, Any],
        *,
        max_concurrency: int,
        stage_specs: list[Banking77StageSpec],
        label_names: list[str],
    ) -> None:
        self.policy_cfg = dict(policy_cfg)
        self.max_concurrency = max(1, int(max_concurrency))
        self.stage_specs = list(stage_specs)
        self.label_names = list(label_names)

    async def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> dict[str, Any]:
        from synth_service_app import call_chat_completion, extract_prediction, _normalize_label

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _call(system: str, user: str, spec: Banking77StageSpec) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
            retry_attempts = 4
            for attempt in range(1, retry_attempts + 1):
                try:
                    async with semaphore:
                        return await call_chat_completion(
                            policy_cfg=self.policy_cfg,
                            system_prompt=system,
                            user_prompt=user,
                            tool_name=spec.tool_name,
                            tool_description=spec.tool_description,
                            tool_result_field=spec.tool_result_field,
                            tool_enum=spec.tool_enum,
                        )
                except Exception as exc:
                    status_code = getattr(getattr(exc, "response", None), "status_code", None)
                    if attempt >= retry_attempts or status_code not in {408, 409, 429, 500, 502, 503, 504}:
                        raise
                    await asyncio.sleep(min(20.0, 1.5 * (2 ** (attempt - 1))))
            raise RuntimeError("unreachable")

        async def evaluate_row(row: dict[str, Any]) -> tuple[dict[str, Any], float, dict[str, Any] | None]:
            prev_stage_outputs: list[dict[str, Any]] = []
            combined_usage = _empty_usage_totals()
            for spec in self.stage_specs:
                system = str(candidate.get(f"{spec.module_prefix}_system") or "") or spec.default_system
                user_tmpl = str(candidate.get(f"{spec.module_prefix}_user") or "") or spec.default_user
                template_vars = spec.get_template_vars(row, prev_stage_outputs)
                user = user_tmpl.format(**template_vars)
                raw, rj, tc = await _call(system, user, spec)
                valid_labels = spec.tool_enum if spec.tool_enum else self.label_names
                prediction = extract_prediction(
                    raw_text=raw, tool_calls=tc, label_names=valid_labels,
                    tool_name=spec.tool_name, result_field=spec.tool_result_field,
                )
                prev_stage_outputs.append({"module_prefix": spec.module_prefix, "prediction": prediction})
                _add_usage_totals(combined_usage, _usage_from_response(rj))

            final_prediction = prev_stage_outputs[-1]["prediction"] if prev_stage_outputs else ""
            expected = str(row["label"])
            score = 1.0 if _normalize_label(final_prediction) == _normalize_label(expected) else 0.0
            output: dict[str, Any] = {
                "seed": row["seed"],
                "index": row["index"],
                "split": row["split"],
                "prediction": final_prediction,
                "expected": expected,
                "correct": score >= 1.0,
                "stage_predictions": {o["module_prefix"]: o["prediction"] for o in prev_stage_outputs},
            }
            trace = None
            if capture_traces:
                trace = {**output, "query": row["text"], "usage": combined_usage}
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
    return {k: str(v) for k, v in selected.items()}


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
    if isinstance(eval_batch, dict):
        outputs = list(eval_batch.get("outputs") or [])
        scores = [float(item) for item in list(eval_batch.get("scores") or [])]
        return outputs, scores
    outputs = list(getattr(eval_batch, "outputs", []) or [])
    scores = [float(item) for item in list(getattr(eval_batch, "scores", []) or [])]
    return outputs, scores


def _eval_batch_payload(eval_batch: Any) -> tuple[list[Any], list[float], list[Any], dict[str, Any]]:
    outputs, scores = _outputs_scores_from_eval_batch(eval_batch)
    if isinstance(eval_batch, dict):
        return outputs, scores, list(eval_batch.get("traces") or []), dict(eval_batch.get("metadata") or {})
    return (
        outputs,
        scores,
        list(getattr(eval_batch, "traces", []) or []),
        dict(getattr(eval_batch, "metadata", {}) or {}),
    )


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


# ── GSM8K task ────────────────────────────────────────────────────────────────

_GSM8K_DEFAULT_SYSTEM = (
    "You are a precise math problem solver. Read the problem carefully, reason step by step, "
    "and call submit_answer with the final numerical answer."
)
_GSM8K_DEFAULT_USER = (
    "Problem: {question}\n\n"
    "Solve step by step. Call submit_answer with only the final number (digits only, no units or commas)."
)


def _gsm8k_extract_answer(text: str) -> str:
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "")
    return text.strip().replace(",", "")


def _gsm8k_normalize(text: str) -> str:
    import re
    text = re.sub(r"[,$%]", "", str(text).strip().lower()).rstrip(".")
    try:
        v = float(text)
        return str(int(v)) if v == int(v) else str(v)
    except (ValueError, OverflowError):
        return text


def _load_rows_gsm8k(config: dict[str, Any], *, smoke: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset

    def rows_for_split(split: str, seeds: list[int]) -> list[dict[str, Any]]:
        dataset = load_dataset("gsm8k", "main", split=split)
        n = len(dataset)
        rows: list[dict[str, Any]] = []
        for seed in seeds:
            idx = int(seed) % n
            row = dataset[idx]
            rows.append({
                "seed": int(seed),
                "index": idx,
                "split": split,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
            })
        return rows

    train_seeds = [int(s) for s in (config.get("train_seeds") or [])]
    heldout_seeds = [int(s) for s in (config.get("heldout_seeds") or [])]
    if smoke:
        train_seeds = train_seeds[:3]
        heldout_seeds = heldout_seeds[:5]
    return rows_for_split("train", train_seeds), rows_for_split("test", heldout_seeds)


# ── HotpotQA task ─────────────────────────────────────────────────────────────

_HOTPOTQA_DEFAULT_SYSTEM = (
    "You are a multi-hop question answering system. Read the provided passages carefully "
    "and reason across them. Call submit_answer with a short, direct answer."
)
_HOTPOTQA_DEFAULT_USER = (
    "Question: {question}\n\n"
    "Passages:\n{context}\n\n"
    "Call submit_answer with the answer (a name, date, number, or short phrase)."
)


def _hotpotqa_normalize(text: str) -> str:
    import re
    text = str(text).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _load_rows_hotpotqa(config: dict[str, Any], *, smoke: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset

    def _fmt_context(ctx: dict[str, Any]) -> str:
        parts = []
        for title, sents in zip(ctx.get("title") or [], ctx.get("sentences") or []):
            parts.append(f"[{title}]\n" + " ".join(str(s) for s in sents))
        return "\n\n".join(parts)

    def rows_for_split(split: str, seeds: list[int]) -> list[dict[str, Any]]:
        dataset = load_dataset("hotpot_qa", "distractor", split=split)
        n = len(dataset)
        rows: list[dict[str, Any]] = []
        for seed in seeds:
            idx = int(seed) % n
            row = dataset[idx]
            rows.append({
                "seed": int(seed),
                "index": idx,
                "split": split,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "context": _fmt_context(dict(row.get("context") or {})),
            })
        return rows

    train_seeds = [int(s) for s in (config.get("train_seeds") or [])]
    heldout_seeds = [int(s) for s in (config.get("heldout_seeds") or [])]
    if smoke:
        train_seeds = train_seeds[:3]
        heldout_seeds = heldout_seeds[:5]
    return rows_for_split("train", train_seeds), rows_for_split("validation", heldout_seeds)


# ── Shared direct LLM tool-call (no container service) ────────────────────────

async def _direct_tool_call(
    policy_cfg: dict[str, Any],
    system: str,
    user: str,
    tool_name: str,
    tool_description: str,
    answer_description: str,
    semaphore: asyncio.Semaphore,
    *,
    retry_attempts: int = 4,
) -> tuple[str, dict[str, Any]]:
    from openai import AsyncOpenAI

    provider = str(policy_cfg.get("provider") or "openai").lower()
    model = str(policy_cfg.get("model") or "gpt-4.1-nano")
    temperature = float(policy_cfg.get("temperature") or 0.0)
    max_tokens = int(policy_cfg.get("max_completion_tokens") or 512)

    if provider == "groq":
        client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY", ""),
        )
    else:
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    tool = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": answer_description}},
                "required": ["answer"],
            },
        },
    }

    resp = None
    _use_tools = True
    for attempt in range(1, retry_attempts + 1):
        try:
            async with semaphore:
                if _use_tools:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        tools=[tool],  # type: ignore[arg-type]
                        tool_choice="required",
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
            break
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            # 400 from Groq means the model hallucinated a wrong tool name; retry without tools
            if status == 400 and _use_tools and attempt < retry_attempts:
                _use_tools = False
                continue
            if attempt >= retry_attempts or status not in {408, 409, 429, 500, 502, 503, 504}:
                raise
            await asyncio.sleep(min(20.0, 1.5 * (2 ** (attempt - 1))))

    rj = resp.model_dump() if resp is not None else {}
    answer = ""
    for choice in rj.get("choices") or []:
        msg = choice.get("message") or {}
        for tc in msg.get("tool_calls") or []:
            args_str = (tc.get("function") or {}).get("arguments") or ""
            try:
                answer = str(json.loads(args_str).get("answer") or "").strip()
                if answer:
                    break
            except (json.JSONDecodeError, KeyError):
                pass
        if not answer:
            answer = str(msg.get("content") or "").strip()
        if answer:
            break
    return answer, rj


class Gsm8kMiproAdapter:
    def __init__(self, policy_cfg: dict[str, Any], *, max_concurrency: int) -> None:
        self.policy_cfg = dict(policy_cfg)
        self.max_concurrency = max(1, int(max_concurrency))

    async def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> dict[str, Any]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def evaluate_row(row: dict[str, Any]) -> tuple[dict[str, Any], float, dict[str, Any] | None]:
            system = str(candidate.get("stage1_system") or _GSM8K_DEFAULT_SYSTEM)
            user_tmpl = str(candidate.get("stage1_user") or _GSM8K_DEFAULT_USER)
            user = user_tmpl.format(question=row["question"])
            answer, rj = await _direct_tool_call(
                self.policy_cfg, system, user,
                tool_name="submit_answer",
                tool_description="Submit the final numerical answer to the math problem.",
                answer_description="The final numerical answer (digits only, no units or commas).",
                semaphore=semaphore,
            )
            gold = _gsm8k_extract_answer(row["answer"])
            pred = _gsm8k_extract_answer(answer)  # handles both bare number and #### N in CoT text
            score = 1.0 if _gsm8k_normalize(pred) == _gsm8k_normalize(gold) else 0.0
            output: dict[str, Any] = {
                "seed": row["seed"],
                "index": row["index"],
                "split": row["split"],
                "prediction": answer,
                "expected": gold,
                "correct": score >= 1.0,
            }
            trace = None
            if capture_traces:
                trace = {**output, "question": row["question"], "usage": _usage_from_response(rj)}
            return output, score, trace

        results = await asyncio.gather(*(evaluate_row(dict(row)) for row in batch))
        outputs = [r[0] for r in results]
        scores = [float(r[1]) for r in results]
        traces = [r[2] for r in results if r[2] is not None]
        usage_totals = _empty_usage_totals()
        for t in traces:
            if isinstance(t, dict):
                _add_usage_totals(usage_totals, dict(t.get("usage") or {}))
        return {
            "outputs": outputs, "scores": scores, "traces": traces,
            "metadata": {"candidate": dict(candidate), "capture_traces": bool(capture_traces), "usage": usage_totals},
        }


class HotpotQaMiproAdapter:
    def __init__(self, policy_cfg: dict[str, Any], *, max_concurrency: int) -> None:
        self.policy_cfg = dict(policy_cfg)
        self.max_concurrency = max(1, int(max_concurrency))

    async def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> dict[str, Any]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def evaluate_row(row: dict[str, Any]) -> tuple[dict[str, Any], float, dict[str, Any] | None]:
            system = str(candidate.get("stage1_system") or _HOTPOTQA_DEFAULT_SYSTEM)
            user_tmpl = str(candidate.get("stage1_user") or _HOTPOTQA_DEFAULT_USER)
            user = user_tmpl.format(question=row["question"], context=row["context"])
            answer, rj = await _direct_tool_call(
                self.policy_cfg, system, user,
                tool_name="submit_answer",
                tool_description="Submit the answer to the multi-hop question.",
                answer_description="A concise answer: a name, date, number, or short phrase.",
                semaphore=semaphore,
            )
            gold = row["answer"]
            score = 1.0 if _hotpotqa_normalize(answer) == _hotpotqa_normalize(gold) else 0.0
            output: dict[str, Any] = {
                "seed": row["seed"],
                "index": row["index"],
                "split": row["split"],
                "prediction": answer,
                "expected": gold,
                "correct": score >= 1.0,
            }
            trace = None
            if capture_traces:
                trace = {**output, "question": row["question"], "usage": _usage_from_response(rj)}
            return output, score, trace

        results = await asyncio.gather(*(evaluate_row(dict(row)) for row in batch))
        outputs = [r[0] for r in results]
        scores = [float(r[1]) for r in results]
        traces = [r[2] for r in results if r[2] is not None]
        usage_totals = _empty_usage_totals()
        for t in traces:
            if isinstance(t, dict):
                _add_usage_totals(usage_totals, dict(t.get("usage") or {}))
        return {
            "outputs": outputs, "scores": scores, "traces": traces,
            "metadata": {"candidate": dict(candidate), "capture_traces": bool(capture_traces), "usage": usage_totals},
        }


def execute_live(
    artifacts_dir: Path,
    *,
    config_path: Path,
    smoke: bool,
    interactive_proposer: bool = False,
    interactive_session_root: Path | None = None,
    interactive_resume_session_id: str | None = None,
) -> int:
    _ensure_local_imports()
    loaded_key_names = _load_local_api_keys()
    from proposer import PROPOSER_SEGMENT_GUIDANCE, resolve_proposer_backend
    from synth_optimizers.miprov2.core import (
        DiscreteMiproOptimizer,
        MiproGroundingHooks,
        MiproModuleTemplate,
        MiproOpenEnvProposerConfig,
        MiproOpenEnvProposerVariant,
        MiproPhase3Config,
        MiproProgramTemplate,
        MiproStageTemplate,
        TpeConfig,
        compile_search_space,
        decode_config,
        export_candidate_train_scores_from_ledger,
        run_phase3_loop,
    )
    from synth_optimizers.miprov2.core.run_ledger import SQLiteMiproRunLedger

    config = _load_config(config_path)
    task = str(config.get("task") or "banking77").lower().strip()
    if task == "gsm8k":
        train_rows, heldout_rows = _load_rows_gsm8k(config, smoke=smoke)
    elif task == "hotpotqa":
        train_rows, heldout_rows = _load_rows_hotpotqa(config, smoke=smoke)
    else:
        train_rows, heldout_rows = [], []
    live_dir = artifacts_dir / ("live_native_smoke" if smoke else "live_native_run")
    output_dir = live_dir / "miprov2_artifacts"
    ledger_path = live_dir / "ledger.sqlite"
    max_concurrency = min(int(config.get("concurrency") or 4), 4 if smoke else 20)
    native_run_id = f"{config.get('run_id')}_native{'_smoke' if smoke else ''}"

    # ── Task-specific program template + adapter ───────────────────────────────
    if task == "gsm8k":
        program_template = MiproProgramTemplate(
            program_id="gsm8k_miprov2",
            stages=(
                MiproStageTemplate(
                    stage_id="stage_1_solve",
                    stage_name="Math problem solving",
                    modules=(
                        MiproModuleTemplate(module_id="stage1_system", instruction_candidates=(_GSM8K_DEFAULT_SYSTEM,)),
                        MiproModuleTemplate(module_id="stage1_user", instruction_candidates=(_GSM8K_DEFAULT_USER,)),
                    ),
                ),
            ),
        )
        adapter: Any = Gsm8kMiproAdapter(_policy_config(config), max_concurrency=max_concurrency)
    elif task == "hotpotqa":
        program_template = MiproProgramTemplate(
            program_id="hotpotqa_miprov2",
            stages=(
                MiproStageTemplate(
                    stage_id="stage_1_answer",
                    stage_name="Multi-hop question answering",
                    modules=(
                        MiproModuleTemplate(module_id="stage1_system", instruction_candidates=(_HOTPOTQA_DEFAULT_SYSTEM,)),
                        MiproModuleTemplate(module_id="stage1_user", instruction_candidates=(_HOTPOTQA_DEFAULT_USER,)),
                    ),
                ),
            ),
        )
        adapter = HotpotQaMiproAdapter(_policy_config(config), max_concurrency=max_concurrency)
    else:
        from synth_service_app import (
            TASK_ID,
            app,
        )
        from synth_containers.http_client import HTTPContainerClient
        from synth_optimizers.miprov2 import (
            ContainerMiproInterceptorAdapter,
            ContainerMiproRolloutBinding,
            program_template_from_prompt_contract,
        )

        client = HTTPContainerClient.from_app(app)
        metadata_payload = asyncio.run(client.metadata())
        route_contract = _miprov2_contract_routes(metadata_payload)
        program_payload = asyncio.run(client.program())
        if not program_payload:
            program_payload = (metadata_payload.get("metadata") or {}).get("policy_prompt_contract")
        if not isinstance(program_payload, dict):
            seed = _seed_candidate()
            program_payload = {
                "version": "prompt_program.v1",
                "program_id": "banking77_2stage_miprov2",
                "pipeline_id": "banking77",
                "search_space": {"initial_candidates": _candidate_variants()},
                "stages": [
                    {
                        "stage_id": "stage_1_coarse",
                        "stage_name": "Coarse category classification",
                        "messages": [
                            {"module_id": "stage1_system", "role": "system", "content": seed["stage1_system"]},
                            {"module_id": "stage1_user", "role": "user", "content": seed["stage1_user"]},
                        ],
                    },
                    {
                        "stage_id": "stage_2_fine",
                        "stage_name": "Fine-grained intent classification",
                        "messages": [
                            {"module_id": "stage2_system", "role": "system", "content": seed["stage2_system"]},
                            {"module_id": "stage2_user", "role": "user", "content": seed["stage2_user"]},
                        ],
                    },
                ],
            }
        program_template = program_template_from_prompt_contract(program_payload)
        train_rows, heldout_rows = asyncio.run(_load_container_rows(client, config, smoke=smoke))
        if not train_rows or not heldout_rows:
            rows_route = route_contract.get("dataset_rows_route") or "/dataset/rows"
            raise RuntimeError(f"container returned no MIPRO rows via {rows_route}")
        policy_config = _policy_config(config)
        component_candidates = _candidate_variants_from_program(program_payload) or _candidate_variants()
        adapter = ContainerMiproInterceptorAdapter(
            client=client,
            binding=ContainerMiproRolloutBinding(
                task_id=TASK_ID,
                extra_request={"policy": {"config": policy_config}},
            ),
            component_candidates=component_candidates,
            max_concurrency=max_concurrency,
            run_id=native_run_id,
            program_template=program_template,
            interceptor_base_url=str(
                config.get("mipro_proxy_base_url")
                or config.get("interceptor_base_url")
                or ""
            ).strip() or None,
            direct_inference_url=str(config.get("mipro_direct_inference_url") or "").strip() or None,
            pipeline_id="banking77",
            stage_id="stage_default",
            interceptor_roles=("system",),
            policy_candidate_fields=("stage1_user", "stage2_user"),
        )
    compiled = compile_search_space(program_template)
    optimizer = asyncio.run(
        DiscreteMiproOptimizer.from_search_space(
            search_space=compiled.search_space,
            tpe_config=TpeConfig(
                n_startup_trials=int(config.get("initial_candidate_count") or 2),
            ),
            rng_seed=int(config.get("rng_seed") or 42),
        )
    )
    metric_call_count = 0
    policy_usage_totals = _empty_usage_totals()
    heldout_details_by_candidate_id: dict[str, dict[str, Any]] = {}
    heldout_details_order: list[dict[str, Any]] = []

    def _frac(score: float, n: int) -> str:
        return f"{round(score * n)}/{n}"

    _prog = {
        "start": time.time(),
        "best_train": -1.0,
        "train_evals": 0,
        "heldout_evals": 0,
        "target_rollouts": int(config.get("target_total_train_rollouts") or 9999),
        "proposer_model": str(config.get("codex_proposer_model") or config.get("proposer_model") or "?"),
        "policy_model": str(config.get("policy_model") or "?"),
        "policy_wall_s": 0.0,
        "proposer_wall_s": 0.0,
    }
    _sep = _dim("─" * 72)
    print(_sep)
    print(
        f"  {_bold('MIPROv2')}  policy={_bold(_cyan(_prog['policy_model']))}  "
        f"proposer={_bold(_cyan(_prog['proposer_model']))}  "
        f"train={len(train_rows)}  heldout={len(heldout_rows)}  "
        f"proposer_trigger={_prog['target_rollouts']} rollouts"
    )
    print(_sep, flush=True)

    async def evaluate_rows(rows: list[dict[str, Any]], candidate: Any, *, split: str) -> tuple[float, dict[str, Any]]:
        nonlocal metric_call_count
        candidate_map = _candidate_to_prompt_map(candidate)
        evaluate_kwargs: dict[str, Any] = {"capture_traces": True}
        if callable(getattr(adapter, "_prepare_interceptor_rollout", None)):
            evaluate_kwargs["candidate_id"] = str(getattr(candidate, "candidate_id", "") or "")
        batch = await adapter.evaluate(rows, candidate_map, **evaluate_kwargs)
        outputs, scores, traces, metadata = _eval_batch_payload(batch)
        metric_call_count += len(rows)
        _add_usage_totals(policy_usage_totals, dict(metadata.get("usage") or {}))
        _achieve_acc: dict[str, list[float]] = {}
        for _tr in traces:
            _tr_dict = _tr if isinstance(_tr, dict) else {}
            _ri = _tr_dict.get("reward_info") or {}
            _tr_achieve = (_ri.get("details") or {}).get("achievements") or {}
            for _ak, _av in _tr_achieve.items():
                _achieve_acc.setdefault(str(_ak), []).append(float(_av))
        details: dict[str, Any] = {
            "candidate_id": str(getattr(candidate, "candidate_id", "") or ""),
            "split": split,
            "scores": scores,
            "outputs": outputs,
            "traces": traces,
            "selected_instructions": dict(getattr(candidate, "selected_instructions", {}) or {}),
        }
        if _achieve_acc:
            details["achievements"] = {k: sum(v) / len(v) for k, v in _achieve_acc.items()}
        if split == "heldout":
            candidate_id = str(getattr(candidate, "candidate_id", "") or "")
            if candidate_id:
                heldout_details_by_candidate_id[candidate_id] = dict(details)
            heldout_details_order.append(dict(details))
        return (sum(scores) / len(scores) if scores else 0.0, details)

    _MODULE_ABBREV = {
        "stage1_system": "s1sys",
        "stage1_user": "s1usr",
        "stage2_system": "s2sys",
        "stage2_user": "s2usr",
    }
    _seen_option_ids: set[str] = set()  # "module_id:option_id" pairs already displayed in proposer banner
    _seen_eval_oids: set[str] = set()   # option_ids already shown in an eval-time diff
    _patch_annotations: dict[str, str] = {}  # option_id → proposer annotation
    _tpe_history: list[dict[str, Any]] = []  # {trial, snap: {mid → {oid → mean}}}
    _cid_to_config: dict[str, dict[str, str]] = {}  # candidate_id → {mid → oid}
    # Populated from baseline compiled space + proposer event patches (working_space is a clone)
    _option_text_cache: dict[str, str] = {
        f"{comp_key.replace('module:', '').replace(':instruction', '')}:{oid}": text
        for comp_key, lookup in compiled.instruction_lookup.items()
        for oid, text in lookup.items()
    }

    def _format_option_ids(candidate: Any) -> str:
        base_ids: dict[str, str] = dict(getattr(candidate, "selected_instruction_base_option_ids", {}) or {})
        if not base_ids:
            return ""
        parts = " ".join(
            f"{_MODULE_ABBREV.get(mid, mid[:4])}={oid}"
            for mid, oid in sorted(base_ids.items())
        )
        return f"  [{parts}]"

    def _print_tpe_beliefs(candidate: Any, current_score: float) -> dict[str, dict[str, float]] | None:
        try:
            before_obs = list(optimizer.tpe.observations)
            base_ids: dict[str, str] = dict(getattr(candidate, "selected_instruction_base_option_ids", {}) or {})
            current_config = {f"module:{mid}:instruction": oid for mid, oid in base_ids.items()}

            def _build(obs_list: list[Any]) -> dict[str, dict[str, list[float]]]:
                c: dict[str, dict[str, list[float]]] = {}
                for obs in obs_list:
                    for k, v in obs.config.items():
                        c.setdefault(k, {}).setdefault(v, []).append(obs.score)
                return c

            before_comp = _build(before_obs)
            after_obs = before_obs + (
                [type("_O", (), {"config": current_config, "score": current_score})()] if current_config else []
            )
            after_comp = _build(after_obs)
            if not after_comp:
                return None

            def _mu(sc: list[float]) -> float:
                return sum(sc) / len(sc)

            def _ranked(opts: dict[str, list[float]]) -> dict[str, int]:
                return {oid: i + 1 for i, (oid, _) in enumerate(sorted(opts.items(), key=lambda x: _mu(x[1]), reverse=True))}

            parts = []
            any_rank_change = False
            for comp_key in sorted(after_comp):
                mod_id = comp_key.replace("module:", "").replace(":instruction", "")
                short = _MODULE_ABBREV.get(mod_id, mod_id[:4])
                before_opts = before_comp.get(comp_key, {})
                after_opts = after_comp[comp_key]
                multi = len(after_opts) > 1
                after_rank = _ranked(after_opts) if multi else {}
                before_rank = _ranked(before_opts) if (multi and before_opts) else {}
                opt_parts = []
                for oid in sorted(after_opts):
                    amu = _mu(after_opts[oid])
                    ar = after_rank.get(oid)
                    if oid not in before_opts:
                        rank_str = f",#{ar}" if ar else ""
                        opt_parts.append(f"{oid}({_cyan('new→' + f'{amu:.3f}')}{rank_str})")
                    else:
                        bmu = _mu(before_opts[oid])
                        br = before_rank.get(oid)
                        delta = amu - bmu
                        if abs(delta) > 1e-6:
                            mu_str = _green(f"{bmu:.3f}→{amu:.3f}") if delta > 0 else _red(f"{bmu:.3f}→{amu:.3f}")
                        else:
                            mu_str = f"{amu:.3f}"
                        if multi and ar is not None:
                            if br is not None and br != ar:
                                any_rank_change = True
                                arrow = _green("↑") if ar < br else _red("↓")
                                rank_str = f",#{br}→#{ar}{arrow}"
                            else:
                                rank_str = f",#{ar}"
                        else:
                            rank_str = ""
                        opt_parts.append(f"{oid}({mu_str}{rank_str})")
                parts.append(f"{short}: {' '.join(opt_parts)}")

            rank_suffix = f"  {_byellow('↕')}" if any_rank_change else ""

            # Explore/exploit mode display
            cfg = optimizer.tpe.config
            def _split(n: int) -> tuple[str, int, int]:
                if n < cfg.n_startup_trials:
                    return "random", 0, 0
                ng = max(1, math.ceil(n * cfg.gamma))
                if ng >= n:
                    ng = n - 1
                return "tpe", ng, n - ng

            bm, bg, bb = _split(len(before_obs))
            am, ag, ab = _split(len(after_obs))
            if bm == "random" and am == "random":
                mode_suffix = f"  {_dim(_yellow(f'[random {len(after_obs)}/{cfg.n_startup_trials}]'))}"
            elif bm == "random" and am == "tpe":
                mode_suffix = f"  {_byellow(f'[random→tpe g={ag}/b={ab} ε={cfg.epsilon}]')}"
            elif bg != ag:
                mode_suffix = f"  {_yellow(f'[tpe g={bg}→{ag}/b={bb}→{ab}]')}"
            else:
                mode_suffix = f"  {_dim(_cyan(f'[tpe g={ag}/b={ab}]'))}"

            print(f"           {_dim('tpe')}  {'   '.join(parts)}{rank_suffix}{mode_suffix}", flush=True)
            return {
                comp_key.replace("module:", "").replace(":instruction", ""): {
                    oid: _mu(scores) for oid, scores in opts.items()
                }
                for comp_key, opts in after_comp.items()
            }
        except Exception:
            return None

    def _print_first_seen_options(candidate: Any) -> None:
        """Show a full diff for each option appearing in eval for the first time."""
        base_ids: dict[str, str] = dict(getattr(candidate, "selected_instruction_base_option_ids", {}) or {})
        _stages = compiled.program_template.stages
        _multi_stage = len(_stages) > 1 or (len(_stages) == 1 and _stages[0].stage_id != "stage_0")
        _mod_to_stage: dict[str, str] = {m.module_id: s.stage_id for s in _stages for m in s.modules}
        _W = 110

        def _fmt(text: str) -> str:
            flat = text.replace("\n", " ↵ ")
            return flat[:_W] + ("…" if len(flat) > _W else "")

        for mid, oid in sorted(base_ids.items()):
            if oid == "i0":
                continue
            if oid in _seen_eval_oids:
                continue
            _seen_eval_oids.add(oid)
            new_text = _option_text_cache.get(f"{mid}:{oid}", "")
            if not new_text:
                continue
            short = _MODULE_ABBREV.get(mid, mid[:4])
            sid = _mod_to_stage.get(mid, "")
            stage_obj = next((s for s in _stages if s.stage_id == sid), None) if sid else None
            stage_label = stage_obj.stage_name if (stage_obj and stage_obj.stage_name) else ""
            prefix = _bold(_cyan(f"[{stage_label}] ")) if stage_label else ""
            old_text = _option_text_cache.get(f"{mid}:i0", "")
            print(f"           {prefix}{_cyan(short)} {_dim(oid)}", flush=True)
            if old_text:
                print(f"             {_red('---')} {_dim(_fmt(old_text))}", flush=True)
            print(f"             {_green('+++')} {_green(_fmt(new_text))}", flush=True)
            annotation = _patch_annotations.get(oid, "")
            if annotation:
                print(f"             {_dim('↳')} {_yellow(annotation[:140])}", flush=True)

    async def evaluate_train(candidate: Any) -> tuple[float, dict[str, Any]]:
        _eval_num = _prog["train_evals"] + 1
        _prog["train_evals"] = _eval_num
        _t0 = time.time()
        result = await evaluate_rows(train_rows, candidate, split="train")
        _prog["policy_wall_s"] += time.time() - _t0
        score = result[0]
        elapsed = time.time() - _prog["start"]
        opts_str = _format_option_ids(candidate)
        is_best = score > _prog["best_train"]
        if is_best:
            _prog["best_train"] = score
        _nt = len(train_rows)
        _score_frac = _frac(score, _nt)
        _best_frac = _frac(_prog["best_train"], _nt)
        score_str = _green(f"{_score_frac} {score:.3f}") if is_best else f"{_score_frac} {score:.3f}"
        marker = f"  {_bgreen('★ new best')}" if is_best else ""
        _train_achievements = result[1].get("achievements") or {}
        _achieve_parts = [
            f"{_ak}={_frac(_av, _nt)} {_av:.3f}"
            for _ak, _av in _train_achievements.items()
        ]
        _achieve_str = ("  " + "  ".join(_achieve_parts)) if _achieve_parts else ""
        print(
            f"  {_dim(f'{elapsed:6.1f}s')}  {_bold('train')} #{_eval_num:>3}  {_dim(f'N={_nt}')}  {opts_str.strip()}"
            f"  score={score_str}{_achieve_str}  best={_best_frac} {_prog['best_train']:.3f}"
            f"  rollouts={metric_call_count}{marker}",
            flush=True,
        )
        _snap = _print_tpe_beliefs(candidate, score)
        if _snap:
            _tpe_history.append({"trial": _eval_num, "snap": _snap})
        _cid = str(getattr(candidate, "candidate_id", "") or "")
        if _cid:
            _cid_to_config[_cid] = dict(getattr(candidate, "selected_instruction_base_option_ids", {}) or {})
        _print_first_seen_options(candidate)
        return result

    async def evaluate_heldout(candidate: Any) -> tuple[float, dict[str, Any]]:
        _prog["heldout_evals"] += 1
        _t0 = time.time()
        result = await evaluate_rows(heldout_rows, candidate, split="heldout")
        _prog["policy_wall_s"] += time.time() - _t0
        score = result[0]
        elapsed = time.time() - _prog["start"]
        _nh = len(heldout_rows)
        cid = str(getattr(candidate, "candidate_id", "") or "")[-12:]
        print(
            f"  {_dim(f'{elapsed:6.1f}s')}  {_cyan('heldout')} #{_prog['heldout_evals']:>2}  {_dim(f'N={_nh}')}  {_dim(cid)}"
            f"  score={_cyan(f'{_frac(score, _nh)} {score:.3f}')}",
            flush=True,
        )
        return result

    _proposer_token_totals = {"prompt": 0, "completion": 0, "total": 0}
    _dashes = _dim("──")

    # Proposer live-status state (in-place overwriting status line, go-explore style)
    _pst: dict[str, Any] = {
        "started_at": None,
        "status": "starting",
        "active_turn_id": "",
        "tokens": 0,
        "raw_events": 0,
        "manifest_status": "missing",
        "manifest_bytes": None,
        "last_detail": "",
        "counts": {"r": 0, "tool": 0, "cmd": 0, "file": 0, "msg": 0},
        "line_active": False,
    }

    def _pst_elapsed() -> float:
        t0 = _pst["started_at"]
        return (time.time() - t0) if t0 else 0.0

    def _status_line(detail: str) -> None:
        el = _pst_elapsed()
        sp = "|/-\\"[int(el * 4) % 4]
        c = _pst["counts"]
        mb = _pst["manifest_bytes"]
        msize = f"{mb}B" if mb is not None else "-"
        turn = _pst["active_turn_id"][-8:] if _pst["active_turn_id"] else "-"
        text = (
            f"proposer {sp} {el:5.1f}s | "
            f"state={_pst['status']} turn={turn} events={_pst['raw_events']} "
            f"tok={_pst['tokens']} manifest={_pst['manifest_status']}:{msize} | "
            f"r={c['r']} tool={c['tool']} cmd={c['cmd']} file={c['file']} msg={c['msg']} | "
            f"{detail[:70]}"
        )
        line = f"  {_dim(text[:180])}"
        if _USE_COLOR:
            print("\r\033[2K" + line, end="", flush=True)
        else:
            print(line, flush=True)
        _pst["line_active"] = True

    def _clear_status_line() -> None:
        if _pst["line_active"] and _USE_COLOR:
            print("\r\033[2K", end="", flush=True)
        elif _pst["line_active"]:
            print(flush=True)
        _pst["line_active"] = False

    def _on_proposer_event(event: dict[str, Any]) -> None:
        elapsed = time.time() - _prog["start"]
        kind = event.get("event")
        round_idx = event.get("round_idx", "?")

        if kind == "codex_workspace_stream_event":
            item_type = str(event.get("item_type") or "")
            if item_type == "command_execution":
                _pst["counts"]["cmd"] += 1
                detail = f"cmd {str(event.get('command') or '')[:60]}"
            elif item_type == "file_change":
                _pst["counts"]["file"] += 1
                detail = f"file {str(event.get('path') or '')[:60]}"
            elif item_type == "agent_message":
                _pst["counts"]["msg"] += 1
                detail = f"msg {str(event.get('text_preview') or '')[:60]}"
            elif item_type == "mcp_tool_call":
                _pst["counts"]["tool"] += 1
                _tname = str(event.get("tool_name") or "")
                _tstatus = str(event.get("status") or "")
                _clear_status_line()
                _status_suffix = f"  {_dim('[' + _tstatus + ']')}" if _tstatus else ""
                print(
                    f"           {_dim('tool')}  {_cyan(_tname)}{_status_suffix}",
                    flush=True,
                )
                _pst["last_detail"] = f"tool {_tname}"
                return
            else:
                detail = item_type or "activity"
            _pst["last_detail"] = detail
            _status_line(detail)
            return

        if kind == "codex_workspace_progress_poll":
            _pst["status"] = str(event.get("session_status") or _pst["status"])
            _pst["active_turn_id"] = str(event.get("active_turn_id") or _pst["active_turn_id"])
            _pst["raw_events"] = int(event.get("raw_event_count") or _pst["raw_events"])
            _pst["manifest_status"] = str(event.get("manifest_status") or _pst["manifest_status"])
            mb = event.get("manifest_bytes")
            if mb is not None:
                _pst["manifest_bytes"] = int(mb)
            usage = dict(event.get("usage") or {})
            _pst["tokens"] = int(usage.get("total_tokens") or _pst["tokens"])
            _status_line(_pst["last_detail"] or "running")
            return

        if kind == "codex_workspace_patch_annotations":
            _patch_annotations.update(dict(event.get("annotations") or {}))
            return

        if kind == "codex_workspace_materialized":
            counts = dict(event.get("counts") or {})
            parts = [
                f"rollouts={counts.get('rollouts', '?')}",
                f"candidates={counts.get('candidates', '?')}",
                f"deltas={counts.get('delta_digests', '?')}",
                f"verdicts={counts.get('verdict_digests', '?')}",
                f"train_rows={counts.get('sampled_train_rows', '?')}",
            ]
            print(f"           {_dim('workspace ready  '  + '  '.join(parts))}", flush=True)
            return

        if kind == "start":
            _pst["started_at"] = time.time()
            _pst["status"] = "starting"
            _pst["active_turn_id"] = ""
            _pst["tokens"] = 0
            _pst["raw_events"] = 0
            _pst["manifest_status"] = "missing"
            _pst["manifest_bytes"] = None
            _pst["last_detail"] = ""
            _pst["counts"] = {"r": 0, "tool": 0, "cmd": 0, "file": 0, "msg": 0}
            _pst["line_active"] = False
            best = event.get("best_train_score")
            score_str = f"best={best:.3f}" if best is not None else "best=—"
            ws = str(event.get("workspace_root") or "")
            print(
                f"  {_dim(f'{elapsed:6.1f}s')}  {_dashes} {_bold(_magenta(f'codex proposer round {round_idx} starting'))}"
                f"  {_dim('model=' + str(event.get('model', '?')))} {_dashes}",
                flush=True,
            )
            print(
                f"           {_dim(score_str + '  obs=' + str(event.get('observation_count', 0)))}"
                + (f"  {_dim('workspace: ' + ws)}" if ws else ""),
                flush=True,
            )
        elif kind == "complete":
            _clear_status_line()
            n = event.get("n_new_candidates", 0)
            secs = event.get("elapsed_s", 0.0)
            prop_prompt = event.get("prompt_tokens", 0)
            prop_comp = event.get("completion_tokens", 0)
            prop_total = event.get("total_tokens", 0)
            _proposer_token_totals["prompt"] += prop_prompt
            _proposer_token_totals["completion"] += prop_comp
            _proposer_token_totals["total"] += prop_total
            _prog["proposer_wall_s"] += float(secs)
            tok_str = (
                f"  proposer_tokens={prop_total} (in={prop_prompt} / out={prop_comp})"
                if prop_total > 0 else ""
            )
            patches = event.get("patches", [])
            # Group patches by stage (if multi-stage) or by module (single-stage)
            _stages = compiled.program_template.stages
            _multi_stage = len(_stages) > 1 or (len(_stages) == 1 and _stages[0].stage_id != "stage_0")
            # module_id → stage_id map for display grouping
            _mod_to_stage: dict[str, str] = {
                m.module_id: s.stage_id for s in _stages for m in s.modules
            }
            # summary: per-module new option IDs, grouped by stage when multi-stage
            # by_mod_new: module_id → [new option_ids]
            by_mod_new: dict[str, list[str]] = {}
            for p in patches:
                mid = str(p.get("module_id", "?"))
                oid = str(p.get("option_id", "?"))
                text = str(p.get("instruction_text", ""))
                by_mod_new.setdefault(mid, []).append(oid)
                if mid and oid and text:
                    _option_text_cache[f"{mid}:{oid}"] = text
            if _multi_stage:
                # group modules under their stage label
                by_stage_summary: dict[str, list[str]] = {}
                for mid, oids in by_mod_new.items():
                    sid = _mod_to_stage.get(mid, mid)
                    short = _MODULE_ABBREV.get(mid, mid[:4])
                    for oid in oids:
                        by_stage_summary.setdefault(sid, []).append(f"{short}={oid}")
                module_summary = "  ".join(
                    f"{sid}: +{' +'.join(parts)}" for sid, parts in by_stage_summary.items()
                )
            else:
                module_summary = "  ".join(
                    f"{_MODULE_ABBREV.get(k, k[:4])}: +{' +'.join(ids)}"
                    for k, ids in by_mod_new.items()
                )
            _tok_str_display = _dim(tok_str) if tok_str else ""
            print(
                _dim("  ") + f"{elapsed:6.1f}s  "
                + _bold(_cyan(f"── codex proposer round {round_idx} done"))
                + f"  {secs:.1f}s  "
                + _bold(f"{n} new transforms")
                + f"  [{module_summary}]"
                + _tok_str_display
                + _bold(_cyan("  ──")),
                flush=True,
            )
            # Print transforms as diffs against the baseline (i0) instruction
            _W = 110

            def _fmt(text: str) -> str:
                flat = text.replace("\n", " ↵ ")
                return flat[:_W] + ("…" if len(flat) > _W else "")

            def _print_transform_diff(mid: str, oid: str, new_text: str, stage_label: str = "", annotation: str = "") -> None:
                short = _MODULE_ABBREV.get(mid, mid[:4])
                prefix = _bold(_cyan(f"[{stage_label}] ")) if stage_label else ""
                old_text = _option_text_cache.get(f"{mid}:i0", "")
                print(f"           {prefix}{_cyan(short)} {_dim(oid)}", flush=True)
                if old_text:
                    print(f"             {_red('---')} {_dim(_fmt(old_text))}", flush=True)
                print(f"             {_green('+++')} {_green(_fmt(new_text))}", flush=True)
                if annotation:
                    print(f"             {_dim('↳')} {_yellow(annotation[:140])}", flush=True)

            if _multi_stage:
                by_stage_patches: dict[str, list[dict[str, Any]]] = {}
                for p in patches:
                    mid = str(p.get("module_id", "?"))
                    sid = _mod_to_stage.get(mid, mid)
                    by_stage_patches.setdefault(sid, []).append(p)
                for sid, stage_patches in by_stage_patches.items():
                    stage_obj = next((s for s in _stages if s.stage_id == sid), None)
                    stage_label = stage_obj.stage_name if (stage_obj and stage_obj.stage_name) else sid
                    for p in stage_patches:
                        _oid = str(p.get("option_id", "?"))
                        _print_transform_diff(
                            str(p.get("module_id", "?")),
                            _oid,
                            str(p.get("instruction_text", "")),
                            stage_label=stage_label,
                            annotation=_patch_annotations.get(_oid, ""),
                        )
            else:
                for p in patches:
                    _oid = str(p.get("option_id", "?"))
                    _print_transform_diff(
                        str(p.get("module_id", "?")),
                        _oid,
                        str(p.get("instruction_text", "")),
                        annotation=_patch_annotations.get(_oid, ""),
                    )
            # Running totals
            pol = policy_usage_totals
            pol_in = int(pol.get("prompt_tokens") or 0)
            pol_out = int(pol.get("completion_tokens") or 0)
            pol_total = pol_in + pol_out
            prop_run_total = _proposer_token_totals["total"]
            prop_run_str = (
                f"  proposer_total={prop_run_total}"
                if prop_run_total > 0 else ""
            )
            print(
                _dim(
                    f"           policy_tokens={pol_total}"
                    f" (in={pol_in} / out={pol_out}){prop_run_str}"
                ),
                flush=True,
            )

    resolved_proposer = resolve_proposer_backend(
        config,
        smoke=smoke,
        output_dir=output_dir,
        interactive_proposer=interactive_proposer,
        interactive_session_root=interactive_session_root,
        interactive_resume_session_id=interactive_resume_session_id,
    )
    proposer_backend = resolved_proposer.backend
    proposer_agent = resolved_proposer.agent
    proposer_control = resolved_proposer.control
    codex_cfg = resolved_proposer.codex_config
    configured_session_root = resolved_proposer.interactive_session_root
    interactive_resume_id = resolved_proposer.interactive_resume_session_id
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
                proposer_control=proposer_control,
                interactive_session_root=str(configured_session_root),
                interactive_resume_session_id=interactive_resume_id or None,
                proposer_config=MiproOpenEnvProposerConfig(
                    max_turns=8 if smoke else int(config.get("proposer_max_turns") or 32),
                    max_noop_turns=4 if smoke else int(config.get("proposer_max_noop_turns") or 12),
                    max_patch_actions_per_session=2 if smoke else int(config.get("proposer_max_patch_actions") or 8),
                    max_instruction_patches=2 if smoke else int(config.get("proposer_max_patch_actions") or 8),
                    max_demo_patches=0,
                    archive_root=str(output_dir / "artifacts" / "proposer_archives"),
                ),
                proposer_variant=MiproOpenEnvProposerVariant(
                    system_prompt_append=PROPOSER_SEGMENT_GUIDANCE,
                ),
            ),
            run_id=native_run_id,
            ledger_path=str(ledger_path),
            resume=bool(interactive_resume_id),
            codex_config=codex_cfg,
            on_proposer_event=_on_proposer_event if proposer_backend == "codex_workspace" else None,
        )
    )

    # ── Top-N heldout evaluation ───────────────────────────────────────────────
    _top_n = 4
    _N_BOOT = 10_000
    _final_space = outcome.final_compiled_space
    _heldout_table: list[dict[str, Any]] = []

    def _boot_ci(scores: list[float]) -> tuple[float, float]:
        n = len(scores)
        if n == 0:
            return (0.0, 0.0)
        rng = random.Random(42)
        means = sorted(sum(rng.choices(scores, k=n)) / n for _ in range(_N_BOOT))
        lo = means[int(0.025 * _N_BOOT)]
        hi = means[int(0.975 * _N_BOOT)]
        return lo, hi

    def _boot_lift_ci(cand_scores: list[float], base_scores: list[float]) -> tuple[float, float, bool]:
        n = len(cand_scores)
        if n == 0 or len(base_scores) != n:
            return (0.0, 0.0, False)
        diffs = [c - b for c, b in zip(cand_scores, base_scores)]
        rng = random.Random(42)
        means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(_N_BOOT))
        lo = means[int(0.025 * _N_BOOT)]
        hi = means[int(0.975 * _N_BOOT)]
        return lo, hi, lo > 0

    if _final_space is not None and not smoke:
        _score_by_cid: dict[str, float] = {}
        _config_by_cid: dict[str, dict[str, str]] = {}
        for obs in outcome.train_observations:
            cid = str(obs.candidate_id or "")
            if not cid:
                continue
            if cid not in _score_by_cid or obs.score > _score_by_cid[cid]:
                _score_by_cid[cid] = float(obs.score)
                _config_by_cid[cid] = dict(obs.config)
        _baseline_cid = str(
            (outcome.train_observations[0].candidate_id if outcome.train_observations else None) or ""
        )
        _best_cid = str((outcome.best_train_candidate.candidate_id if outcome.best_train_candidate else None) or "")
        _ranked = sorted(_score_by_cid.items(), key=lambda x: x[1], reverse=True)
        _top_cids = [cid for cid, _ in _ranked[:_top_n]]
        if _baseline_cid and _baseline_cid not in _top_cids:
            _top_cids.append(_baseline_cid)
        _N_h = len(heldout_rows)
        print(_sep)
        print(
            f"  {_bold('heldout sweep')}  N={_N_h}  {len(_top_cids)} candidates"
            f"  {_dim(f'bootstrap 95% CI  {_N_BOOT:,} samples')}",
            flush=True,
        )
        for _cid in _top_cids:
            _train_score = _score_by_cid.get(_cid, 0.0)
            _cfg = _config_by_cid.get(_cid, {})
            try:
                _candidate = decode_config(_final_space, _cfg)
            except Exception:
                continue
            _elapsed_h = time.time() - _prog["start"]
            _is_baseline = _cid == _baseline_cid
            _is_best = _cid == _best_cid
            _tag = "baseline" if _is_baseline else ("best" if _is_best else "")
            _tag_str = f"  {_dim('[' + _tag + ']')}" if _tag else ""
            print(
                f"  {_dim(f'{_elapsed_h:6.1f}s')}  heldout  {_dim(f'N={_N_h}')}  {_dim(_cid[-12:])}  train={_train_score:.3f}{_tag_str}",
                flush=True,
            )
            _h_t0 = time.time()
            _h_score, _h_details = asyncio.run(evaluate_rows(heldout_rows, _candidate, split="heldout"))
            _prog["policy_wall_s"] += time.time() - _h_t0
            _row_scores = [float(s) for s in (_h_details.get("scores") or [])]
            heldout_details_by_candidate_id[_cid] = dict(_h_details)
            heldout_details_order.append(dict(_h_details))
            _ci_lo, _ci_hi = _boot_ci(_row_scores)
            _heldout_table.append({
                "cid": _cid,
                "train": _train_score,
                "heldout": _h_score,
                "scores": _row_scores,
                "outputs": list(_h_details.get("outputs") or []),
                "achievements": dict(_h_details.get("achievements") or {}),
                "eval_idx": len(_heldout_table),
                "ci_lo": _ci_lo,
                "ci_hi": _ci_hi,
                "is_baseline": _is_baseline,
                "is_best": _is_best,
            })
            print(
                f"           heldout={_cyan(f'{_frac(_h_score, _N_h)} {_h_score:.3f}')}  {_dim(f'95% CI [{_ci_lo:.3f}, {_ci_hi:.3f}]')}",
                flush=True,
            )

        if _heldout_table:
            _base_row = next((r for r in _heldout_table if r["is_baseline"]), None)
            _base_scores = _base_row["scores"] if _base_row else []
            _base_h = _base_row["heldout"] if _base_row else None
            # Compute paired lift CIs for all candidates
            for _row in _heldout_table:
                if _row["is_baseline"]:
                    _row["lift_lo"] = _row["lift_hi"] = 0.0
                    _row["sig"] = False
                else:
                    _lo, _hi, _sig = _boot_lift_ci(_row["scores"], _base_scores)
                    _row["lift_lo"] = _lo
                    _row["lift_hi"] = _hi
                    _row["sig"] = _sig
            print(_sep)
            _N_str = f"N={_N_h}"
            _col_tr = max(len(_frac(s, len(train_rows))) + 6 for s in _score_by_cid.values()) if _score_by_cid else 12
            _col_h = max(len(_frac(r["heldout"], _N_h)) + 6 for r in _heldout_table) if _heldout_table else 12
            print(
                _bold(f"  {'candidate':>12}  {'train':>{_col_tr}}  {'heldout':>{_col_h}}  {'95% CI':^13}  {'lift':>6}  {'lift 95% CI':^15}  sig  note"),
                flush=True,
            )
            for _row in sorted(_heldout_table, key=lambda r: r["heldout"], reverse=True):
                _lift = (_row["heldout"] - _base_h) if _base_h is not None else 0.0
                _sig = _row.get("sig", False)
                _sig_str = _bgreen("✓") if _sig else _dim("·")
                _note = ("baseline" if _row["is_baseline"] else "") + ("  ★" if _row["is_best"] else "")
                _ci_str = f"[{_row['ci_lo']:.3f} {_row['ci_hi']:.3f}]"
                _lift_str = f"{_lift:+.3f}" if _base_h is not None else "   —"
                if _row["is_baseline"]:
                    _lift_ci_str = _dim("     —     ")
                else:
                    _lift_ci_str = f"[{_row['lift_lo']:+.3f} {_row['lift_hi']:+.3f}]"
                    _lift_ci_str = _green(_lift_ci_str) if _sig else _dim(_lift_ci_str)
                _tr_frac = _frac(_row["train"], len(train_rows))
                _h_frac = _frac(_row["heldout"], _N_h)
                _tr_str = f"{_tr_frac} {_row['train']:.3f}"
                _h_raw = f"{_h_frac} {_row['heldout']:.3f}"
                _h_col = _bgreen(_h_raw) if _sig else (_dim(_h_raw) if _row["is_baseline"] else _h_raw)
                _lift_col = _green(_lift_str) if _lift > 0 else (_red(_lift_str) if _lift < 0 else _dim(_lift_str))
                print(
                    f"  {_dim(_row['cid'][-12:]):>12}  {_tr_str:>{_col_tr}}  {_h_col:>{_col_h}}  {_dim(_ci_str):^13}  {_lift_col:>6}  {_lift_ci_str:^15}  {_sig_str}  {_dim(_note)}",
                    flush=True,
                )
            print(_dim(f"  {_N_str}  sig = paired bootstrap 95% CI on lift excludes 0"), flush=True)

            # ── Achievement breakdown ──────────────────────────────────────────
            _achieve_base_row = next((r for r in _heldout_table if r["is_baseline"]), None)
            _achieve_best_row = next((r for r in _heldout_table if r["is_best"]), None)
            if _achieve_best_row is None:
                _achieve_best_row = max(
                    (r for r in _heldout_table if not r["is_baseline"]),
                    key=lambda r: r["heldout"], default=None,
                )
            _achieve_keys: list[str] = []
            for _ar in _heldout_table:
                for _ak in (_ar.get("achievements") or {}).keys():
                    if _ak not in _achieve_keys:
                        _achieve_keys.append(_ak)
            if _achieve_keys and (_achieve_base_row or _achieve_best_row):
                print(_sep)
                _col_a = max(len(k) for k in _achieve_keys)
                print(
                    _bold(f"  {'achievement':<{_col_a}}  {'base':>12}  {'best':>12}  {'Δ':>6}"),
                    flush=True,
                )
                for _ak in _achieve_keys:
                    _a_base = (_achieve_base_row.get("achievements") or {}).get(_ak) if _achieve_base_row else None
                    _a_best = (_achieve_best_row.get("achievements") or {}).get(_ak) if _achieve_best_row else None
                    _ab_str = f"{_frac(_a_base, _N_h)} {_a_base:.3f}" if _a_base is not None else "—"
                    _ax_str = f"{_frac(_a_best, _N_h)} {_a_best:.3f}" if _a_best is not None else "—"
                    if _a_base is not None and _a_best is not None:
                        _ad = _a_best - _a_base
                        _ad_str = _green(f"{_ad:+.3f}") if _ad > 0.005 else (_red(f"{_ad:+.3f}") if _ad < -0.005 else _dim(f"{_ad:+.3f}"))
                        _a_star = f"  {_bgreen('★')}" if _ad > 0.005 else ""
                    else:
                        _ad_str = _dim("     —")
                        _a_star = ""
                    print(
                        f"  {_ak:<{_col_a}}  {_ab_str:>12}  {_ax_str:>12}  {_ad_str}{_a_star}",
                        flush=True,
                    )
                _ab_cid = _achieve_base_row["cid"][-8:] if _achieve_base_row else "—"
                _ax_cid = _achieve_best_row["cid"][-8:] if _achieve_best_row else "—"
                print(_dim(f"  base={_ab_cid}  best={_ax_cid}"), flush=True)

            # ── Candidate timeline: train run order vs heldout score ───────────
            _cid_to_train_order: dict[str, int] = {}
            for _oi, _obs in enumerate(outcome.train_observations):
                _ocid = str(_obs.candidate_id or "")
                if _ocid and _ocid not in _cid_to_train_order:
                    _cid_to_train_order[_ocid] = _oi
            _chart_pts = [
                (_cid_to_train_order.get(r["cid"], 0), r["heldout"], r["is_baseline"], r["is_best"])
                for r in _heldout_table
            ]
            if len(_chart_pts) > 1:
                _CW, _CH = 58, 10
                _x_vals = [p[0] for p in _chart_pts]
                _y_vals = [p[1] for p in _chart_pts]
                _xmin, _xmax = min(_x_vals), max(_x_vals)
                _ymin = max(0.0, min(_y_vals) - 0.04)
                _ymax = min(1.0, max(_y_vals) + 0.04)
                _xr = max(1, _xmax - _xmin)
                _yr = max(0.001, _ymax - _ymin)
                _cgrid = [[" "] * _CW for _ in range(_CH)]
                for _px, _py, _pbase, _pbest in _chart_pts:
                    _col = int(round((_px - _xmin) / _xr * (_CW - 1)))
                    _row = _CH - 1 - int(round((_py - _ymin) / _yr * (_CH - 1)))
                    _col, _row = max(0, min(_CW - 1, _col)), max(0, min(_CH - 1, _row))
                    _cgrid[_row][_col] = "○" if _pbase else ("★" if _pbest else "●")
                print(_sep)
                print(
                    f"  {_bold('heldout score vs train run order')}  {_dim('○=baseline  ●=candidate  ★=best train')}",
                    flush=True,
                )
                for _ri in range(_CH):
                    _yv = _ymax - (_ri / max(_CH - 1, 1)) * _yr
                    print(f"  {_yv:.3f} │{''.join(_cgrid[_ri])}", flush=True)
                print(f"  {'':5} └{'─' * _CW}", flush=True)
                _xlbl = [" "] * _CW
                for _px, _, _, _ in _chart_pts:
                    _col = int(round((_px - _xmin) / _xr * (_CW - 1)))
                    _lbl = str(_px)
                    for _li, _lc in enumerate(_lbl):
                        if _col + _li < _CW:
                            _xlbl[_col + _li] = _lc
                print(f"  {'':5}  {''.join(_xlbl)}  {_dim('← train rollout #')}", flush=True)

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
                task_id=task,
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
        "run_status": outcome.run_status,
        "pending_interactive_session": outcome.pending_interactive_session,
        "consumed_interactive_session": outcome.consumed_interactive_session,
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
    _total_elapsed = time.time() - _prog["start"]
    _best_cid = str((best_candidate or {}).get("candidate_id", "") or "")[-12:]
    _baseline = payload.get("baseline_train_score", 0.0)
    _best = payload.get("best_train_score", 0.0)
    _heldout_baseline = payload.get("baseline_heldout_score")
    _heldout_best = payload.get("best_heldout_score")
    _pol = policy_usage_totals
    _pol_in = int(_pol.get("prompt_tokens") or 0)
    _pol_out = int(_pol.get("completion_tokens") or 0)
    _prop_total = _proposer_token_totals["total"]
    _improved = _best > _baseline
    _star = f"  {_bgreen('★')}" if _improved else ""
    _policy_cost_usd, _ = _cost_estimate_from_config(
        config=config, policy_usage=policy_usage_totals, proposer_usage=_empty_usage_totals()
    )
    _proposer_cost_usd, _ = _cost_estimate_from_config(
        config=config, policy_usage=_empty_usage_totals(), proposer_usage=proposer_usage_totals
    )
    _policy_wall = _prog.get("policy_wall_s", 0.0)
    _proposer_wall = _prog.get("proposer_wall_s", 0.0)
    print(_sep)
    _nt_done = len(train_rows)
    _nh_done = len(heldout_rows)
    print(
        f"  {_bold('done')}  {_dim(f'{_total_elapsed:.1f}s')}  "
        f"train: {_dim(f'{_frac(_baseline, _nt_done)} {_baseline:.3f}')} → "
        f"{(_bgreen if _improved else _bold)(f'{_frac(_best, _nt_done)} {_best:.3f}')}"
        + (
            f"  heldout: {_dim(f'{_frac(_heldout_baseline, _nh_done)} {_heldout_baseline:.3f}')} → "
            f"{_cyan(f'{_frac(_heldout_best, _nh_done)} {_heldout_best:.3f}')}"
            if _heldout_best is not None else ""
        )
        + f"  {_dim(f'best={_best_cid}  rollouts={metric_call_count}')}"
        + _star
    )
    print(
        _dim(
            f"  policy tokens:   in={_pol_in:,}  out={_pol_out:,}  total={_pol_in+_pol_out:,}"
            + (f"\n  proposer tokens: total={_prop_total:,}" if _prop_total > 0 else "")
        )
    )
    print(_sep)
    print(f"  {_bold('run stats')}", flush=True)
    print(f"  {'rollouts':>16}  {metric_call_count}", flush=True)

    def _fmt_cost(c: float | None) -> str:
        return f"${c:.4f}" if c is not None else "—"

    print(f"  {'policy cost':>16}  {_fmt_cost(_policy_cost_usd)}", flush=True)
    print(f"  {'proposer cost':>16}  {_fmt_cost(_proposer_cost_usd)}", flush=True)
    print(f"  {'policy time':>16}  {_policy_wall:.1f}s", flush=True)
    print(f"  {'proposer time':>16}  {_proposer_wall:.1f}s", flush=True)
    print(_sep, flush=True)
    # ── Best candidate full instructions ──────────────────────────────────────
    _best_h_row_for_display = max(_heldout_table, key=lambda r: r["heldout"]) if _heldout_table else None
    if _best_h_row_for_display and not smoke:
        _bh_cid = _best_h_row_for_display["cid"]
        _bh_details = heldout_details_by_candidate_id.get(_bh_cid, {})
        _bh_instrs = dict(_bh_details.get("selected_instructions") or {})
        if _bh_instrs:
            def _wrap_text(text: str, first_indent: str = "    ", cont_indent: str = "    ", width: int = 96) -> list[str]:
                lines = []
                for para in text.split("\n"):
                    wrapped = textwrap.fill(para or " ", width=width, initial_indent=first_indent, subsequent_indent=cont_indent)
                    lines.extend(wrapped.splitlines())
                return lines
            _bh_score = _best_h_row_for_display["heldout"]
            print(flush=True)
            print(
                f"  {_bold('best candidate')}  {_dim(_bh_cid[-12:])}  "
                f"heldout={_cyan(f'{_bh_score:.3f}')}",
                flush=True,
            )
            _stages_disp = compiled.program_template.stages
            for _s in _stages_disp:
                _slabel = _s.stage_name or _s.stage_id
                print(f"\n  {_bold(_cyan(f'[{_slabel}]'))}", flush=True)
                for _m in _s.modules:
                    _mid = _m.module_id
                    _text = _bh_instrs.get(_mid, "")
                    if not _text:
                        continue
                    _short = _MODULE_ABBREV.get(_mid, _mid)
                    _base = _option_text_cache.get(f"{_mid}:i0", "")
                    _changed = bool(_base) and _base != _text
                    print(f"  {_bold(_short)}", flush=True)
                    if _changed:
                        for _wl in _wrap_text(_base, "    ", "    "):
                            print(f"  {_red('---')} {_dim(_wl.strip())}", flush=True)
                        for _wl in _wrap_text(_text, "    ", "    "):
                            print(f"  {_green('+++')} {_green(_wl.strip())}", flush=True)
                    else:
                        for _wl in _wrap_text(_text, "    ", "    "):
                            print(f"  {_dim(_wl)}", flush=True)
            print(flush=True)
    # ── TPE evolution table ────────────────────────────────────────────────────
    if _tpe_history and not smoke:
        print(_sep, flush=True)
        print(f"  {_bold('transform evolution')}  {_dim('TPE mean per training trial')}", flush=True)
        _ev_stages = compiled.program_template.stages
        _ev_mids = [m.module_id for s in _ev_stages for m in s.modules]
        _ev_mids_present = [mid for mid in _ev_mids if any(mid in e["snap"] for e in _tpe_history)]
        for _ev_mid in _ev_mids_present:
            _ev_short = _MODULE_ABBREV.get(_ev_mid, _ev_mid)
            # Collect all option IDs ever seen for this module, sorted
            _ev_oids: list[str] = sorted({
                oid
                for e in _tpe_history
                for oid in (e["snap"].get(_ev_mid) or {}).keys()
            })
            if not _ev_oids:
                continue
            _cw = 7  # cell width
            print(f"\n  {_bold(_cyan(_ev_short)):<14}" + "".join(f"{oid:>{_cw}}" for oid in _ev_oids), flush=True)
            for _ev_entry in _tpe_history:
                _mod_snap = _ev_entry["snap"].get(_ev_mid, {})
                row = f"  {'#' + str(_ev_entry['trial']):<14}"
                for _ev_oid in _ev_oids:
                    if _ev_oid in _mod_snap:
                        row += f"{_mod_snap[_ev_oid]:>{_cw}.3f}"
                    else:
                        row += f"{'':>{_cw}}"
                print(row, flush=True)
        # Top candidates composition
        if _heldout_table and _cid_to_config:
            _ev_sorted = sorted(_heldout_table, key=lambda r: r["heldout"], reverse=True)
            _ev_top = _ev_sorted[:4]
            print(flush=True)
            print(f"  {_bold('top candidates by heldout score')}", flush=True)
            _ev_shorts = [_MODULE_ABBREV.get(m, m) for m in _ev_mids_present]
            _tcw = 8
            print(
                f"  {'rank':<5}{'heldout':<10}{'train':<10}" +
                "".join(f"{s:>{_tcw}}" for s in _ev_shorts),
                flush=True,
            )
            for _ev_rank, _ev_row in enumerate(_ev_top, 1):
                _ev_cid = _ev_row["cid"]
                _ev_cfg = _cid_to_config.get(_ev_cid, {})
                _ev_h = _ev_row["heldout"]
                _ev_tr = _ev_row["train"]
                _ev_tag = " [baseline]" if _ev_row["is_baseline"] else ""
                line = f"  {_ev_rank:<5}{_ev_h:.3f}{'':>4}{_ev_tr:.3f}{'':>4}"
                for _ev_mid in _ev_mids_present:
                    _ev_oid = _ev_cfg.get(_ev_mid, "?")
                    _ev_star = "★" if _ev_oid not in ("i0", "?") else ""
                    line += f"{(_ev_star + _ev_oid):>{_tcw}}"
                print(line + _ev_tag, flush=True)
    print(_sep, flush=True)
    print(_dim(f"wrote native MIPROv2 OpenEnv artifacts to {live_dir}"))
    close_adapter = getattr(adapter, "aclose", None)
    if callable(close_adapter):
        asyncio.run(close_adapter())
    return 0


def execute_gepa(artifacts_dir: Path, *, config_path: Path, smoke: bool) -> int:
    _ensure_local_imports()
    _load_local_api_keys()
    from gepa import optimize
    from gepa.core.adapter import EvaluationBatch
    from synth_containers.http_client import HTTPContainerClient
    from synth_optimizers.miprov2 import ContainerMiproAdapter, ContainerMiproRolloutBinding
    from synth_service_app import app

    config = _load_config(config_path)
    train_rows, heldout_rows = _load_rows(config, smoke=smoke)
    live_dir = artifacts_dir / ("live_gepa_smoke" if smoke else "live_gepa_run")
    output_dir = live_dir / "miprov2_artifacts"
    live_dir.mkdir(parents=True, exist_ok=True)
    policy_config = _policy_config(config)
    client = HTTPContainerClient.from_app(app)
    adapter = ContainerMiproAdapter(
        client=client,
        binding=ContainerMiproRolloutBinding(
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
    parser.add_argument(
        "--interactive-proposer",
        action="store_true",
        help="Pause at proposer boundaries and expose a checkpoint-backed proposer session.",
    )
    parser.add_argument(
        "--interactive-session-root",
        type=Path,
        default=None,
        help="Directory containing interactive proposer sessions.",
    )
    parser.add_argument(
        "--interactive-resume-session-id",
        default=None,
        help="Committed interactive proposer session id to consume before continuing.",
    )
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
    return execute_live(
        artifacts_dir,
        config_path=Path(args.config),
        smoke=bool(args.smoke),
        interactive_proposer=bool(args.interactive_proposer),
        interactive_session_root=args.interactive_session_root,
        interactive_resume_session_id=args.interactive_resume_session_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())
