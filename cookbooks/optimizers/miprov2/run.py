from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
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


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "cookbook": "optimizers/miprov2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "config": str(CONFIG_PATH.relative_to(COOKBOOK_ROOT)),
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
            trace = None
            if capture_traces:
                trace = {
                    **output,
                    "query": row["text"],
                    "raw_response": raw_response,
                    "response_id": response_json.get("id") if isinstance(response_json, dict) else None,
                }
            return output, score, trace

        results = await asyncio.gather(*(evaluate_row(dict(row)) for row in batch))
        outputs = [item[0] for item in results]
        scores = [float(item[1]) for item in results]
        traces = [item[2] for item in results if item[2] is not None]
        return {
            "outputs": outputs,
            "scores": scores,
            "traces": traces,
            "metadata": {
                "candidate": dict(candidate),
                "capture_traces": bool(capture_traces),
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


def execute_live(artifacts_dir: Path, *, smoke: bool) -> int:
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
        run_phase3_loop,
    )
    from synth_optimizers.miprov2.core.run_ledger import SQLiteMiproRunLedger

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
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

    async def evaluate_rows(rows: list[dict[str, Any]], candidate: Any, *, split: str) -> tuple[float, dict[str, Any]]:
        nonlocal metric_call_count
        batch = await adapter.evaluate(rows, _candidate_to_prompt_map(candidate), capture_traces=True)
        scores = [float(score) for score in batch["scores"]]
        metric_call_count += len(rows)
        return (
            sum(scores) / len(scores) if scores else 0.0,
            {
                "split": split,
                "scores": scores,
                "outputs": batch["outputs"],
                "traces": batch["traces"],
                "selected_instructions": dict(getattr(candidate, "selected_instructions", {}) or {}),
            },
        )

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
            events = list(reversed(ledger.query_events(limit=1000)))
        finally:
            ledger.close()

    best_candidate = outcome.best_train_candidate.to_dict() if outcome.best_train_candidate is not None else None
    proposer_trace_paths = [
        _relative_to(path, live_dir) for path in outcome.proposer_trace_paths
    ]
    ledger_path_relative = _relative_to(outcome.ledger_path, live_dir)
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
        "train_observation_count": len(outcome.train_observations),
        "heldout_snapshots": [asdict(snapshot) for snapshot in outcome.heldout_snapshots],
        "run_read_model": run_read_model,
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
            "artifacts": {
                "best_candidate": "artifacts/best_candidate.json",
                "heldout_eval": "artifacts/heldout_eval.json",
                "run_summary": "artifacts/miprov2_run_summary.json",
                "result_manifest": "artifacts/result_manifest.json",
                "run_read_model": "artifacts/run_read_model.json",
                "run_events": "artifacts/run_events.jsonl",
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
            "proposer_round_count": len(outcome.proposer_sessions),
            "proposer_trace_paths": proposer_trace_paths,
            "local_key_fallback_used": bool(loaded_key_names),
            "loaded_local_key_count": len(loaded_key_names),
        },
    )
    print(f"wrote native MIPROv2 OpenEnv artifacts to {live_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or run the Banking77 MIPROv2 cookbook.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
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
    return execute_live(artifacts_dir, smoke=bool(args.smoke))


if __name__ == "__main__":
    raise SystemExit(main())
