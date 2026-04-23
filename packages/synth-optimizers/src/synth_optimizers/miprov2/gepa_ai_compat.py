"""GEPA-shaped compatibility facade for MIPROv2."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from synth_optimizers.miprov2.artifacts import (
    MiproArtifactManifest,
    MiproArtifactPaths,
    MiproRunArtifactSummary,
    write_miprov2_artifacts,
)
from synth_optimizers.miprov2.core import (
    DiscreteMiproOptimizer,
    MiproCandidateExecutionMode,
    MiproCandidateRecord,
    MiproCompatResult,
    MiproCompatRunConfig,
    MiproComponentSpec,
    MiproEvaluationBatch,
    MiproExecutionContract,
    MiproModuleTemplate,
    MiproPhase2Config,
    MiproProgramCandidate,
    MiproProgramTemplate,
    TpeConfig,
    compile_search_space,
    decode_config,
    run_train_loop,
)
from synth_optimizers.miprov2.core.run_ledger import SQLiteMiproRunLedger
from synth_optimizers.miprov2.requirements import assert_mipro_runtime_supported


class GepaCompatAdapter(Protocol):
    def evaluate(
        self,
        batch: Sequence[Any],
        candidate: Mapping[str, str],
        capture_traces: bool = False,
    ) -> MiproEvaluationBatch | dict[str, Any] | Awaitable[MiproEvaluationBatch | dict[str, Any]]:
        ...

    def make_reflective_dataset(
        self,
        candidate: Mapping[str, str],
        eval_batch: MiproEvaluationBatch,
        components_to_update: Sequence[str],
    ) -> dict[str, list[dict[str, Any]]] | Mapping[str, Sequence[Any]]:
        ...


class MiproCompatInterruptionRequested(RuntimeError):
    """Raised by benchmark harnesses to stop a compat run mid-flight."""


def _normalized_candidate_map(value: Mapping[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if not key:
            continue
        normalized[key] = str(raw_value).strip()
    if not normalized:
        raise ValueError("candidate must contain at least one non-empty component")
    return dict(sorted(normalized.items(), key=lambda item: item[0]))


def _candidate_map_from_program(candidate: MiproProgramCandidate) -> dict[str, str]:
    return dict(sorted(candidate.selected_instructions.items(), key=lambda item: item[0]))


def _extract_candidate_texts(items: Sequence[Any], *, component_id: str) -> list[str]:
    out: list[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, Mapping):
            text = ""
            for key in (component_id, "candidate", "instruction", "text", "content", "value", "prompt", "system_prompt"):
                raw = item.get(key) if isinstance(item, Mapping) else None
                text = str(raw or "").strip()
                if text:
                    break
        else:
            text = ""
        if text:
            out.append(text)
    return out


def _component_candidates_from_reflection(
    *,
    seed_candidate: dict[str, str],
    reflective_dataset: Mapping[str, Sequence[Any]] | None,
    config: MiproCompatRunConfig,
) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {}
    for component_id, seed_text in seed_candidate.items():
        options = [seed_text]
        if component_id in config.component_candidates:
            options.extend(config.component_candidates.get(component_id) or [])
        elif reflective_dataset is not None:
            options.extend(_extract_candidate_texts(reflective_dataset.get(component_id, ()), component_id=component_id))
        deduped: list[str] = []
        seen: set[str] = set()
        for option in options:
            text = str(option).strip()
            if text and text not in seen:
                deduped.append(text)
                seen.add(text)
        candidates[component_id] = deduped or [seed_text]
    return candidates


def _coerce_eval_batch(value: Any) -> MiproEvaluationBatch:
    batch = MiproEvaluationBatch.from_value(value)
    if not batch.outputs:
        raise ValueError("adapter.evaluate must return at least one output")
    return batch


def _validate_batch_size(batch: Sequence[Any], eval_batch: MiproEvaluationBatch) -> None:
    if len(batch) != len(eval_batch.outputs) or len(batch) != len(eval_batch.scores):
        raise ValueError(
            "adapter.evaluate must return one output and one score per input example"
        )


def _result_output_root(
    *,
    config: MiproCompatRunConfig,
    ledger_path: str | None,
) -> Path | None:
    if config.output_dir:
        return Path(config.output_dir).expanduser().resolve()
    if ledger_path:
        return Path(ledger_path).expanduser().resolve().parent
    return None


async def async_optimize(
    *,
    seed_candidate: Mapping[str, str],
    trainset: Sequence[Any],
    valset: Sequence[Any] | None = None,
    adapter: GepaCompatAdapter,
    task_lm: str | None = None,
    reflection_lm: str | None = None,
    max_metric_calls: int | None = None,
    config: MiproCompatRunConfig | None = None,
) -> MiproCompatResult:
    normalized_seed = _normalized_candidate_map(seed_candidate)
    if not trainset:
        raise ValueError("trainset must be non-empty")
    cfg = config or MiproCompatRunConfig(
        dataset="unknown",
        task_model=str(task_lm or ""),
        proposer_model=str(reflection_lm or ""),
    )
    effective_config = MiproCompatRunConfig(
        dataset=cfg.dataset,
        task=cfg.task,
        train_n=cfg.train_n if cfg.train_n is not None else len(trainset),
        heldout_n=cfg.heldout_n if cfg.heldout_n is not None else (len(valset) if valset is not None else 0),
        seed=cfg.seed,
        task_model=str(task_lm or cfg.task_model or ""),
        proposer_model=str(reflection_lm or cfg.proposer_model or ""),
        optimizer_budget=int(cfg.optimizer_budget),
        max_concurrency=cfg.max_concurrency,
        use_proposer=cfg.use_proposer,
        resume=cfg.resume,
        run_id=cfg.run_id,
        output_dir=cfg.output_dir,
        ledger_path=cfg.ledger_path,
        runtime_binding=cfg.runtime_binding,
        runtime_capabilities=cfg.runtime_capabilities,
        container_contract=cfg.container_contract,
        execution_mode=cfg.execution_mode,
        sft_config=cfg.sft_config,
        component_candidates=dict(cfg.component_candidates),
        metadata=dict(cfg.metadata),
    )
    metric_call_cap = (
        max(1, int(max_metric_calls))
        if max_metric_calls is not None
        else None
    )
    if effective_config.execution_mode != MiproCandidateExecutionMode.PROMPT_ONLY:
        raise ValueError(
            "GEPA compatibility mode currently supports only prompt_only execution mode"
        )
    if effective_config.runtime_capabilities is not None:
        assert_mipro_runtime_supported(
            effective_config.runtime_capabilities,
            execution_mode=effective_config.execution_mode,
            use_proposer=effective_config.use_proposer,
        )
    if effective_config.container_contract is not None:
        assert_mipro_runtime_supported(
            effective_config.container_contract,
            execution_mode=effective_config.execution_mode,
            use_proposer=effective_config.use_proposer,
        )
    metric_call_count = 0
    interrupt_after_metric_calls = int(
        effective_config.metadata.get("interrupt_after_metric_calls") or 0
    )

    def persist_metric_call_count() -> None:
        if not effective_config.ledger_path:
            return
        try:
            ledger = SQLiteMiproRunLedger(
                run_id=str(effective_config.run_id or Path(effective_config.ledger_path).stem),
                ledger_path=str(effective_config.ledger_path),
                program_id=compiled.program_template.program_id if "compiled" in locals() else "mipro_gepa_compat",
                mode="phase2",
                resume=True,
            )
        except Exception:
            return
        try:
            ledger.upsert_state(key="total_metric_calls", value=int(metric_call_count))
        finally:
            ledger.close()

    async def evaluate_adapter(
        batch_rows: Sequence[Any],
        candidate_map: Mapping[str, str],
        *,
        capture_traces: bool,
    ) -> MiproEvaluationBatch:
        nonlocal metric_call_count
        raw = adapter.evaluate(batch_rows, dict(candidate_map), capture_traces=capture_traces)
        if inspect.isawaitable(raw):
            raw = await cast(
                Awaitable[MiproEvaluationBatch | dict[str, Any]],
                raw,
            )
        eval_batch = _coerce_eval_batch(raw)
        _validate_batch_size(batch_rows, eval_batch)
        metric_call_count += len(batch_rows)
        persist_metric_call_count()
        if metric_call_cap is not None and metric_call_count >= metric_call_cap:
            raise MiproCompatInterruptionRequested(
                f"compat benchmark exhausted max_metric_calls={metric_call_cap}"
            )
        if interrupt_after_metric_calls > 0 and metric_call_count >= interrupt_after_metric_calls:
            raise MiproCompatInterruptionRequested(
                f"compat benchmark interrupted after {metric_call_count} metric calls"
            )
        return eval_batch

    component_specs = [
        MiproComponentSpec(component_id=component_id)
        for component_id in normalized_seed
    ]
    seed_train_eval = await evaluate_adapter(trainset, normalized_seed, capture_traces=True)
    reflective_dataset = adapter.make_reflective_dataset(
        normalized_seed,
        seed_train_eval,
        [spec.component_id for spec in component_specs],
    )
    component_candidates = _component_candidates_from_reflection(
        seed_candidate=normalized_seed,
        reflective_dataset=reflective_dataset,
        config=effective_config,
    )
    template = MiproProgramTemplate(
        program_id=f"mipro_gepa_{effective_config.dataset or 'task'}",
        modules=tuple(
            MiproModuleTemplate(
                module_id=component_id,
                instruction_candidates=tuple(options),
                demo_slots=(),
            )
            for component_id, options in component_candidates.items()
        ),
    )
    compiled = compile_search_space(template)
    top_k = max(1, min(int(effective_config.max_concurrency), 4))
    optimizer = await DiscreteMiproOptimizer.from_search_space(
        search_space=compiled.search_space,
        tpe_config=TpeConfig(),
        rng_seed=int(effective_config.seed),
    )
    consumed_seed_train_cache = False

    async def evaluate_train(candidate: MiproProgramCandidate) -> tuple[float, dict[str, Any]]:
        nonlocal consumed_seed_train_cache
        candidate_map = _candidate_map_from_program(candidate)
        if not consumed_seed_train_cache and candidate_map == normalized_seed:
            eval_batch = seed_train_eval
            consumed_seed_train_cache = True
        else:
            eval_batch = await evaluate_adapter(trainset, candidate_map, capture_traces=False)
        return eval_batch.aggregate_score(), {
            "split": "train",
            "scores": list(eval_batch.scores),
            "outputs": list(eval_batch.outputs),
            "traces_captured": bool(eval_batch.traces),
            "eval_metadata": dict(eval_batch.metadata),
        }

    async def evaluate_heldout(candidate: MiproProgramCandidate) -> tuple[float, dict[str, Any]]:
        batch_rows = valset if valset is not None else trainset
        split = "val" if valset is not None else "train"
        eval_batch = await evaluate_adapter(batch_rows, _candidate_map_from_program(candidate), capture_traces=False)
        return eval_batch.aggregate_score(), {
            "split": split,
            "scores": list(eval_batch.scores),
            "outputs": list(eval_batch.outputs),
            "eval_metadata": dict(eval_batch.metadata),
        }

    phase2_outcome = await run_train_loop(
        compiled_space=compiled,
        optimizer=optimizer,
        evaluate_train=evaluate_train,
        evaluate_heldout=evaluate_heldout,
        config=MiproPhase2Config(
            rounds=effective_config.phase2_rounds(top_k=top_k),
            top_k=top_k,
            max_concurrency=top_k,
            seed_with_baseline=True,
            heldout_interval=None,
            compute_final_heldout=True,
        ),
        run_id=effective_config.run_id,
        ledger_path=effective_config.ledger_path,
        resume=effective_config.resume,
    )

    discovery_candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for trial in phase2_outcome.train_observations:
        candidate = decode_config(compiled, trial.config)
        if candidate.candidate_id in seen_ids:
            continue
        discovery_candidates.append(
            {
                "candidate": candidate,
                "score": float(trial.score),
                "metadata": dict(trial.details),
            }
        )
        seen_ids.add(str(candidate.candidate_id))

    eval_rows = valset if valset is not None else trainset
    candidate_records: list[MiproCandidateRecord] = []
    candidates: list[dict[str, str]] = []
    parents: list[str | None] = []
    val_aggregate_scores: list[float] = []
    val_subscores: list[list[float]] = []
    for entry in discovery_candidates:
        candidate = entry["candidate"]
        candidate_map = _candidate_map_from_program(candidate)
        batch = await evaluate_adapter(eval_rows, candidate_map, capture_traces=False)
        candidates.append(candidate_map)
        parents.append(candidate.parent_candidate_id)
        val_aggregate_scores.append(batch.aggregate_score())
        val_subscores.append([float(score) for score in batch.scores])
        candidate_records.append(
            MiproCandidateRecord(
                candidate_id=str(candidate.candidate_id),
                component_values=candidate_map,
                parent_candidate_id=candidate.parent_candidate_id,
                score=entry["score"],
                metadata={
                    "discovery_score": float(entry["score"]),
                    "runtime": effective_config.runtime_binding.to_dict(),
                    **dict(entry["metadata"]),
                },
            )
        )

    if not candidate_records:
        raise RuntimeError("compat optimize produced no candidate records")
    best_idx = max(range(len(val_aggregate_scores)), key=lambda idx: val_aggregate_scores[idx])
    baseline_idx = next(
        (
            idx
            for idx, candidate_map in enumerate(candidates)
            if candidate_map == normalized_seed
        ),
        None,
    )
    if baseline_idx is None:
        raise RuntimeError("compat optimize did not preserve the seed candidate in final heldout candidates")
    final_baseline_score = float(val_aggregate_scores[baseline_idx])
    final_best_score = float(val_aggregate_scores[best_idx])
    final_lift = final_best_score - final_baseline_score
    ledger_read_model: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    run_status: dict[str, Any] = {}
    if phase2_outcome.ledger_path:
        ledger = SQLiteMiproRunLedger(
            run_id=str(phase2_outcome.run_id),
            ledger_path=str(phase2_outcome.ledger_path),
            program_id=compiled.program_template.program_id,
            mode="phase2",
            resume=True,
        )
        try:
            run_status = ledger.get_run_status()
            ledger_read_model = ledger.build_run_read_model()
            events = list(reversed(ledger.query_events(limit=500)))
        finally:
            ledger.close()

    artifact_root = _result_output_root(config=effective_config, ledger_path=phase2_outcome.ledger_path)
    artifacts_payload: dict[str, Any] = {}
    execution_contract = MiproExecutionContract(
        dataset=effective_config.dataset,
        task=effective_config.task,
        runtime_binding=effective_config.runtime_binding,
        metadata={"component_specs": [spec.to_dict() for spec in component_specs]},
    )
    run_summary = MiproRunArtifactSummary(
        run_id=str(phase2_outcome.run_id or ""),
        task_id=effective_config.dataset,
        best_candidate_id=candidate_records[best_idx].candidate_id,
        best_score=final_best_score,
        baseline_score=final_baseline_score,
        heldout_score=final_best_score,
        metadata={
            "task": effective_config.task,
            "optimizer_budget": effective_config.optimizer_budget,
            "phase2_rounds": effective_config.phase2_rounds(top_k=top_k),
            "execution_contract": execution_contract.to_dict(),
            "baseline_candidate_id": candidate_records[baseline_idx].candidate_id,
            "baseline_candidate_index": baseline_idx,
            "best_candidate_index": best_idx,
            "phase2_heldout_baseline_score": phase2_outcome.heldout_baseline_score,
            "phase2_heldout_best_score": phase2_outcome.heldout_best_score,
            "phase2_heldout_lift": phase2_outcome.heldout_lift,
        },
    )
    manifest = MiproArtifactManifest(
        run_id=str(phase2_outcome.run_id or ""),
        task_id=effective_config.dataset,
        mode="gepa_ai_compat",
        ledger_path=phase2_outcome.ledger_path,
        workspace_root=run_status.get("workspace_root"),
        metadata={
            "task": effective_config.task,
            "component_specs": [spec.to_dict() for spec in component_specs],
            "execution_contract": execution_contract.to_dict(),
        },
    )
    if artifact_root is not None:
        artifacts_payload = write_miprov2_artifacts(
            output_dir=str(artifact_root),
            run_summary=run_summary.as_dict(),
            best_candidate={
                "candidate_id": candidate_records[best_idx].candidate_id,
                "candidate": candidates[best_idx],
                "score": final_best_score,
            },
            heldout_eval={
                "baseline_score": final_baseline_score,
                "best_score": final_best_score,
                "lift": final_lift,
                "baseline_candidate_id": candidate_records[baseline_idx].candidate_id,
                "baseline_candidate_index": baseline_idx,
                "best_candidate_id": candidate_records[best_idx].candidate_id,
                "best_candidate_index": best_idx,
                "scores_by_candidate": val_aggregate_scores,
                "phase2_heldout_baseline_score": phase2_outcome.heldout_baseline_score,
                "phase2_heldout_best_score": phase2_outcome.heldout_best_score,
                "phase2_heldout_lift": phase2_outcome.heldout_lift,
            },
            result_manifest=manifest.as_dict(),
            run_read_model=ledger_read_model or None,
            event_stream=events or None,
            artifacts=MiproArtifactPaths(),
        )

    persist_metric_call_count()

    return MiproCompatResult(
        run_id=phase2_outcome.run_id,
        ledger_path=phase2_outcome.ledger_path,
        best_idx=best_idx,
        best_candidate=candidates[best_idx],
        candidates=candidates,
        candidate_records=candidate_records,
        parents=parents,
        val_aggregate_scores=val_aggregate_scores,
        val_subscores=val_subscores,
        total_metric_calls=metric_call_count,
        artifacts={
            "files": artifacts_payload,
            "execution_contract": execution_contract.to_dict(),
            "manifest": manifest.as_dict(),
        },
        metadata={
            "task_lm": effective_config.task_model,
            "reflection_lm": effective_config.proposer_model,
            "train_observations": len(phase2_outcome.train_observations),
            "component_specs": [spec.to_dict() for spec in component_specs],
            "baseline_score": final_baseline_score,
            "best_score": final_best_score,
            "lift": final_lift,
            "baseline_candidate_index": baseline_idx,
            "best_candidate_index": best_idx,
            "phase2_heldout_baseline_score": phase2_outcome.heldout_baseline_score,
            "phase2_heldout_best_score": phase2_outcome.heldout_best_score,
            "phase2_heldout_lift": phase2_outcome.heldout_lift,
            "optimizer_budget": int(effective_config.optimizer_budget),
            "max_metric_calls": metric_call_cap,
        },
    )


def optimize(
    *,
    seed_candidate: Mapping[str, str],
    trainset: Sequence[Any],
    valset: Sequence[Any] | None = None,
    adapter: GepaCompatAdapter,
    task_lm: str | None = None,
    reflection_lm: str | None = None,
    max_metric_calls: int | None = None,
    config: MiproCompatRunConfig | None = None,
) -> MiproCompatResult:
    return asyncio.run(
        async_optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            task_lm=task_lm,
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            config=config,
        )
    )


__all__ = [
    "GepaCompatAdapter",
    "MiproCompatInterruptionRequested",
    "async_optimize",
    "optimize",
]
