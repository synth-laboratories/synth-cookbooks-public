"""DSPy-facing adapter and entrypoint for prompt-only MIPROv2 optimization.

This module intentionally keeps the runtime contract aligned with ``MiproCompatRunConfig`` and
``MiproCompatResult`` so benchmark/report artifacts remain on the existing schema.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import ceil
from typing import Any

from synth_optimizers.miprov2.core import (
    MiproCandidateExecutionMode,
    MiproCompatResult,
    MiproCompatRunConfig,
    MiproEvaluationBatch,
)
from synth_optimizers.miprov2.mipro_compat import optimize as compat_optimize

DspyMetric = Callable[..., Any]


def _require_dspy() -> None:
    try:
        import dspy  # ty: ignore[unresolved-import]  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "dspy is required for DSPy compatibility mode. "
            "Install it in-run, for example: `uv run --with dspy ...`."
        ) from exc


def _coerce_metric_score(value: Any, failure_score: float) -> float:
    if isinstance(value, Mapping):
        if value.get("score") is not None:
            return float(value["score"])
        if value.get("value") is not None:
            return float(value["value"])
    if hasattr(value, "score"):
        try:
            return float(value.score)
        except Exception:  # noqa: BLE001
            return float(failure_score)
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float(failure_score)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(item) for item in value]
    for method_name in ("to_dict", "model_dump", "dict", "as_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _json_safe(method())
            except Exception:  # noqa: BLE001
                continue
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _clone_student(student: Any) -> Any:
    # Prefer the module-aware clone path when available (DSPy modules define ``deepcopy``).
    deep_copy = getattr(student, "deepcopy", None)
    if callable(deep_copy):
        return deep_copy()
    return copy.deepcopy(student)


@dataclass(frozen=True, slots=True)
class DspyProgramComponent:
    component_id: str
    predictor_name: str
    signature_class: str | None
    instructions: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DspyProgramCandidateModel:
    seed_candidate: dict[str, str]
    components: tuple[DspyProgramComponent, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_candidate": dict(self.seed_candidate),
            "components": [component.to_dict() for component in self.components],
        }


@dataclass(frozen=True, slots=True)
class DspyBudgetSemantics:
    optimizer_budget: int
    max_metric_calls: int
    phase2_rounds: int
    top_k: int
    full_eval_equivalent: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_dspy_candidate_model(student: Any) -> DspyProgramCandidateModel:
    components: list[DspyProgramComponent] = []
    seed_candidate: dict[str, str] = {}
    for raw_name, predictor in list(student.named_predictors()):
        name = str(raw_name)
        signature = getattr(predictor, "signature", None)
        instructions = str(getattr(signature, "instructions", "") or "")
        signature_class = None
        if signature is not None:
            signature_class = signature.__class__.__name__
        seed_candidate[name] = instructions
        components.append(
            DspyProgramComponent(
                component_id=name,
                predictor_name=name,
                signature_class=signature_class,
                instructions=instructions,
            )
        )
    if not seed_candidate:
        raise ValueError("student must expose at least one predictor via named_predictors()")
    return DspyProgramCandidateModel(
        seed_candidate=dict(sorted(seed_candidate.items(), key=lambda item: item[0])),
        components=tuple(components),
    )


def resolve_dspy_budget(
    *,
    trainset_size: int,
    valset_size: int,
    max_concurrency: int,
    optimizer_budget: int | None = None,
    max_metric_calls: int | None = None,
    max_full_evals: int | None = None,
) -> DspyBudgetSemantics:
    if trainset_size <= 0:
        raise ValueError("trainset_size must be > 0")
    if valset_size <= 0:
        raise ValueError("valset_size must be > 0")
    configured = [
        optimizer_budget is not None,
        max_metric_calls is not None,
        max_full_evals is not None,
    ]
    if sum(configured) != 1:
        raise ValueError(
            "Exactly one of optimizer_budget, max_metric_calls, or max_full_evals must be set."
        )
    if optimizer_budget is not None:
        resolved_optimizer_budget = max(1, int(optimizer_budget))
        resolved_max_metric_calls = max(
            1,
            resolved_optimizer_budget * (int(trainset_size) + int(valset_size)),
        )
        source = "optimizer_budget"
    elif max_metric_calls is not None:
        resolved_max_metric_calls = max(1, int(max_metric_calls))
        resolved_optimizer_budget = max(
            1,
            ceil(resolved_max_metric_calls / max(1, int(trainset_size))),
        )
        source = "max_metric_calls"
    else:
        assert max_full_evals is not None
        resolved_max_metric_calls = max(
            1,
            int(max_full_evals) * (int(trainset_size) + int(valset_size)),
        )
        resolved_optimizer_budget = max(
            1,
            ceil(resolved_max_metric_calls / max(1, int(trainset_size))),
        )
        source = "max_full_evals"
    top_k = max(1, min(int(max_concurrency), 4))
    phase2_rounds = max(1, ceil(max(1, resolved_optimizer_budget - 1) / top_k))
    full_eval_equivalent = float(
        resolved_max_metric_calls / max(1, int(trainset_size) + int(valset_size))
    )
    return DspyBudgetSemantics(
        optimizer_budget=resolved_optimizer_budget,
        max_metric_calls=resolved_max_metric_calls,
        phase2_rounds=phase2_rounds,
        top_k=top_k,
        full_eval_equivalent=full_eval_equivalent,
        source=source,
    )


@dataclass(slots=True)
class DspyMiproAdapter:
    student: Any
    metric_fn: DspyMetric
    num_threads: int | None = None
    failure_score: float = 0.0
    component_candidates: dict[str, list[str]] = field(default_factory=dict)
    _seed_model: DspyProgramCandidateModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._seed_model = extract_dspy_candidate_model(self.student)
        normalized: dict[str, list[str]] = {}
        for key, values in dict(self.component_candidates or {}).items():
            out: list[str] = []
            seen: set[str] = set()
            for raw in list(values or []):
                text = str(raw or "").strip()
                if text and text not in seen:
                    out.append(text)
                    seen.add(text)
            normalized[str(key)] = out
        self.component_candidates = normalized

    @property
    def seed_candidate(self) -> dict[str, str]:
        return dict(self._seed_model.seed_candidate)

    @property
    def model(self) -> DspyProgramCandidateModel:
        return self._seed_model

    def build_program(self, candidate: Mapping[str, str]) -> Any:
        program = _clone_student(self.student)
        for raw_name, predictor in list(program.named_predictors()):
            name = str(raw_name)
            if name not in candidate:
                continue
            signature = getattr(predictor, "signature", None)
            if signature is None:
                continue
            update = getattr(signature, "with_instructions", None)
            if callable(update):
                predictor.signature = update(str(candidate[name]))
        return program

    def evaluate(
        self,
        batch: Sequence[Any],
        candidate: Mapping[str, str],
        capture_traces: bool = False,
    ) -> MiproEvaluationBatch:
        _require_dspy()
        program = self.build_program(candidate)
        if capture_traces:
            from dspy.teleprompt import bootstrap_trace as bootstrap_trace_module  # ty: ignore[unresolved-import]

            trajectories = bootstrap_trace_module.bootstrap_trace_data(
                program=program,
                dataset=list(batch),
                metric=self.metric_fn,
                num_threads=self.num_threads,
                raise_on_error=False,
                capture_failed_parses=True,
                failure_score=float(self.failure_score),
                format_failure_score=float(self.failure_score),
                callback_metadata={"disable_logging": True},
            )
            outputs: list[Any] = []
            scores: list[float] = []
            for row in list(trajectories or []):
                outputs.append(_json_safe(row.get("prediction")))
                scores.append(_coerce_metric_score(row.get("score"), self.failure_score))
            return MiproEvaluationBatch(
                outputs=outputs,
                scores=scores,
                traces=[_json_safe(row) for row in list(trajectories or [])],
                metadata={"capture_traces": True},
            )

        from dspy.evaluate.evaluate import Evaluate  # ty: ignore[unresolved-import]

        evaluator = Evaluate(
            devset=list(batch),
            metric=self.metric_fn,
            num_threads=self.num_threads,
            return_all_scores=True,
            failure_score=float(self.failure_score),
            provide_traceback=True,
            max_errors=max(100, len(batch) * 100),
        )
        result = evaluator(program)
        raw_results = list(getattr(result, "results", []) or [])
        outputs = [
            _json_safe(entry[1]) for entry in raw_results if isinstance(entry, tuple) and len(entry) >= 2
        ]
        scores = [
            _coerce_metric_score(entry[2], self.failure_score)
            for entry in raw_results
            if isinstance(entry, tuple) and len(entry) >= 3
        ]
        if not outputs and not scores:
            aggregate = _coerce_metric_score(result, self.failure_score)
            outputs = [None for _ in list(batch)]
            scores = [aggregate for _ in list(batch)]
        return MiproEvaluationBatch(
            outputs=outputs,
            scores=scores,
            traces=[],
            metadata={"capture_traces": False},
        )

    def make_reflective_dataset(
        self,
        candidate: Mapping[str, str],
        eval_batch: MiproEvaluationBatch,
        components_to_update: Sequence[str],
    ) -> dict[str, list[dict[str, Any]]]:
        del eval_batch
        seed_candidate = dict(candidate)
        out: dict[str, list[dict[str, Any]]] = {}
        for component in list(components_to_update):
            key = str(component)
            options = self.component_candidates.get(key, [])
            out[key] = [
                {
                    "instruction": option,
                    "source": "dspy_component_candidates",
                }
                for option in options
                if option != seed_candidate.get(key)
            ]
        return out


def optimize_dspy_program(
    *,
    student: Any,
    metric: DspyMetric,
    trainset: Sequence[Any],
    valset: Sequence[Any] | None = None,
    dataset: str = "unknown",
    task: str = "",
    task_lm: str = "",
    reflection_lm: str = "",
    optimizer_budget: int = 8,
    max_metric_calls: int | None = None,
    max_full_evals: int | None = None,
    max_concurrency: int = 4,
    num_threads: int | None = None,
    component_candidates: Mapping[str, Sequence[str]] | None = None,
    config: MiproCompatRunConfig | None = None,
) -> MiproCompatResult:
    _require_dspy()
    if not trainset:
        raise ValueError("trainset must be non-empty")
    heldout_rows = list(valset) if valset is not None else list(trainset)
    budget = resolve_dspy_budget(
        trainset_size=len(trainset),
        valset_size=len(heldout_rows),
        max_concurrency=max_concurrency,
        optimizer_budget=optimizer_budget if max_metric_calls is None and max_full_evals is None else None,
        max_metric_calls=max_metric_calls,
        max_full_evals=max_full_evals,
    )
    normalized_component_candidates: dict[str, list[str]] = {}
    for key, values in dict(component_candidates or {}).items():
        normalized_component_candidates[str(key)] = [
            str(value) for value in list(values or []) if str(value or "").strip()
        ]
    adapter = DspyMiproAdapter(
        student=student,
        metric_fn=metric,
        num_threads=num_threads,
        failure_score=0.0,
        component_candidates=normalized_component_candidates,
    )
    base = config or MiproCompatRunConfig(
        dataset=str(dataset),
        task=str(task),
        train_n=len(trainset),
        heldout_n=len(heldout_rows),
        seed=42,
        task_model=str(task_lm or ""),
        proposer_model=str(reflection_lm or ""),
        optimizer_budget=budget.optimizer_budget,
        max_concurrency=max(1, int(max_concurrency)),
        execution_mode=MiproCandidateExecutionMode.PROMPT_ONLY,
    )
    merged_component_candidates = dict(base.component_candidates)
    for key, values in adapter.component_candidates.items():
        merged_component_candidates[key] = list(values)
    merged_metadata = {
        **dict(base.metadata),
        "dspy_candidate_model": adapter.model.to_dict(),
        "dspy_budget_semantics": budget.to_dict(),
    }
    effective_config = MiproCompatRunConfig(
        dataset=str(base.dataset or dataset),
        task=str(base.task or task),
        train_n=base.train_n if base.train_n is not None else len(trainset),
        heldout_n=base.heldout_n if base.heldout_n is not None else len(heldout_rows),
        seed=int(base.seed),
        task_model=str(base.task_model or task_lm or ""),
        proposer_model=str(base.proposer_model or reflection_lm or ""),
        optimizer_budget=int(budget.optimizer_budget),
        max_concurrency=max(1, int(base.max_concurrency)),
        use_proposer=bool(base.use_proposer),
        resume=bool(base.resume),
        run_id=base.run_id,
        output_dir=base.output_dir,
        ledger_path=base.ledger_path,
        runtime_binding=base.runtime_binding,
        runtime_capabilities=base.runtime_capabilities,
        container_contract=base.container_contract,
        execution_mode=MiproCandidateExecutionMode.PROMPT_ONLY,
        sft_config=None,
        component_candidates=merged_component_candidates,
        metadata=merged_metadata,
    )
    return compat_optimize(
        seed_candidate=adapter.seed_candidate,
        trainset=list(trainset),
        valset=list(heldout_rows),
        adapter=adapter,
        task_lm=str(task_lm or effective_config.task_model or ""),
        reflection_lm=str(reflection_lm or effective_config.proposer_model or ""),
        max_metric_calls=int(budget.max_metric_calls),
        config=effective_config,
    )


__all__ = [
    "DspyBudgetSemantics",
    "DspyMetric",
    "DspyMiproAdapter",
    "DspyProgramCandidateModel",
    "DspyProgramComponent",
    "extract_dspy_candidate_model",
    "optimize_dspy_program",
    "resolve_dspy_budget",
]
