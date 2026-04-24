"""MIPROv2-oriented discrete optimizer facade for Synth optimizer packages.

This module adds the phase-0 core loop from ``core/instructions.md``:
suggest candidate configs, observe scored trials, and run async batched
evaluation loops on top of :class:`AsyncDiscreteTpeOptimizer`.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, TypeAlias, cast

from synth_optimizers.miprov2.async_discrete_tpe import (
    AsyncDiscreteTpeOptimizer,
    TpeConfig,
)

_CANDIDATE_PREFIX = "candidate_"


def canonicalize_lever_bundle(config: Mapping[str, str]) -> dict[str, str]:
    """Return a stable, string-only lever bundle mapping."""

    normalized: dict[str, str] = {}
    for raw_key, raw_value in config.items():
        key = str(raw_key).strip()
        if not key:
            continue
        normalized[key] = str(raw_value).strip()
    return dict(sorted(normalized.items(), key=lambda item: item[0]))


def lever_bundle_hash(config: Mapping[str, str]) -> str:
    """Stable sha256 hash for a config bundle (canonical JSON serialization)."""

    payload = json.dumps(
        canonicalize_lever_bundle(config),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class MiproTrialResult:
    """Observed score and metadata for one candidate evaluation."""

    config: dict[str, str]
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    cost_proxy: float = 0.0
    latency_ms: float = 0.0
    candidate_id: str | None = None
    lever_bundle_hash: str | None = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.config = canonicalize_lever_bundle(self.config)
        self.score = float(self.score)
        self.cost_proxy = float(self.cost_proxy)
        self.latency_ms = float(self.latency_ms)
        self.details = dict(self.details)
        bundle_hash = self.lever_bundle_hash or lever_bundle_hash(self.config)
        self.lever_bundle_hash = bundle_hash
        if self.candidate_id is None:
            self.candidate_id = f"{_CANDIDATE_PREFIX}{bundle_hash[:12]}"


@dataclass(slots=True, frozen=True)
class MiproObservationSummary:
    count: int
    mean_score: float | None
    min_score: float | None
    max_score: float | None


@dataclass(slots=True, frozen=True)
class MiproSearchState:
    """Serializable snapshot of optimizer state for orchestration/ledger layers."""

    search_space: dict[str, list[str]]
    rng_seed: int | None
    observation_summary: MiproObservationSummary
    best_config: dict[str, str] | None
    best_score: float | None
    best_candidate_id: str | None
    last_candidate_id: str | None


class MiproOptimizer(Protocol):
    """Protocol for MIPRO-style optimizers."""

    async def suggest(
        self, *, top_k: int = 1, exclude_keys: set[str] | None = None
    ) -> list[dict[str, str]]: ...

    async def observe(self, result: MiproTrialResult) -> None: ...

    async def run_batch(
        self,
        *,
        evaluate: "EvaluateCandidate",
        rounds: int,
        top_k: int = 1,
        max_concurrency: int = 1,
        exclude_keys: set[str] | None = None,
    ) -> list[MiproTrialResult]: ...

    async def get_state(self) -> MiproSearchState: ...


ScoreWithDetails: TypeAlias = tuple[float, Mapping[str, Any]]
EvaluateOutcome: TypeAlias = float | MiproTrialResult | ScoreWithDetails
EvaluateCandidate: TypeAlias = Callable[
    [dict[str, str]],
    EvaluateOutcome | Awaitable[EvaluateOutcome],
]


class DiscreteMiproOptimizer:
    """Thin MIPRO facade over :class:`AsyncDiscreteTpeOptimizer`."""

    def __init__(
        self,
        *,
        tpe: AsyncDiscreteTpeOptimizer,
        rng_seed: int | None,
    ) -> None:
        self._tpe = tpe
        self._rng_seed = rng_seed
        self._history: list[MiproTrialResult] = []
        self._best: MiproTrialResult | None = None
        self._last: MiproTrialResult | None = None
        self._state_lock = asyncio.Lock()

    @classmethod
    async def from_search_space(
        cls,
        *,
        search_space: Mapping[str, list[str]],
        tpe_config: TpeConfig | None = None,
        rng_seed: int | None = 42,
    ) -> "DiscreteMiproOptimizer":
        rng = random.Random(rng_seed)
        tpe = AsyncDiscreteTpeOptimizer(config=tpe_config or TpeConfig(), rng=rng)
        instance = cls(tpe=tpe, rng_seed=rng_seed)
        await instance.update_search_space(search_space)
        return instance

    @property
    def observations(self) -> list[MiproTrialResult]:
        return list(self._history)

    @property
    def tpe(self) -> AsyncDiscreteTpeOptimizer:
        return self._tpe

    async def update_search_space(self, search_space: Mapping[str, list[str]]) -> None:
        await self._tpe.update_search_space(
            {
                str(component): [str(value) for value in values]
                for component, values in search_space.items()
            }
        )

    async def suggest(
        self, *, top_k: int = 1, exclude_keys: set[str] | None = None
    ) -> list[dict[str, str]]:
        return await self._tpe.ask(top_k=max(1, int(top_k)), exclude_keys=exclude_keys)

    async def preview_suggest(
        self, *, top_k: int = 1, exclude_keys: set[str] | None = None
    ) -> list[dict[str, str]]:
        return await self._tpe.preview_ask(
            top_k=max(1, int(top_k)),
            exclude_keys=exclude_keys,
        )

    async def suggest_one(
        self, *, exclude_keys: set[str] | None = None
    ) -> dict[str, str] | None:
        candidates = await self.suggest(top_k=1, exclude_keys=exclude_keys)
        return candidates[0] if candidates else None

    async def observe(self, result: MiproTrialResult) -> None:
        await self._tpe.tell(result.config, result.score)
        await self._record_observation(result)

    async def observe_batch(self, results: list[MiproTrialResult]) -> None:
        if not results:
            return
        await self._tpe.tell_batch([(item.config, item.score) for item in results])
        async with self._state_lock:
            for result in results:
                self._record_observation_unlocked(result)

    async def run_batch(
        self,
        *,
        evaluate: EvaluateCandidate,
        rounds: int,
        top_k: int = 1,
        max_concurrency: int = 1,
        exclude_keys: set[str] | None = None,
    ) -> list[MiproTrialResult]:
        """Run ``rounds`` of suggest/evaluate/observe and return new observations."""

        if rounds <= 0:
            return []

        semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        all_results: list[MiproTrialResult] = []

        for _ in range(int(rounds)):
            candidates = await self.suggest(top_k=top_k, exclude_keys=exclude_keys)
            if not candidates:
                break

            batch_results = await asyncio.gather(
                *[
                    self._evaluate_candidate(evaluate, cfg, semaphore)
                    for cfg in candidates
                ]
            )
            await self.observe_batch(batch_results)
            all_results.extend(batch_results)

        return all_results

    async def get_state(self) -> MiproSearchState:
        snapshot = self._tpe.snapshot_state()
        async with self._state_lock:
            scores = [item.score for item in self._history]
            count = len(scores)
            summary = MiproObservationSummary(
                count=count,
                mean_score=(sum(scores) / count) if count else None,
                min_score=min(scores) if scores else None,
                max_score=max(scores) if scores else None,
            )
            best_config = dict(self._best.config) if self._best is not None else None
            best_score = self._best.score if self._best is not None else None
            best_candidate_id = (
                self._best.candidate_id if self._best is not None else None
            )
            last_candidate_id = (
                self._last.candidate_id if self._last is not None else None
            )

        return MiproSearchState(
            search_space={
                k: list(v) for k, v in dict(snapshot.get("search_space") or {}).items()
            },
            rng_seed=self._rng_seed,
            observation_summary=summary,
            best_config=best_config,
            best_score=best_score,
            best_candidate_id=best_candidate_id,
            last_candidate_id=last_candidate_id,
        )

    async def _evaluate_candidate(
        self,
        evaluate: EvaluateCandidate,
        config: dict[str, str],
        semaphore: asyncio.Semaphore,
    ) -> MiproTrialResult:
        start = time.perf_counter()
        async with semaphore:
            raw = evaluate(dict(config))
            if inspect.isawaitable(raw):
                raw = await cast(Awaitable[EvaluateOutcome], raw)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return self._coerce_result(config=config, outcome=raw, latency_ms=latency_ms)

    def _coerce_result(
        self,
        *,
        config: dict[str, str],
        outcome: EvaluateOutcome,
        latency_ms: float,
    ) -> MiproTrialResult:
        if isinstance(outcome, MiproTrialResult):
            merged = MiproTrialResult(
                config=outcome.config or config,
                score=outcome.score,
                details=outcome.details,
                cost_proxy=outcome.cost_proxy,
                latency_ms=outcome.latency_ms if outcome.latency_ms > 0 else latency_ms,
                candidate_id=outcome.candidate_id,
                lever_bundle_hash=outcome.lever_bundle_hash,
                timestamp=outcome.timestamp,
            )
            return merged

        if isinstance(outcome, tuple):
            score, details = outcome
            return MiproTrialResult(
                config=config,
                score=float(score),
                details=dict(details),
                latency_ms=latency_ms,
            )

        return MiproTrialResult(
            config=config, score=float(outcome), latency_ms=latency_ms
        )

    async def _record_observation(self, result: MiproTrialResult) -> None:
        async with self._state_lock:
            self._record_observation_unlocked(result)

    def _record_observation_unlocked(self, result: MiproTrialResult) -> None:
        self._history.append(result)
        self._last = result
        if self._best is None or result.score > self._best.score:
            self._best = result


__all__ = [
    "MiproOptimizer",
    "MiproTrialResult",
    "MiproSearchState",
    "MiproObservationSummary",
    "DiscreteMiproOptimizer",
    "EvaluateCandidate",
    "EvaluateOutcome",
    "ScoreWithDetails",
    "canonicalize_lever_bundle",
    "lever_bundle_hash",
]
