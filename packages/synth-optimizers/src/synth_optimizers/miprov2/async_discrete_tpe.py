"""Async-friendly discrete TPE (Tree-structured Parzen for categorical search spaces).

Algorithm matches the custom ``DiscreteTpeOptimizer`` in ``nanoprogram/baselines/basic.py``:
good vs bad observation split, smoothed frequency tables, expected-improvement scoring,
startup random trials.  Public API is ``async`` so orchestrators can ``await suggest()``
/ ``await tell()`` alongside asyncio rollouts without mixing in ``ThreadPoolExecutor``.

Reference: ``~/Documents/GitHub/nanoprogram/baselines/basic.py`` (sync implementation).
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass
from typing import Any

NONE_TRANSFORM = "__none__"


@dataclass(frozen=True)
class TpeConfig:
    enabled: bool = True
    gamma: float = 0.25
    n_candidates: int = 24
    n_startup_trials: int = 8
    epsilon: float = 0.08
    alpha: float = 1.0


@dataclass(frozen=True)
class TpeObservation:
    config: dict[str, str]
    score: float
    timestamp: float


class _SyncDiscreteTpeCore:
    """Sync core; wrapped by :class:`AsyncDiscreteTpeOptimizer` with an ``asyncio.Lock``."""

    def __init__(self, *, config: TpeConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng
        self.search_space: dict[str, list[str]] = {}
        self.observations: list[TpeObservation] = []
        self.tabu_config_keys: set[str] = set()

    def update_search_space(self, choices_by_component: dict[str, list[str]]) -> None:
        normalized: dict[str, list[str]] = {}
        for component, values in choices_by_component.items():
            unique: list[str] = []
            seen: set[str] = set()
            for value in values:
                text = str(value).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                unique.append(text)
            if unique:
                normalized[str(component)] = unique
        self.search_space = normalized

    def config_key(self, config: dict[str, str]) -> str:
        items = []
        for component in sorted(self.search_space):
            items.append(f"{component}:{config.get(component, NONE_TRANSFORM)}")
        return "|".join(items)

    def tell(self, config: dict[str, str], score: float) -> None:
        key = self.config_key(config)
        self.observations.append(
            TpeObservation(
                config=dict(config), score=float(score), timestamp=time.time()
            )
        )
        self.tabu_config_keys.add(key)

    def ask(
        self, *, top_k: int, exclude_keys: set[str] | None = None
    ) -> list[dict[str, str]]:
        exclude = set(exclude_keys or set()) | self.tabu_config_keys
        if top_k <= 0 or not self.search_space:
            return []

        if len(self.observations) < self.config.n_startup_trials:
            return self._sample_random_configs(top_k=top_k, exclude_keys=exclude)

        ranked = self._ranked_candidates(exclude_keys=exclude)
        if ranked:
            return [config for _score, config in ranked[:top_k]]
        return self._sample_random_configs(top_k=top_k, exclude_keys=exclude)

    def _sample_random_config(self) -> dict[str, str]:
        sampled: dict[str, str] = {}
        for component, values in self.search_space.items():
            sampled[component] = self.rng.choice(values)
        return sampled

    def _sample_random_configs(
        self, *, top_k: int, exclude_keys: set[str]
    ) -> list[dict[str, str]]:
        configs: list[dict[str, str]] = []
        seen = set(exclude_keys)
        max_attempts = max(20, top_k * max(8, len(self.search_space)))
        while len(configs) < top_k and max_attempts > 0:
            max_attempts -= 1
            candidate = self._sample_random_config()
            key = self.config_key(candidate)
            if key in seen:
                continue
            configs.append(candidate)
            seen.add(key)
        return configs

    def _ranked_candidates(
        self, *, exclude_keys: set[str]
    ) -> list[tuple[float, dict[str, str]]]:
        good_obs, bad_obs = self._split_observations()
        if not good_obs or not bad_obs:
            return []
        good_tables, bad_tables = self._build_probability_tables(good_obs, bad_obs)
        ranked: list[tuple[float, dict[str, str]]] = []
        seen = set(exclude_keys)
        attempts = max(self.config.n_candidates * 4, 32)
        for _ in range(attempts):
            if self.rng.random() < self.config.epsilon:
                candidate = self._sample_random_config()
            else:
                candidate = self._sample_from_tables(good_tables)
            key = self.config_key(candidate)
            if key in seen:
                continue
            ranked.append(
                (
                    self._expected_improvement(candidate, good_tables, bad_tables),
                    candidate,
                )
            )
            seen.add(key)
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _split_observations(self) -> tuple[list[TpeObservation], list[TpeObservation]]:
        if len(self.observations) < 2:
            return [], []
        ordered = sorted(self.observations, key=lambda item: item.score, reverse=True)
        n_good = max(1, int(math.ceil(len(ordered) * self.config.gamma)))
        if n_good >= len(ordered):
            n_good = len(ordered) - 1
        return ordered[:n_good], ordered[n_good:]

    def _build_probability_tables(
        self,
        good_obs: list[TpeObservation],
        bad_obs: list[TpeObservation],
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        good_tables: dict[str, dict[str, float]] = {}
        bad_tables: dict[str, dict[str, float]] = {}
        for component, values in self.search_space.items():
            good_counts = {value: float(self.config.alpha) for value in values}
            bad_counts = {value: float(self.config.alpha) for value in values}
            for obs in good_obs:
                selected = obs.config.get(component, NONE_TRANSFORM)
                good_counts[selected] = (
                    good_counts.get(selected, float(self.config.alpha)) + 1.0
                )
            for obs in bad_obs:
                selected = obs.config.get(component, NONE_TRANSFORM)
                bad_counts[selected] = (
                    bad_counts.get(selected, float(self.config.alpha)) + 1.0
                )
            good_total = sum(good_counts.values()) or 1.0
            bad_total = sum(bad_counts.values()) or 1.0
            good_tables[component] = {
                value: count / good_total for value, count in good_counts.items()
            }
            bad_tables[component] = {
                value: count / bad_total for value, count in bad_counts.items()
            }
        return good_tables, bad_tables

    def _sample_from_tables(
        self, tables: dict[str, dict[str, float]]
    ) -> dict[str, str]:
        sampled: dict[str, str] = {}
        for component, weights in tables.items():
            options = list(weights.keys())
            probs = [max(1e-9, float(weights[option])) for option in options]
            sampled[component] = self.rng.choices(options, weights=probs, k=1)[0]
        return sampled

    def _expected_improvement(
        self,
        config: dict[str, str],
        good_tables: dict[str, dict[str, float]],
        bad_tables: dict[str, dict[str, float]],
    ) -> float:
        score = 0.0
        for component in self.search_space:
            value = config.get(component, NONE_TRANSFORM)
            p_good = max(1e-9, float(good_tables.get(component, {}).get(value, 1e-9)))
            p_bad = max(1e-9, float(bad_tables.get(component, {}).get(value, 1e-9)))
            score += math.log(p_good) - math.log(p_bad)
        return score


class AsyncDiscreteTpeOptimizer:
    """Discrete TPE with ``async`` API; safe for concurrent awaiting callers."""

    def __init__(self, *, config: TpeConfig, rng: random.Random) -> None:
        self._core = _SyncDiscreteTpeCore(config=config, rng=rng)
        self._lock = asyncio.Lock()

    @property
    def config(self) -> TpeConfig:
        return self._core.config

    @property
    def search_space(self) -> dict[str, list[str]]:
        return self._core.search_space

    @property
    def observations(self) -> list[TpeObservation]:
        return list(self._core.observations)

    async def update_search_space(
        self, choices_by_component: dict[str, list[str]]
    ) -> None:
        async with self._lock:
            self._core.update_search_space(choices_by_component)

    async def config_key(self, config: dict[str, str]) -> str:
        async with self._lock:
            return self._core.config_key(config)

    async def tell(self, config: dict[str, str], score: float) -> None:
        async with self._lock:
            self._core.tell(config, score)

    async def tell_batch(self, items: list[tuple[dict[str, str], float]]) -> None:
        """Record several observations atomically (single lock acquisition)."""

        async with self._lock:
            for cfg, score in items:
                self._core.tell(cfg, score)

    async def ask(
        self, *, top_k: int, exclude_keys: set[str] | None = None
    ) -> list[dict[str, str]]:
        async with self._lock:
            return self._core.ask(top_k=top_k, exclude_keys=exclude_keys)

    def snapshot_state(self) -> dict[str, Any]:
        """Read-only snapshot without locking; for debugging only (may be slightly stale)."""

        return {
            "search_space": dict(self._core.search_space),
            "n_observations": len(self._core.observations),
            "n_tabu": len(self._core.tabu_config_keys),
        }


__all__ = [
    "NONE_TRANSFORM",
    "TpeConfig",
    "TpeObservation",
    "AsyncDiscreteTpeOptimizer",
]
