"""The declared blog experiments (active launch A/C plus post-launch D draft).

Each entry encodes the locked parity from the README parity tables. The verdict
is NOT declared here — it is computed from the evidence by `Experiment.verdict()`.
A `mandate` records a human decision to rerun regardless of what the evidence says
(e.g. a route switch or an auth-config fix), so the override is explicit and dated.
"""

from __future__ import annotations

from .model import Experiment, ExperimentKind, ParityLock

# Shared proposer lock for the launch tier (README §"Locked model parity").
_MINI_CHATGPT = dict(proposer_model="gpt-5.4-mini", proposer_auth="chatgpt")

REGISTRY: tuple[Experiment, ...] = (
    # --- Head-to-head (Charts A, C): Synth GEPA vs gepa-ai --------------------
    Experiment(
        container="healthbench",
        label="HealthBench Pro",
        kind=ExperimentKind.HEAD_TO_HEAD,
        charts=("A", "C"),
        parity=ParityLock(policy_model="google/gemini-2.5-flash-lite",
                          policy_route="openrouter", **_MINI_CHATGPT),
    ),
    Experiment(
        container="banking77",
        label="Banking77",
        kind=ExperimentKind.HEAD_TO_HEAD,
        charts=("A", "C"),
        parity=ParityLock(policy_model="google/gemini-2.5-flash-lite",
                          policy_route="openrouter", **_MINI_CHATGPT),
    ),
    Experiment(
        container="hotpotqa",
        label="HotpotQA",
        kind=ExperimentKind.HEAD_TO_HEAD,
        charts=("A", "C"),
        parity=ParityLock(policy_model="google/gemini-2.5-flash-lite",
                          policy_route="openrouter", **_MINI_CHATGPT),
    ),
    Experiment(
        container="tau2_retail",
        label="tau2-bench retail",
        kind=ExperimentKind.HEAD_TO_HEAD,
        charts=("A", "C"),
        # Decision 2026-06-03: move policy off the OpenRouter rpm cap to the Gemini
        # API directly (same model/quality, ~3x throughput). LiteLLM native provider
        # `gemini/...` routes to Google; the existing OpenRouter evidence flags drift.
        parity=ParityLock(policy_model="gemini/gemini-3.1-flash-lite",
                          policy_route="gemini_direct", **_MINI_CHATGPT),
    ),
    # --- Post-launch draft proposer sweep (Chart D): nano / mini / gpt-5.4 ---
    Experiment(
        container="healthbench",
        label="HealthBench Pro",
        kind=ExperimentKind.PROPOSER_SWEEP,
        charts=("D",),
        arms=("gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"),
        parity=ParityLock(policy_model="google/gemini-2.5-flash-lite",
                          policy_route="openrouter", **_MINI_CHATGPT),
    ),
    Experiment(
        container="tau2_retail",
        label="tau2-bench retail",
        kind=ExperimentKind.PROPOSER_SWEEP,
        charts=("D",),
        arms=("gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"),
        parity=ParityLock(policy_model="gemini/gemini-3.1-flash-lite",
                          policy_route="gemini_direct", **_MINI_CHATGPT),
    ),
)


def in_scope() -> tuple[Experiment, ...]:
    return REGISTRY
