"""The Experiment unit and the checks that decide its verdict.

Nouns:
  ParityLock    the recorded proposer/policy/route conditions for an experiment
  Experiment    one container x chart with its arms, under a single ParityLock
  CheckResult   the outcome of one validation against the evidence
  Verdict       the experiment's status, derived from its checks (never hand-set)

Evidence sources (all real files, no fabrication):
  benchmarks/<container>/summary.json          per-arm rollouts, candidates, recorded parity
  evals/evidence/heldout_evaluations.jsonl     the LIVE aggregate the charts read
"""

from __future__ import annotations

import functools
import importlib.util
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from blog_paths import EVALS_DIR, EVIDENCE_DIR

# --- recognized routes ---------------------------------------------------------
# A route is matched either by a host substring in the recorded policy_base_url,
# or by a LiteLLM provider prefix on the recorded policy_model (the native
# `gemini/` provider carries no base_url — it routes to Google directly).
ROUTE_HOST = {
    "openrouter": "openrouter.ai",
    "gemini_direct": "generativelanguage.googleapis.com",
    "openai_direct": "api.openai.com",
}
ROUTE_MODEL_PREFIX = {
    "openrouter": "openrouter/",
    "gemini_direct": "gemini/",
}

# arms of a head-to-head are the two stacks, by their evidence keys
HEAD_TO_HEAD_ARMS = ("synth_gepa", "gepa_ai")


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"  # cannot be auto-verified from here; needs a human/other tool


class Verdict(str, Enum):
    FINISHED = "finished"  # present, parity-checked, budget-floor-checked, wired into aggregate
    REPLUMB = "replumb"    # evidence valid but not in the live aggregate -> file merge, no rollouts
    RERUN = "rerun"        # present but invalid (unfair budget / parity drift) OR mandated
    MISSING = "missing"    # no evidence at all


class ExperimentKind(str, Enum):
    HEAD_TO_HEAD = "head_to_head"      # Synth GEPA vs gepa-ai (Charts A, C)
    PROPOSER_SWEEP = "proposer_sweep"  # proposer-model arms (Chart D)


@dataclass(frozen=True)
class ParityLock:
    """The locked conditions. A run is a valid comparison only if it matches."""

    proposer_model: str
    proposer_auth: str   # "chatgpt" | "api_key"  (only gpt-5.4-nano may use api_key)
    policy_model: str
    policy_route: str    # key of ROUTE_HOST


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str


@functools.lru_cache(maxsize=1)
def live_aggregate_benchmarks() -> frozenset[str]:
    """Distinct benchmark keys present in the LIVE aggregate the charts read."""
    path = EVIDENCE_DIR / "heldout_evaluations.jsonl"
    if not path.exists():
        return frozenset()
    keys: set[str] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                keys.add(json.loads(line)["benchmark"])
    return frozenset(keys)


@dataclass(frozen=True)
class Experiment:
    """One blog experiment: a container x chart, its arms, and the parity it locks."""

    container: str
    label: str
    kind: ExperimentKind
    charts: tuple[str, ...]
    parity: ParityLock
    arms: tuple[str, ...] = HEAD_TO_HEAD_ARMS
    match_floor: float = 0.8      # both arms' candidate AND rollout counts must be within this ratio
    mandate: str = ""             # if set, a human decision forces RERUN regardless of checks

    @property
    def key(self) -> str:
        return f"{self.container}::{'/'.join(self.charts)}"

    @property
    def summary_path(self) -> Path:
        return EVIDENCE_DIR / "benchmarks" / self.container / "summary.json"

    def _summary(self) -> dict | None:
        if not self.summary_path.exists():
            return None
        return json.loads(self.summary_path.read_text())

    # --- checks ---------------------------------------------------------------
    def checks(self) -> list[CheckResult]:
        if self.kind is ExperimentKind.PROPOSER_SWEEP:
            return self._proposer_sweep_checks()

        sm = self._summary()
        if sm is None:
            return [CheckResult("evidence_present", CheckStatus.FAIL,
                                f"no summary.json at {self.summary_path}")]
        out = [CheckResult("evidence_present", CheckStatus.PASS,
                           f"generated {sm.get('generated_at', '?')}")]
        out.append(self._check_live_aggregate())
        out.append(self._check_budget_floor(sm))
        out.extend(self._check_parity(sm))
        return out

    def _proposer_sweep_checks(self) -> list[CheckResult]:
        """Validate Chart D through its own manifest authority.

        Proposer-sweep cells do not live in the heldout aggregate used by Chart
        A/C. They are complete only when Chart D's producer can extract every
        task × proposer manifest into an available numeric cell.
        """
        chart = _chart_d_builder()
        proposers = [p for p in chart.PROPOSER_MODELS if p.get("label") in self.arms]
        by_label = {p.get("label"): p for p in proposers}
        out: list[CheckResult] = []

        for arm in self.arms:
            proposer = by_label.get(arm)
            if proposer is None:
                out.append(CheckResult(
                    f"chart_d_{arm}_declared",
                    CheckStatus.FAIL,
                    "arm is not declared in Chart D PROPOSER_MODELS"))
                continue

            slug = proposer["slug"]
            path = chart._manifest_path(self.container, slug)
            manifest = chart._read_manifest(path)
            if manifest is None:
                out.append(CheckResult(
                    f"chart_d_manifest_{slug}",
                    CheckStatus.FAIL,
                    f"missing or unreadable manifest at {path}"))
                continue

            out.append(CheckResult(
                f"chart_d_manifest_{slug}",
                CheckStatus.PASS,
                f"manifest={_rel(path)}"))

            cell = chart._extract_cell(self.container, proposer, manifest)
            try:
                chart._assert_launch_ready([cell])
            except SystemExit as exc:
                out.append(CheckResult(
                    f"chart_d_cell_{slug}",
                    CheckStatus.FAIL,
                    str(exc)))
            else:
                out.append(CheckResult(
                    f"chart_d_cell_{slug}",
                    CheckStatus.PASS,
                    f"initial_observed={cell.get('initial_observed_reward')} "
                    f"best_observed={cell.get('best_observed_reward')} "
                    f"source={cell.get('best_observed_reward_source')} "
                    f"calls={cell.get('proposer_calls')}"))

        return out

    def _check_live_aggregate(self) -> CheckResult:
        present = self.container in live_aggregate_benchmarks()
        if present:
            return CheckResult("in_live_aggregate", CheckStatus.PASS,
                               "rows in evals/evidence/*.jsonl")
        return CheckResult("in_live_aggregate", CheckStatus.FAIL,
                           "absent from live aggregate (likely in a backup bundle)")

    def _check_budget_floor(self, sm: dict) -> CheckResult:
        """Budget-parity floor (README E04).

        The launch comparison requires both arms' candidate counts and rollout
        counts to be within `match_floor`. This is a comparability guard, not
        proof of identical compute.
        """
        per = sm.get("per_stack", {})
        rollouts = {a: per.get(a, {}).get("total_rollouts") for a in self.arms}
        cands = {a: per.get(a, {}).get("num_candidates") for a in self.arms}
        if any(v is None for v in rollouts.values()) or any(v is None for v in cands.values()):
            return CheckResult("budget_floor", CheckStatus.UNKNOWN,
                               f"missing per-arm counts: rollouts={rollouts} cands={cands}")
        cand_ratio = min(cands.values()) / max(cands.values()) if max(cands.values()) else 0.0
        roll_ratio = min(rollouts.values()) / max(rollouts.values()) if max(rollouts.values()) else 0.0
        ok = cand_ratio >= self.match_floor and roll_ratio >= self.match_floor
        return CheckResult(
            "budget_floor", CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"cands={cands} (ratio={cand_ratio:.2f}); rollouts={rollouts} (ratio={roll_ratio:.2f})")

    def _check_parity(self, sm: dict) -> list[CheckResult]:
        pc = sm.get("parity_controls", {})
        out: list[CheckResult] = []

        rec_proposer = pc.get("proposer_model")
        out.append(CheckResult(
            "parity_proposer",
            CheckStatus.PASS if rec_proposer == self.parity.proposer_model else CheckStatus.FAIL,
            f"recorded={rec_proposer} lock={self.parity.proposer_model}"))

        rec_policy = pc.get("policy_model")
        out.append(CheckResult(
            "parity_policy",
            CheckStatus.PASS if rec_policy == self.parity.policy_model else CheckStatus.FAIL,
            f"recorded={rec_policy} lock={self.parity.policy_model}"))

        rec_url = pc.get("policy_base_url", "")
        want_host = ROUTE_HOST.get(self.parity.policy_route, "")
        want_prefix = ROUTE_MODEL_PREFIX.get(self.parity.policy_route, "")
        route_ok = (bool(want_host) and want_host in rec_url) or \
                   (bool(want_prefix) and str(rec_policy or "").startswith(want_prefix))
        out.append(CheckResult(
            "parity_route",
            CheckStatus.PASS if route_ok else CheckStatus.FAIL,
            f"recorded_url={rec_url or '-'} model={rec_policy} lock={self.parity.policy_route}"))
        return out

    # --- verdict --------------------------------------------------------------
    def verdict(self) -> Verdict:
        if self.mandate:
            return Verdict.RERUN
        checks = self.checks()
        if self.kind is ExperimentKind.PROPOSER_SWEEP:
            if any(c.name.startswith("chart_d_manifest_") and c.status is CheckStatus.FAIL for c in checks):
                return Verdict.MISSING
            if any(c.status is CheckStatus.FAIL for c in checks):
                return Verdict.RERUN
            return Verdict.FINISHED

        by_name = {c.name: c for c in checks}
        if by_name["evidence_present"].status is CheckStatus.FAIL:
            return Verdict.MISSING
        invalidating = ("budget_floor", "parity_proposer", "parity_policy", "parity_route")
        if any(by_name.get(n, _passing(n)).status is CheckStatus.FAIL for n in invalidating):
            return Verdict.RERUN
        if by_name["in_live_aggregate"].status is CheckStatus.FAIL:
            return Verdict.REPLUMB
        return Verdict.FINISHED

    # --- reproduce ------------------------------------------------------------
    def reproduce(self) -> list[str]:
        v = self.verdict()
        if self.kind is ExperimentKind.PROPOSER_SWEEP:
            return [f"cd {_rel(CHART_D_DIR())}",
                    "./run_sweep.sh   # full Chart D: HealthBench + tau2 retail x proposer ladder",
                    "python build_chart.py   # rebuild figures/frontend mirror from existing manifests"]
        if v is Verdict.REPLUMB:
            return [
                f"# evidence is valid but absent from the live aggregate -> file merge, NO rollouts.",
                f"# Append benchmark={self.container} rows from the backup into the live aggregate:",
                f"#   src: evals/evidence/backups/pre_blog_final_scope/"
                "{heldout_evaluations,train_evaluations,candidate_timeline}.jsonl",
                f"#   dst: evals/evidence/<same files>",
                "# then rebuild the charts that read it (chart-c BENCHES + langprobe addendum).",
            ]
        cd = f"cd {_rel(EVALS_DIR)}"
        return [cd,
                f"uv run --project . python scripts/run_stack.py --benchmark {self.container} --stack gepa_ai",
                f"uv run --project . python scripts/run_stack.py --benchmark {self.container} --stack synth_gepa",
                f"uv run --project . python scripts/evaluate_heldout.py --benchmark {self.container}",
                f"uv run --project . python scripts/build_evidence.py --benchmark {self.container}"]


def _passing(name: str) -> CheckResult:
    return CheckResult(name, CheckStatus.PASS, "n/a")


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(EVALS_DIR.parents[3]))
    except ValueError:
        return str(p)


def CHART_D_DIR() -> Path:
    from blog_paths import CHARTS_DIR
    return CHARTS_DIR / "chart-d-proposer-scaling"


@functools.lru_cache(maxsize=1)
def _chart_d_builder():
    path = CHART_D_DIR() / "build_chart.py"
    spec = importlib.util.spec_from_file_location("_gepa_platform_chart_d_build", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Chart D producer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
