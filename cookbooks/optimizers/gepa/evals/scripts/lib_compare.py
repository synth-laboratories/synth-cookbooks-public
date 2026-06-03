"""Shared helpers for the Banking77 GEPA comparison.

This module centralizes the two things that must be computed IDENTICALLY for
both stacks so the comparison is apples-to-apples:

  1. Dollar cost — derived from recorded token usage via one shared price table
     (configs/banking77.toml [pricing]). Both the policy (rollout) model and the
     proposer/reflection model are priced the same way for both stacks.
  2. Wall-clock time — derived from event timestamps relative to a single run
     start, parsed uniformly across both stacks' timestamp formats.

Proposer cost parity note:
  - gepa-ai's reflection runs through the OpenAI API; usage is read from the
    response (prompt/completion/cached tokens).
  - synth_gepa's proposer runs through the codex app server, which writes its
    real token usage into proposer_workspaces/<gen>/.agent_artifacts/
    opencode_messages.json (thread/tokenUsage/updated events). We parse the
    final cumulative `total` per generation, so the synth proposer's gpt-5.4-mini
    spend is measured too — not dropped.
  Both proposers are gpt-5.4-mini and are priced with the same proposer rates.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path


# ── timestamps ────────────────────────────────────────────────────────────────

def iso_to_epoch(ts: str | None) -> float | None:
    """Parse an ISO-8601 timestamp (with 'Z' or '+00:00') into epoch seconds."""
    if not ts:
        return None
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


# ── pricing ───────────────────────────────────────────────────────────────────

class Pricing:
    """USD-per-token rates, loaded from the [pricing] config block.

    Rates in the config are per 1M tokens; stored here per token.
    """

    def __init__(self, block: dict):
        self.policy_input = float(block["policy_input_per_1m"]) / 1e6
        self.policy_output = float(block["policy_output_per_1m"]) / 1e6
        self.proposer_input = float(block["proposer_input_per_1m"]) / 1e6
        self.proposer_cached_input = float(block["proposer_cached_input_per_1m"]) / 1e6
        self.proposer_output = float(block["proposer_output_per_1m"]) / 1e6

    def rollout_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens * self.policy_input + completion_tokens * self.policy_output

    def proposer_cost(self, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
        """input_tokens is the TOTAL input (cached is a subset of it)."""
        uncached = max(0, int(input_tokens) - int(cached_input_tokens))
        return (
            uncached * self.proposer_input
            + int(cached_input_tokens) * self.proposer_cached_input
            + int(output_tokens) * self.proposer_output
        )


def load_pricing(cfg: dict) -> Pricing:
    """Pricing lives in the [limits.cost] subblock."""
    cost = (cfg.get("limits") or {}).get("cost")
    if not cost:
        raise SystemExit("config is missing the [limits.cost] block")
    return Pricing(cost)


def load_limits(cfg: dict) -> dict:
    """Return the [limits] block (budgets shared by both stacks)."""
    limits = cfg.get("limits")
    if not limits:
        raise SystemExit("config is missing the [limits] block")
    return limits


# ── synth proposer token usage (codex app server artifacts) ─────────────────────

def parse_synth_proposer_usage(run_dir: Path) -> list[dict]:
    """Return one usage record per proposer generation, time-ordered.

    Each record:
      {generation, input_tokens, cached_input_tokens, output_tokens,
       reasoning_output_tokens, total_tokens}

    input_tokens is the full input (cached ⊆ input). Two proposer backends write
    their token usage to different artifacts under
    proposer_workspaces/<gen>/.agent_artifacts/:
      - codex_app_server → opencode_messages.json (thread/tokenUsage/updated total)
      - deepseek_chat     → deepseek_chat_response.json (OpenAI-style usage block,
        with DeepSeek's prompt_cache_hit_tokens as the cached portion)
    """
    ws_root = run_dir / "proposer_workspaces"
    out: list[dict] = []
    if not ws_root.is_dir():
        return out
    for gen_dir in sorted(ws_root.glob("generation_*")):
        adir = gen_dir / ".agent_artifacts"
        try:
            gen_idx = int(gen_dir.name.split("_")[-1])
        except ValueError:
            gen_idx = len(out)
        rec = _parse_codex_usage(adir / "opencode_messages.json", gen_idx)
        if rec is None:
            rec = _parse_deepseek_usage(adir / "deepseek_chat_response.json", gen_idx)
        if rec is not None:
            out.append(rec)
    return out


def _parse_codex_usage(artifact: Path, gen_idx: int) -> dict | None:
    if not artifact.is_file():
        return None
    try:
        doc = json.loads(artifact.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    last_total = None
    for ev in doc.get("received") or []:
        if isinstance(ev, dict) and ev.get("method") == "thread/tokenUsage/updated":
            tu = (ev.get("params") or {}).get("tokenUsage") or {}
            if tu.get("total"):
                last_total = tu["total"]
    if not last_total:
        return None
    return {
        "generation": gen_idx,
        "input_tokens": int(last_total.get("inputTokens", 0)),
        "cached_input_tokens": int(last_total.get("cachedInputTokens", 0)),
        "output_tokens": int(last_total.get("outputTokens", 0)),
        "reasoning_output_tokens": int(last_total.get("reasoningOutputTokens", 0)),
        "total_tokens": int(last_total.get("totalTokens", 0)),
    }


def _parse_deepseek_usage(artifact: Path, gen_idx: int) -> dict | None:
    if not artifact.is_file():
        return None
    try:
        usage = (json.loads(artifact.read_text()) or {}).get("usage") or {}
    except (json.JSONDecodeError, OSError):
        return None
    if not usage:
        return None
    return {
        "generation": gen_idx,
        "input_tokens": int(usage.get("prompt_tokens", 0)),
        "cached_input_tokens": int(usage.get("prompt_cache_hit_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
        "reasoning_output_tokens": int(usage.get("reasoning_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }


# ── cost ledger / cumulative attribution ────────────────────────────────────────

class CostLedger:
    """A time-ordered list of cost-bearing events for one run.

    Each entry: {ts_epoch, rollout_cost, proposer_cost, rollout_count}.
    cumulative_at(t) sums all entries with ts_epoch <= t.
    """

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def add(self, ts_epoch: float | None, *, rollout_cost: float = 0.0,
            proposer_cost: float = 0.0, rollout_count: int = 0) -> None:
        if ts_epoch is None:
            return
        self.entries.append({
            "ts_epoch": ts_epoch,
            "rollout_cost": rollout_cost,
            "proposer_cost": proposer_cost,
            "rollout_count": rollout_count,
        })

    def finalize(self) -> None:
        self.entries.sort(key=lambda e: e["ts_epoch"])

    def cumulative_at(self, ts_epoch: float | None) -> dict:
        """Sum of all ledger entries at or before ts_epoch.

        If ts_epoch is None, returns the full ledger total (best effort).
        """
        rc = pc = 0.0
        n = 0
        for e in self.entries:
            if ts_epoch is None or e["ts_epoch"] <= ts_epoch + 1e-6:
                rc += e["rollout_cost"]
                pc += e["proposer_cost"]
                n += e["rollout_count"]
        return {
            "rollout_cost_usd": round(rc, 8),
            "proposer_cost_usd": round(pc, 8),
            "total_cost_usd": round(rc + pc, 8),
            "rollout_count": n,
        }

    def total(self) -> dict:
        return self.cumulative_at(None)
