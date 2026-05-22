#!/usr/bin/env bash
# Compute-parity head-to-head on Banking77.
# Boots the live container, runs gepa-ai + Synth GEPA against it at matched budgets,
# parses tokens + computes USD cost (gpt-4.1-nano pricing), prints comparison.
#
# Usage (from repo root):
#   source /Users/joshpurtell/Documents/GitHub/synth-ai/.env
#   bash cookbooks/blogs/oss-containers-and-gepa/chart-a-head-to-head/run_banking77_parity.sh

set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY not set" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/Users/joshpurtell/Documents/GitHub/synth-cookbooks-public}"
COOKBOOK_DIR="${REPO_ROOT}/cookbooks/optimizers/gepa"
CHART_DIR="${REPO_ROOT}/cookbooks/blogs/oss-containers-and-gepa/chart-a-head-to-head"
PORT="${PORT:-8810}"
GEPA_AI_MAX_METRIC_CALLS="${GEPA_AI_MAX_METRIC_CALLS:-200}"

cd "${REPO_ROOT}"

echo "[parity] booting Banking77 container on :${PORT} ..."
LOG=/tmp/b77_parity.log
( cd "${COOKBOOK_DIR}" && uv run --project banking77_container python banking77_container/synth_service_app.py --host 127.0.0.1 --port "${PORT}" >"${LOG}" 2>&1 ) &
CPID=$!
trap 'kill "${CPID}" 2>/dev/null || true' EXIT

# wait for /health
for i in {1..30}; do
  sleep 1
  if curl -sS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[parity] container up after ${i}s"
    break
  fi
  if [[ "${i}" == "30" ]]; then
    echo "ERROR: container did not become healthy in 30s. Log tail:" >&2
    tail -20 "${LOG}" >&2
    exit 1
  fi
done

# -------- gepa-ai --------
echo ""
echo "=== gepa-ai (via container adapter, max_metric_calls=${GEPA_AI_MAX_METRIC_CALLS}) ==="
GEPA_AI_T0=$(python3 -c 'import time; print(time.time())')
CONTAINER_URL="http://127.0.0.1:${PORT}" \
  GEPA_AI_MAX_METRIC_CALLS="${GEPA_AI_MAX_METRIC_CALLS}" \
  REFLECTION_MODEL="${REFLECTION_MODEL:-gpt-4.1-nano}" \
  python3 "${CHART_DIR}/configs/gepa_ai/banking77_via_container.py" 2>&1 | tail -3 | tee /tmp/gepa_ai_tail.txt
GEPA_AI_T1=$(python3 -c 'import time; print(time.time())')

# Find the most recent gepa-ai run summary
GEPA_AI_SUMMARY=$(ls -t "${CHART_DIR}/runs/gepa_ai_via_container/"banking77_*/summary.json 2>/dev/null | head -1)

# -------- synth gepa --------
echo ""
echo "=== Synth GEPA (via gepa.toml, max_total_rollouts=200) ==="
# Synth GEPA's run_id is fixed; clear prior workspace state so we get a fresh run.
rm -rf "${CHART_DIR}/runs/synth_gepa/banking77_parity_synth_gepa" 2>/dev/null || true
SYNTH_T0=$(python3 -c 'import time; print(time.time())')
uv run --project packages/synth-optimizers synth-optimizers gepa run \
  --config "${CHART_DIR}/configs/synth_gepa/banking77_parity.toml" 2>&1 | tail -25 | tee /tmp/synth_gepa_tail.txt
SYNTH_T1=$(python3 -c 'import time; print(time.time())')

SYNTH_MANIFEST="${CHART_DIR}/runs/synth_gepa/banking77_parity_synth_gepa/result_manifest.json"

# -------- comparison table --------
echo ""
echo "=== HEAD-TO-HEAD SUMMARY (Banking77 / gpt-4.1-nano via container :${PORT}) ==="
python3 - <<PYEOF
import json, os
PRICE_IN  = 1e-7   # gpt-4.1-nano: $0.10 / 1M input
PRICE_OUT = 4e-7   # gpt-4.1-nano: $0.40 / 1M output

def usd(prompt, completion):
    return prompt * PRICE_IN + completion * PRICE_OUT

# gepa-ai
ga_path = "${GEPA_AI_SUMMARY}"
ga = json.load(open(ga_path)) if ga_path and os.path.exists(ga_path) else None

# synth gepa — parse manifest + the CLI tail for runtime tokens
sg_path = "${SYNTH_MANIFEST}"
sg = json.load(open(sg_path)) if os.path.exists(sg_path) else None

print()
print(f"{'metric':<30}  {'gepa-ai':>16}  {'synth-gepa':>16}")
print("-" * 68)

def line(label, ga_val, sg_val, fmt="{!s:>16}"):
    print(f"{label:<30}  {fmt.format(ga_val if ga_val is not None else '—'):>16}  {fmt.format(sg_val if sg_val is not None else '—'):>16}")

if ga and sg:
    line("seed heldout reward",       round(ga.get('seed_val_score') or 0, 4), "see manifest")
    line("best heldout reward",       round(ga.get('best_val_score') or 0, 4), sg.get('best_candidate', {}).get('heldout_reward'))
    line("metric / rollout calls",    ga.get('rollout_calls'), sg.get('summary',{}).get('rollout_calls'))
    line("rollout prompt tokens",     ga.get('rollout_prompt_tokens'), sg.get('summary',{}).get('rollout_prompt_tokens'))
    line("rollout completion tokens", ga.get('rollout_completion_tokens'), sg.get('summary',{}).get('rollout_completion_tokens'))
    line("reflection prompt tokens",  ga.get('reflection_prompt_tokens'), sg.get('summary',{}).get('proposer_prompt_tokens'))
    line("reflection completion tok", ga.get('reflection_completion_tokens'), sg.get('summary',{}).get('proposer_completion_tokens'))
    line("rollout USD",               f"{ga.get('rollout_usd', 0):.4f}", None)
    line("reflection USD",            f"{ga.get('reflection_usd', 0):.4f}", None)
    line("total USD",                 f"{ga.get('total_usd', 0):.4f}", None)
    line("wall clock (s)",            ga.get('wall_clock_s'), round(${SYNTH_T1} - ${SYNTH_T0}, 1))
    print()
    print(f"gepa-ai summary:    {ga_path}")
    print(f"synth-gepa manifest: {sg_path}")
else:
    print("[parity] one or both summaries missing. ga_path=", ga_path, " sg_path=", sg_path)
PYEOF

echo ""
echo "[parity] done."
