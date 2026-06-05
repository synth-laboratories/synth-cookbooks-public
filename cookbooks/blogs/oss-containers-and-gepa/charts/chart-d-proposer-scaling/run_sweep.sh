#!/usr/bin/env bash
# run_sweep.sh — Chart D proposer sweep (HealthBench + tau2 retail size ladder)
#
# Boots the HealthBench (:8815) and tau2 retail (:8775) containers and runs the
# proposer size ladder (gpt-5.4-nano -> gpt-5.4-mini -> gpt-5.4) on each, so the
# chart shows what changes as the proposer tier changes within one model
# generation. Policy model is held fixed per task. Then builds figures.
#
# Usage:
#   ./run_sweep.sh          # full: nano, mini, gpt-5.4 on HealthBench + tau2 retail (6 runs)
#   ./run_sweep.sh --smoke  # smoke: nano + mini on HealthBench only (skips gpt-5.4 + tau2)
#
# Requirements:
#   - OPENROUTER_API_KEY set in env
#   - OPENAI_API_KEY set in env for gpt-5.4-nano proposer API auth
#   - Run from repo root (the parent of cookbooks/):
#       bash cookbooks/blogs/oss-containers-and-gepa/charts/chart-d-proposer-scaling/run_sweep.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
CHART_D="$REPO_ROOT/cookbooks/blogs/oss-containers-and-gepa/charts/chart-d-proposer-scaling"
GEPA_DIR="$REPO_ROOT/cookbooks/optimizers/gepa"
GEPA_EVALS_DIR="$GEPA_DIR/evals"
CONFIG_DIR="$CHART_D/configs/proposer_sweep"

SMOKE=false
for arg in "$@"; do
  if [[ "$arg" == "--smoke" ]]; then
    SMOKE=true
  fi
done

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "[chart-d] ERROR: OPENROUTER_API_KEY is required for Gemini policy runs." >&2
  exit 1
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[chart-d] ERROR: OPENAI_API_KEY is required for gpt-5.4-nano proposer API auth." >&2
  exit 1
fi

echo "[chart-d] Repo root: $REPO_ROOT"
echo "[chart-d] Smoke mode: $SMOKE"

# ---------------------------------------------------------------------------
# Container lifecycle
# ---------------------------------------------------------------------------

HEALTHBENCH_PID=""
TAU2_PID=""
cleanup() {
  echo "[chart-d] Shutting down containers..."
  for pid in "$HEALTHBENCH_PID" "$TAU2_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

wait_health() {
  local name="$1" port="$2"
  echo "[chart-d] Waiting for $name /health on :$port..."
  for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
      echo "[chart-d] $name container ready."
      return 0
    fi
    if [[ $i -eq 30 ]]; then
      echo "[chart-d] ERROR: $name container did not become healthy after 30s." >&2
      exit 1
    fi
    sleep 1
  done
}

cd "$REPO_ROOT"

echo "[chart-d] Starting HealthBench container on :8815..."
HEALTHBENCH_POLICY_MODEL=google/gemini-2.5-flash-lite \
HEALTHBENCH_JUDGE_MODEL=google/gemini-2.5-flash-lite \
HEALTHBENCH_POLICY_BASE_URL=https://openrouter.ai/api/v1 \
HEALTHBENCH_POLICY_API_KEY_ENV=OPENROUTER_API_KEY \
uv run --project "$GEPA_DIR/healthbench_container" \
  python "$GEPA_DIR/healthbench_container/synth_service_app.py" \
  --host 127.0.0.1 --port 8815 &
HEALTHBENCH_PID=$!
wait_health "HealthBench" 8815

if [[ "$SMOKE" == "false" ]]; then
  echo "[chart-d] Starting tau2 retail container on :8775..."
  OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
  TAU2_RETAIL_AGENT_MODEL=openrouter/google/gemini-3.1-flash-lite \
  TAU2_RETAIL_USER_MODEL=gpt-4.1-nano \
  uv run --project "$GEPA_DIR/tau2_retail_container" \
    python "$GEPA_DIR/tau2_retail_container/synth_service_app.py" \
    --host 127.0.0.1 --port 8775 &
  TAU2_PID=$!
  wait_health "tau2 retail" 8775
fi

# ---------------------------------------------------------------------------
# Helper: run one GEPA config
# ---------------------------------------------------------------------------

run_gepa() {
  local config="$1"
  echo ""
  echo "[chart-d] ---- Running: $config ----"
  uv run --project "$GEPA_EVALS_DIR" synth-optimizers gepa run \
    --config "$config"
  echo "[chart-d] ---- Done: $config ----"
}

RUN_PIDS=()
run_gepa_async() {
  local config="$1"
  local label
  label="$(basename "$config" .toml)"
  (
    run_gepa "$config"
  ) &
  RUN_PIDS+=("$!:$label")
}

wait_gepa_jobs() {
  local failed=0
  local entry pid label
  for entry in "${RUN_PIDS[@]}"; do
    pid="${entry%%:*}"
    label="${entry#*:}"
    if wait "$pid"; then
      echo "[chart-d] PASS: $label"
    else
      echo "[chart-d] FAIL: $label" >&2
      failed=1
    fi
  done
  return "$failed"
}

# ---------------------------------------------------------------------------
# Sweep: nano -> mini -> gpt-5.4 (size ladder), HealthBench then tau2 retail
# ---------------------------------------------------------------------------

echo ""
echo "[chart-d] === Launching proposer size ladder in parallel ==="
run_gepa_async "$CONFIG_DIR/healthbench_gpt-5.4-nano.toml"
run_gepa_async "$CONFIG_DIR/healthbench_gpt-5.4-mini.toml"

if [[ "$SMOKE" == "false" ]]; then
  run_gepa_async "$CONFIG_DIR/healthbench_gpt-5.4.toml"
  run_gepa_async "$CONFIG_DIR/tau2_retail_gpt-5.4-nano.toml"
  run_gepa_async "$CONFIG_DIR/tau2_retail_gpt-5.4-mini.toml"
  run_gepa_async "$CONFIG_DIR/tau2_retail_gpt-5.4.toml"
else
  echo "[chart-d] Smoke mode: skipping full-size gpt-5.4 and all tau2 retail runs."
fi

wait_gepa_jobs

# ---------------------------------------------------------------------------
# Build chart figures
# ---------------------------------------------------------------------------

echo ""
echo "[chart-d] Building chart figures..."
cd "$CHART_D"
uv run python build_chart.py

echo ""
echo "[chart-d] Done. Figures written to $CHART_D/figures/"
