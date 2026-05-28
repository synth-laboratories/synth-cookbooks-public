#!/usr/bin/env bash
# Run Synth GEPA from a single self-contained config TOML.
#
#   ./run_gepa.sh --cfg configs/minigrid_concurrent.toml
#
# The TOML fully specifies the run (container command + env, policy, proposer,
# dataset, budgets, concurrency, cache). This script handles the operational
# gotchas so a run is reproducible:
#   - sources provider keys (OPENAI_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY)
#   - kills any stale container from the cfg (else the runner reuses a live port
#     and config changes silently don't apply)
#   - stamps a fresh run_id + cache namespace per invocation (avoids stale-lease
#     and stale-cache reuse)
#   - runs synth-optimizers from the pinned public org repo
#   - streams the live GEPA terminal visualizer (SYNTH_OPTIMIZERS_TERMINAL=1)
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)"
OPTIMIZERS_SPEC="synth-optimizers @ git+https://github.com/synth-laboratories/optimizers.git@d5acd9c3464fd28fda0a0d417b0cd4418152ab62"

CFG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg) CFG="$2"; shift 2 ;;
    --cfg=*) CFG="${1#--cfg=}"; shift ;;
    *) echo "unknown arg: $1" >&2; echo "usage: $0 --cfg <config.toml>" >&2; exit 2 ;;
  esac
done
if [[ -z "$CFG" ]]; then
  echo "usage: $0 --cfg <config.toml>" >&2
  echo "configs:" >&2; ls "$SCRIPT_DIR"/configs/*.toml 2>/dev/null | sed 's/^/  /' >&2 || true
  exit 2
fi
[[ "$CFG" = /* ]] || CFG="$(CDPATH= cd -- "$(pwd)" && pwd)/$CFG"
[[ -f "$CFG" ]] || CFG="$SCRIPT_DIR/$CFG"
[[ -f "$CFG" ]] || { echo "config not found: $CFG" >&2; exit 2; }

# provider keys
ENV_FILE="${SYNTH_ENV_FILE:-$REPO_ROOT/../synth-ai/.env}"
[[ -f "$ENV_FILE" ]] && { set -a; source "$ENV_FILE"; set +a; }

# kill any stale container referenced by the cfg (match its synth_service_app path)
APP="$(grep -oE '[A-Za-z0-9_./-]*synth_service_app\.py' "$CFG" | head -1 || true)"
if [[ -n "$APP" ]]; then
  pkill -f "$APP" 2>/dev/null || true
  sleep 2
fi

# fresh per-run copy: timestamped run_id + cache namespace. Written INTO the gepa
# dir (not /tmp) so the config's relative [container].cwd / output_dir / cache
# paths resolve against the cookbook, not the temp dir.
STAMP="$(date +%Y%m%d%H%M%S)"
RUNCFG="$SCRIPT_DIR/.gepa_run_${STAMP}.toml"
trap 'rm -f "$RUNCFG"' EXIT
sed -E "s/(run_id = \"[^\"]*)\"/\1_${STAMP}\"/; s/(namespace = \"[^\"]*)\"/\1_${STAMP}\"/" "$CFG" > "$RUNCFG"

export PATH="$HOME/.cargo/bin:$PATH"   # git install builds the Rust extension when needed
RUNNER=(uv run --with "$OPTIMIZERS_SPEC" synth-optimizers)

echo "GEPA config : $CFG"
echo "run config  : $RUNCFG  (run_id stamped ${STAMP})"
echo "runner      : ${RUNNER[*]}"
cd "$SCRIPT_DIR"
exec env SYNTH_OPTIMIZERS_TERMINAL=1 "${RUNNER[@]}" gepa run --config "$RUNCFG"
