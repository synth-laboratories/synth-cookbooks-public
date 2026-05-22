#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../../../.." && pwd)"
PROFILE_DIR="$SCRIPT_DIR/run_profiles"

# GEPA writes per-run artifacts (events.jsonl, rollouts, traces) under
# `cookbooks/optimizers/gepa/runs/<run_id>/` and a single run can reach
# multiple GiB. Mid-run ENOSPC corrupts the run's jsonl and leaves a
# half-finalized container. Pre-flight a free-space check so a doomed
# launch fails loudly at start instead of partway through. Override the
# threshold with `GEPA_MIN_FREE_MB=...` if you want a tighter or looser
# floor; bypass entirely with `GEPA_SKIP_DISK_CHECK=1` (do this only
# when you know what you're doing).
gepa_runs_root="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)/runs"
mkdir -p "$gepa_runs_root"
gepa_min_free_mb="${GEPA_MIN_FREE_MB:-5120}"
if [[ "${GEPA_SKIP_DISK_CHECK:-0}" != "1" ]]; then
  gepa_free_mb="$(df -m "$gepa_runs_root" | awk 'NR==2 {print $4}')"
  if [[ -z "$gepa_free_mb" || ! "$gepa_free_mb" =~ ^[0-9]+$ ]]; then
    echo "error: could not read free disk space for $gepa_runs_root" >&2
    exit 1
  fi
  if (( gepa_free_mb < gepa_min_free_mb )); then
    echo "error: insufficient disk space for GEPA run" >&2
    echo "  runs_dir=$gepa_runs_root" >&2
    echo "  free=${gepa_free_mb} MiB" >&2
    echo "  required>=${gepa_min_free_mb} MiB (override with GEPA_MIN_FREE_MB)" >&2
    echo "  largest existing runs:" >&2
    du -sh "$gepa_runs_root"/*/ 2>/dev/null | sort -h | tail -5 | sed 's/^/    /' >&2
    echo "  prune with: rm -rf $gepa_runs_root/<run_id>" >&2
    echo "  or bypass:  GEPA_SKIP_DISK_CHECK=1 $0 $*" >&2
    exit 1
  fi
fi

PROFILE_ARG="${1:-default}"
if [[ "$PROFILE_ARG" == "--profile" ]]; then
  PROFILE_ARG="${2:-default}"
fi
if [[ "$PROFILE_ARG" == "--list" ]]; then
  echo "GEPA profiles:"
  find "$PROFILE_DIR" -maxdepth 1 -type f -name '*.toml' -print | sed 's|.*/||; s|\.toml$||' | sort
  exit 0
fi
if [[ "$PROFILE_ARG" == *.toml || "$PROFILE_ARG" == */* ]]; then
  PROFILE_PATH="$PROFILE_ARG"
else
  PROFILE_PATH="$PROFILE_DIR/$PROFILE_ARG.toml"
fi
if [[ ! -f "$PROFILE_PATH" ]]; then
  echo "error: GEPA profile not found: $PROFILE_PATH" >&2
  echo "hint: run 'bash run_fresh_gepa.sh --list'" >&2
  exit 1
fi

toml_get() {
  local section="$1"
  local key="$2"
  local default_value="$3"
  awk -v section="$section" -v key="$key" -v default_value="$default_value" '
    BEGIN { current = "" }
    /^[[:space:]]*\[/ {
      current = $0
      gsub(/^[[:space:]]*\[/, "", current)
      gsub(/\][[:space:]]*$/, "", current)
      next
    }
    current == section {
      line = $0
      sub(/[[:space:]]+#.*/, "", line)
      split(line, parts, "=")
      name = parts[1]
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
      if (name == key) {
        value = substr(line, index(line, "=") + 1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        gsub(/^"/, "", value)
        gsub(/"$/, "", value)
        print value
        found = 1
        exit
      }
    }
    END { if (!found) print default_value }
  ' "$PROFILE_PATH"
}

PROFILE_NAME="$(toml_get profile name "$PROFILE_ARG")"
PROFILE_DESCRIPTION="$(toml_get profile description "")"
BASE_CONFIG_VALUE="$(toml_get base config "gepa.user50_20_100.toml")"
if [[ "$BASE_CONFIG_VALUE" == /* ]]; then
  BASE_CONFIG="$BASE_CONFIG_VALUE"
else
  BASE_CONFIG="$SCRIPT_DIR/$BASE_CONFIG_VALUE"
fi
BASE_RUN_ID="$(toml_get base run_id "banking77_gepa_async_t50_mb20_h100")"
POLICY_PROVIDER="${GEPA_POLICY_PROVIDER:-$(toml_get policy provider "openrouter")}"
POLICY_MODEL="${GEPA_POLICY_MODEL:-$(toml_get policy model "qwen/qwen-2.5-7b-instruct")}"
POLICY_BASE_URL="${GEPA_POLICY_BASE_URL:-$(toml_get policy base_url "https://openrouter.ai/api/v1")}"
POLICY_API_KEY_ENV="${GEPA_POLICY_API_KEY_ENV:-$(toml_get policy api_key_env "OPENROUTER_API_KEY")}"
POLICY_API_MODE="${GEPA_POLICY_API_MODE:-$(toml_get policy api_mode "auto")}"
POLICY_CONCURRENCY="${GEPA_POLICY_CONCURRENCY:-$(toml_get policy concurrency "20")}"
POLICY_RETRIES="${GEPA_POLICY_RETRIES:-$(toml_get policy retries "1")}"
POLICY_MAX_TOKENS="${GEPA_POLICY_MAX_TOKENS:-$(toml_get policy max_tokens "16")}"
POLICY_DISABLE_REASONING="${GEPA_POLICY_DISABLE_REASONING:-$(toml_get policy disable_reasoning "auto")}"
POLICY_TIMEOUT_SECONDS="${GEPA_POLICY_TIMEOUT_SECONDS:-$(toml_get timeouts policy_seconds "20")}"
ROLLOUT_TIMEOUT_SECONDS="${GEPA_ROLLOUT_TIMEOUT_SECONDS:-$(toml_get timeouts rollout_seconds "25")}"
CONTAINER_HTTP_TIMEOUT_SECONDS="${GEPA_CONTAINER_HTTP_TIMEOUT_SECONDS:-$(toml_get timeouts container_http_seconds "30")}"
ROLLOUT_HTTP_RETRIES="${GEPA_ROLLOUT_HTTP_RETRIES:-$(toml_get timeouts rollout_http_retries "0")}"
PROPOSER_MODEL="${GEPA_PROPOSER_MODEL:-$(toml_get proposer model "gpt-5.4-nano")}"
PROPOSER_AUTH_MODE="${GEPA_PROPOSER_AUTH_MODE:-$(toml_get proposer auth_mode "api_key")}"
PROPOSER_API_KEY_ENV="${GEPA_PROPOSER_API_KEY_ENV:-$(toml_get proposer api_key_env "OPENAI_API_KEY")}"
PROPOSER_COPY_HOST_AUTH="${GEPA_PROPOSER_COPY_HOST_AUTH:-$(toml_get proposer copy_host_auth "false")}"
TRAIN_SIZE="${GEPA_TRAIN_SIZE:-$(toml_get dataset train_size "50")}"
HELDOUT_SIZE="${GEPA_HELDOUT_SIZE:-$(toml_get dataset heldout_size "100")}"
TRAIN_SHUFFLE_SEED="${GEPA_TRAIN_SHUFFLE_SEED:-$(toml_get dataset train_shuffle_seed "random")}"
HELDOUT_SHUFFLE_SEED="${GEPA_HELDOUT_SHUFFLE_SEED:-$(toml_get dataset heldout_shuffle_seed "random")}"
MAX_GENERATIONS="${GEPA_MAX_GENERATIONS:-$(toml_get pipeline max_generations "1")}"
PROPOSAL_COUNT="${GEPA_PROPOSALS_PER_GENERATION:-$(toml_get pipeline proposals_per_generation "4")}"
MINIBATCH_SIZE="${GEPA_MINIBATCH_SIZE:-$(toml_get pipeline minibatch_size "20")}"
ROLLOUT_WORKERS="${GEPA_ROLLOUT_WORKERS:-$(toml_get pipeline rollout_workers "20")}"
IN_FLIGHT_CANDIDATES="${GEPA_MAX_IN_FLIGHT_CANDIDATES:-$(toml_get pipeline max_in_flight_candidates "$PROPOSAL_COUNT")}"
ADAPTIVE_ROLLOUT_ENABLED="${GEPA_ADAPTIVE_ROLLOUT_ENABLED:-$(toml_get pipeline.adaptive_rollout_concurrency enabled "true")}"
ADAPTIVE_ROLLOUT_INITIAL="${GEPA_ADAPTIVE_ROLLOUT_INITIAL:-$(toml_get pipeline.adaptive_rollout_concurrency initial "50")}"
ADAPTIVE_ROLLOUT_MIN="${GEPA_ADAPTIVE_ROLLOUT_MIN:-$(toml_get pipeline.adaptive_rollout_concurrency min "1")}"
ADAPTIVE_ROLLOUT_MAX="${GEPA_ADAPTIVE_ROLLOUT_MAX:-$(toml_get pipeline.adaptive_rollout_concurrency max "120")}"
ADAPTIVE_ROLLOUT_INCREASE_STEP="${GEPA_ADAPTIVE_ROLLOUT_INCREASE_STEP:-$(toml_get pipeline.adaptive_rollout_concurrency increase_step "10")}"
ADAPTIVE_ROLLOUT_DECREASE_STEP="${GEPA_ADAPTIVE_ROLLOUT_DECREASE_STEP:-$(toml_get pipeline.adaptive_rollout_concurrency decrease_step "10")}"
ADAPTIVE_ROLLOUT_INCREASE_AFTER="${GEPA_ADAPTIVE_ROLLOUT_INCREASE_AFTER:-$(toml_get pipeline.adaptive_rollout_concurrency increase_after_successes "40")}"
TRAIN_BUDGET="${GEPA_MAX_TRAIN_ROLLOUTS:-$(toml_get budgets train_rollouts "520")}"
HELDOUT_BUDGET="${GEPA_MAX_HELDOUT_ROLLOUTS:-$(toml_get budgets heldout_rollouts "220")}"
CHUNK_SIZE="${SYNTH_OPTIMIZERS_GEPA_ROLLOUT_CHUNK_SIZE:-$(toml_get pipeline rollout_chunk_size "")}"
if [[ "$CHUNK_SIZE" == "auto" ]]; then
  CHUNK_SIZE=""
fi

require_uint() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "error: $name must be a non-negative integer, got: $value" >&2
    exit 1
  fi
}

random_u32() {
  od -An -N4 -tu4 /dev/urandom | tr -d ' '
}

if [[ "$TRAIN_SHUFFLE_SEED" == "random" || "$TRAIN_SHUFFLE_SEED" == "auto" || -z "$TRAIN_SHUFFLE_SEED" ]]; then
  TRAIN_SHUFFLE_SEED="$(random_u32)"
fi
if [[ "$HELDOUT_SHUFFLE_SEED" == "random" || "$HELDOUT_SHUFFLE_SEED" == "auto" || -z "$HELDOUT_SHUFFLE_SEED" ]]; then
  HELDOUT_SHUFFLE_SEED="$(random_u32)"
fi

for numeric_value in \
  "GEPA_TRAIN_SIZE=$TRAIN_SIZE" \
  "GEPA_HELDOUT_SIZE=$HELDOUT_SIZE" \
  "GEPA_TRAIN_SHUFFLE_SEED=$TRAIN_SHUFFLE_SEED" \
  "GEPA_HELDOUT_SHUFFLE_SEED=$HELDOUT_SHUFFLE_SEED" \
  "GEPA_MINIBATCH_SIZE=$MINIBATCH_SIZE" \
  "GEPA_ADAPTIVE_ROLLOUT_INITIAL=$ADAPTIVE_ROLLOUT_INITIAL" \
  "GEPA_ADAPTIVE_ROLLOUT_MIN=$ADAPTIVE_ROLLOUT_MIN" \
  "GEPA_ADAPTIVE_ROLLOUT_MAX=$ADAPTIVE_ROLLOUT_MAX" \
  "GEPA_ADAPTIVE_ROLLOUT_INCREASE_STEP=$ADAPTIVE_ROLLOUT_INCREASE_STEP" \
  "GEPA_ADAPTIVE_ROLLOUT_DECREASE_STEP=$ADAPTIVE_ROLLOUT_DECREASE_STEP" \
  "GEPA_ADAPTIVE_ROLLOUT_INCREASE_AFTER=$ADAPTIVE_ROLLOUT_INCREASE_AFTER"
do
  require_uint "${numeric_value%%=*}" "${numeric_value#*=}"
done

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "error: base config not found: $BASE_CONFIG" >&2
  exit 1
fi
if [[ -z "${!POLICY_API_KEY_ENV:-}" ]]; then
  for env_file in \
    "$SCRIPT_DIR/.env" \
    "$REPO_ROOT/../synth-ai/.env" \
    "$REPO_ROOT/../synth-dev/.env.shared" \
    "$REPO_ROOT/../backend/.env.local"
  do
    if [[ -f "$env_file" ]]; then
      env_value="$(KEY="$POLICY_API_KEY_ENV" awk -F= '$1 == ENVIRON["KEY"] { sub(/^[^=]*=/, ""); print; exit }' "$env_file")"
      env_value="${env_value%\"}"
      env_value="${env_value#\"}"
      if [[ -n "$env_value" ]]; then
        export "$POLICY_API_KEY_ENV=$env_value"
        echo "GEPA loaded $POLICY_API_KEY_ENV from $env_file"
        break
      fi
    fi
  done
fi
if [[ "$POLICY_PROVIDER" == "openrouter" && -z "${!POLICY_API_KEY_ENV:-}" ]]; then
  echo "error: $POLICY_API_KEY_ENV is not set; OpenRouter policy rollouts need an API key" >&2
  exit 1
fi
if [[ "$PROPOSER_AUTH_MODE" == "api_key" && -z "${!PROPOSER_API_KEY_ENV:-}" ]]; then
  for env_file in \
    "$SCRIPT_DIR/.env" \
    "$REPO_ROOT/../synth-ai/.env" \
    "$REPO_ROOT/../synth-dev/.env.shared" \
    "$REPO_ROOT/../backend/.env.local"
  do
    if [[ -f "$env_file" ]]; then
      env_value="$(KEY="$PROPOSER_API_KEY_ENV" awk -F= '$1 == ENVIRON["KEY"] { sub(/^[^=]*=/, ""); print; exit }' "$env_file")"
      env_value="${env_value%\"}"
      env_value="${env_value#\"}"
      if [[ -n "$env_value" ]]; then
        export "$PROPOSER_API_KEY_ENV=$env_value"
        echo "GEPA loaded $PROPOSER_API_KEY_ENV from $env_file"
        break
      fi
    fi
  done
fi
if [[ "$PROPOSER_AUTH_MODE" == "api_key" && -z "${!PROPOSER_API_KEY_ENV:-}" ]]; then
  echo "error: $PROPOSER_API_KEY_ENV is not set; Codex proposer api_key auth needs an API key" >&2
  exit 1
fi

SUFFIX="$(uuidgen 2>/dev/null | tr '[:upper:]' '[:lower:]' | cut -c1-8 || true)"
if [[ -z "$SUFFIX" ]]; then
  SUFFIX="$(date +%Y%m%d%H%M%S)-$$"
fi

RUN_ID="${BASE_RUN_ID}_${SUFFIX}"
CONFIG="$SCRIPT_DIR/gepa.${RUN_ID}.toml"
CONTAINER_PORT="${GEPA_CONTAINER_PORT:-}"
if [[ -z "$CONTAINER_PORT" ]]; then
  PORT_DIGITS="$(printf '%s' "$SUFFIX" | tr -cd '0-9' | cut -c1-5)"
  if [[ -z "$PORT_DIGITS" ]]; then
    PORT_DIGITS="$(date +%S%M)"
  fi
  CONTAINER_PORT="$((20000 + 10#$PORT_DIGITS % 20000))"
fi
if [[ ! "$CONTAINER_PORT" =~ ^[0-9]+$ ]]; then
  echo "error: GEPA container port is not numeric: $CONTAINER_PORT" >&2
  exit 1
fi
CONTAINER_URL="http://127.0.0.1:$CONTAINER_PORT"

set_toml_value() {
  local section="$1"
  local key="$2"
  local value="$3"
  SECTION="$section" KEY="$key" VALUE="$value" perl -0pi -e 'my $section = $ENV{SECTION}; my $key = $ENV{KEY}; my $value = $ENV{VALUE}; if (!s/(^\[\Q$section\E\]\n(?:(?!^\[).)*?^\Q$key\E[[:space:]]*=[[:space:]]*)[^\n]*/$1 . "\"" . $value . "\""/mse) { s/(^\[\Q$section\E\]\n)/$1 . $key . " = \"" . $value . "\"\n"/mse; }' "$CONFIG"
}

ensure_toml_section() {
  local section="$1"
  if ! grep -Eq "^\\[$section\\][[:space:]]*$" "$CONFIG"; then
    printf '\n[%s]\n' "$section" >> "$CONFIG"
  fi
}

set_toml_raw_value() {
  local section="$1"
  local key="$2"
  local value="$3"
  ensure_toml_section "$section"
  SECTION="$section" KEY="$key" VALUE="$value" perl -0pi -e 'my $section = $ENV{SECTION}; my $key = $ENV{KEY}; my $value = $ENV{VALUE}; if (!s/(^\[\Q$section\E\]\n(?:(?!^\[).)*?^\Q$key\E[[:space:]]*=[[:space:]]*)[^\n]*/$1 . $value/mse) { s/(^\[\Q$section\E\]\n)/$1 . $key . " = " . $value . "\n"/mse; }' "$CONFIG"
}

set_command_env() {
  local name="$1"
  local value="$2"
  NAME="$name" VALUE="$value" perl -0pi -e 'my $name = $ENV{NAME}; my $value = $ENV{VALUE}; if (!s/"\Q$name\E=[^"]*"/"\"" . $name . "=" . $value . "\""/e) { s/("\/usr\/bin\/env",\n)/$1 . "  \"" . $name . "=" . $value . "\",\n"/e; }' "$CONFIG"
}

seed_array_literal() {
  local count="$1"
  local result=$'[\n'
  local line=""
  local i=0
  while (( i < count )); do
    if [[ -n "$line" ]]; then
      line="$line, "
    fi
    line="$line$i"
    if (( (i + 1) % 10 == 0 || i + 1 == count )); then
      result="$result  $line"
      if (( i + 1 < count )); then
        result="$result,"
      fi
      result="$result"$'\n'
      line=""
    fi
    i=$((i + 1))
  done
  result="$result]"
  printf '%s' "$result"
}

replace_toml_array() {
  local key="$1"
  local value="$2"
  KEY="$key" VALUE="$value" perl -0pi -e 'my $key = $ENV{KEY}; my $value = $ENV{VALUE}; if (!s/(^\Q$key\E[[:space:]]*=[[:space:]]*)\[[^\]]*\]/$1 . $value/mse) { die "missing TOML array: $key\n"; }' "$CONFIG"
}

cp "$BASE_CONFIG" "$CONFIG"
set_toml_value "run" "run_id" "$RUN_ID"
set_toml_value "cache" "namespace" "$RUN_ID"
perl -0pi -e "s|^url[[:space:]]*=.*$|url = \"$CONTAINER_URL\"|m" "$CONFIG"
perl -0pi -e "s/\"--port\", \"[0-9]+\"/\"--port\", \"$CONTAINER_PORT\"/g" "$CONFIG"
set_toml_value "policy" "provider" "$POLICY_PROVIDER"
set_toml_value "policy" "model" "$POLICY_MODEL"
set_toml_value "policy" "base_url" "$POLICY_BASE_URL"
set_toml_value "policy" "api_key_env" "$POLICY_API_KEY_ENV"
set_toml_value "proposer" "model" "$PROPOSER_MODEL"
set_toml_value "proposer" "auth_mode" "$PROPOSER_AUTH_MODE"
set_toml_value "proposer" "api_key_env" "$PROPOSER_API_KEY_ENV"
set_toml_raw_value "proposer" "copy_host_auth" "$PROPOSER_COPY_HOST_AUTH"
set_command_env "BANKING77_POLICY_MODEL" "$POLICY_MODEL"
set_command_env "BANKING77_POLICY_BASE_URL" "$POLICY_BASE_URL"
set_command_env "BANKING77_POLICY_TIMEOUT_SECONDS" "$POLICY_TIMEOUT_SECONDS"
set_command_env "BANKING77_ROLLOUT_TIMEOUT_SECONDS" "$ROLLOUT_TIMEOUT_SECONDS"
set_command_env "BANKING77_POLICY_API_MODE" "$POLICY_API_MODE"
set_command_env "BANKING77_POLICY_CONCURRENCY" "$POLICY_CONCURRENCY"
set_command_env "BANKING77_POLICY_RETRIES" "$POLICY_RETRIES"
set_command_env "BANKING77_POLICY_MAX_TOKENS" "$POLICY_MAX_TOKENS"
set_command_env "BANKING77_POLICY_DISABLE_REASONING" "$POLICY_DISABLE_REASONING"
set_command_env "BANKING77_TRAIN_SAMPLE" "$TRAIN_SIZE"
set_command_env "BANKING77_TEST_SAMPLE" "$HELDOUT_SIZE"
set_command_env "BANKING77_TRAIN_SHUFFLE_SEED" "$TRAIN_SHUFFLE_SEED"
set_command_env "BANKING77_TEST_SHUFFLE_SEED" "$HELDOUT_SHUFFLE_SEED"
replace_toml_array "train_seeds" "$(seed_array_literal "$TRAIN_SIZE")"
replace_toml_array "heldout_seeds" "$(seed_array_literal "$HELDOUT_SIZE")"
perl -0pi -e "s/^proposals_per_generation[[:space:]]*=.*$/proposals_per_generation = $PROPOSAL_COUNT/m" "$CONFIG"
perl -0pi -e "s/^max_generations[[:space:]]*=.*$/max_generations = $MAX_GENERATIONS/m" "$CONFIG"
perl -0pi -e "s/^minibatch_size[[:space:]]*=.*$/minibatch_size = $MINIBATCH_SIZE/m" "$CONFIG"
perl -0pi -e "s/^max_in_flight_candidates[[:space:]]*=.*$/max_in_flight_candidates = $IN_FLIGHT_CANDIDATES/m" "$CONFIG"
perl -0pi -e "s/^rollout[[:space:]]*=.*$/rollout = $ROLLOUT_WORKERS/m" "$CONFIG"
set_toml_raw_value "gepa.pipeline.adaptive_rollout_concurrency" "enabled" "$ADAPTIVE_ROLLOUT_ENABLED"
set_toml_raw_value "gepa.pipeline.adaptive_rollout_concurrency" "initial" "$ADAPTIVE_ROLLOUT_INITIAL"
set_toml_raw_value "gepa.pipeline.adaptive_rollout_concurrency" "min" "$ADAPTIVE_ROLLOUT_MIN"
set_toml_raw_value "gepa.pipeline.adaptive_rollout_concurrency" "max" "$ADAPTIVE_ROLLOUT_MAX"
set_toml_raw_value "gepa.pipeline.adaptive_rollout_concurrency" "increase_step" "$ADAPTIVE_ROLLOUT_INCREASE_STEP"
set_toml_raw_value "gepa.pipeline.adaptive_rollout_concurrency" "decrease_step" "$ADAPTIVE_ROLLOUT_DECREASE_STEP"
set_toml_raw_value "gepa.pipeline.adaptive_rollout_concurrency" "increase_after_successes" "$ADAPTIVE_ROLLOUT_INCREASE_AFTER"
if ! grep -Eq '^proposals_per_generation[[:space:]]*=' "$CONFIG"; then
  perl -0pi -e "s/(^max_generations[[:space:]]*=[[:space:]]*[0-9]+[[:space:]]*\n)/\$1proposals_per_generation = $PROPOSAL_COUNT\n/m" "$CONFIG"
fi
if ! grep -Eq '^max_in_flight_candidates[[:space:]]*=' "$CONFIG"; then
  perl -0pi -e "s/(^\\[gepa\\.pipeline\\][[:space:]]*\n)/\$1max_in_flight_candidates = $IN_FLIGHT_CANDIDATES\n/m" "$CONFIG"
fi
perl -0pi -e "s/^max_train_rollouts[[:space:]]*=.*$/max_train_rollouts = $TRAIN_BUDGET/m" "$CONFIG"
perl -0pi -e "s/^max_heldout_rollouts[[:space:]]*=.*$/max_heldout_rollouts = $HELDOUT_BUDGET/m" "$CONFIG"
if ! grep -Eq '^max_train_rollouts[[:space:]]*=' "$CONFIG"; then
  perl -0pi -e "s/(^max_total_rollouts[[:space:]]*=[[:space:]]*[0-9]+[[:space:]]*\n)/\$1max_train_rollouts = $TRAIN_BUDGET\nmax_heldout_rollouts = $HELDOUT_BUDGET\n/m" "$CONFIG"
fi
if ! grep -Eq '^max_heldout_rollouts[[:space:]]*=' "$CONFIG"; then
  perl -0pi -e "s/(^max_train_rollouts[[:space:]]*=[[:space:]]*[0-9]+[[:space:]]*\n)/\$1max_heldout_rollouts = $HELDOUT_BUDGET\n/m" "$CONFIG"
fi

echo "GEPA config: $CONFIG"
echo "GEPA profile: $PROFILE_NAME"
if [[ -n "$PROFILE_DESCRIPTION" ]]; then
  echo "GEPA profile_description: $PROFILE_DESCRIPTION"
fi
echo "GEPA profile_toml: $PROFILE_PATH"
echo "GEPA base_config: $BASE_CONFIG"
echo "GEPA run_id: $RUN_ID"
echo "GEPA container_url: $CONTAINER_URL"
echo "GEPA container: banking77_container"
echo "GEPA dataset: train=$TRAIN_SIZE heldout=$HELDOUT_SIZE sampler=balanced_random_per_label"
echo "GEPA dataset_shuffle_seeds: train=$TRAIN_SHUFFLE_SEED heldout=$HELDOUT_SHUFFLE_SEED"
echo "GEPA policy: provider=$POLICY_PROVIDER model=$POLICY_MODEL base_url=$POLICY_BASE_URL api_key_env=$POLICY_API_KEY_ENV"
echo "GEPA policy_timeout_seconds: $POLICY_TIMEOUT_SECONDS"
echo "GEPA rollout_timeout_seconds: $ROLLOUT_TIMEOUT_SECONDS"
echo "GEPA container_http_timeout_seconds: $CONTAINER_HTTP_TIMEOUT_SECONDS"
echo "GEPA policy_api_mode: $POLICY_API_MODE"
echo "GEPA policy_concurrency: $POLICY_CONCURRENCY"
echo "GEPA policy_retries: $POLICY_RETRIES"
echo "GEPA rollout_http_retries: $ROLLOUT_HTTP_RETRIES"
echo "GEPA policy_max_tokens: $POLICY_MAX_TOKENS"
echo "GEPA policy_disable_reasoning: $POLICY_DISABLE_REASONING"
echo "GEPA proposer_model: $PROPOSER_MODEL"
echo "GEPA proposer_auth: mode=$PROPOSER_AUTH_MODE api_key_env=$PROPOSER_API_KEY_ENV copy_host_auth=$PROPOSER_COPY_HOST_AUTH"
echo "GEPA max_generations: $MAX_GENERATIONS"
echo "GEPA proposals_per_generation: $PROPOSAL_COUNT"
echo "GEPA minibatch_size: $MINIBATCH_SIZE"
echo "GEPA max_in_flight_candidates: $IN_FLIGHT_CANDIDATES"
echo "GEPA rollout_workers: $ROLLOUT_WORKERS"
echo "GEPA adaptive_rollout_concurrency: enabled=$ADAPTIVE_ROLLOUT_ENABLED initial=$ADAPTIVE_ROLLOUT_INITIAL min=$ADAPTIVE_ROLLOUT_MIN max=$ADAPTIVE_ROLLOUT_MAX up=$ADAPTIVE_ROLLOUT_INCREASE_STEP down=$ADAPTIVE_ROLLOUT_DECREASE_STEP after=$ADAPTIVE_ROLLOUT_INCREASE_AFTER"
if [[ -n "$CHUNK_SIZE" ]]; then
  echo "GEPA rollout_chunk_size: $CHUNK_SIZE"
else
  echo "GEPA rollout_chunk_size: auto"
fi
echo "GEPA rollout budgets: train=$TRAIN_BUDGET heldout=$HELDOUT_BUDGET"

cd "$SCRIPT_DIR"
export SYNTH_OPTIMIZERS_CONTAINER_HTTP_TIMEOUT_SECONDS="$CONTAINER_HTTP_TIMEOUT_SECONDS"
export SYNTH_OPTIMIZERS_GEPA_ROLLOUT_HTTP_RETRIES="$ROLLOUT_HTTP_RETRIES"
if [[ -n "$CHUNK_SIZE" ]]; then
  export SYNTH_OPTIMIZERS_GEPA_ROLLOUT_CHUNK_SIZE="$CHUNK_SIZE"
fi
exec uv run --project "$REPO_ROOT/packages/synth-optimizers" --group dev --reinstall-package synth-optimizers synth-optimizers gepa run --config "$CONFIG"
