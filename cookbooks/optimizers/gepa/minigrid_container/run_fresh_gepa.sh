#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../../../.." && pwd)"
PROFILE_DIR="$SCRIPT_DIR/run_profiles"
CONTAINER_NAME="$(basename "$SCRIPT_DIR")"

case "$CONTAINER_NAME" in
  minigrid_container)
    DEFAULT_RUN_ID="minigrid_gepa_public"
    DEFAULT_POLICY_MODEL="gpt-4.1-nano"
    DEFAULT_TRAIN_SIZE="4"
    DEFAULT_HELDOUT_SIZE="2"
    DEFAULT_TRAIN_SEED_START="1"
    DEFAULT_HELDOUT_SEED_START="100"
    TASK_LABEL="MiniGrid"
    ;;
  tblite_container)
    DEFAULT_RUN_ID="tblite_gepa_public"
    DEFAULT_POLICY_MODEL="gpt-4.1-nano"
    DEFAULT_TRAIN_SIZE="3"
    DEFAULT_HELDOUT_SIZE="2"
    DEFAULT_TRAIN_SEED_START="0"
    DEFAULT_HELDOUT_SEED_START="100"
    TASK_LABEL="TBLite"
    ;;
  crafter_container)
    DEFAULT_RUN_ID="crafter_gepa_public"
    DEFAULT_POLICY_MODEL="gpt-4.1-nano"
    DEFAULT_TRAIN_SIZE="4"
    DEFAULT_HELDOUT_SIZE="2"
    DEFAULT_TRAIN_SEED_START="11"
    DEFAULT_HELDOUT_SEED_START="101"
    TASK_LABEL="Crafter"
    ;;
  *)
    echo "error: unsupported GEPA container runner location: $CONTAINER_NAME" >&2
    exit 1
    ;;
esac

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

ensure_toml_section() {
  local section="$1"
  if ! grep -Eq "^\\[$section\\][[:space:]]*$" "$CONFIG"; then
    printf '\n[%s]\n' "$section" >> "$CONFIG"
  fi
}

set_toml_value() {
  local section="$1"
  local key="$2"
  local value="$3"
  ensure_toml_section "$section"
  SECTION="$section" KEY="$key" VALUE="$value" perl -0pi -e 'my $section = $ENV{SECTION}; my $key = $ENV{KEY}; my $value = $ENV{VALUE}; if (!s/(^\[\Q$section\E\]\n(?:(?!^\[).)*?^\Q$key\E[[:space:]]*=[[:space:]]*)[^\n]*/$1 . "\"" . $value . "\""/mse) { s/(^\[\Q$section\E\]\n)/$1 . $key . " = \"" . $value . "\"\n"/mse; }' "$CONFIG"
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

seed_array_range_literal() {
  local start="$1"
  local count="$2"
  local result=$'[\n'
  local line=""
  local i=0
  local value=0
  while (( i < count )); do
    value=$((start + i))
    if [[ -n "$line" ]]; then
      line="$line, "
    fi
    line="$line$value"
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

load_api_key_if_needed() {
  local key_name="$1"
  if [[ -n "${!key_name:-}" ]]; then
    return
  fi
  local env_file=""
  local env_value=""
  for env_file in \
    "$SCRIPT_DIR/.env" \
    "$REPO_ROOT/../synth-ai/.env" \
    "$REPO_ROOT/../synth-dev/.env.shared" \
    "$REPO_ROOT/../backend/.env.local"
  do
    if [[ -f "$env_file" ]]; then
      env_value="$(KEY="$key_name" awk -F= '$1 == ENVIRON["KEY"] { sub(/^[^=]*=/, ""); print; exit }' "$env_file")"
      env_value="${env_value%\"}"
      env_value="${env_value#\"}"
      if [[ -n "$env_value" ]]; then
        export "$key_name=$env_value"
        echo "GEPA loaded $key_name from $env_file"
        return
      fi
    fi
  done
}

PROFILE_NAME="$(toml_get profile name "$PROFILE_ARG")"
PROFILE_DESCRIPTION="$(toml_get profile description "")"
BASE_CONFIG_VALUE="$(toml_get base config "gepa.toml")"
if [[ "$BASE_CONFIG_VALUE" == /* ]]; then
  BASE_CONFIG="$BASE_CONFIG_VALUE"
else
  BASE_CONFIG="$SCRIPT_DIR/$BASE_CONFIG_VALUE"
fi
BASE_RUN_ID="$(toml_get base run_id "$DEFAULT_RUN_ID")"
POLICY_PROVIDER="${GEPA_POLICY_PROVIDER:-$(toml_get policy provider "openai")}"
POLICY_MODEL="${GEPA_POLICY_MODEL:-$(toml_get policy model "$DEFAULT_POLICY_MODEL")}"
POLICY_API_KEY_ENV="${GEPA_POLICY_API_KEY_ENV:-$(toml_get policy api_key_env "OPENAI_API_KEY")}"
PROPOSER_MODEL="${GEPA_PROPOSER_MODEL:-$(toml_get proposer model "gpt-5.4-nano")}"
PROPOSER_AUTH_MODE="${GEPA_PROPOSER_AUTH_MODE:-$(toml_get proposer auth_mode "api_key")}"
PROPOSER_API_KEY_ENV="${GEPA_PROPOSER_API_KEY_ENV:-$(toml_get proposer api_key_env "OPENAI_API_KEY")}"
PROPOSER_COPY_HOST_AUTH="${GEPA_PROPOSER_COPY_HOST_AUTH:-$(toml_get proposer copy_host_auth "false")}"
TRAIN_SIZE="${GEPA_TRAIN_SIZE:-$(toml_get dataset train_size "$DEFAULT_TRAIN_SIZE")}"
HELDOUT_SIZE="${GEPA_HELDOUT_SIZE:-$(toml_get dataset heldout_size "$DEFAULT_HELDOUT_SIZE")}"
TRAIN_SEED_START="${GEPA_TRAIN_SEED_START:-$(toml_get dataset train_seed_start "$DEFAULT_TRAIN_SEED_START")}"
HELDOUT_SEED_START="${GEPA_HELDOUT_SEED_START:-$(toml_get dataset heldout_seed_start "$DEFAULT_HELDOUT_SEED_START")}"
MAX_GENERATIONS="${GEPA_MAX_GENERATIONS:-$(toml_get pipeline max_generations "1")}"
PROPOSAL_COUNT="${GEPA_PROPOSALS_PER_GENERATION:-$(toml_get pipeline proposals_per_generation "1")}"
MINIBATCH_SIZE="${GEPA_MINIBATCH_SIZE:-$(toml_get pipeline minibatch_size "$TRAIN_SIZE")}"
ROLLOUT_WORKERS="${GEPA_ROLLOUT_WORKERS:-$(toml_get pipeline rollout_workers "10")}"
IN_FLIGHT_CANDIDATES="${GEPA_MAX_IN_FLIGHT_CANDIDATES:-$(toml_get pipeline max_in_flight_candidates "$PROPOSAL_COUNT")}"
TRAIN_BUDGET="${GEPA_MAX_TRAIN_ROLLOUTS:-$(toml_get budgets train_rollouts "16")}"
HELDOUT_BUDGET="${GEPA_MAX_HELDOUT_ROLLOUTS:-$(toml_get budgets heldout_rollouts "4")}"
CONTAINER_HTTP_TIMEOUT_SECONDS="${GEPA_CONTAINER_HTTP_TIMEOUT_SECONDS:-$(toml_get timeouts container_http_seconds "60")}"
ROLLOUT_ASYNC_TIMEOUT_SECONDS="${GEPA_ROLLOUT_ASYNC_TIMEOUT_SECONDS:-$(toml_get timeouts rollout_async_seconds "$CONTAINER_HTTP_TIMEOUT_SECONDS")}"
ROLLOUT_HTTP_RETRIES="${GEPA_ROLLOUT_HTTP_RETRIES:-$(toml_get timeouts rollout_http_retries "0")}"
CHUNK_SIZE="${SYNTH_OPTIMIZERS_GEPA_ROLLOUT_CHUNK_SIZE:-$(toml_get pipeline rollout_chunk_size "")}"
ADAPTIVE_ENABLED="${GEPA_ADAPTIVE_ROLLOUT_ENABLED:-$(toml_get pipeline.adaptive_rollout_concurrency enabled "true")}"
ADAPTIVE_INITIAL="${GEPA_ADAPTIVE_ROLLOUT_INITIAL:-$(toml_get pipeline.adaptive_rollout_concurrency initial "$ROLLOUT_WORKERS")}"
ADAPTIVE_MIN="${GEPA_ADAPTIVE_ROLLOUT_MIN:-$(toml_get pipeline.adaptive_rollout_concurrency min "1")}"
ADAPTIVE_MAX="${GEPA_ADAPTIVE_ROLLOUT_MAX:-$(toml_get pipeline.adaptive_rollout_concurrency max "$ROLLOUT_WORKERS")}"
ADAPTIVE_UP="${GEPA_ADAPTIVE_ROLLOUT_INCREASE_STEP:-$(toml_get pipeline.adaptive_rollout_concurrency increase_step "5")}"
ADAPTIVE_DOWN="${GEPA_ADAPTIVE_ROLLOUT_DECREASE_STEP:-$(toml_get pipeline.adaptive_rollout_concurrency decrease_step "5")}"
ADAPTIVE_AFTER="${GEPA_ADAPTIVE_ROLLOUT_INCREASE_AFTER:-$(toml_get pipeline.adaptive_rollout_concurrency increase_after_successes "20")}"

if [[ "$TRAIN_SEED_START" == "random" || "$TRAIN_SEED_START" == "auto" || -z "$TRAIN_SEED_START" ]]; then
  TRAIN_SEED_START="$(random_u32)"
fi
if [[ "$HELDOUT_SEED_START" == "random" || "$HELDOUT_SEED_START" == "auto" || -z "$HELDOUT_SEED_START" ]]; then
  HELDOUT_SEED_START="$(random_u32)"
fi

for numeric_value in \
  "GEPA_TRAIN_SIZE=$TRAIN_SIZE" \
  "GEPA_HELDOUT_SIZE=$HELDOUT_SIZE" \
  "GEPA_TRAIN_SEED_START=$TRAIN_SEED_START" \
  "GEPA_HELDOUT_SEED_START=$HELDOUT_SEED_START" \
  "GEPA_MINIBATCH_SIZE=$MINIBATCH_SIZE" \
  "GEPA_ROLLOUT_WORKERS=$ROLLOUT_WORKERS" \
  "GEPA_MAX_IN_FLIGHT_CANDIDATES=$IN_FLIGHT_CANDIDATES" \
  "GEPA_MAX_TRAIN_ROLLOUTS=$TRAIN_BUDGET" \
  "GEPA_MAX_HELDOUT_ROLLOUTS=$HELDOUT_BUDGET"
do
  require_uint "${numeric_value%%=*}" "${numeric_value#*=}"
done

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "error: base config not found: $BASE_CONFIG" >&2
  exit 1
fi

load_api_key_if_needed "$POLICY_API_KEY_ENV"
load_api_key_if_needed "$PROPOSER_API_KEY_ENV"
if [[ -z "${!POLICY_API_KEY_ENV:-}" ]]; then
  echo "error: $POLICY_API_KEY_ENV is not set; live policy rollouts need an API key" >&2
  exit 1
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
CONTAINER_URL="http://127.0.0.1:$CONTAINER_PORT"

cp "$BASE_CONFIG" "$CONFIG"
perl -0pi -e "s/\\Q$BASE_RUN_ID\\E/$RUN_ID/g" "$CONFIG"
perl -0pi -e "s|^url[[:space:]]*=.*$|url = \"$CONTAINER_URL\"|m" "$CONFIG"
perl -0pi -e "s/\"--port\", \"[0-9]+\"/\"--port\", \"$CONTAINER_PORT\"/g" "$CONFIG"
set_toml_value "policy" "provider" "$POLICY_PROVIDER"
set_toml_value "policy" "model" "$POLICY_MODEL"
set_toml_value "policy" "api_key_env" "$POLICY_API_KEY_ENV"
set_toml_value "proposer" "model" "$PROPOSER_MODEL"
set_toml_value "proposer" "auth_mode" "$PROPOSER_AUTH_MODE"
set_toml_value "proposer" "api_key_env" "$PROPOSER_API_KEY_ENV"
set_toml_raw_value "proposer" "copy_host_auth" "$PROPOSER_COPY_HOST_AUTH"
replace_toml_array "train_seeds" "$(seed_array_range_literal "$TRAIN_SEED_START" "$TRAIN_SIZE")"
replace_toml_array "heldout_seeds" "$(seed_array_range_literal "$HELDOUT_SEED_START" "$HELDOUT_SIZE")"
set_toml_raw_value "gepa" "max_generations" "$MAX_GENERATIONS"
set_toml_raw_value "gepa" "proposals_per_generation" "$PROPOSAL_COUNT"
set_toml_raw_value "gepa" "minibatch_size" "$MINIBATCH_SIZE"
set_toml_raw_value "gepa" "max_total_rollouts" "$((TRAIN_BUDGET + HELDOUT_BUDGET))"
set_toml_raw_value "gepa" "max_train_rollouts" "$TRAIN_BUDGET"
set_toml_raw_value "gepa" "max_heldout_rollouts" "$HELDOUT_BUDGET"
set_toml_raw_value "gepa" "rollout_async_timeout_seconds" "$ROLLOUT_ASYNC_TIMEOUT_SECONDS"
set_toml_raw_value "gepa.pipeline" "max_in_flight_candidates" "$IN_FLIGHT_CANDIDATES"
set_toml_raw_value "gepa.pipeline.workers" "rollout" "$ROLLOUT_WORKERS"
set_toml_raw_value "gepa.pipeline.adaptive_rollout_concurrency" "enabled" "$ADAPTIVE_ENABLED"
set_toml_raw_value "gepa.pipeline.adaptive_rollout_concurrency" "initial" "$ADAPTIVE_INITIAL"
set_toml_raw_value "gepa.pipeline.adaptive_rollout_concurrency" "min" "$ADAPTIVE_MIN"
set_toml_raw_value "gepa.pipeline.adaptive_rollout_concurrency" "max" "$ADAPTIVE_MAX"
set_toml_raw_value "gepa.pipeline.adaptive_rollout_concurrency" "increase_step" "$ADAPTIVE_UP"
set_toml_raw_value "gepa.pipeline.adaptive_rollout_concurrency" "decrease_step" "$ADAPTIVE_DOWN"
set_toml_raw_value "gepa.pipeline.adaptive_rollout_concurrency" "increase_after_successes" "$ADAPTIVE_AFTER"
set_toml_value "cache" "namespace" "$RUN_ID"

if [[ "$CONTAINER_NAME" == "minigrid_container" ]]; then
  MINIGRID_MAX_STEPS="${GEPA_MINIGRID_MAX_STEPS:-$(toml_get task max_steps "48")}"
  MINIGRID_ENV_ID="${GEPA_MINIGRID_ENV_ID:-$(toml_get task env_id "MiniGrid-DoorKey-5x5-v0")}"
  require_uint "GEPA_MINIGRID_MAX_STEPS" "$MINIGRID_MAX_STEPS"
  set_command_env "MINIGRID_POLICY_MODEL" "$POLICY_MODEL"
  set_command_env "MINIGRID_MAX_STEPS" "$MINIGRID_MAX_STEPS"
  set_command_env "MINIGRID_ENV_ID" "$MINIGRID_ENV_ID"
fi
if [[ "$CONTAINER_NAME" == "tblite_container" ]]; then
  TBLITE_TEST_TIMEOUT_SECONDS="${GEPA_TBLITE_TEST_TIMEOUT_SECONDS:-$(toml_get task test_timeout_seconds "30")}"
  require_uint "GEPA_TBLITE_TEST_TIMEOUT_SECONDS" "$TBLITE_TEST_TIMEOUT_SECONDS"
  set_command_env "TBLITE_POLICY_MODEL" "$POLICY_MODEL"
  set_command_env "TBLITE_TEST_TIMEOUT_SECONDS" "$TBLITE_TEST_TIMEOUT_SECONDS"
fi
if [[ "$CONTAINER_NAME" == "crafter_container" ]]; then
  CRAFTER_MAX_TURNS="${GEPA_CRAFTER_MAX_TURNS:-$(toml_get task max_turns "20")}"
  CRAFTER_MIN_BATCH="${GEPA_CRAFTER_MIN_BATCH:-$(toml_get task min_batch "1")}"
  CRAFTER_MAX_BATCH="${GEPA_CRAFTER_MAX_BATCH:-$(toml_get task max_batch "5")}"
  require_uint "GEPA_CRAFTER_MAX_TURNS" "$CRAFTER_MAX_TURNS"
  require_uint "GEPA_CRAFTER_MIN_BATCH" "$CRAFTER_MIN_BATCH"
  require_uint "GEPA_CRAFTER_MAX_BATCH" "$CRAFTER_MAX_BATCH"
  set_command_env "CRAFTER_POLICY_MODEL" "$POLICY_MODEL"
  set_command_env "CRAFTER_MAX_TURNS" "$CRAFTER_MAX_TURNS"
  set_command_env "CRAFTER_MIN_BATCH" "$CRAFTER_MIN_BATCH"
  set_command_env "CRAFTER_MAX_BATCH" "$CRAFTER_MAX_BATCH"
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
echo "GEPA container: $CONTAINER_NAME"
echo "GEPA dataset: train=$TRAIN_SIZE heldout=$HELDOUT_SIZE train_seed_start=$TRAIN_SEED_START heldout_seed_start=$HELDOUT_SEED_START"
echo "GEPA policy: provider=$POLICY_PROVIDER model=$POLICY_MODEL api_key_env=$POLICY_API_KEY_ENV"
echo "GEPA proposer_model: $PROPOSER_MODEL"
echo "GEPA proposer_auth: mode=$PROPOSER_AUTH_MODE api_key_env=$PROPOSER_API_KEY_ENV copy_host_auth=$PROPOSER_COPY_HOST_AUTH"
echo "GEPA rollout_async_timeout_seconds: $ROLLOUT_ASYNC_TIMEOUT_SECONDS"
echo "GEPA container_http_timeout_seconds: $CONTAINER_HTTP_TIMEOUT_SECONDS"
echo "GEPA max_generations: $MAX_GENERATIONS"
echo "GEPA proposals_per_generation: $PROPOSAL_COUNT"
echo "GEPA minibatch_size: $MINIBATCH_SIZE"
echo "GEPA max_in_flight_candidates: $IN_FLIGHT_CANDIDATES"
echo "GEPA rollout_workers: $ROLLOUT_WORKERS"
echo "GEPA adaptive_rollout_concurrency: enabled=$ADAPTIVE_ENABLED initial=$ADAPTIVE_INITIAL min=$ADAPTIVE_MIN max=$ADAPTIVE_MAX up=$ADAPTIVE_UP down=$ADAPTIVE_DOWN after=$ADAPTIVE_AFTER"
if [[ -n "$CHUNK_SIZE" && "$CHUNK_SIZE" != "auto" ]]; then
  echo "GEPA rollout_chunk_size: $CHUNK_SIZE"
else
  echo "GEPA rollout_chunk_size: auto"
fi
echo "GEPA rollout budgets: train=$TRAIN_BUDGET heldout=$HELDOUT_BUDGET"

cd "$SCRIPT_DIR"
export SYNTH_OPTIMIZERS_CONTAINER_HTTP_TIMEOUT_SECONDS="$CONTAINER_HTTP_TIMEOUT_SECONDS"
export SYNTH_OPTIMIZERS_GEPA_ROLLOUT_HTTP_RETRIES="$ROLLOUT_HTTP_RETRIES"
if [[ -n "$CHUNK_SIZE" && "$CHUNK_SIZE" != "auto" ]]; then
  export SYNTH_OPTIMIZERS_GEPA_ROLLOUT_CHUNK_SIZE="$CHUNK_SIZE"
fi
exec uv run --project "$REPO_ROOT/packages/synth-optimizers" --group dev --reinstall-package synth-optimizers synth-optimizers gepa run --config "$CONFIG"
