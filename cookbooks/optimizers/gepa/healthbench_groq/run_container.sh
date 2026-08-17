#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
storage_root="$script_dir/.healthbench2-runs"
port="8114"
grader_model="gpt-4.1-2025-04-14"
grader_api_key_env="OPENAI_API_KEY"
grader_base_url="https://api.openai.com/v1"

while (( $# )); do
  case "$1" in
    --port)
      (( $# >= 2 )) || { echo "--port requires a value" >&2; exit 2; }
      port="$2"
      shift 2
      ;;
    --storage-root)
      (( $# >= 2 )) || { echo "--storage-root requires a value" >&2; exit 2; }
      storage_root="$2"
      shift 2
      ;;
    --grader-model)
      (( $# >= 2 )) || { echo "--grader-model requires a value" >&2; exit 2; }
      grader_model="$2"
      shift 2
      ;;
    --grader-api-key-env)
      (( $# >= 2 )) || { echo "--grader-api-key-env requires a value" >&2; exit 2; }
      grader_api_key_env="$2"
      shift 2
      ;;
    --grader-base-url)
      (( $# >= 2 )) || { echo "--grader-base-url requires a value" >&2; exit 2; }
      grader_base_url="$2"
      shift 2
      ;;
    *)
      echo "usage: $0 [--port PORT] [--storage-root PATH] [--grader-model MODEL] [--grader-api-key-env ENV_NAME] [--grader-base-url URL]" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${!grader_api_key_env:-}" ]]; then
  echo "$grader_api_key_env is required by the canonical HealthBench scorer; refusing to advertise an unscorable container." >&2
  exit 78
fi

exec uv run --frozen --project "$script_dir" python "$script_dir/synth_service_app.py" \
  --host 127.0.0.1 \
  --port "$port" \
  --storage-root "$storage_root" \
  --grader-model "$grader_model" \
  --grader-api-key-env "$grader_api_key_env" \
  --grader-base-url "$grader_base_url"
