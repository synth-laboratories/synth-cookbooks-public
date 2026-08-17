#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
storage_root="$script_dir/.banking77-runs"
port="8765"

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
    *) echo "usage: $0 [--port PORT] [--storage-root PATH]" >&2; exit 2 ;;
  esac
done

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required by the advertised Banking77 desktop_eval policy; refusing to advertise an unusable container." >&2
  exit 78
fi

cd "$script_dir"
exec uv run --frozen --project "$script_dir" python synth_service_app.py \
  --host 127.0.0.1 \
  --port "$port" \
  --storage-root "$storage_root"
