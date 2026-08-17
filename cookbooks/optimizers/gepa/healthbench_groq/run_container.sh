#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
storage_root="${1:-$script_dir/.healthbench2-runs}"
port="${HEALTHBENCH2_PORT:-8114}"

exec uv run --project "$script_dir" python "$script_dir/synth_service_app.py" \
  --host 127.0.0.1 \
  --port "$port" \
  --storage-root "$storage_root"
