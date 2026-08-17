#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
port="${BANKING77_PORT:-8765}"

cd "$script_dir"
exec uv run --project "$script_dir" python synth_service_app.py --host 127.0.0.1 --port "$port"
