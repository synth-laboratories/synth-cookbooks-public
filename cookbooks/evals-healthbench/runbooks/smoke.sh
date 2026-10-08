#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/packages:${PYTHONPATH:-}"
PORT="${HEALTHBENCH_PORT:-8114}"
python3 -m cookbook_tools.serve_compat --target healthbench_chat --port "$PORT" --storage-root "${HERE}/.out/storage" &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 50); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  sleep 0.1
done
python3 "${HERE}/runbooks/inspect.py" --url "http://127.0.0.1:${PORT}"
