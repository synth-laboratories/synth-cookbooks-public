#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/packages:${PYTHONPATH:-}"
PORT="${BANKING77_PORT:-8765}"
python3 -m cookbook_tools.serve_compat --target banking77_classify --port "$PORT" --storage-root "${HERE}/.out/storage" &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 50); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  sleep 0.1
done
python3 "${HERE}/runbooks/gold_rollout.py" --url "http://127.0.0.1:${PORT}"
