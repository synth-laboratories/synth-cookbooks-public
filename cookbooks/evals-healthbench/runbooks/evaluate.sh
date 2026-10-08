#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
python3 "${HERE}/runbooks/evaluate.py" --url "http://127.0.0.1:8114" --config "${HERE}/configs/eval.toml"
