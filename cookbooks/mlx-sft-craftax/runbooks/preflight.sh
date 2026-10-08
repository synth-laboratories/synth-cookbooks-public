#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}/packages:${PYTHONPATH:-}"
python3 -m cookbook_tools.cli preflight --cookbook mlx-sft-craftax
