#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
python3 "${HERE}/runbooks/cispo_smoke.py"
python3 "${HERE}/runbooks/hosted_preflight.py" || true
