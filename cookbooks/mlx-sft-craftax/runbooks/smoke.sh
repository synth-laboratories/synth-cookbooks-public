#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
python3 "${HERE}/runbooks/eval_craftax_scripted.py"
python3 "${HERE}/runbooks/sft_identity_smoke.py"
