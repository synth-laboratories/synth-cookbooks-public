#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "${AUTHORIZE_TINKER_CISPO:-}" != "1" ]]; then
  echo "Refusing paid Tinker CISPO: set AUTHORIZE_TINKER_CISPO=1" >&2
  exit 2
fi
cd "${HERE}"
uv run --with 'tinker==0.24.0' --with jinja2 --with datasets python runbooks/tinker_cispo.py --authorize "$@"
