#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
synth-optimizers gepa run --config "${HERE}/configs/gepa.toml"
