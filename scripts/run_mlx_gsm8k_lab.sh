#!/usr/bin/env bash
# Real MLX Tinker RL lab from synth-mlx-rl (Qwen/Qwen3.5-0.8B, Metal).
set -euo pipefail
MLX_ROOT="/Users/joshuapurtell/Documents/GitHub/synth-mlx-rl"
OUT_ROOT="/Users/joshuapurtell/Documents/GitHub/worktrees/synth-cookbooks-public-v0.6/catalog/generated"
PY="${MLX_ROOT}/.venv/bin/python"
export SYNTH_GSM8K_SOURCE=hf
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export SYNTH_MLX_RL_STORE="${OUT_ROOT}/mlx-store"
mkdir -p "$OUT_ROOT"
cd "$MLX_ROOT"
MODE="${1:-sft}"
shift || true
exec "$PY" scripts/gsm8k_lab.py "$MODE" \
  --eval-n 40 \
  --train-prompts 80 \
  --group 4 \
  --epochs 1 \
  --rounds 8 \
  --max-tokens 256 \
  --out "${OUT_ROOT}/gsm8k_${MODE}_lab.json" \
  "$@"
