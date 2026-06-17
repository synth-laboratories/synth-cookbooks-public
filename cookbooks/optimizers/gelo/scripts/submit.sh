#!/usr/bin/env bash
set -euo pipefail

: "${SYNTH_API_KEY:?Set SYNTH_API_KEY before submitting hosted GELO.}"
: "${CRAFTER_CONTAINER_URL:?Set CRAFTER_CONTAINER_URL to your local GELO-compatible Crafter container, for example http://127.0.0.1:8943.}"

synth-optimizers gelo submit \
  --preset crafter_smoke \
  --tunnel-url "${CRAFTER_CONTAINER_URL}" \
  --tunnel-provider synth_tunnel \
  --proposer-rounds 1 \
  --train-seed-count 4 \
  --heldout-seed-count 2 \
  --max-rollouts 80 \
  --follow
