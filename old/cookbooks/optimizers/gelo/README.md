# GELO Hosted Launch Guide

GELO is the hosted Go-Explore optimizer for prompt-space search. Use this guide
when you already have a GELO-compatible task container with checkpoint and
resume endpoints.

## Launch Promo

The launch promo gives the first 20 organizations a `$500` hosted Go-Ex
proposer-spend grant, valid for 14 days after claim. Hosted `go-ex` submits
auto-claim the grant when slots remain and require at least `$1` of
`optimizer_go_ex_llm_spend` headroom before the run is queued.

The promo does not cover in-container policy LLM calls. Those calls are owned by
your task container and its provider credentials.

Repeat submits from the same organization reuse the same promo grant. Each
organization may have at most one hosted GELO run in `queued` or `running`
status at once while using the promo.

## Submit Requirements

- Submit with `algorithm: "go-ex"`.
- Provide a materialized `config_json` or launcher `config_toml`.
- Use GPT models for these paid proposer roles:
  `core_proposer`, `aux_hill_climb_proposer`, `aux_data_miner_proposer`,
  `aux_consolidate_proposer`, and `aux_consolidate_hill_climb_proposer`.
- `theme_verifier_agent` and `terminator_agent` may use non-GPT models.
- Start the task container first and verify `GET /health`.

## Container Contract

GELO needs a task container that supports long-horizon rollouts, scheduled
checkpoints, terminal checkpoint retrieval, and resume from checkpoint. The public
contract lives in the hosted optimizer docs and the `optimizers` repository:

- [Hosted Optimizers docs](https://docs.usesynth.ai/sdk/hosted-optimizers)
- [`synth-laboratories/optimizers`](https://github.com/synth-laboratories/optimizers)

## Crafter Container Target

Use a Crafter-compatible container you control, or a public GELO-compatible task
container when one is published, and expose it to hosted GELO with SynthTunnel.

Before submitting:

1. Start the Crafter container locally.
2. Confirm `GET /health` returns 200.
3. Confirm the container implements the GELO task contract above.

```bash
export SYNTH_API_KEY="..."
export CRAFTER_CONTAINER_URL="http://127.0.0.1:8943"
curl -fsS "$CRAFTER_CONTAINER_URL/health"
./scripts/submit.sh
```

The submit script opens a SynthTunnel lease to `CRAFTER_CONTAINER_URL`, submits
the public `crafter_smoke` preset, and follows the hosted run. The preset uses
GPT-family paid proposer roles, so it is promo-eligible; in-container policy LLM
calls remain your container's responsibility.

## Watch State

After submit prints a `run_id`, inspect the public GELO state:

```bash
synth-optimizers gelo watch "$RUN_ID" --slice board
synth-optimizers gelo watch "$RUN_ID" --slice themes
synth-optimizers gelo watch "$RUN_ID" --slice frontier
```

## Run Evidence

Example production proof run, captured on hosted GELO. This run used a smoke
config derived from `crafter_smoke`, with the paid proposer path exercised while
keeping rollout volume small.

- hosted `run_id`: `prod_gelo_promo_chargeable_full_lane_20260612053153`
- terminal status: `succeeded`
- storage mode: `hosted`; finalize state: `deleted`
- billing receipt: `optimizer:prod_gelo_promo_chargeable_full_lane_20260612053153:terminal-usage`
- charged cost: `$0.0197228` from `optimizer_go_ex_llm_spend`
- promo headroom: `$500.0000000` before, `$499.9802772` after
- rollout calls: 14 total, 10 search, 4 heldout measurement
- proposer calls: 1 core proposer and 2 aux proposer calls
- best candidate: `goex_prompt_e44c36546488`
- search mean reward: `1.1`; heldout mean reward: `1.5`
- concurrency proof: a second submit while the run was active returned
  `409 gelo_launch_promo_concurrency`
