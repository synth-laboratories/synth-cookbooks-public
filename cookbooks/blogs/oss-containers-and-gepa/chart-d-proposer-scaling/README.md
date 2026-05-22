# Chart D — Proposer Scaling

Does optimizer quality ride the proposer-model curve? Sweep the
proposer model across generations and measure best heldout reward at
fixed compute budget per cookbook.

## Sweep grid

- **Proposer models** (X-axis, ordered chronologically):
  - `gpt-5`
  - `gpt-5.1-codex`
  - `gpt-5.2`
  - `gpt-5.3-codex`
  - `gpt-5.4`
  - `gpt-5.5`
- **Cookbooks** (Y-axis series, one line per cookbook):
  - Banking77 · TBLite · Code Review · Crafter
- **Held constant**:
  - Optimizer = Synth GEPA
  - Student/policy = cookbook fixture (or one fixed live model — TBD)
  - `max_total_rollouts` = same per cell
  - Train/heldout seeds = same per cookbook

## Companion chart (Pareto)

For each cell, log `total_cost_usd` from the run manifest and emit a
secondary Pareto scatter (cost-per-reward-point) — nano / mini
proposers may sit on the Pareto frontier even if they aren't peak
quality.

## Layout

```
chart-d-proposer-scaling/
  README.md
  configs/
    proposer_sweep/
      banking77_gpt-5.toml
      banking77_gpt-5.1-codex.toml
      ...                        # 6 proposers × 4 cookbooks = 24 configs
  run_sweep.sh                   # Launches all 24 runs
  build_chart.py                 # Reads runs/, emits figures/proposer_scaling.svg
                                  # and figures/proposer_pareto.svg
  runs/
    banking77_gpt-5_<ts>/
    ...
  figures/
    proposer_scaling.svg
    proposer_pareto.svg
```

## Reproduce

```bash
./run_sweep.sh                 # All 24 runs (cache-aware; reruns are cheap)
python build_chart.py
```

## Status

- [ ] Configs for all 6 proposers × 4 cookbooks generated.
- [ ] Sweep launched.
- [ ] Both chart SVGs rendered.
- [ ] Section in blog MDX embeds chart.

## Open questions

- Do we have proposer auth for all 6 generations through Codex app
  server, or do some need direct OpenAI API routing?
- Cost cap per cell so the full sweep is bounded?
