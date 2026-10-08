# Chart E — Policy Model Variation

Does a GEPA-discovered prompt carry across student/policy models, or
is it overfit to the model it was optimized against?

## Sweep grid

- **Student/policy models** (one bar per model):
  - `gpt-4.1-nano`
  - `gpt-5-nano`
  - `gpt-5-mini`
  - `gemini-3.1-flash-lite`
  - `claude-haiku` (latest)
- **Held constant**:
  - Optimizer = Synth GEPA
  - Proposer = `gpt-5.4-mini` (fixed across all cells)
  - `max_total_rollouts` = same per cell

## Two views in the chart

1. **In-distribution bars** — optimize on model X, evaluate on model X.
   Per-cookbook bar chart.
2. **Cross-transfer matrix** — optimize on model A, evaluate on model
   B. 5×5 heatmap per cookbook. Diagonal-vs-off-diagonal tells the
   transfer story.

## Layout

```
chart-e-policy-model-variation/
  README.md
  configs/
    policy_sweep/
      banking77_nano.toml
      banking77_haiku.toml
      ...                        # 5 policies × 4 cookbooks = 20 configs
  run_sweep.sh
  run_cross_transfer.sh          # Optional second pass: evaluate-only on other policies
  build_chart.py                 # Emits figures/policy_bars.svg + figures/policy_transfer.svg
  runs/
    banking77_nano_<ts>/
    ...
  figures/
    policy_bars.svg
    policy_transfer.svg
```

## Reproduce

```bash
./run_sweep.sh
./run_cross_transfer.sh        # Skips if cross-transfer is out of scope for v1
python build_chart.py
```

## Status

- [ ] Cookbook containers wired to accept all 5 policy models.
  (Non-OpenAI students may need additional plumbing — check the
  container `[policy]` block before assuming this works.)
- [ ] Sweep launched.
- [ ] Cross-transfer matrix in scope for v1? — decide before tonight.
- [ ] Chart SVGs rendered.
- [ ] Section in blog MDX embeds chart.
