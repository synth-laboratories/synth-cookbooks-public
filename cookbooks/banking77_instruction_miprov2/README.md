# Banking77 Instruction MIPROv2

Second MVP cookbook for instruction optimization with MIPROv2 on Banking77.

## Goal

Show a compact instruction-optimization workflow where MIPROv2 improves a
Banking77 classifier prompt and reports the result honestly on heldout data.

The target public example is the Banking77 confusable slice because the private
draft has a measured positive heldout lift. Easier Banking77 slices can be
included as diagnostics, but they should not be marketed as lift examples when
the baseline saturates.

## Source

Initial migration references:

- `synth-cookbooks-private/recipes/miprov2/README.md`
- `synth-cookbooks-private/recipes/miprov2/scripts/run_banking77_miprov2_openai_split.py`
- `synth-cookbooks-private/recipes/miprov2/configs/banking77_openai_split_confusable_30x300_100heldout.json`
- `evals/containers/arbitrary/banking77/`
- `evals/benchmarks/reportbench_synth/banking77_offline_gepa_pool/`

## MVP Shape

1. Install `synth-optimizers` with MIPROv2.
2. Load or serve the public-safe Banking77 task data.
3. Run instruction MIPROv2 on the confusable slice.
4. Write `summary.json`, artifact manifest, proposer traces, and rollout
   evidence.
5. Report baseline train, best train, heldout baseline, heldout best, and lift.

## Known Evidence To Preserve

Private draft result for Banking77 confusable 30x300, heldout 100, `K=10`:

- baseline train: `0.7666666667`
- best train: `0.8333333333`
- heldout baseline: `0.7900`
- heldout best: `0.8200`
- heldout lift: `+0.0300`

## Public Safety

Before publishing, confirm the dataset source, configs, traces, and generated
artifacts are public-safe and contain no private credentials, private endpoints,
or unredacted internal logs.
