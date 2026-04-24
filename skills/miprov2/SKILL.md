---
name: miprov2
description: Use when running, reproducing, or adapting Synth MIPROv2 optimizer cookbooks, especially the native Banking77 instruction-optimization example.
---

# Synth MIPROv2 Cookbook Skill

Use this skill when a task mentions Synth MIPROv2, instruction optimization,
Banking77, prompt proposal loops, or the public `synth-optimizers` package.

## First Files To Read

- `packages/synth-optimizers/README.md`
- `cookbooks/optimizers/miprov2/README.md`
- `cookbooks/optimizers/miprov2/run.py`
- `cookbooks/optimizers/miprov2/miprov2/banking77_openai_split_confusable_30x300_100heldout.json`

## Default Workflow

1. Start from the cookbook README and confirm the desired mode: dry-run plan,
   service rollout, or live native MIPROv2 execution.
2. Use the public package path, not private optimizer code:

```bash
PYTHONPATH=packages/synth-optimizers/src:packages/synth-containers/src \
python cookbooks/optimizers/miprov2/run.py
```

3. For a live reproduction, run the native Synth MIPRO phase-3 OpenEnv proposer
   loop:

```bash
PYTHONPATH=packages/synth-optimizers/src:packages/synth-containers/src \
python cookbooks/optimizers/miprov2/run.py \
  --execute \
  --artifacts-dir cookbooks/optimizers/miprov2/run_artifacts/stratified_100x200_retry
```

4. Summarize both training and heldout metrics. Never report train-only lift as
   the cookbook result.

## Expected Public Evidence

The current reproducible public run is the stratified Banking77 example:

- optimizer path: `native_phase3_openenv`
- policy model: `gpt-4.1-nano`
- proposer model: `gpt-5.4-mini`
- train rows: `100`
- heldout rows: `200`
- proposer rounds: `8`
- total metric calls: `6000`
- heldout lift: `+0.075`

Committed lightweight evidence should live under:

```text
cookbooks/optimizers/miprov2/run_artifacts/stratified_100x200_retry/live_native_run/
```

## Guardrails

- Use native Synth MIPROv2. Do not route this cookbook through GEPA compatibility
  shims unless the task explicitly asks for migration or comparison work.
- Keep heavy SQLite ledgers, full run read models, and full proposer transcripts
  out of committed lightweight artifacts unless the user explicitly asks to
  publish them.
- Do not hardcode API keys or print raw environment values.
- Treat provider `502`/transient failures as retryable run noise. Record the
  retry behavior in artifacts or notes when it affects a reproduction.
- If a small slice saturates the baseline, label it as a diagnostic rather than
  a lift example.

