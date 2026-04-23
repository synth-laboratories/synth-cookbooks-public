# synth-optimizers

Public Synth optimizer package source.

PyPI distribution: `synth-optimizers`

Current package version: `0.1.1`

The initial public package contains the Synth Lab-derived MIPROv2 optimizer
surface under `synth_optimizers.miprov2`.

## Install

```bash
pip install synth-optimizers
```

## MIPROv2 Usage

```python
from synth_optimizers.miprov2 import MiproCompatRunConfig, optimize

result = optimize(
    seed_candidate={"system_prompt": "You are a careful assistant."},
    trainset=train_rows,
    valset=heldout_rows,
    adapter=my_adapter,
    task_lm="openai/gpt-4.1-mini",
    reflection_lm="openai/gpt-5",
    config=MiproCompatRunConfig(
        dataset="banking77",
        task="first5_intents",
        optimizer_budget=8,
        max_concurrency=4,
    ),
)

print(result.best_candidate)
print(result.val_aggregate_scores[result.best_idx])
```

## Layout

```text
src/
  synth_optimizers/
    miprov2/
```

## Release

This package releases independently from other packages in this monorepo. See
`RELEASE.md` for the manual release workflow.
