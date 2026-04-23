# MIPROv2 Banking77 Glue

This folder holds the optimizer glue for the public Banking77 MIPROv2
cookbook.

The target package import is:

```python
from synth_optimizers.miprov2 import MiproCompatRunConfig, optimize
```

The package source has been migrated into
`packages/synth-optimizers/src/synth_optimizers/miprov2/`. `../run.py --execute`
now runs native Synth MIPRO phase-3 with the OpenEnv proposer against the
Banking77 container adapter, while the default dry run still writes a
public-safe artifact manifest and preserves the known Banking77 confusable
evidence.

## Config

`banking77_openai_split_confusable_30x300_100heldout.json` is the first MVP
config. It uses:

- `30` train seeds
- `100` heldout seeds
- `10` target train candidates
- `300` target train rollouts
- OpenAI policy/proposer/verifier role split

## Success Bar

The first release should report:

- baseline train
- best train
- heldout baseline
- heldout best
- heldout lift
- result label: `positive`, `negative`, or `inconclusive`
