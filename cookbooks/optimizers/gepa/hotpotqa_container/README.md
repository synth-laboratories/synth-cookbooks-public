# HotpotQA GEPA Container

Public Rust GEPA cookbook for HotpotQA multi-hop QA.

The container serves the GEPA HTTP contract over FastAPI and scores each rollout
with HotpotQA-style token F1 against the labeled `hotpot_qa/distractor` split.
The mutable prompt field is `stage1_system`; the user prompt always includes
the question and distractor passages.

## Run

```bash
cd ~/Documents/GitHub/synth-cookbooks-public/cookbooks/optimizers/gepa/hotpotqa_container
bash run_fresh_gepa.sh --profile long
```

Profiles live under `run_profiles/`:

- `smoke` - small shape check
- `default` - modest run
- `long` - train=100, heldout=200, minibatch=40

The runner loads `OPENROUTER_API_KEY` from the local Synth `.env` locations when
it is not already set. It writes a fresh generated TOML beside the base config
for each run and then calls the public Rust `synth-optimizers gepa run` command.
