# HotpotQA GEPA Container

Public Rust GEPA cookbook for HotpotQA multi-hop QA.

The container serves the GEPA HTTP contract over FastAPI and scores each rollout
with HotpotQA-style token F1 against the labeled `hotpot_qa/distractor` split.
The mutable prompt field is `stage1_system`; the user prompt always includes
the question and distractor passages.

## GEPA v2 contract

- `GET /metadata` advertises `synth_optimizers.gepa.v2` plus its absolute
  `program_route`, `taskset_route`, `taskset_tasks_route`, and `rollout_route`.
- `GET /taskset` returns the taskset id and per-split sizes.
- `POST /taskset/tasks` resolves the stable `<split>:<seed>` task ids declared in
  the config's `[taskset]` and `[gepa.task_pools]` blocks (for example `train:0`, `validation:1000`),
  and echoes each requested id back on its row. This is the route the optimizer
  loads rows from.
- `POST /dataset/rows` and the other older seed-based routes remain available for
  compatibility clients, but GEPA v2 does not use them.
- `GET /program` exposes one mutable module: `stage1_system`.
- `POST /rollout` answers one question and scores it with HotpotQA token F1.

Heldout ids use the `validation` split (`heldout_split = "validation"`), matching
the `hotpot_qa/distractor` split names.

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
