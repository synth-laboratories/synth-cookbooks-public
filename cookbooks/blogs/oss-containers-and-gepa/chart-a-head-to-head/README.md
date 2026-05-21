# Chart A — Compute-Parity Head-to-Head

The anchor table for the post. Rows are optimizer implementations
(seed candidate · gepa-ai · Synth GEPA). Columns are the public reference
cookbooks (Banking77 · TBLite · Crafter — code-review stays private and
is not in this comparison). Cells are heldout reward at fixed compute
budget.

## Parity conditions

- `max_total_rollouts = 256`
- Proposer: `gpt-5.4-mini`, `reasoning_effort = medium`
- Student: cookbook fixture policy (deterministic; ensures cache-replay
  works for both stacks)
- Train seeds / heldout seeds: cookbook defaults (see each cookbook's
  `gepa.toml`)
- Minibatch size: cookbook defaults

## Layout

```
chart-a-head-to-head/
  README.md
  configs/
    synth_gepa/                # Our gepa.toml per cookbook
      banking77.toml
      tblite.toml
      crafter.toml
    gepa_ai/                   # gepa-ai run configs (their YAML/JSON shape)
      banking77.yaml
      tblite.yaml
      crafter.yaml
  run_parity.sh                # Top-level: runs both stacks end-to-end
  build_chart.py               # Reads runs/, emits figures/head_to_head.svg
  runs/
    synth_gepa/
      banking77_<timestamp>/   # result_manifest.json, events.jsonl, ...
      tblite_<timestamp>/
      crafter_<timestamp>/
    gepa_ai/
      banking77_<timestamp>/
      ...
  figures/
    head_to_head.svg           # The chart as it appears in the post
```

## Reproduce

```bash
./run_parity.sh                # Launches both stacks on all 4 cookbooks
python build_chart.py          # Generates figures/head_to_head.svg
```

## Status

- [ ] Synth GEPA runs on all 3 public cookbooks at parity budget.
- [ ] gepa-ai runs on all 3 public cookbooks at parity budget.
- [ ] `build_chart.py` reads from `runs/` and emits SVG.
- [ ] Chart embedded in blog MDX.
