# Chart A — Final Same-Container Head-to-Head

The anchor table for the post. Rows are the final four launch benchmarks:
HealthBench Pro, Harvey Lab Tax, tau2-bench retail, and DungeonGrid. Columns
show each stack's own seed baseline, best posthoc heldout reward, the
`Synth - gepa-ai` best-heldout delta, and the winner. Seed baselines are
stack-specific whenever the underlying optimizer adapters start from different
candidate prompts.

## Parity conditions

- Both stacks are evaluated through the same public HTTP container boundary for
  each task.
- Train and heldout splits are fixed per task:
  - HealthBench Pro: `100` train / `200` heldout rows.
  - Harvey Lab Tax: `25` train / `9` heldout matters.
  - tau2-bench retail: `20` train / `94` heldout tasks.
  - DungeonGrid: `8` train / `8` heldout episodes.
- Proposer and reflection models are fixed at `gpt-5.4-mini` in the final
  head-to-head evidence.
- Policy and judge models are task-specific and fixed inside each task's
  `parity_controls.json`.
- `build_chart.py` reads committed final evidence summaries from
  `cookbooks/optimizers/gepa/evals/evidence/benchmarks/*/summary.json`.

## Layout

```
chart-a-head-to-head/
  README.md
  configs/
    synth_gepa/                # Historical per-cookbook run configs
    gepa_ai/                   # Same-container gepa-ai adapters
  run_matrix.sh                # Historical parity launcher
  build_chart.py               # Reads final eval evidence, emits figures/ and frontend mirror
  figures/
    head_to_head_data.json     # Cookbook copy of frontend chart data
    source_evidence.json       # Checksums and run ids for final summaries
    head_to_head.md            # Markdown table emitted by producer
    head_to_head.svg           # The chart as it appears in the post
```

## Reproduce

```bash
cd cookbooks/blogs/oss-containers-and-gepa/chart-a-head-to-head
python build_chart.py
```

## Status

- [x] Synth GEPA and gepa-ai final evidence summaries selected for all four
      launch benchmarks.
- [x] `build_chart.py` reads committed evidence summaries and emits JSON,
      Markdown, SVG, frontend mirror, and source-evidence checksum packets.
- [x] `figures/source_evidence.json` preserves a tracked compact snapshot
      of the final summary inputs.
- [x] Chart table embedded in blog MDX.
