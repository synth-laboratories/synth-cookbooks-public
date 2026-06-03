# Chart G - DungeonGrid Agent Evidence

DungeonGrid evidence for the GEPA blog post. This folder owns the public chart
producer for the DungeonGrid multi-agent example.

No DungeonGrid number, curve, table cell, rollout visual, or achievement
frequency should appear in the blog unless it is emitted by `build_chart.py`
from the final evidence artifacts listed below.

## Scope

- Task config: `cookbooks/optimizers/gepa/evals/configs/dungeongrid.toml`.
- Blog framing: two-agent DungeonGrid policy hillclimb.
- Agent rule: every rollout uses exactly two heroes.
- Product surface: agent example, hillclimb curve, progress-signal frequency,
  and source-linked evidence packets.

## Required Evidence

`build_chart.py` reads the final GEPA eval evidence:

- `cookbooks/optimizers/gepa/evals/evidence/candidate_timeline.jsonl`
- `cookbooks/optimizers/gepa/evals/evidence/heldout_evaluations.jsonl`
- `cookbooks/optimizers/gepa/evals/evidence/benchmarks/dungeongrid/summary.json`

The producer emits:

- `figures/dungeongrid_data.json`
- `frontend/src/components/blog/posts/introducing-gepa-platform/data/dungeongrid_data.json`

If those source artifacts are missing or the final rows are incomplete, the
producer fails before writing JSON.

## Reproduce

Run the producer from the repository root:

```bash
python cookbooks/blogs/oss-containers-and-gepa/chart-g-dungeongrid/build_chart.py
```

## Status

- [x] Producer refuses to output chart JSON without complete final evidence rows.
- [x] Frontend consumes generated JSON rather than inline DungeonGrid data.
- [x] Two-agent DungeonGrid final evidence is present.
- [x] `figures/dungeongrid_data.json` is generated from completed artifacts.
- [x] Frontend renders real hillclimb and achievement series.
