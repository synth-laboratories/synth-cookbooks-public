# Chart C — Heldout Coverage

Heldout coverage curves for the final same-container comparison. This chart
answers a different question from Chart A: instead of showing the single best
heldout reward per stack, it counts how many distinct heldout rows each stack
covers cumulatively as candidates are added.

Coverage means reward greater than or equal to the benchmark's configured
threshold:

- HealthBench Pro, Harvey Lab Tax, and DungeonGrid use positive reward
  (`>= 0.000001`).
- tau2-bench retail uses task success (`>= 1.0`).

The producer asserts each curve's final value against the final
`summary.json` packet emitted by the shared GEPA eval evidence pipeline.

## Final Coverage Counts

| Task | Heldout rows | Synth GEPA | gepa-ai |
|---|---:|---:|---:|
| HealthBench Pro | 200 | 158 | 157 |
| Harvey Lab Tax | 9 | 8 | 8 |
| tau2-bench retail | 94 | 58 | 75 |
| DungeonGrid | 8 | 8 | 8 |

## Layout

```
chart-c-use-case-coverage/
  README.md
  build_heldout_coverage.py    # Reads final eval evidence, emits JSON
  figures/
    use_case_heldout_coverage_data.json
```

The producer also writes the frontend mirror at
`frontend/src/components/blog/posts/introducing-gepa-platform/data/use_case_heldout_coverage_data.json`.

## Reproduce

```bash
python cookbooks/blogs/oss-containers-and-gepa/chart-c-use-case-coverage/build_heldout_coverage.py
```

## Status

- [x] Final heldout evidence exists for HealthBench Pro, Harvey Lab Tax,
      tau2-bench retail, and DungeonGrid.
- [x] Producer asserts final curve values against benchmark summaries.
- [x] Producer emits the cookbook JSON and frontend mirror.
- [x] Frontend consumes the generated JSON in `ParetoCoverageChart`.
