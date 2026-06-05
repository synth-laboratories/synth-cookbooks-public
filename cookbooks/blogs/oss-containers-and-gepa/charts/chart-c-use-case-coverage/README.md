# Chart C - Heldout Coverage

Heldout coverage curves for the current four-task same-container
comparison. This chart answers a different question from the head-to-head score:
instead of showing the single best heldout reward per stack, it counts how many
distinct heldout rows each stack covers cumulatively as candidates are added.

Coverage means reward greater than or equal to the benchmark's configured
threshold:

- HealthBench Pro uses positive rubric credit (`>= 0.000001`).
- tau2-bench retail, Banking77, and HotpotQA use task success / exact match
  (`>= 1.0`).

The producer asserts each curve's final value against the `summary.json` packet
emitted by the shared GEPA eval evidence pipeline. The chart data and checked-in
producer output are tracked in the public evidence packet; the frontend launch
commit must carry the byte-matched mirror and cite that packet.

## Final Coverage Counts

| Task | Heldout rows | Synth GEPA | gepa-ai |
|---|---:|---:|---:|
| HealthBench Pro | 200 | 158 | 157 |
| tau2-bench retail | 100 | 63 | 62 |
| Banking77 | 200 | 177 | 182 |
| HotpotQA | 200 | 142 | 145 |

## Layout

```
chart-c-use-case-coverage/
  README.md
  build_heldout_coverage.py    # Reads final eval evidence, emits JSON
  figures/
    use_case_heldout_coverage_data.json
    source_evidence.json
```

The producer also writes the frontend mirror at
`frontend/src/components/blog/posts/introducing-gepa-platform/data/use_case_heldout_coverage_data.json`.

## Reproduce

```bash
python cookbooks/blogs/oss-containers-and-gepa/charts/chart-c-use-case-coverage/build_heldout_coverage.py
```

## Status

- [x] Heldout evidence exists for HealthBench Pro, tau2-bench retail,
      Banking77, and HotpotQA.
- [x] Producer asserts final curve values against benchmark summaries.
- [x] Producer emits the cookbook JSON and frontend mirror.
- [x] Producer emits checksums for the JSONL aggregate and per-benchmark
      summaries in `figures/source_evidence.json`.
- [x] Frontend consumes the generated JSON in `ParetoCoverageChart`.
- [x] Public evidence commit and checked-in producer output are published from
      the same commit authority.
- [ ] Frontend launch commit cites the public evidence packet and carries the
      byte-matched mirror.
