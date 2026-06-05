# Chart A - Same-Container Head-to-Head

The anchor table for the current post. Rows are the four current
same-container comparison tasks: HealthBench Pro, tau2-bench retail, Banking77,
and HotpotQA. Columns show each stack's seed baseline, best posthoc heldout
reward, the `Synth - gepa-ai` best-heldout delta, and the winner.

The current frontend mirror is generated from launch evidence. The chart data
and checked-in producer output are tracked in the public evidence packet; the
frontend launch commit must carry the byte-matched mirror and cite that packet.

## Parity conditions

- Both stacks are evaluated through the same public HTTP container boundary for
  each task.
- Heldout splits are fixed per task. Train denominators are shown only where
  the producer summary exposes them:
  - HealthBench Pro: `100` train / `200` heldout rows.
  - tau2-bench retail: `30` train / `100` heldout tasks.
  - Banking77: `100` train / `200` heldout rows.
  - HotpotQA: `200` heldout rows; the current summary reports no train
    denominator, so do not claim shared train coverage for this task.
- Proposer model is fixed at `gpt-5.4-mini` for the current comparison rows.
- Policy/judge models are locked per task in the generated evidence rows.
- `build_chart.py` reads committed final evidence summaries from
  `cookbooks/optimizers/gepa/evals/evidence/benchmarks/*/summary.json`.

## Reproducible records

The eval configs live in `cookbooks/optimizers/gepa/evals/configs/*.toml`.
Each stack run writes full artifacts to
`cookbooks/optimizers/gepa/evals/runs/<stack>/<benchmark>/<run_id>/` and appends
its command row to `cookbooks/optimizers/gepa/evals/evidence/commands.jsonl`.
After posthoc heldout/train evaluation, `build_evidence.py --benchmark <name>`
writes the compact launch packet to
`cookbooks/optimizers/gepa/evals/evidence/benchmarks/<benchmark>/`.

`build_langprobe_addendum.py` is inactive for the current post. Do not
regenerate the old Banking77/HotpotQA/HoVer/Heart Disease addendum unless it is
explicitly re-added to the active MDX with fresh evidence.

## Layout

```text
charts/chart-a-head-to-head/
  README.md
  build_chart.py               # reads experiment_records / eval evidence → figures/
  build_langprobe_addendum.py   # inactive historical addendum
  run_matrix.sh
  figures/
    head_to_head_data.json
    source_evidence.json
    head_to_head.md
    head_to_head.svg
  frontend mirror:
    frontend/src/components/blog/posts/introducing-gepa-platform/data/core_head_to_head_data.json
```

Chart A launch configs do not live in this folder. Use
`cookbooks/optimizers/gepa/evals/configs/*.toml` plus the committed benchmark
evidence summaries listed above.

## Reproduce

```bash
cd cookbooks/blogs/oss-containers-and-gepa/charts/chart-a-head-to-head
python build_chart.py
```

## Status

- [x] Synth GEPA and gepa-ai launch evidence summaries selected for
      HealthBench Pro, tau2-bench retail, Banking77, and HotpotQA.
- [x] `build_chart.py` reads committed evidence summaries and emits JSON,
      Markdown, SVG, frontend mirror, and source-evidence checksum packets.
- [x] `figures/source_evidence.json` preserves a tracked compact snapshot
      of the final summary inputs.
- [x] Chart table embedded in blog MDX.
- [x] Public evidence commit and checked-in producer output are published from
      the same commit authority.
- [x] Current frontend review commit cites the public evidence packet and
      carries the byte-matched mirror.
