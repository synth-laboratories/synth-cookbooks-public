# Chart A — Compute-Parity Head-to-Head

The anchor table for the post. Rows are optimizer implementations
(seed candidate · gepa-ai · Synth GEPA). Columns are the public reference
cookbooks (Banking77 · TBLite · Crafter — code-review stays private and
is not in this comparison). Cells are heldout reward at fixed compute
budget.

## Parity conditions

- Both stacks call the same public HTTP container for each task.
- Banking77 is the broadest run: `24` train rows, `200` heldout rows,
  and roughly `2400` metric calls.
- TBLite and Crafter are smoke-scale public parity splits:
  TBLite uses `3` train / `2` heldout rows, Crafter uses `2` train /
  `1` heldout row.
- Policy model: `gpt-4.1-nano`.
- Synth proposer: `gpt-5.3-codex-spark` via `codex_app_server`.

## Layout

```
chart-a-head-to-head/
  README.md
  configs/
    synth_gepa/                # Our gepa.toml per cookbook
      banking77_parity.toml
      tblite_parity.toml
      crafter_parity.toml
    gepa_ai/                   # gepa-ai adapters against the same containers
      banking77_via_container.py
      container_task.py
  run_banking77_parity.sh      # Banking77 full-budget parity runner
  build_chart.py               # Reads runs/, emits figures/head_to_head.svg
  runs/
    synth_gepa/
      banking77_parity_synth_gepa/
      tblite_parity_synth_gepa/
      crafter_parity_synth_gepa/
    gepa_ai_via_container/
      banking77_<timestamp>/   # summary.json
      ...
  figures/
    source_evidence.json       # Compact tracked snapshot of ignored raw runs
    head_to_head.svg           # The chart as it appears in the post
```

## Reproduce

```bash
python configs/gepa_ai/container_task.py --task tblite
python configs/gepa_ai/container_task.py --task crafter
synth-optimizers gepa run --config configs/synth_gepa/tblite_parity.toml
synth-optimizers gepa run --config configs/synth_gepa/crafter_parity.toml
python build_chart.py
```

## Status

- [x] Synth GEPA same-container manifests selected for all 3 public cookbooks.
- [x] gepa-ai same-container runs completed for all 3 public cookbooks.
- [x] `build_chart.py` reads from `runs/` and emits JSON, Markdown, and SVG.
- [x] `figures/source_evidence.json` preserves a tracked compact snapshot
      of the ignored raw run manifests/summaries.
- [x] Chart table embedded in blog MDX.

Current caveat: Banking77 is the strongest true full same-container comparison
(`24` train rows, `200` heldout rows, `2400` metric-call budget). TBLite and
Crafter now have fresh same-container runs for both stacks, but they are
small public smoke-scale splits.
