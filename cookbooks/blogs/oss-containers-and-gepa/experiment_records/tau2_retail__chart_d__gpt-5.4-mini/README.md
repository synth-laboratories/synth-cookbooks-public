# tau2 retail - Chart D - gpt-5.4-mini

Status: draft Chart D snapshot only. The compact manifest is tracked, but the raw all-cell rerun gate is not green and this record must not be cited as launch-scope run evidence.

## Result

| Field | Value |
|---|---|
| Initial observed optimization reward | 0.5666666666666667 |
| Best observed optimization reward | 0.6 |
| Best observed reward source | train_reward |
| A/C heldout seed context (not Chart D metric) | 0.38 |
| Proposer calls | 3 |
| Metered cost USD | 2.6915854749999992 |

## Run

- Run id: `tau2_retail_proposer_mini_rerun_20260604T0523`
- Task: `tau2_retail`
- Proposer: `gpt-5.4-mini`
- Chart: D proposer sweep

## Evidence

- Compact manifest snapshot: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-d-proposer-scaling/figures/manifest_snapshots/tau2_retail_mini.result_manifest.json`
- Snapshot sha256: `b96f3976967172d17f441af8acf6ae82359562a1e1474618ee37c71b035e8e77`
- Chart D source packet: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-d-proposer-scaling/figures/source_evidence.json`

## Notes

Chart D reports observed train optimization reward from Synth GEPA runs, not posthoc heldout reward. The heldout seed value above is context from the separate Chart A/C comparison and is not plotted as the Chart D baseline. Raw run artifacts stay in gitignored Chart D run folders; this record cites the compact public manifest snapshot.
