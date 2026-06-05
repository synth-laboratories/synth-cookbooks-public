# tau2 retail - Chart D - gpt-5.4-nano

Status: launch-scope run evidence is tracked in the public evidence packet; final launch still requires the frontend mirror and release checklist to cite that packet.

## Result

| Field | Value |
|---|---|
| Initial observed optimization reward | 0.6 |
| Best observed optimization reward | 0.6 |
| Best observed reward source | train_reward |
| A/C heldout seed context (not Chart D metric) | 0.38 |
| Proposer calls | 3 |
| Metered cost USD | 2.5996006749999996 |

## Run

- Run id: `tau2_retail_proposer_nano_rerun_20260604T0523`
- Task: `tau2_retail`
- Proposer: `gpt-5.4-nano`
- Chart: D proposer sweep

## Evidence

- Compact manifest snapshot: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-d-proposer-scaling/figures/manifest_snapshots/tau2_retail_nano.result_manifest.json`
- Snapshot sha256: `a8ea62328a99d9d69443f6bb6606fed1627d0010b8e06a55c836b2484af679f1`
- Chart D source packet: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-d-proposer-scaling/figures/source_evidence.json`

## Notes

Chart D reports observed train optimization reward from Synth GEPA runs, not posthoc heldout reward. The heldout seed value above is context from the separate Chart A/C comparison and is not plotted as the Chart D baseline. Raw run artifacts stay in gitignored Chart D run folders; this record cites the compact public manifest snapshot.
