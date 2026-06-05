# tau2 retail - Chart D - gpt-5.4

Status: launch-scope run evidence is tracked in the public evidence packet; final launch still requires the frontend mirror and release checklist to cite that packet.

## Result

| Field | Value |
|---|---|
| Initial observed optimization reward | 0.6333333333333333 |
| Best observed optimization reward | 0.6666666666666666 |
| Best observed reward source | train_reward |
| A/C heldout seed context (not Chart D metric) | 0.38 |
| Proposer calls | 3 |
| Metered cost USD | 2.696295100000001 |

## Run

- Run id: `tau2_retail_proposer_gpt54_rerun_20260604T0523`
- Task: `tau2_retail`
- Proposer: `gpt-5.4`
- Chart: D proposer scaling

## Evidence

- Compact manifest snapshot: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-d-proposer-scaling/figures/manifest_snapshots/tau2_retail_gpt54.result_manifest.json`
- Snapshot sha256: `be1b9c98c11b7037478f04d61790ba2218828630342453786a3f638ce4af6236`
- Chart D source packet: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-d-proposer-scaling/figures/source_evidence.json`

## Notes

Chart D reports observed train optimization reward from Synth GEPA runs, not posthoc heldout reward. The heldout seed value above is context from the separate Chart A/C comparison and is not plotted as the Chart D baseline. Raw run artifacts stay in gitignored Chart D run folders; this record cites the compact public manifest snapshot.
