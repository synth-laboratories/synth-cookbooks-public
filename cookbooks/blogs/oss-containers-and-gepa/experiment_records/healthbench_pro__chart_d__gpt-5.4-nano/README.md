# HealthBench Pro - Chart D - gpt-5.4-nano

Status: launch-scope run evidence is tracked in the public evidence packet; final launch still requires the frontend mirror and release checklist to cite that packet.

## Result

| Field | Value |
|---|---|
| Initial observed optimization reward | 0.2509448010317407 |
| Best observed optimization reward | 0.3469341151460078 |
| Best observed reward source | train_reward |
| A/C heldout seed context (not Chart D metric) | 0.2547314326739686 |
| Proposer calls | 4 |
| Metered cost USD | 0.0 |

## Run

- Run id: `healthbench_proposer_nano`
- Task: `healthbench`
- Proposer: `gpt-5.4-nano`
- Chart: D proposer scaling

## Evidence

- Compact manifest snapshot: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-d-proposer-scaling/figures/manifest_snapshots/healthbench_nano.result_manifest.json`
- Snapshot sha256: `c5068f77fa5d2c55962e573ff515a996f697a1ba57ea4985e95641810601fe16`
- Chart D source packet: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-d-proposer-scaling/figures/source_evidence.json`

## Notes

Chart D reports observed train optimization reward from Synth GEPA runs, not posthoc heldout reward. The heldout seed value above is context from the separate Chart A/C comparison and is not plotted as the Chart D baseline. Raw run artifacts stay in gitignored Chart D run folders; this record cites the compact public manifest snapshot.
