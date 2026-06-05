# HealthBench Pro - Chart D - gpt-5.4-mini

Status: draft Chart D snapshot only. The compact manifest is tracked, but the raw all-cell rerun gate is not green and this record must not be cited as launch-scope run evidence.

## Result

| Field | Value |
|---|---|
| Initial observed optimization reward | 0.2553621913211898 |
| Best observed optimization reward | 0.3143681513570261 |
| Best observed reward source | train_reward |
| A/C heldout seed context (not Chart D metric) | 0.2547314326739686 |
| Proposer calls | 4 |
| Metered cost USD | 0.0 |

## Run

- Run id: `healthbench_proposer_mini`
- Task: `healthbench`
- Proposer: `gpt-5.4-mini`
- Chart: D proposer sweep

## Evidence

- Compact manifest snapshot: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-d-proposer-scaling/figures/manifest_snapshots/healthbench_mini.result_manifest.json`
- Snapshot sha256: `05a32c98325f8bd376637e15f333ff75f471a241c05f78d11a7ea08acfce22b4`
- Chart D source packet: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-d-proposer-scaling/figures/source_evidence.json`

## Notes

Chart D reports observed train optimization reward from Synth GEPA runs, not posthoc heldout reward. The heldout seed value above is context from the separate Chart A/C comparison and is not plotted as the Chart D baseline. Raw run artifacts stay in gitignored Chart D run folders; this record cites the compact public manifest snapshot.
