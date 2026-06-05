# HealthBench Pro - Chart D - gpt-5.4

Status: launch-scope run evidence present; publication still requires the public evidence commit.

## Result

| Field | Value |
|---|---|
| Initial observed optimization reward | 0.25565582187364394 |
| Best observed optimization reward | 0.3386648064383831 |
| Best observed reward source | train_reward |
| A/C heldout seed context (not Chart D metric) | 0.2547314326739686 |
| Proposer calls | 2 |
| Metered cost USD | 0.0 |

## Run

- Run id: `healthbench_proposer_gpt54_rerun_20260604T0555`
- Task: `healthbench`
- Proposer: `gpt-5.4`
- Chart: D proposer scaling

## Evidence

- Compact manifest snapshot: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-d-proposer-scaling/figures/manifest_snapshots/healthbench_gpt54.result_manifest.json`
- Snapshot sha256: `34787db0577da109f134bb3010b1fa1aab72584f5794217a328200341d47351c`
- Chart D source packet: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-d-proposer-scaling/figures/source_evidence.json`

## Notes

Chart D reports observed train optimization reward from Synth GEPA runs, not posthoc heldout reward. The heldout seed value above is context from the separate Chart A/C comparison and is not plotted as the Chart D baseline. Raw run artifacts stay in gitignored Chart D run folders; this record cites the compact public manifest snapshot.
