# Experiment records

**Primary organization for this blog cookbook.** Each runnable cell gets one
folder: container × chart × setup (for example HealthBench Pro · Chart A · Synth
GEPA).

Chart producers under [`../charts/`](../charts/) currently read the shared eval
evidence for launch A/C and compact Chart D manifest snapshots for post-launch
draft
proposer view, then write derived JSON to `charts/<chart>/figures/`. These
records are the compact per-cell provenance index for that generated evidence;
they do not mean the Chart D raw rerun gate is green.

Master status grid: [blog README § Experiments](../README.md#experiments).

## Folder layout

```text
experiment_records/
  README.md
  <container>__chart_<letter>__<setup>/
    README.md           # run id, key metrics, source packets, checksums
    config/             # optional future copy of TOML/adapter config
    run_manifest.json   # optional future compact run manifest
    evidence/           # optional future copied summary/checksum packet
    derived/            # optional future chart-specific slice
```

Naming convention: lowercase container slug, double underscore separators,
setup slug with dots preserved (`gpt-5.4-nano`).

Use [`../blog_paths.py`](../blog_paths.py) helpers:

```python
from blog_paths import experiment_dir
experiment_dir("healthbench_pro", "a", "synth_gepa")
# → experiment_records/healthbench_pro__chart_a__synth_gepa/
```

## Where raw runs live

Full run artifacts (manifests, SQLite, traces) stay in gitignored `runs/` trees:

| Experiment type | Raw run location |
|-----------------|------------------|
| E01 parity (Charts A, C) | `cookbooks/optimizers/gepa/evals/runs/<stack>/<benchmark>/` |
| E05 proposer (Chart D draft) | `charts/chart-d-proposer-scaling/runs/final_20260603/` |

The current launch packet uses committed per-cell `README.md` records rather
than copying raw run artifacts. Each README points at the authoritative
checksummed summary packet, aggregate JSONL packet, or compact Chart D manifest
snapshot.

## Migration status

Current launch scope is limited to HealthBench Pro, tau2-bench retail,
Banking77, and HotpotQA for Charts A/C. The HealthBench Pro and tau2-bench
retail Chart D proposer sweep is post-launch draft-only until the raw rerun gate
is green and the MDX explicitly re-adds it.

Backfill status:

1. DONE cells: launch-scope Chart A/C rows for HealthBench Pro, tau2-bench
   retail, Banking77, and HotpotQA have per-cell README records.
2. SNAPSHOT cells: Chart D proposer sweep rows for HealthBench Pro and tau2-bench
   retail have per-cell README records, but the raw rerun evidence is still
   gated by `experiment_unit status`.
3. POST-LAUNCH ONLY: Harvey, TaxCalcBench, FinQA, TBLite, Crafter, MiniGrid,
   DungeonGrid, and LangProbe-style addenda require fresh evidence packets and
   explicit MDX re-add before they can appear in any public chart.
