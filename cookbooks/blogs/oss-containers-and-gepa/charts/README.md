# Charts

Chart **producers** live here. They read committed experiment data and emit the
JSON, Markdown, and SVG the blog post consumes.

Current launch producers are **A** and **C**. Chart **D** is present only as
post-launch draft/debug context after the fresh all-cell rerun failed. Historical
and future addenda are not committed in this launch packet; re-add them only with
fresh evidence packets and an explicit active-MDX section.

Experiments are the source of truth. Chart scripts never own raw rollout data —
they aggregate from [`../experiment_records/`](../experiment_records/) (per-cell
folders) or, during migration, from the shared eval harness at
`cookbooks/optimizers/gepa/evals/evidence/`.

Shared path helpers: [`../blog_paths.py`](../blog_paths.py).

## Layout

```text
charts/
  README.md
  chart-a-head-to-head/       build_chart.py — Synth vs gepa-ai table (Chart A)
  chart-c-use-case-coverage/  build_heldout_coverage.py — cumulative coverage (Chart C)
  chart-d-proposer-scaling/   build_chart.py — proposer model sweep (Chart D)
```

Each chart folder contains:

- **Producer script(s)** — read experiment evidence, write `figures/`
- **`figures/`** — committed derived data + `source_evidence.json` checksums
- **`README.md`** — parity conditions and reproduce commands
- **`runs/`** (optional, gitignored) — local sweep outputs when the chart owns
  its own run launcher (currently Chart D)

## Data flow

```text
experiment_records/<cell>/evidence/     ─┐
evals/evidence/benchmarks/<task>/       ─┼──► charts/<chart>/build_*.py ──► figures/*.json
experiment_records/<cell>/derived/      ─┘                              └──► frontend/data/*.json
```

## Reproduce launch charts

From the repo root:

```bash
python cookbooks/blogs/oss-containers-and-gepa/charts/chart-a-head-to-head/build_chart.py
python cookbooks/blogs/oss-containers-and-gepa/charts/chart-c-use-case-coverage/build_heldout_coverage.py
```

## Chart registry

| Chart | Folder | Producer | Launch | Inputs |
|:-----:|--------|----------|:------:|--------|
| A | [chart-a-head-to-head/](./chart-a-head-to-head/) | `build_chart.py` | in post | E01 parity summaries per task |
| C | [chart-c-use-case-coverage/](./chart-c-use-case-coverage/) | `build_heldout_coverage.py` | in post | E01 heldout eval JSONL + summaries |
| D | [chart-d-proposer-scaling/](./chart-d-proposer-scaling/) | `build_chart.py` | post-launch draft only | E05 compact proposer-sweep manifests; raw rerun failed |

Master experiment status grid: [blog README § Experiments](../README.md#experiments).
