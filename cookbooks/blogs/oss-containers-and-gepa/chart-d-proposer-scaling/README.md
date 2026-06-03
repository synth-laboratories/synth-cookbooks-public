# Chart D — Proposer Scaling

Does optimizer quality ride the proposer-model curve? Sweep the
proposer model across three generations and measure best heldout reward
at a fixed compute budget, holding all other variables constant.

## Sweep grid

- **Proposer models** (X-axis, ordered by capability):
  - `gpt-5.4-nano`  (`reasoning_effort = "low"`)
  - `gpt-5.4-mini`  (`reasoning_effort = "medium"`)
  - `gpt-5.4`       (`reasoning_effort = "high"`)
- **Tasks** (two panels in the bar chart):
  - HealthBench Pro  (medical QA scored by rubric)
  - tau2-bench retail  (tool-using customer-service episodes)
- **Held constant across all cells**:
  - Optimizer: Synth GEPA
  - HealthBench policy/judge: `google/gemini-2.5-flash-lite` via OpenRouter
  - tau2 retail agent: `openrouter/google/gemini-3.1-flash-lite`
  - Train + heldout seeds: same per task
  - Proposer auth: nano uses API-key auth because ChatGPT Codex auth rejects
    `gpt-5.4-nano`; mini and `gpt-5.4` use ChatGPT Codex auth.

## Configs

```
configs/proposer_sweep/
  healthbench_gpt-5.4-nano.toml
  healthbench_gpt-5.4-mini.toml
  healthbench_gpt-5.4.toml
  tau2_retail_gpt-5.4-nano.toml
  tau2_retail_gpt-5.4-mini.toml
  tau2_retail_gpt-5.4.toml
```

## Layout

```
chart-d-proposer-scaling/
  README.md
  configs/
    proposer_sweep/
      healthbench_gpt-5.4-nano.toml
      healthbench_gpt-5.4-mini.toml
      healthbench_gpt-5.4.toml
      tau2_retail_gpt-5.4-nano.toml
      tau2_retail_gpt-5.4-mini.toml
      tau2_retail_gpt-5.4.toml
  run_sweep.sh            # boots containers, runs all 6 configs in parallel, builds chart
  build_chart.py          # reads runs/, emits figures/ plus public manifest snapshots
  runs/
    final_20260603/
      healthbench_nano/
      healthbench_mini/
      healthbench_gpt54/
      tau2_retail_nano/
      tau2_retail_mini/
      tau2_retail_gpt54/
  figures/
    manifest_snapshots/
      healthbench_nano.result_manifest.json
      healthbench_mini.result_manifest.json
      healthbench_gpt54.result_manifest.json
      tau2_retail_nano.result_manifest.json
      tau2_retail_mini.result_manifest.json
      tau2_retail_gpt54.result_manifest.json
    proposer_scaling_data.json
    proposer_scaling.md
    proposer_scaling.svg
    source_evidence.json
```

## Reproduce

Boot containers and run all 6 cells from the repo root:

```bash
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
bash cookbooks/blogs/oss-containers-and-gepa/chart-d-proposer-scaling/run_sweep.sh
```

Smoke mode (nano + mini on HealthBench only):

```bash
bash cookbooks/blogs/oss-containers-and-gepa/chart-d-proposer-scaling/run_sweep.sh --smoke
```

Rebuild figures from existing runs (no new rollouts):

```bash
cd cookbooks/blogs/oss-containers-and-gepa/chart-d-proposer-scaling
uv run python build_chart.py
```

## Status

- [x] All 6 configs verified against running containers.
- [x] Sweep launched and completed.
- [x] `figures/proposer_scaling.svg` rendered.
- [x] Section in blog MDX embeds chart.

## Design notes

- `run_sweep.sh` launches all cells in parallel against shared task containers;
  the containers handle concurrent rollout requests.
- `build_chart.py` requires all six launch manifests, validates core numeric
  fields, and fails before writing JSON if any cell is incomplete. It mirrors
  the six result manifests into `figures/manifest_snapshots/` so the public
  chart directory remains inspectable without committing ignored run databases,
  traces, or failed attempts.
- The producer writes both `figures/proposer_scaling_data.json` and the sibling
  frontend mirror used by the blog component.
- The launch run group is `runs/final_20260603`.
