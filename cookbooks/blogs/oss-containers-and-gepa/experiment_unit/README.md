# The blog experiment unit

A blog chart is a *view*. The durable unit underneath it is an **Experiment**: one
task container compared across a set of **arms**, under a single locked set of
**parity** conditions, producing evidence the charts consume.

This package makes the run-evidence audit **executable** instead of
hand-maintained in the top-level README: each `Experiment` declares its
`ParityLock`, then computes its own `Verdict` by reading the real evidence
files. The CLI also prints a separate **publication packet** footer. Run evidence
can be `FINISHED` while publication remains `PENDING`.

## Verdicts

| Verdict | Meaning | Cost to resolve |
|---|---|---|
| `FINISHED` | run evidence is present, parity checks pass, the candidate/rollout floor passes, and the rows are in the chart authority | no rerun |
| `REPLUMB` | evidence valid but not in the live aggregate | file merge, **no rollouts** |
| `RERUN` | present but invalid (unfair budget / parity drift) **or** mandated | rollouts |
| `MISSING` | no evidence at all | rollouts |

## Checks (run against real evidence, no fabrication)

- `evidence_present` — `evals/evidence/benchmarks/<container>/summary.json` exists
- `in_live_aggregate` — container rows in `evals/evidence/heldout_evaluations.jsonl`
  (a backup bundle does **not** count — that's what `REPLUMB` catches)
- `budget_floor` — per-arm candidate counts and rollout counts both pass the
  parity-counter floor (`arm_ratio >= 0.8`); it does not prove identical budgets
- `parity_proposer` / `parity_policy` / `parity_route` — recorded `parity_controls`
  match the declared `ParityLock`
- `chart_d_manifest_*` / `chart_d_cell_*` — proposer-sweep cells are verified
  through Chart D's own manifest producer, not through the Chart A/C aggregate

`mandate` records a *human* decision to rerun regardless of the checks (a route
switch, an auth-config fix). It forces `RERUN` and is shown verbatim, so an
override is never silent.

## Publication Packet

The footer is intentionally separate from experiment verdicts. It checks whether
the launch packet is ready to publish:

- Chart A/C/D `figures/source_evidence.json` files exist.
- Every path referenced by those `source_evidence.json` packets exists and its
  recorded `sha256` / byte count matches the current file.
- Chart A/C/D frontend mirrors byte-match the producer JSON.
- Active launch docs/producers avoid stale draft or overclaim terms such as old
  source SHAs, `matched-budget`, and draft-local publication language.
- Per-cell `experiment_records/*/README.md` records have been backfilled.
- Launch evidence paths are committed instead of dirty local work:
  `cookbooks/blogs/oss-containers-and-gepa/` plus the Chart A/C
  `evals/evidence` aggregate files and launch benchmark folders.
- Ignored local launch-adjacent nuisance files, such as obsolete handoffs,
  local Chart A configs, or one-off sweep helpers, are reported as a warning so
  a dirty worktree cannot look cleaner than it is.
- TBLite container dirt is reported separately as a nonblocking quarantine
  warning because TBLite is not launch evidence.
- Other dirty GEPA workspace paths are reported separately as a nonblocking
  warning so unrelated Harbor, Harvey, Minigrid, or local optimizer work does
  not get mistaken for launch evidence.

If the footer says `PENDING`, do not publish even when all run evidence says
`FINISHED`.

## Use

```bash
cd cookbooks/blogs/oss-containers-and-gepa
uv run python -m experiment_unit            # status table (the executable audit)
uv run python -m experiment_unit show tau2  # per-check detail + reproduce command
uv run python -m experiment_unit plan       # only what needs work, with commands
uv run python -m experiment_unit packet     # publication paths + full dirty list
```

## Running (parallel, time-boxed)

```bash
uv run python -m experiment_unit run --dry-run            # show the plan
uv run python -m experiment_unit run --max-parallel 4     # execute all RERUN experiments
uv run python -m experiment_unit run --only tau2 --time-limit 1800
```

`run` (see `runner.py`) executes every `RERUN` experiment arm as capped
subprocesses:

- **Hard time limit** (default 30 min/arm). On timeout the run's **process group**
  is killed — which takes its container child with it — so a hung run can't block
  the batch.
- **Abort → heldout-on-partial.** After the arm phase it extracts surviving
  candidates and re-scores them on heldout. synth_gepa checkpoints candidates
  incrementally so an aborted run is still scoreable; **gepa-ai loses in-memory
  candidates on a kill until its adapter checkpoints per generation** (gap in
  `scripts/run_stack.py`) — the runner reports this rather than hiding it.
- **Massively parallel.** Each arm boots its own container on a free port (own
  uvicorn loop) and uses the Gemini-direct route (no shared rpm cap) — the two
  reasons the legacy sweep ran arms sequentially are gone.
- The 30-min cap is a **hang backstop**, not the normal exit: size budgets to
  finish under it. An arm that aborts will have a different candidate count from
  its sibling, so `budget_floor` correctly keeps it out of the published chart.

Proposer sweeps (Chart D) are checked against the six manifest cells consumed by
`charts/chart-d-proposer-scaling/build_chart.py`. The older one-off sweep helper
scripts are local archaeology; `run_sweep.sh` plus the producer are the public
surfaces.

## Adding / changing an experiment

Edit `registry.py`. One `Experiment(...)` per container × chart, with the
`ParityLock` from the README parity tables. The verdict is **never** declared here
— it is computed. To record a decision to rerun, set `mandate=`.

This is the throughput primitive: as new containers land, declare them once and
`status` tells you what is real, what needs replumbing, and what must rerun —
with the exact command for each.
