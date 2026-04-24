# MIPROv2 Optimizer Cookbook

Second MVP cookbook for instruction optimization with MIPROv2. The first
concrete run is Banking77.

## Goal

Show a compact instruction-optimization workflow where MIPROv2 improves an
instruction prompt and reports the result honestly on heldout data.

The target public example is the Banking77 confusable slice because the private
draft has a measured positive heldout lift. Easier Banking77 slices can be
included as diagnostics, but they should not be marketed as lift examples when
the baseline saturates.

## Source

Initial migration references:

- private MIPROv2 draft runbook and runner
- private Banking77 confusable 30x300 / 100-heldout config
- `evals/containers/arbitrary/banking77/`
- `evals/benchmarks/reportbench_synth/banking77_offline_gepa_pool/`

## MVP Shape

Current structure:

```text
optimizers/miprov2/
  README.md
  miprov2/
  banking77_container/
  run.py
  run_artifacts/
```

- `banking77_container/` holds the public-safe Banking77 task runtime, dataset
  adapter, scoring, and container metadata.
- `banking77_container/synth_service_app.py` is the public `synth-containers`
  contract entrypoint.
- `miprov2/` holds the MIPROv2 instruction-optimization glue.
- `run.py` is the single public example entrypoint.
- `run_artifacts/` holds small committed example outputs from `run.py`.

## Run

Generate the public-safe dry-run artifacts:

```bash
python cookbooks/optimizers/miprov2/run.py
```

This writes:

- `run_artifacts/dry_run/plan.json`
- `run_artifacts/dry_run/known_evidence.json`
- `run_artifacts/dry_run/summary.md`

The live `--execute` path runs through native Synth MIPRO phase-3 with the
OpenEnv proposer loop.

## Interactive Proposer And Rollout Queues

Native phase-3 can pause at proposer boundaries for manual/Codex-driven
inspection:

```bash
python cookbooks/optimizers/miprov2/run.py \
  --execute \
  --smoke \
  --interactive-proposer \
  --artifacts-dir .out/miprov2_interactive_smoke
```

At each proposer boundary the run writes a `before_proposer` checkpoint, creates
a file-backed proposer session, and includes a tentative rollout queue in the
session state. The proposer flow is:

```text
read evidence -> inspect memory -> refine hypothesis
-> make/resolve bet -> curate/query labels
-> score/query open-endedness -> preview/override/commit queue
-> patch candidate -> finish/commit session -> resume
```

TPE is the default scheduler, not the final authority. If the proposer does not
commit a queue, phase-3 auto-commits the post-proposer TPE default. If the
proposer commits a queue, phase-3 preserves both the original TPE queue and the
committed queue under `miprov2_artifacts/rollout_queues/`, then executes the
committed candidate plan before pausing at the next proposer boundary.

Hypotheses and bets are durable proposer memory, not optimizer rewards. Use
`register_hypothesis`, `append_hypothesis_adjustment`, and `query_hypotheses` to
track claims about agent x dataset x preference structure. Use `register_bet`,
`resolve_bet`, and `query_bets` to make those claims falsifiable against future
rollouts. Queue overrides and commits can link to hypothesis/bet ids so later
analysis can see why budget was spent.

Labels are also durable proposer memory. Use
`register_rollout_label_definition` to curate a small task-specific schema,
`assign_rollout_label` for completed rollout evidence, and
`query_rollouts_by_label` before planning queue overrides. Phase-3 can accept an
optional caller-owned labeller hook; if there are no active label definitions,
the labeller pass is skipped and no free-form labels are invented.

Open-endedness scores are proposer memory too. Use
`score_rollout_open_endedness` and `query_open_ended_rollouts` to track novelty,
unexpectedness, and learnability for high-signal rollouts. These scores are not
task reward; they describe information value for future search.

The proposer runbook policy is controlled by
`MiproOpenEnvProposerConfig.proposer_runbook_policy`:

- `warn` is the default for normal experiments. It records counters, warnings,
  and next-step reminders while preserving proposer freedom.
- `enforce_core` is for controlled evals. It blocks only candidate patches
  before evidence reads, nontrivial queue overrides before a hypothesis/bet, and
  finish before meaningful progress.
- `off` suppresses runbook diagnostics and keeps the older proposer behavior.

Runbook summaries are written into proposer outcomes, traces, interactive
session summaries, and checkpoint replay outputs so prompt/tool/model variants
can be compared on both reward and process quality.

Resume a committed session with:

```bash
python cookbooks/optimizers/miprov2/run.py \
  --execute \
  --smoke \
  --interactive-proposer \
  --interactive-resume-session-id <session_id> \
  --artifacts-dir .out/miprov2_interactive_smoke
```

Run the public `synth-containers` service:

```bash
PYTHONPATH=packages/synth-containers/src:cookbooks/optimizers/miprov2/banking77_container \
PORT=8942 python cookbooks/optimizers/miprov2/banking77_container/synth_service_app.py
```

Then call `/health`, `/task_info`, and `/rollout` on `http://127.0.0.1:8942`.

## Reproduce Native MIPROv2 Result

The current release-candidate config is:

```text
miprov2/banking77_openai_split_confusable_30x300_100heldout.json
```

Despite the historical filename, the config now runs the stratified native
OpenEnv proposer setup:

- `100` stratified train rows across the `20` configured confusable labels
- `200` stratified heldout rows
- `8` native OpenEnv proposer sessions
- `16` target train candidates
- policy model: `gpt-4.1-nano`
- proposer model: `gpt-5.4-mini`

Run it with:

```bash
python cookbooks/optimizers/miprov2/run.py \
  --execute \
  --artifacts-dir cookbooks/optimizers/miprov2/run_artifacts/stratified_100x200_retry
```

The committed lightweight evidence lives under:

```text
run_artifacts/stratified_100x200_retry/live_native_run/
```

Latest successful native run:

- optimizer path: `native_phase3_openenv`
- train rows: `100`
- heldout rows: `200`
- total metric calls: `6000`
- proposer rounds: `8`
- baseline train: `0.39`
- best train: `0.51`
- heldout baseline: `0.495`
- heldout best: `0.57`
- heldout lift: `+0.075`

The full local run also writes SQLite ledger state, a large run read model, and
full proposer transcripts. Those are reproducible from the command above and
are intentionally kept out of the lightweight committed artifact set.

Implementation steps:

1. Install `synth-optimizers` with MIPROv2. Done in `packages/synth-optimizers`.
2. Load or serve the public-safe Banking77 task data. Done in
   `banking77_container/synth_service_app.py`.
3. Run instruction MIPROv2 on the confusable slice with native phase-3 OpenEnv
   proposer. Done in `run.py --execute`.
4. Write `summary.json`, artifact manifest, proposer traces, and rollout
   evidence. The native path writes run summary, manifest, heldout evaluation,
   event stream, read model, best-candidate artifacts, and proposer traces.
5. Report baseline train, best train, heldout baseline, heldout best, and lift.
   Known evidence is preserved in `run_artifacts/dry_run/known_evidence.json`.

## Known Evidence To Preserve

Private draft result for Banking77 confusable 30x300, heldout 100, `K=10`:

- baseline train: `0.7666666667`
- best train: `0.8333333333`
- heldout baseline: `0.7900`
- heldout best: `0.8200`
- heldout lift: `+0.0300`

## Public Safety

Before publishing, confirm the dataset source, configs, traces, and generated
artifacts are public-safe and contain no private credentials, private endpoints,
or unredacted internal logs.
