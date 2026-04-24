# Harbor/TBLite Codex PipelineRL/Tinker

MVP cookbook for packaging a Harbor task pool behind the public
`synth-containers` contract, running a Codex-style worker against it, and
testing PipelineRL via Tinker on that applied loop.

## Goal

Show how a Harbor task pool maps into the shared container runtime surface:

- metadata
- task catalog
- rollout
- artifacts
- reward / verdict output

This is the first applied public cookbook. The experiment is to test whether
PipelineRL via Tinker can improve the Codex worker loop on Harbor/TBLite tasks.
MIPROv2 can enter later as a follow-up optimizer for worker instructions, task
handoffs, or rubric text.

## Source

Initial migration source:

- `evals/containers/harbor/tblite/README.md`
- `evals/containers/harbor/tblite/tb_lite_harbor_pool.json`
- `evals/containers/harbor/tblite/run_local_eval.py`
- `evals/containers/harbor/tblite/run_tblite_codex_nano_once.py`
- `evals/containers/harbor/tblite/smoke.py`
- `evals/containers/harbor/tblite/tb_lite_dataset.py`

## MVP Shape

Current structure:

```text
harbor_tblite_codex/
  README.md
  pipelinerl_tinker/
  tblite_codex_container/
  run.py
  run_artifacts/
```

- `tblite_codex_container/` holds the public-safe Harbor/TBLite task-pool
  runtime, task data adapter, container metadata, and reward/verdict formatting.
- `tblite_codex_container/synth_service_app.py` is the public
  `synth-containers` contract entrypoint.
- `pipelinerl_tinker/` holds the PipelineRL/Tinker glue for the Codex worker
  loop.
- `run.py` is the single public example entrypoint.
- `run_artifacts/` holds small committed example outputs from `run.py`.

## Run

Generate the public-safe dry-run artifacts:

```bash
python cookbooks/harbor_tblite_codex/run.py
```

This writes:

- `run_artifacts/dry_run/plan.json`
- `run_artifacts/dry_run/summary.md`

Run one live backend rollout only after a Harbor backend and `SYNTH_API_KEY`
are configured:

```bash
uv run --with synth-ai==0.9.11 \
  python cookbooks/harbor_tblite_codex/run.py --execute-one-shot
```

Run the public `synth-containers` contract wrapper:

```bash
PYTHONPATH=packages/synth-containers/src:cookbooks/harbor_tblite_codex/tblite_codex_container \
PORT=8952 python cookbooks/harbor_tblite_codex/tblite_codex_container/synth_service_app.py
```

Then call `/health`, `/task_info`, and `/rollout` on `http://127.0.0.1:8952`.
The old Harbor backend one-shot path is migration reference only; it is not the
public cookbook surface.

The live Harbor/Codex path pins `synth-ai==0.9.11` because that is the SDK
version whose pool client import is `synth_ai.sdk.container_pools`.

Implementation steps:

1. Copy only public-safe source files and configs. Done for the MVP source set.
2. Remove generated files and local caches. Done.
3. Add a `synth-containers` compatibility wrapper if the source runner does not
   already expose the normalized routes. Done via `synth_service_app.py`.
4. Provide one local smoke command. Source copied; live execution remains opt-in.
5. Provide one Codex-style worker command or documented dry-run. Done.
6. Add the PipelineRL/Tinker execution path. Scoped in `pipelinerl_tinker/`.
7. Capture expected artifact, reward, verdict, and uplift paths.

## Result Reporting

Treat this as an applied experiment. Report the outcome as positive, negative,
or inconclusive based on measured reward/verdict output. Do not turn a negative
or inconclusive PipelineRL/Tinker result into a positive cookbook claim.

## Public Safety

Before publishing, confirm that task data, traces, logs, and generated artifacts
are public-safe and do not include private credentials, customer-like data, or
internal service endpoints.
