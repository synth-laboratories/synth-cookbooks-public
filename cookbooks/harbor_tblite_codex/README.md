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

1. Copy only public-safe source files and configs.
2. Remove generated files and local caches.
3. Add a `synth-containers` compatibility wrapper if the source runner does not
   already expose the normalized routes.
4. Provide one local smoke command.
5. Provide one Codex-style worker command or documented dry-run.
6. Add the PipelineRL/Tinker execution path.
7. Capture expected artifact, reward, verdict, and uplift paths.

## Result Reporting

Treat this as an applied experiment. Report the outcome as positive, negative,
or inconclusive based on measured reward/verdict output. Do not turn a negative
or inconclusive PipelineRL/Tinker result into a positive cookbook claim.

## Public Safety

Before publishing, confirm that task data, traces, logs, and generated artifacts
are public-safe and do not include private credentials, customer-like data, or
internal service endpoints.
