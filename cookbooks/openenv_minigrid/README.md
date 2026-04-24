# MiniGrid PipelineRL via Modal

Third MVP cookbook for adapting a MiniGrid-style environment into the public
`synth-containers` contract and running a scoped PipelineRL training path via
Modal.

## Goal

Show how an OpenEnv-style environment maps into the shared container runtime
surface:

- `reset`
- `step`
- `state`
- task registry
- blocking rollout
- async rollout lifecycle
- snapshot or checkpoint proof

This is the first public OpenEnv adapter example and the first Modal PipelineRL
cookbook. The local adapter smoke comes first; Modal training comes only after
the MiniGrid runtime produces stable rollouts, rewards, and snapshot/checkpoint
proofs.

## Source

Initial migration source:

- `evals/containers/minigrid/README.md`
- `evals/containers/minigrid/container_spec.json`
- `evals/containers/minigrid/service_app.py`
- `evals/containers/minigrid/synth_service_app.py`
- `evals/containers/minigrid/task_registry.json`
- `evals/containers/minigrid/smoke.py`
- `evals/containers/minigrid/snapshot_proof.py`
- `nanolong/reference/nanolong_qwen35_modal/README.md`
- `nanolong/reference/nanolong_qwen35_modal/train_rl_cispo_modal.py`
- `nanolong/reference/nanolong_qwen35_modal/modal_common.py`
- `nanolong/reference/nanolong_qwen35_modal/synth_task_app.py`

## MVP Shape

Current structure:

```text
openenv_minigrid/
  README.md
  pipelinerl_modal/
  minigrid_container/
  run.py
  run_artifacts/
```

- `minigrid_container/` holds the public-safe MiniGrid runtime, task registry,
  container metadata, smoke path, and snapshot/checkpoint proof path.
- `minigrid_container/synth_service_app.py` is the public `synth-containers`
  contract entrypoint.
- `pipelinerl_modal/` holds the Modal PipelineRL training/inference glue.
- `run.py` is the single public example entrypoint.
- `run_artifacts/` holds small committed example outputs from `run.py`.

## Run

Generate the public-safe dry-run artifacts:

```bash
python cookbooks/openenv_minigrid/run.py
```

This writes:

- `run_artifacts/dry_run/plan.json`
- `run_artifacts/dry_run/summary.md`

Run the public `synth-containers` service:

```bash
PYTHONPATH=packages/synth-containers/src:cookbooks/openenv_minigrid/minigrid_container \
PORT=8932 python cookbooks/openenv_minigrid/minigrid_container/synth_service_app.py
```

Run the local smoke and snapshot proof only after that service is already
running:

```bash
python cookbooks/openenv_minigrid/run.py --execute-local-smoke
```

The default command does not start local services and does not launch Modal.

Implementation steps:

1. Copy only public-safe source files and configs. Done for the MVP source set.
2. Remove generated files and local caches. Done.
3. Prefer `synth_service_app.py` as the normalized public entrypoint if it
   already matches the `synth-containers` route vocabulary. Done.
4. Provide one local service startup command. Captured in `run.py` artifacts.
5. Provide one smoke command. Captured in `run.py` artifacts.
6. Provide one snapshot/checkpoint proof command. Captured in `run.py` artifacts.
7. Add a Modal PipelineRL training config. Scoped in `pipelinerl_modal/`.
8. Run a small baseline eval before training.
9. Run Modal PipelineRL.
10. Run a final eval and report whether the result is positive, negative, or
    inconclusive.

## Modal PipelineRL Scope

Use Modal for the training/inference path, but keep MiniGrid itself behind the
same public `synth-containers` runtime contract. The cookbook should make the
boundary clear:

- MiniGrid runtime: tasks, observations, actions, rewards, rollouts, snapshots
- PipelineRL trainer: trajectory collection, policy update, artifacts
- Modal: remote training/inference compute

The first version should be intentionally small: short rollout budget, small
model, explicit cost/credential prerequisites, and no headline claim unless the
final eval proves it.

## Expected Outputs

- local smoke result
- snapshot/checkpoint proof result
- baseline eval summary
- Modal training artifact directory
- final eval summary
- result note labeling the run as positive, negative, or inconclusive

## Public Safety

MiniGrid should be public-safe by default, but still confirm that no local
absolute paths, generated logs, or private endpoints are included in the final
cookbook.
