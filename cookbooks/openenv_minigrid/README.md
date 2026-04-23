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

1. Copy only public-safe source files and configs.
2. Remove generated files and local caches.
3. Prefer `synth_service_app.py` as the normalized public entrypoint if it
   already matches the `synth-containers` route vocabulary.
4. Provide one local service startup command.
5. Provide one smoke command.
6. Provide one snapshot/checkpoint proof command.
7. Add a Modal PipelineRL training config.
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
