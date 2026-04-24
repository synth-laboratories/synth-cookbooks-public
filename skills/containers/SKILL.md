---
name: containers
description: Use when building, running, or adapting public synth-containers cookbooks for Harbor, OpenEnv, Archipelago, MiniGrid, TBLite, or rollout APIs.
---

# Synth Containers Cookbook Skill

Use this skill when a task mentions `synth-containers`, container contracts,
rollouts, checkpoints, resume, Harbor, OpenEnv, Archipelago, MiniGrid, TBLite,
or PipelineRL data collection from a public Synth cookbook.

## First Files To Read

- `packages/synth-containers/README.md`
- `packages/synth-containers/src/synth_containers/README.md`
- `cookbooks/harbor_tblite_codex/README.md`
- `cookbooks/openenv_minigrid/README.md`
- `cookbooks/archipelago_eval/README.md`

## Default Workflow

1. Identify the framework adapter:
   - Harbor/TBLite: `cookbooks/harbor_tblite_codex/tblite_codex_container/`
   - OpenEnv/MiniGrid: `cookbooks/openenv_minigrid/minigrid_container/`
   - Archipelago: `cookbooks/archipelago_eval/`
2. Prefer the normalized public entrypoint named `synth_service_app.py` when a
   cookbook provides one.
3. Start services with the public package source on `PYTHONPATH`.

Harbor/TBLite:

```bash
PYTHONPATH=packages/synth-containers/src:cookbooks/harbor_tblite_codex/tblite_codex_container \
PORT=8952 python cookbooks/harbor_tblite_codex/tblite_codex_container/synth_service_app.py
```

OpenEnv/MiniGrid:

```bash
PYTHONPATH=packages/synth-containers/src:cookbooks/openenv_minigrid/minigrid_container \
PORT=8932 python cookbooks/openenv_minigrid/minigrid_container/synth_service_app.py
```

4. Check the normalized HTTP surface before running training:
   - `GET /health`
   - `GET /task_info`
   - `POST /rollout`
   - checkpoint/resume routes when the cookbook claims resumability
5. Commit only small public-safe example artifacts under each cookbook's
   `run_artifacts/`.

## Compatibility Expectations

The public package should preserve the shared contract vocabulary for:

- task metadata and catalogs
- rollout request/response formatting
- reward and verifier reporting
- artifacts and traces
- checkpoint, resume, pause, and terminate semantics where supported
- framework adapters for Harbor, OpenEnv, and Archipelago

PipelineRL cookbooks should keep the boundary explicit:

- container runtime owns tasks, observations, actions, rewards, rollouts, and
  checkpoints
- trainer code owns trajectory collection, policy updates, and training
  artifacts
- Modal or Tinker owns remote compute/training when used

## Guardrails

- Use the new public `synth-containers` package, not old
  `synth_ai.sdk.container.server` wrappers.
- Pin legacy `synth-ai` only for migration paths that still require old clients;
  do not make that the default public surface.
- Do not commit private Harbor backend URLs, raw secrets, local absolute paths in
  generated traces, or unredacted logs.
- Label applied PipelineRL results as positive, negative, or inconclusive based
  on measured reward/verdict output.
- For NLE-style rewards, prefer the reusable Scout reward helper in
  `synth_containers.rewards` so the cookbook does not fork reward semantics.

