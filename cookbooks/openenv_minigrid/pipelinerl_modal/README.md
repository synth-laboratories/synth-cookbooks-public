# PipelineRL via Modal

This folder holds the Modal training side of the MiniGrid cookbook.

The implementation is separate from the MiniGrid container so readers can see
the boundary clearly:

- `../minigrid_container/` exposes tasks, rollouts, rewards, state, checkpoints,
  and resume.
- this folder owns remote training and inference plumbing.
- `../run.py` writes a small artifact plan by default and can optionally run
  local smoke/proof commands against a MiniGrid service.

The copied `train_rl_cispo_modal.py` reference is intentionally not launched by
default. Modal jobs require credentials, budget, and a task-app endpoint.

## MVP Modal Command Shape

```bash
modal run pipelinerl_modal/train_rl_cispo_modal.py --cmd '
python train_rl_cispo_modal.py
  --reward-backend synth_task_app
  --task-app-url https://YOUR-TASK-APP.modal.run
  --dataset-path /data/minigrid_rollouts.jsonl
  --output-dir /checkpoints/minigrid-pipelinerl
'
```

The first publishable result should include baseline reward, final reward, the
heldout split size, and an explicit `positive`, `negative`, or `inconclusive`
label.
