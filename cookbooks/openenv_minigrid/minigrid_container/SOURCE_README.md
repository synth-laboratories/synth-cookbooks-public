# MiniGrid Go-Ex Container

This bundle is the first-class MiniGrid environment surface for Go-Explore prompt bring-up.

It exposes:

- `GET /health`
- `GET /task_info`
- `POST /rollout`
- `GET /rollouts/{rollout_id}/state`
- `POST /rollouts/{rollout_id}/pause`
- `POST /rollouts/{rollout_id}/terminate`
- `POST /rollouts/{rollout_id}/checkpoints`
- `GET /checkpoints/{checkpoint_id}`
- `POST /rollouts/{rollout_id}/resume`

Checkpoint semantics are advertised as:

- `restore_semantics = "true_environment_snapshot"`
- `true_environment_snapshot = true`
- `supports_branching = true`

MiniGrid v1 uses fully observed symbolic JSON state, not RGB.

The rollout reward is environment-grounded but shaped with small milestone bonuses for
symbolically verified progress such as key acquisition, door unlocks, new-room entry,
goal visibility, and goal completion.

External long-horizon consumers should treat `GET /task_info` as self-contained.
The canonical dataset metadata now includes:

- `dataset.task_registry`: inline curated task registry payload
- `dataset.task_registry_path`: local compatibility path only

External callers should prefer the inline registry and not rely on a local file
path being meaningful across process or container boundaries.

## Run locally

```bash
PORT=8922 python minigrid_container/service_app.py
```

## Smoke

With the service already running:

```bash
python minigrid_container/smoke.py --base-url http://127.0.0.1:8922
```

## Snapshot proof

This captures a checkpoint after one short rollout step, resumes from it, and compares the restored symbolic observation frontier:

```bash
python minigrid_container/snapshot_proof.py --base-url http://127.0.0.1:8922
```
