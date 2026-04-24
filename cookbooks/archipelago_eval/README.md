# Archipelago Eval Wrapper

This cookbook wraps an existing Archipelago-compatible Synth HTTP service or pool URL with
`synth-containers`. It is intentionally an eval wrapper, not a vendored copy of the Rhodes
Archipelago runtime.

## Inputs

- `ARCHIPELAGO_SERVICE_URL` or `--service-url`: base URL for the service or pool that exposes
  `/health`, `/metadata`, `/task_info`, and `/rollout`.
- `--task-path`: optional path to the Archipelago task bundle. This is recorded as a `data`
  resource with subtype `task_bundle`.
- `--world-id` and `--dataset-name`: optional selectors recorded in task metadata and forwarded
  through rollout env config.
- `--auth-token`: optional service auth token. Tokens are not printed.

## Dry Run

```bash
python cookbooks/archipelago_eval/run.py --service-url http://127.0.0.1:9000 --dry-run
```

## Single Rollout

```bash
python cookbooks/archipelago_eval/run.py \
  --service-url http://127.0.0.1:9000 \
  --task-path /path/to/task-bundle \
  --dataset-name crafter \
  --seed 0 \
  --rollout
```
