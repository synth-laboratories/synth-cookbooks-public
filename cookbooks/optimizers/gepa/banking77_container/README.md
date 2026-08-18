# Banking77 Container

Public Banking77 `synth-containers` service used by the GEPA cookbook.

This container is the only task boundary for the Banking77 optimizer slice. It
must expose prompt-program metadata, dataset rows, and rollout execution over
the standard HTTP routes.

The runnable GEPA config for this example lives beside the container at
`gepa.toml`.

## Quick start

```bash
./run_container.sh
uv run --project . --with pytest pytest -q test_rollout_retention.py
```

The default endpoint is `http://127.0.0.1:8765`. Set `BANKING77_PORT` when a
dedicated Workshop instance needs another port. Terminal sync and async
rollouts are written atomically and remain retrievable by rollout id and
`/reward` after process-memory loss or restart.

## Required Routes

- `GET /metadata`
- `GET /task_info`
- `GET /program`
- `GET /taskset`
- `POST /taskset/tasks`
- `GET /dataset`
- `POST /dataset/rows`
- `POST /rollout`
- `POST /rollouts/prepare`
- `GET /rollouts/{rollout_id}/events?after={sequence}`
- `GET /rollouts/{rollout_id}/events/sse`
- `GET /rollouts/{rollout_id}`
- `GET /rollouts/{rollout_id}/state`
- `GET /reward?rollout_id={rollout_id}`
- `POST /reward`

The metadata payload advertises:

```json
{
  "metadata": {
    "optimizer_contracts": {
      "gepa": {
        "version": "synth_optimizers.gepa.v2",
        "taskset_route": "/taskset",
        "taskset_tasks_route": "/taskset/tasks"
      }
    }
  }
}
```

Current GEPA v2 runners load stable ids such as `train:0` and `test:0` through
`/taskset/tasks`. The older dataset routes remain available for cookbook and
compatibility clients.

Each rollout returns a `synth.rollout.stream.v1` descriptor. Its declared poll
route reads an fsynced `synth.trace-stream-event.v1` journal with a sequence
cursor; the non-advancing `stream.subscribed` record is emitted before rollout
execution. Journals use `BANKING77_STREAM_ROOT` when configured and otherwise
live under `.banking77-streams/` in the service working directory.

## Prompt Program Shape

The `/program` payload describes:

- `program_id`: stable Banking77 program id.
- `modules`: mutable prompt modules.
- `target_modules`: modules GEPA is allowed to mutate.
- `seed_candidate`: baseline candidate payload.
- `rollout_overlay_schema`: how candidate fields map into rollout requests.

Example module:

```json
{
  "module_id": "stage2_system",
  "role": "system",
  "mutable": true,
  "candidate_field": "stage2_system",
  "content": "Classify the customer banking query into exactly one Banking77 intent."
}
```

## Dataset Shape

`POST /dataset/rows` accepts:

```json
{
  "split": "train",
  "seeds": [0, 1, 2],
  "filters": {}
}
```

It returns rows with stable seed, input text, label, and metadata fields.

## Rollout Shape

`POST /rollout` accepts a candidate overlay:

```json
{
  "rollout_id": "optional-client-id",
  "submission_mode": "sync",
  "task_id": "banking77.intent_classification",
  "seed": 0,
  "candidate": {
    "stage2_system": "..."
  }
}
```

It returns the standard `synth-containers` rollout payload with `reward_info`,
`summary`, `usage`, `trace`, `metadata`, and `split_identity`.

`GET /metadata` advertises:

- `literal_training_targets` default `forbid` (opt in with `BANKING77_LITERAL_TRAINING_TARGETS=allow`)
- `leakage_contract {policy, protected_split, span_digest_route}` and per-example sha256 digests at `GET /leakage/span_digests`
- `execution {policy_concurrency, timeout, retries}`
- `split_identity {dataset_id, dataset_digest, train_seed, test_seed, samples, sampling_mode}`

## Non-Goals

- No private policy router.
- No optimizer code in the container.
- No MIPROv2 contract.
- No direct private dataset paths.

## Engineering ownership

See [`../CONTAINER_ENGINEERING.md`](../CONTAINER_ENGINEERING.md) for the shared
quality bar, paid-run preflight, and the corresponding HealthBench 2 path.
