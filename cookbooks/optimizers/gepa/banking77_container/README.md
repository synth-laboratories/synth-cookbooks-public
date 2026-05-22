# Banking77 Container

Public Banking77 `synth-containers` service used by the GEPA cookbook.

This container is the only task boundary for the Banking77 optimizer slice. It
must expose prompt-program metadata, dataset rows, and rollout execution over
the standard HTTP routes.

The runnable GEPA config for this example lives beside the container at
`gepa.toml`.

## Required Routes

- `GET /metadata`
- `GET /task_info`
- `GET /program`
- `GET /dataset`
- `POST /dataset/rows`
- `POST /rollout`

The metadata payload advertises:

```json
{
  "metadata": {
    "optimizer_contracts": {
      "gepa": {
        "version": "synth_optimizers.gepa.v1"
      }
    }
  }
}
```

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
`summary`, `usage`, `trace`, and `metadata`.

## Non-Goals

- No private policy router.
- No optimizer code in the container.
- No MIPROv2 contract.
- No direct private dataset paths.
