# HealthBench GEPA · weak Groq policy

This recipe optimizes one mutable health-assistant system prompt against the
normalized `healthbench_chat` Containers target.

This directory is the canonical public HealthBench 2 container and GEPA recipe.
It now includes the runnable service boundary rather than assuming an engineer
already has an unpublished service listening on port 8114.

## Quick start

```bash
./run_container.sh --storage-root /absolute/path/to/a/dedicated/run-store
uv run --project . --group dev pytest -q test_container_contract.py
```

The service binds only to `127.0.0.1` and defaults to port `8114`. Every
Workshop instance should receive a separate durable run store. The policy uses
`GROQ_API_KEY`; canonical physician-rubric grading uses `OPENAI_API_KEY`.
These are independent model roles: the policy generates the candidate response,
while the scorer makes one grading call per physician-authored rubric item.
Their token, cost, and failure telemetry is retained in separate `policy` and
`grader` usage lanes as well as the combined rollout total. `/metadata` declares
both roles so callers do not mistake HealthBench for a single-model container.
Until the HealthBench runtime release reaches PyPI, `pyproject.toml` pins the
exact reviewed public Containers commit; it never floats on a branch head.

The task contract is intentionally honest:

- one mutable `system_prompt` declared by `/program`;
- open-text output, so literal training-target memorization is forbidden;
- physician-rubric reward authority with missing grader evidence preserved as
  missing rather than zero;
- no fabricated live frames, checkpoints, restore, or fork support;
- token-derived costs labeled as estimates and unknown costs left null;
- optimizer-facing GEPA v2 routes declared in `/metadata`.

- Policy: Groq `llama-3.1-8b-instant`.
- Search: 60 train tasks, 50 isolated heldout tasks, three generations, three
  proposals per generation, and 12-task minibatches.
- Spend guard: hard `$4.75` optimizer budget. Missing provider cost remains
  `null`; token-derived estimates carry `cost_kind` and `cost_source`.
- Grading: `gpt-4.1-2025-04-14` is the canonical HealthBench grader. For a
  bounded local search set `HEALTHBENCH_GRADER_MODEL=gpt-4.1-mini-2025-04-14`;
  that run is explicitly labeled `healthbench_scaled_grader.v1`, not canonical
  HealthBench.

The recipe does not call a run successful merely because it completes. A
meaningful receipt must include a distinct candidate, measured train and
heldout scores, positive heldout uplift, complete cost provenance, and the live
optimizer event stream used by Workshop's right-panel visual.

`gepa.toml` is the reviewed current profile. The dated `*.sdk.toml` files are
historical experiment records and should not be copied as defaults.

`eval_smoke.toml` is a bounded, zero-generation baseline receipt: two train and
two heldout examples, two parallel rollout workers, a `$0.50` hard ceiling, and
no proposer calls. It intentionally uses an OpenAI mini policy plus the
canonical OpenAI grader so one healthy credential can validate both independent
model lanes. It measures execution and baseline quality only; it cannot establish
optimizer uplift.

See [`../CONTAINER_ENGINEERING.md`](../CONTAINER_ENGINEERING.md) for the shared
quality standard and paid-run preflight.
