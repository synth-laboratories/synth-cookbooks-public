# GEPA Cookbook

Public GEPA examples for `synth-optimizers`.

The current public launch examples are **Banking77** (live OpenAI
classifier), **HotpotQA** (multi-hop QA with token-F1 verifier),
**TBLite** (real Python coding tasks with pytest verifier), and
**Crafter** (real Craftax episodes). **MiniGrid** is included as the
next public container scaffold. Every rollout exercises real models and
real environments. Each
container declares its own deps in a per-container `pyproject.toml` so
installing one cookbook does not pull deps for the others.

Additional containers (LangProbe set, Tau-Bench 3, and the vertical-domain
roadmap) flip public over follow-up releases; see
`cookbooks/blogs/oss-containers-and-gepa/README.md` for the full catalog.

## Directory Map

```text
cookbooks/optimizers/gepa/
  README.md
  banking77_container/
    README.md
    gepa.toml
    synth_service_app.py     # live OpenAI classifier
  hotpotqa_container/
    README.md
    pyproject.toml           # datasets, openai, fastapi
    gepa.toml
    synth_service_app.py     # HotpotQA token-F1 verifier
  tblite_container/
    README.md
    pyproject.toml           # openai, pytest
    gepa.toml
    synth_service_app.py     # real pytest subprocess verifier
  minigrid_container/
    README.md
    pyproject.toml           # gymnasium, minigrid, openai
    gepa.toml
    synth_service_app.py     # real MiniGrid DoorKey env
  crafter_container/
    README.md
    pyproject.toml           # craftax, jax[cpu], openai
    crafter_text_env.py
    gepa.toml
    synth_service_app.py     # real Craftax episodes
```

Each example keeps its optimizer config beside the container it exercises. There
is no shared cookbook-level config directory. The public cookbooks use the
Codex app-server proposer so the proposer can inspect a materialized workspace
and write `proposal/manifest.json`.

## Quickstart

Run GEPA from a container directory. The helper script keeps all run parameters
in TOML profiles, generates a one-off local TOML next to the container, and
writes large run artifacts under `cookbooks/optimizers/gepa/runs/`.

```bash
cd cookbooks/optimizers/gepa/banking77_container
export OPENAI_API_KEY="..."       # Codex proposer auth
export OPENROUTER_API_KEY="..."   # policy model auth for the shipped profiles
bash run_fresh_gepa.sh --profile smoke
```

Use `bash run_fresh_gepa.sh --list` to see available profiles. The long
Banking77 profile is:

```bash
cd cookbooks/optimizers/gepa/banking77_container
bash run_fresh_gepa.sh --profile long
```

The HotpotQA profile is the same shape:

```bash
cd cookbooks/optimizers/gepa/hotpotqa_container
bash run_fresh_gepa.sh --profile long
```

Canonical base configs are also runnable directly:

- `cookbooks/optimizers/gepa/banking77_container/gepa.toml` — live Banking77 classifier
- `cookbooks/optimizers/gepa/hotpotqa_container/gepa.toml` — HotpotQA multi-hop QA
- `cookbooks/optimizers/gepa/tblite_container/gepa.toml` — real pytest verifier
- `cookbooks/optimizers/gepa/crafter_container/gepa.toml` — real Craftax episodes

The MiniGrid scaffold is present under
`cookbooks/optimizers/gepa/minigrid_container/` for the next public container
flip.

## Codex Proposer Auth

The public path is API-key-only. Do not rely on a local Codex login for cookbook
runs. Set `OPENAI_API_KEY` in the shell that launches `synth-optimizers` or the
container helper script, and configure the proposer like this:

```toml
[proposer]
backend = "codex_app_server"
execution_mode = "local_process"
model = "gpt-5.4-nano"
auth_mode = "api_key"
api_key_env = "OPENAI_API_KEY"
copy_host_auth = false
sandbox_mode = "workspace-write"
approval_policy = "never"
timeout_seconds = 900
```

With `auth_mode = "api_key"`, Rust GEPA starts `codex app-server` with an
isolated run-local `CODEX_HOME` and passes only the configured API key into the
Codex process. This keeps public cookbook runs independent of any developer's
personal Codex auth state.

The policy model is separate from the proposer. The shipped Banking77 and
HotpotQA profiles use OpenRouter for policy rollouts, so they also need
`OPENROUTER_API_KEY`. If you switch `[policy]` to OpenAI, use
`api_key_env = "OPENAI_API_KEY"` there as well.

Cookbooks ship with the Codex app-server proposer (`backend =
"codex_app_server"`). GEPA materializes a run-local proposer workspace with
`state/algorithm_read_model.json`, `state/candidates.json`,
`state/rollouts.json`, `state/evidence_frames.json`, and
`proposal/PROPOSAL_SCHEMA.md`; Codex must inspect those files and write
`proposal/manifest.json`.

## Container Requirements

A GEPA container is an HTTP service. It can be written in any language as long
as it implements this contract:

- `GET /health`: returns 200 when the container is ready.
- `GET /metadata`: declares the optimizer protocol and route names.
- `GET /task_info`: describes the task, objective, output contract, dataset,
  constraints, and useful prompt-writing guidance for the proposer.
- `GET /program`: returns the prompt program, mutable fields, scoring
  objectives, and rollout overlay schema.
- `GET /dataset`: returns dataset metadata and split names.
- `POST /dataset/rows`: accepts split/seeds and returns concrete dataset rows.
- `POST /rollout`: runs one blocking rollout and returns scores plus metadata.
- Optional async rollout routes: `POST /rollouts`, `GET /rollouts/{id}`,
  `GET /rollouts/{id}/state`, and `POST /rollouts/{id}/terminate`.

The container owns data loading, policy calls, evaluation, and retries at the
task boundary. Rust GEPA only sees HTTP responses and mutable prompt payloads.

Container responses should include enough metadata for the proposer to learn:
the original input, expected output or verifier target when public-safe, model
output, per-objective scores, failure reason, and any task-specific hints. The
`/task_info` route is what keeps the proposer general across classification,
QA, coding, and environment-control tasks.

Each container should declare its dependencies in its own `pyproject.toml`; the
cookbook configs launch containers through `uv run --project <container_dir>`.

## Config Shape

The TOML section shape is frozen for public GEPA v1:

- `[run]`: `run_id`, `output_dir`, and `seed`.
- `[container]`: `url` for a standing container, or `command` and `cwd` for a
  cookbook local process, plus `startup_timeout_seconds`.
- `[dataset]`: `train_split`, `heldout_split`, `train_seeds`, `heldout_seeds`,
  and optional `filters`.
- `[candidate]`: `target_modules` and optional `candidate_id_prefix`.
- `[seed_candidate]`: baseline candidate payload. Keys must match mutable prompt
  fields from `GET /program`.
- `[policy]`: student policy route fields such as `provider`, `model`,
  optional `base_url`, and optional `api_key_env`.
- `[proposer]`: `backend`, `execution_mode`, optional `command`,
  `sandbox_mode`, `approval_policy`, `reasoning_effort`, `auth_mode`,
  `api_key_env`, `copy_host_auth`, `timeout_seconds`, and `model`. Public
  cookbook configs should use `auth_mode = "api_key"` and
  `copy_host_auth = false`.
- `[gepa]`: `max_generations`, `proposals_per_generation`, `minibatch_size`,
  `max_total_rollouts`, `max_cost_usd`, optional `max_time_seconds`, optional
  token limits, and optional proposer/rollout budget estimates used for
  pre-dispatch admission.
- `[cache]`: `mode = "off" | "readwrite" | "readonly"`, `path`, and
  `namespace`.

Readonly mode must fail on the first uncached external boundary. Relative paths
are resolved against the config file directory.

For repeatable cookbook runs, the Rust loader accepts these narrow environment
overrides:

- `SYNTH_OPTIMIZERS_RUN_ID`
- `SYNTH_OPTIMIZERS_OUTPUT_DIR`
- `SYNTH_OPTIMIZERS_CACHE_MODE`
- `SYNTH_OPTIMIZERS_CACHE_PATH`
- `SYNTH_OPTIMIZERS_CACHE_NAMESPACE`
- `SYNTH_OPTIMIZERS_PROPOSER_BACKEND`

The legacy `GEPA_PLATFORM_*` names are accepted as aliases for the same fields.

## Expected Artifacts

Each accepted run writes:

- `result_manifest.json`
- `events.jsonl`
- `events.normalized.jsonl`
- `cache_profile.json`
- `best_candidate.json`
- `candidate_registry.json`
- `frontier.json`
- `run_registry.jsonl` under the configured output root

## Public-Safe Constraints

- The container owns dataset access and rollout execution.
- The optimizer talks to the container over HTTP only.
- The cookbook does not depend on private backend services.
- Every container ships with a real env or real verifier and a live OpenAI
  policy — no fixture scoring, no string-matching rewards.
- Cached replay proves the run can be inspected without spending more model or
  rollout calls.
