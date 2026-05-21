# GEPA Cookbook

Public GEPA examples for `synth-optimizers`.

The current public examples are **Banking77** (live OpenAI classifier),
**TBLite** (real Python coding tasks with pytest verifier), **MiniGrid**
(real gymnasium DoorKey env), and **Crafter** (real Craftax episodes).
Every rollout exercises real models and real environments — no fixture
scoring, no string matching. Each container declares its own deps in a
per-container `pyproject.toml` so installing one cookbook does not pull
deps for the others.

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
    synth_service_app.py
  banking77_container/
    README.md
    gepa.toml
    synth_service_app.py     # live OpenAI classifier
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
  _proposers/
    openai_api/
      README.md
      proposer.py            # local_process_json wrapper, OPENAI_API_KEY auth
```

Each example keeps its optimizer config beside the container it exercises. There
is no shared cookbook-level config directory; shared proposer wrappers live
under `_proposers/`.

## Acceptance Path

Set `CONFIG` and `NAME` for the example:

```bash
CONFIG=cookbooks/optimizers/gepa/banking77_container/gepa.toml
NAME=banking77_gepa
```

Available configs (all four use the OpenAI-API proposer + a live policy):

- `cookbooks/optimizers/gepa/banking77_container/gepa.toml` — live OpenAI classifier
- `cookbooks/optimizers/gepa/tblite_container/gepa.toml` — real pytest verifier
- `cookbooks/optimizers/gepa/minigrid_container/gepa.toml` — real gymnasium MiniGrid env
- `cookbooks/optimizers/gepa/crafter_container/gepa.toml` — real Craftax episodes

All require `OPENAI_API_KEY` set in the shell that launches `synth-optimizers`.
Each container installs its own deps on first boot via `uv run --project <container_dir>`.

1. Fresh readwrite run:

   ```bash
   SYNTH_OPTIMIZERS_RUN_ID=${NAME}_fresh \
     synth-optimizers gepa run --config "$CONFIG"
   ```

2. Immediate cached rerun with the same config and cache path:

   ```bash
   SYNTH_OPTIMIZERS_RUN_ID=${NAME}_cached \
     synth-optimizers gepa run --config "$CONFIG"
   ```

3. Readonly replay with the populated cache:

   ```bash
   SYNTH_OPTIMIZERS_RUN_ID=${NAME}_readonly \
   SYNTH_OPTIMIZERS_CACHE_MODE=readonly \
     synth-optimizers gepa run --config "$CONFIG"
   ```

4. Normalized event comparison:

   ```bash
   synth-optimizers events compare \
     --left cookbooks/optimizers/gepa/runs/${NAME}_fresh/events.normalized.jsonl \
     --right cookbooks/optimizers/gepa/runs/${NAME}_cached/events.normalized.jsonl
   ```

TBLite ships with the OpenAI API-key proposer (`backend = "local_process_json"`
pointing at `_proposers/openai_api/proposer.py`); set `OPENAI_API_KEY` and
invoke from the repo root. Banking77 and Crafter still use the deterministic
fixture proposer for cache-replayable cookbooks; live runs against either
container can be configured by editing their `[proposer]` block to point at
the same OpenAI wrapper.

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
  `sandbox_mode`, `approval_policy`, `reasoning_effort`, `copy_host_auth`,
  `api_key_env`, `timeout_seconds`, and `model`.
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
