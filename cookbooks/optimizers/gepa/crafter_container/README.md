# Crafter GEPA Container

Real Craftax episodes optimized via the public GEPA contract.

This container runs full Craftax episodes against a live OpenAI-driven agent
on each rollout. GEPA optimizes one mutable lever — `react_system_prompt` —
and reward is the actual total environment reward from the episode.

No fixture, no string-matching scorer. Each rollout costs real model tokens.

## Required env

- `OPENAI_API_KEY` — passed through to the container process.
- Optional: `CRAFTER_POLICY_MODEL` (default `gpt-4.1-nano`),
  `CRAFTER_MAX_TURNS` (default `20`), `CRAFTER_MIN_BATCH` (default `1`),
  `CRAFTER_MAX_BATCH` (default `5`), and `CRAFTER_STREAM_ROOT` (default
  `.crafter-streams` in the service working directory).

## Per-container dependencies

Declared in [`pyproject.toml`](./pyproject.toml). Installing this cookbook
does **not** require dependencies for other cookbooks.

- `craftax`, `jax[cpu]` — the real Crafter env (Craftax fork on jax)
- `openai` — live policy + (separately) live proposer
- `fastapi`, `uvicorn`, `numpy`

First-time boot installs ~70 packages and processes Craftax textures
(~30s). Cached for subsequent runs.

## GEPA v2 contract

- `GET /metadata` advertises `synth_optimizers.gepa.v2` and its absolute
  program, taskset, task-loading, and rollout routes.
- `GET /program` exposes one mutable module: `react_system_prompt`.
- `GET /taskset` and `POST /taskset/tasks` expose stable identifiers such as
  `train:11` and return those exact identifiers with each episode seed row.
- The older dataset routes remain available for compatibility clients.
- `POST /rollout` runs a real episode:
  - Instantiates `CrafterTextEnv` (real Craftax env)
  - Resets with the row's seed
  - For up to `CRAFTER_MAX_TURNS` turns:
    - Calls OpenAI with the candidate's `react_system_prompt` as system,
      a compact text observation as user, and the `crafter_interact` tool.
    - Parses native tool calls (or `<tool_call>` XML fallback) into actions.
    - Steps the env once per action; accumulates real env reward.
  - Returns `reward_info.outcome_reward = total_episode_reward` and a
    `synth.rollout.stream.v1` descriptor.
- `POST /rollouts/prepare`, `GET /rollouts/{rollout_id}/events`, and
  `GET /reward` expose the poll transport and authoritative environment reward.
  The sequence journal is fsynced under `CRAFTER_STREAM_ROOT`, beginning with a
  non-advancing `stream.subscribed` control record.

## Cost notes

Real episodes are not free. The default `gepa.toml` is small on purpose:
2 train seeds, 1 heldout seed, 1 generation, 1 proposal — about 30–60
OpenAI calls per full GEPA pass. Scale `train_seeds`, `heldout_seeds`,
`max_generations`, `proposals_per_generation`, and `CRAFTER_MAX_TURNS`
once you're confident in the setup.
