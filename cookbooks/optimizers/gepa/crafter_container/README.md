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
  `CRAFTER_MAX_BATCH` (default `5`).

## Per-container dependencies

Declared in [`pyproject.toml`](./pyproject.toml). Installing this cookbook
does **not** require dependencies for other cookbooks.

- `craftax`, `jax[cpu]` — the real Crafter env (Craftax fork on jax)
- `openai` — live policy + (separately) live proposer
- `fastapi`, `uvicorn`, `numpy`

First-time boot installs ~70 packages and processes Craftax textures
(~30s). Cached for subsequent runs.

## Contract

- `GET /metadata` advertises `synth_optimizers.gepa.v1`.
- `GET /program` exposes one mutable module: `react_system_prompt`.
- `POST /dataset/rows` returns episode seed rows (each row is a Craftax env seed).
- `POST /rollout` runs a real episode:
  - Instantiates `CrafterTextEnv` (real Craftax env)
  - Resets with the row's seed
  - For up to `CRAFTER_MAX_TURNS` turns:
    - Calls OpenAI with the candidate's `react_system_prompt` as system,
      a compact text observation as user, and the `crafter_interact` tool.
    - Parses native tool calls (or `<tool_call>` XML fallback) into actions.
    - Steps the env once per action; accumulates real env reward.
  - Returns `reward_info.outcome_reward = total_episode_reward`.

## Cost notes

Real episodes are not free. The default `gepa.toml` is small on purpose:
2 train seeds, 1 heldout seed, 1 generation, 1 proposal — about 30–60
OpenAI calls per full GEPA pass. Scale `train_seeds`, `heldout_seeds`,
`max_generations`, `proposals_per_generation`, and `CRAFTER_MAX_TURNS`
once you're confident in the setup.
