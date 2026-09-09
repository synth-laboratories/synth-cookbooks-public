# MiniGrid GEPA Container

Real MiniGrid gymnasium episodes optimized via the public GEPA contract.

Each rollout runs a real `MiniGrid-DoorKey-5x5-v0` (default) episode with
a live OpenAI-driven agent. GEPA optimizes one mutable lever —
`system_prompt` — and reward is the actual env reward (positive only when
the agent reaches the goal).

No fixture, no string-matching scorer.

## Required env

- `OPENAI_API_KEY` — required.
- Optional: `MINIGRID_POLICY_MODEL` (default `gpt-4.1-nano`),
  `MINIGRID_MAX_STEPS` (default `48`),
  `MINIGRID_ENV_ID` (default `MiniGrid-DoorKey-5x5-v0`).

## Per-container dependencies

Declared in [`pyproject.toml`](./pyproject.toml). Installing this cookbook
does **not** require dependencies for other cookbooks.

- `gymnasium`, `minigrid` — real env
- `openai` — live policy
- `fastapi`, `uvicorn`, `numpy`

## GEPA v2 contract

- `GET /metadata` advertises `synth_optimizers.gepa.v2` plus its absolute
  `program_route`, `taskset_route`, `taskset_tasks_route`, and `rollout_route`.
- `GET /taskset` returns the taskset id and per-split sizes.
- `POST /taskset/tasks` resolves the stable `<split>:<seed>` task ids declared in
  the config's `[taskset]` and `[gepa.task_pools]` blocks (for example `train:1`, `test:100`),
  and echoes each requested id back on its row. This is the route the optimizer
  loads rows from.
- `POST /dataset/rows` and the other older seed-based routes remain available for
  compatibility clients, but GEPA v2 does not use them.
- `GET /program` exposes one mutable module: `system_prompt`.
- `POST /rollout` runs a real episode:
  - Instantiates the env via `gymnasium.make(MINIGRID_ENV_ID)` + `FullyObsWrapper`
  - Resets with the row's seed
  - For up to `MINIGRID_MAX_STEPS` steps:
    - Calls OpenAI with the candidate's `system_prompt` as system,
      a text observation as user, JSON response format
    - Parses the `{"action": "..."}` JSON into a MiniGrid action index
    - Steps env; accumulates real reward
  - Returns `reward_info.outcome_reward = total_episode_reward`

MiniGrid rewards 0 on every step until success; on success it pays out
roughly `1 - 0.9 * (steps / max_steps)`. So `outcome_reward > 0` means
the agent solved the task.
