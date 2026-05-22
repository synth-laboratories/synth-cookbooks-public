# Terminal-Bench-Lite GEPA Container

Real Python coding tasks verified by real pytest. No fixture, no string
matching, no Docker.

Each rollout:
1. Picks a task by seed (function signature + spec + hidden pytest).
2. Calls OpenAI with the candidate's `starting_prompt` as system prompt
   and the task spec as user prompt.
3. Writes the agent's response as `solution.py` plus the hidden tests as
   `test_solution.py` into a temp dir.
4. Runs `pytest -q test_solution.py` in a subprocess (hard timeout).
5. Reward = `passed / (passed + failed + errors)`.

5 tasks ship with the container: 3 train, 2 heldout. Each task is small,
self-contained, and verifiable end-to-end without network access.

## Required env

- `OPENAI_API_KEY` — required.
- Optional: `TBLITE_POLICY_MODEL` (default `gpt-4.1-nano`),
  `TBLITE_TEST_TIMEOUT_SECONDS` (default `30`).

## Per-container dependencies

Declared in [`pyproject.toml`](./pyproject.toml). Installing this cookbook
does **not** require dependencies for other cookbooks.

- `openai` — live agent + (separately) live proposer
- `pytest` — the real verifier
- `fastapi`, `uvicorn`

## Contract

- `GET /metadata` advertises `synth_optimizers.gepa.v1`.
- `GET /program` exposes one mutable module: `starting_prompt`.
- `POST /dataset/rows` returns rows **with hidden tests stripped** — the
  optimizer side never sees the verifier tests.
- `POST /rollout` runs the agent + verifier loop and returns real reward.

## Safety notes

- The agent's response is executed as Python in a temp dir via a
  subprocess. We don't sandbox beyond the subprocess + the pytest
  timeout. Run in a trusted environment.
- The container does not write outside the temp dir or call out to the
  network from inside the subprocess.

## Task list (shipped)

| Seed | Split | Function | What it tests |
|---|---|---|---|
| 0 | train | `is_palindrome` | string normalization + reverse-equality |
| 1 | train | `two_sum_indices` | list scan + index return |
| 2 | train | `rle_encode` | run-length encoding |
| 100 | test | `balanced_brackets` | stack-based bracket matching |
| 101 | test | `merge_sorted` | O(n) two-pointer merge |
