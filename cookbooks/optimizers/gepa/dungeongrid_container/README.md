# DungeonGrid GEPA cookbook container

A public GEPA container for **DungeonGrid** — a turn-based, multi-hero dungeon —
that mirrors the [`crafter_container`](../crafter_container) pattern:

- GEPA optimizes a single mutable module, `react_system_prompt`.
- The container runs a **real DungeonGrid episode** per rollout, driven by an
  LLM policy (default **`gemini-3.1-flash-lite`**) making typed tool-call actions.
- Reward = total environment reward for the episode (achievement bonuses
  included). No string matching, no fixtures.

It speaks the public synth-optimizers GEPA contract: `/metadata`, `/task_info`,
`/program`, `/dataset`, `/dataset/rows`, `/rollout`.

## Run

```bash
# from cookbooks/optimizers/gepa/
synth-optimizers gepa run --config dungeongrid_container/gepa.toml
```

The `gepa.toml` boots the container itself (via `uv run`), so you only install
DungeonGrid-side deps when you run this cookbook.

## Env

| Var | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `OPENAI_BASE_URL` | — | point at a gemini OpenAI-compatible gateway for the default policy |
| `DUNGEONGRID_POLICY_MODEL` | `gemini-3.1-flash-lite` | policy model |
| `DUNGEONGRID_QUEST` | `lantern_crypt` | quest/dungeon to load |
| `DUNGEONGRID_NUM_HEROES` | `2` | party size the policy controls |
| `DUNGEONGRID_MAX_TURNS` | `30` | per-episode policy-call cap |

The Warden (adversary) is environment-controlled: on its turn the container
steps it with `warden_auto`, so the policy only ever controls heroes.

## Files
- `synth_service_app.py` — the GEPA contract service (mirrors Crafter's).
- `dungeongrid_text_env.py` — thin text wrapper over the `dungeongrid` env.
- `gepa.toml` — GEPA run config (gemini-3.1-flash-lite policy).
- `pyproject.toml` — container deps (`dungeongrid`, `openai`, `fastapi`, `uvicorn`).
