# Banking77 operating plan

## Ready now

- Self-contained locked `uv` environment and loopback-only service entrypoint.
- Stable program, dataset, taskset, sync/async rollout, poll/SSE, reward, and usage contracts.
- Atomic durable terminal records, idempotent rollout ids, and restart retrieval.
- Exact Desktop policy reference plus conservative capability advertisement.

## Before every paid workflow

1. Run `uv run --frozen --with pytest pytest -q test_rollout_retention.py`.
2. Start `./run_container.sh --storage-root <dedicated-absolute-path> --port <port>`.
3. Probe `/health`, `/metadata`, `/task_info`, `/program`, and both dataset splits.
4. Run one rollout, retrieve its record/reward/events, restart, and retrieve again.
5. Run a 10×2 parallel eval before a larger GEPA search.

## Next hardening

- Move the durable record primitives into the released `synth-containers` SDK once its public API stabilizes.
- Add CI against the pinned Workshop eval driver and the next published Containers release.
- Maintain one smoke profile and one meaningful-search profile; treat generated run configs as artifacts.
