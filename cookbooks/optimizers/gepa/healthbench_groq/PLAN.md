# HealthBench 2 operating plan

## Ready now

- Self-contained locked entrypoint pinned to an immutable public Containers commit.
- GEPA v2 program/taskset contract with one narrow mutable system prompt.
- Canonical physician-rubric authority, explicit scaled-grader labeling, honest null cost/reward handling.
- Dedicated durable storage, idempotent terminal lifecycle, and restart recovery.
- No invented frames, checkpoint, restore, fork, or literal training-target exposure.

## Before every paid workflow

1. Run `uv run --frozen --group dev pytest -q test_container_contract.py`.
2. Verify `GROQ_API_KEY` and `OPENAI_API_KEY` exist without printing them.
3. Start `./run_container.sh --storage-root <dedicated-absolute-path> --port 8114`.
4. Probe `/health`, `/metadata`, `/task_info`, `/program`, and the requested task ids.
5. Run one paid policy+grader rollout; retrieve reward, usage, rubric events, and the record after restart.
6. Only then launch the reviewed `gepa.toml` profile and monitor failure rate/cost provenance.

## Next hardening

- Replace the git pin with the first PyPI Containers release containing the same HealthBench and recovery contracts.
- Add a bounded fixture grader for CI shape validation while keeping paid acceptance canonical.
- Establish repeated-run ETA bands and a small public result manifest without publishing medical prompt content that leaks heldout evidence.
