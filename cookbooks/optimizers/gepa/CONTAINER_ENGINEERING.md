# Banking77 and HealthBench 2 container engineering

These are the two canonical, public task boundaries used for fast eval and optimizer development.

| Container | Canonical directory | Runtime profile |
| --- | --- | --- |
| Banking77 | `banking77_container/` | closed-label classifier with environment-authored accuracy |
| HealthBench 2 | `healthbench_groq/` | open-text medical response with physician-rubric grading |

## Shared quality contract

Both containers must:

- own task data, policy invocation, scoring, and public-safe evidence;
- advertise only implemented routes and capabilities;
- keep train/heldout identity stable and never turn missing evidence into zero;
- accept narrow candidate overlays declared by `/program`;
- provide stable `/health`, `/metadata`, `/task_info`, `/program`, and rollout surfaces;
- make retries idempotent at the paid-call boundary;
- retain terminal rollout, reward, usage, and event records for later inspection;
- expose declared poll/SSE URLs instead of requiring clients to guess routes;
- keep credentials in the process environment and out of traces, configs, and artifacts;
- distinguish provider-settled cost, token-derived estimates, and unknown cost.

Containers do not own optimizer campaigns, candidate scheduling, acceptance decisions, or reports. Those remain optimizer/Workshop concerns.

## Banking77

```bash
cd banking77_container
./run_container.sh
uv run --project . --with pytest pytest -q test_rollout_retention.py
```

The service is standalone and defaults to `127.0.0.1:8765`. Its durable rollout journal is stored under `.banking77-streams/` unless `BANKING77_STREAM_ROOT` is deliberately set by an isolated launcher. The Desktop profile uses the exact advertised `desktop_eval/banking77_gpt_4_1_nano` policy reference.

## HealthBench 2

```bash
cd healthbench_groq
./run_container.sh --storage-root /absolute/path/to/a/dedicated/run-store
uv run --project . --group dev pytest -q test_container_contract.py
```

The service defaults to `127.0.0.1:8114`. A storage root is mandatory so concurrent Workshop instances cannot accidentally share rollout ownership. `GROQ_API_KEY` supplies the policy model and `OPENAI_API_KEY` supplies the rubric grader. The canonical grader is `gpt-4.1-2025-04-14`; a scaled grader must remain explicitly labeled non-canonical.

## Preflight before paid runs

1. Run the contract tests.
2. Start the service with a fresh dedicated storage directory.
3. Inspect `/health`, `/metadata`, `/task_info`, and `/program`.
4. Verify the configured policy/grader credentials exist without printing them.
5. Prepare one rollout and observe `stream.subscribed` before start.
6. Run one paid smoke rollout, retrieve it again by id, then retrieve `/reward` and events.
7. Restart the service and confirm required retained records remain available.
8. Only then launch a parallel eval or optimizer run.

Historical `*.sdk.toml` files in `healthbench_groq/` are experiment records, not recommended defaults. `gepa.toml` is the current reviewed profile.
