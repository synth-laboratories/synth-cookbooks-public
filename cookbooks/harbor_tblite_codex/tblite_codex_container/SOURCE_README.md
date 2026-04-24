# Harbor TBLite Eval

Canonical local eval assets for the Rhodes Harbor-backed Terminal-Bench Lite path.

This target runs a real OpenThoughts TBLite task through:
- `synth-ai` `ContainerPoolsClient`
- `backend/rhodes` Harbor pools
- a task-scoped Harbor image
- Codex inside the task image
- the real task verifier
- the Python `/v1/pools` control plane, not host-local Docker orchestration

## Local run

```bash
export SYNTH_API_KEY=...
uv run --with synth-ai==0.9.11 python tblite_codex_container/run_local_eval.py \
  --backend-base http://127.0.0.1:8001 \
  --open-thoughts-task application-debug \
  --model openai/gpt-5.4-mini \
  --seed 0 \
  --seed 1

uv run --with synth-ai==0.9.11 python tblite_codex_container/run_tblite_codex_nano_once.py \
  --backend-base http://127.0.0.1:8001 \
  --open-thoughts-task application-debug \
  --output-root /tmp/harbor-tblite-once
```

For staging/main, set `HARBOR_EXECUTION_BACKEND` explicitly. Local `docker` is a
valid Harbor backend; deployed environments should not rely on an implicit
fallback.

## Files

- `tb_lite_harbor_pool.json`: canonical example pool shape for Harbor/TBLite
- `run_local_eval.py`: runnable local eval wrapper
- `run_tblite_codex_nano_once.py`: one-shot Harbor/TBLite runner with staged artifacts
- `smoke.py`: TBLite Harbor smoke harness
- `tb_lite_dataset.py`: OpenThoughts task packager
- `codex_harbor_runner.py`: shared Codex Harbor runner
