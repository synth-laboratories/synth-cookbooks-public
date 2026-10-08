# Remote CISPO on GSM8K

Hosted CISPO here means **Tinker**, not local MLX and not `/api/v1/optimizers` submit (`gepa` / `ohco` / `go-ex` only).

```bash
bash runbooks/preflight.sh
bash runbooks/smoke.sh
AUTHORIZE_TINKER_CISPO=1 bash runbooks/run.sh --model Qwen/Qwen3.5-4B
```

`run.sh` needs `TINKER_API_KEY`. The learner uses Tinker's `cispo` loss (`clip_low_threshold=0`, `clip_high_threshold=4`).
