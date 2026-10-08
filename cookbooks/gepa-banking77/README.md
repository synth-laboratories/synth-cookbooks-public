# GEPA on Banking77

Prompt evolution against the current `synth_containers` `banking77_classify` target.

Pins: Containers `origin/main` `a2a316b`, Optimizers `origin/main` `1c89092`.

## Commands

```bash
bash runbooks/preflight.sh
bash runbooks/smoke.sh          # dataset_gold rollout, no policy spend
bash runbooks/run.sh            # synth-optimizers gepa run --config configs/gepa.toml
```

The container is served with `python -m cookbook_tools.serve_compat --target banking77_classify`. Do not copy `old/.../synth_service_app.py`.
