# Local MLX SFT on Craftax

Train with current `synth-mlx-rl` (Tinker-shaped `/v1` learner). Evaluate with current Containers `craftax_engine` (scripted fixture). Gold `/evals/craftax` stays a separate product page.

```bash
bash runbooks/preflight.sh
bash runbooks/smoke.sh
# live Apple silicon:
# synth-mlx-rl serve --host 127.0.0.1 --port 8787
# python3 runbooks/train_sft.py --base-url http://127.0.0.1:8787
```
