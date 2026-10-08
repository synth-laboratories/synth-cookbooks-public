# Evals on HealthBench

Eval-only. Serves `healthbench_chat` from current Containers. No GEPA proposer.

```bash
bash runbooks/preflight.sh
bash runbooks/smoke.sh       # /health + GEPA v2 contract + credential flags
# live (needs OPENAI_API_KEY for policy and grader):
python3 runbooks/evaluate.py --url http://127.0.0.1:8114 --config configs/eval.toml
```
