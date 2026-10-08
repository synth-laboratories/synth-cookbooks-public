# AGENT

Objective: score HealthBench rows with the physician rubric. Do not run an optimizer.

Authoritative: `cookbook.yaml`, `configs/eval.toml`.

Safe writes: `.out/`.

Credentials: `OPENAI_API_KEY` (policy + canonical grader). Do not log values.

Smoke: `runbooks/smoke.sh` — contract only.

Live eval: `runbooks/evaluate.py` after the container is up.

Success: JSON with per-task `status`/`reward`. Missing grader evidence must stay missing, not coerced to 0.
