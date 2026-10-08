# AGENT

Objective: run GEPA on Banking77 using Optimizers + Containers, not a forked HTTP server.

Authoritative files: `cookbook.yaml`, `configs/gepa.toml`, `configs/smoke.toml`.

Safe writes: `.out/` only.

Credentials: `OPENAI_API_KEY` for live policy/proposer. Never print secrets.

Bounded path: `runbooks/smoke.sh` (dataset_gold, reward must be 1.0 on seed 0).

Full path: `runbooks/run.sh` after preflight compatible=true.

Completion: GEPA result JSON under `.out/runs` plus `REPORT.md` metrics. Failure: non-zero smoke, missing `/health`, reward != 1.0 on gold.
