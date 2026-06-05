# HealthBench Pro - Chart A/C - Synth GEPA

Status: launch-scope run evidence is tracked in the public evidence packet; final launch still requires the frontend mirror and release checklist to cite that packet.

## Result

| Field | Value |
|---|---:|
| Best heldout reward | 0.3609311821772041 |
| Seed heldout reward | 0.2547314326739686 |
| Heldout coverage | 158 / 200 |
| Candidates | 17 |
| Rollout calls | 1240 |

## Run

- Run id: `synth_gepa_healthbench_20260603T010401`
- Stack: `synth_gepa`
- Task: `healthbench`
- Charts: A head-to-head, C heldout coverage

## Evidence

- Summary packet: `cookbooks/optimizers/gepa/evals/evidence/benchmarks/healthbench/summary.json`
- Summary sha256: `507c515535fe25be91f7e7b2ccfafaecf16d3074ffcf071c88d367fc32d208cb`
- Chart A source packet: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-a-head-to-head/figures/source_evidence.json`
- Chart C source packet: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-c-use-case-coverage/figures/source_evidence.json`
- Chart C aggregate inputs:
  - `cookbooks/optimizers/gepa/evals/evidence/heldout_evaluations.jsonl` sha256 `50626f095145bf1ab1e8334ddfd95132f409ff714ae833a8597d476199a9ae4e`
  - `cookbooks/optimizers/gepa/evals/evidence/train_evaluations.jsonl` sha256 `7b15b16775ebc1d981ace67831bb382e48cdfc997efbda699e621ad2e99a6e73`
  - `cookbooks/optimizers/gepa/evals/evidence/candidate_timeline.jsonl` sha256 `3c426a874a736ff9d4ede88c6055e59ea3637ebdd4377e03b4f037f1772991ee`

## Notes

This record is a compact provenance index. Raw run artifacts stay under the gitignored eval run tree; the public chart authority is the checksummed summary and source-evidence packets above.
