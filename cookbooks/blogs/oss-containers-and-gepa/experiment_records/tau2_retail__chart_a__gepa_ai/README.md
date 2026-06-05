# tau2 retail - Chart A/C - gepa-ai

Status: launch-scope A/C run evidence is tracked in the public evidence packet; final launch still requires a clean evidence commit and release checklist citation.

## Result

| Field | Value |
|---|---:|
| Best heldout reward | 0.4 |
| Seed heldout reward | 0.38 |
| Heldout coverage | 62 / 100 |
| Candidates | 4 |
| Rollout calls | 129 |

## Run

- Run id: `gepa_ai_tau2_retail_20260604T062038`
- Stack: `gepa_ai`
- Task: `tau2_retail`
- Charts: A head-to-head, C heldout coverage

## Evidence

- Summary packet: `cookbooks/optimizers/gepa/evals/evidence/benchmarks/tau2_retail/summary.json`
- Summary sha256: `810c9356333cd674bd34cff472883af9207365fde74c6e3ed6f009b3dcaed188`
- Chart A source packet: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-a-head-to-head/figures/source_evidence.json`
- Chart C source packet: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-c-use-case-coverage/figures/source_evidence.json`
- Chart C aggregate inputs:
  - `cookbooks/optimizers/gepa/evals/evidence/heldout_evaluations.jsonl` sha256 `50626f095145bf1ab1e8334ddfd95132f409ff714ae833a8597d476199a9ae4e`
  - `cookbooks/optimizers/gepa/evals/evidence/train_evaluations.jsonl` sha256 `7b15b16775ebc1d981ace67831bb382e48cdfc997efbda699e621ad2e99a6e73`
  - `cookbooks/optimizers/gepa/evals/evidence/candidate_timeline.jsonl` sha256 `3c426a874a736ff9d4ede88c6055e59ea3637ebdd4377e03b4f037f1772991ee`

## Notes

This record is a compact provenance index. Raw run artifacts stay under the gitignored eval run tree; the public chart authority is the checksummed summary and source-evidence packets above.
