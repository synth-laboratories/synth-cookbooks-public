# Banking77 - Chart A/C - Synth GEPA

Status: launch-scope run evidence present; publication still requires the public evidence commit.

## Result

| Field | Value |
|---|---:|
| Best heldout reward | 0.785 |
| Seed heldout reward | 0.75 |
| Heldout coverage | 177 / 200 |
| Candidates | 17 |
| Rollout calls | 1140 |

## Run

- Run id: `synth_gepa_banking77_20260602T012632`
- Stack: `synth_gepa`
- Task: `banking77`
- Charts: A head-to-head, C heldout coverage

## Evidence

- Summary packet: `cookbooks/optimizers/gepa/evals/evidence/benchmarks/banking77/summary.json`
- Summary sha256: `81dc155842af817538e7fe22b9263264a4a316b626636c773e1f2c8e8e310429`
- Chart A source packet: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-a-head-to-head/figures/source_evidence.json`
- Chart C source packet: `cookbooks/blogs/oss-containers-and-gepa/charts/chart-c-use-case-coverage/figures/source_evidence.json`
- Chart C aggregate inputs:
  - `cookbooks/optimizers/gepa/evals/evidence/heldout_evaluations.jsonl` sha256 `50626f095145bf1ab1e8334ddfd95132f409ff714ae833a8597d476199a9ae4e`
  - `cookbooks/optimizers/gepa/evals/evidence/train_evaluations.jsonl` sha256 `7b15b16775ebc1d981ace67831bb382e48cdfc997efbda699e621ad2e99a6e73`
  - `cookbooks/optimizers/gepa/evals/evidence/candidate_timeline.jsonl` sha256 `3c426a874a736ff9d4ede88c6055e59ea3637ebdd4377e03b4f037f1772991ee`

## Notes

This record is a compact provenance index. Raw run artifacts stay under the gitignored eval run tree; the public chart authority is the checksummed summary and source-evidence packets above.
