# REPORT — Evals on HealthBench

## Hypothesis
HealthBench can be scored as an eval-only lane with independent policy and grader usage.

## Verdict
**Live batch complete.** Mean scaled score **0.527** over **8** seeds. Estimated cost **$0.052**.

## Confidence
High for this scaled packet. Not canonical HealthBench.

## Why
Policy `openai_gpt41_mini`, grader `gpt-4.1-mini-2025-04-14` (`healthbench_scaled_grader.v1`). No optimizer.

## Results
| Seed | Reward |
| --- | --- |
| 0 | 0.890 |
| 1 | 0.492 |
| 2 | 1.000 |
| 3 | 0.556 |
| 4 | 0.429 |
| 5 | 0.298 |
| 6 | 0.362 |
| 7 | 0.190 |
| mean | 0.527 |

## Limitations
Canonical grader is `gpt-4.1-2025-04-14`. Spread across seeds is large; n=8 is still a small eval.
