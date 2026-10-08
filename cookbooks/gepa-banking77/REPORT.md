# REPORT — GEPA on Banking77

## Hypothesis
A live classify policy on current `banking77_classify` can be scored with env-authored exact match.

## Verdict
**Live batch complete.** `gpt-4.1-nano` heldout exact-match **0/16 (0%)**. Gold replay on train seed 0 remains **1.0**.

## Confidence
High.

## Why
Requested 32 heldout seeds; the current split scored 16 and returned `absent` for 16–31 (out of range). All 16 scored rows are real misses, not transport errors. This is a weak seed policy, which is why GEPA search is the next paid step — not yet billed.

## Results
| Path | n scored | Hits | Accuracy |
| --- | --- | --- | --- |
| dataset_gold train seed 0 | 1 | 1 | 1.0 |
| gpt-4.1-nano heldout 0–15 | 16 | 0 | 0.0 |

## Limitations
No proposer generations. Do not cite prompt-evolution uplift from this packet.
