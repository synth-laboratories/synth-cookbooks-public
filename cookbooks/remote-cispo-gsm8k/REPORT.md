# REPORT — Remote CISPO on GSM8K

## Hypothesis
Hosted CISPO is Tinker's `loss_fn="cispo"` on a LoRA client, scored on HuggingFace `openai/gsm8k` test.

## Verdict
**Ran to completion.** `Qwen/Qwen3.5-4B` LoRA, 6 CISPO rounds, **9** optimizer steps. Held-out test seeds 0–31: **30/32 = 0.9375** before and after. Uplift **+0.00**.

## Confidence
High that this is hosted Tinker CISPO. Low that 9 LoRA steps move a 4B already at 94% on this 32-seed slice.

## Why
Sequential Tinker sampling (32 greedy eval + 6×32 train samples + 32 greedy eval) took ~36 minutes. Train rewards 0.75–0.97; non-zero groups produced 4–8 datums per round.

## Limitations
n=32 is noisy. Model was already near ceiling. Hosted Optimizers still has no CISPO slug.
