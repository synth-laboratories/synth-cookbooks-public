# REPORT — Local MLX SFT on Craftax

## Hypothesis
LoRA SFT through current `synth-mlx-rl` plus Craftax fixture eval is a valid local-training cookbook.

## Verdict
**Proved** on FakeEngine SFT + `craftax_engine` scripted fixture.

## Confidence
High for those two contracts. Low for Apple-silicon base-model transfer.

## Why
Fixture seeds 0 and 1 each scored **0.5** on `env:craftax_fixture`. FakeEngine `cross_entropy` loss **0.25**, `training_version` 1, checkpoint `sft-craftax-smoke`.

## Results
| Seed | Reward | Env |
| --- | --- | --- |
| 0 | 0.5 | env:craftax_fixture |
| 1 | 0.5 | env:craftax_fixture |

## Limitations
Not `/evals/craftax` gold. Not a downloaded Qwen/MLX weights run.
