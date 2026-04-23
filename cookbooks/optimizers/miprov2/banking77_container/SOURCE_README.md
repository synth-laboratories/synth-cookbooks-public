# Banking77 Container

## Why this belongs first

`banking77` is a strong first container because it is compact, cheap, and easy
to reason about while still being non-trivial for prompt optimization. It is a
single-query classification task with a large intent label set, so small prompt
changes can matter without requiring long-horizon execution.

## Task shape

- domain: banking support intent classification
- output: exactly one label from the Banking77 taxonomy
- primary metric: accuracy
- expected program length: 1 step
- canonical action shape: one tool call selecting one intent

## Why it fits Trinity Nano

- short context
- cheap per-example evaluation
- low-latency scoring
- easy to run many seeds for optimizer comparison

## Benchmark role in nanoprogram

This should be the cleanest lane for:

- few-shot selection
- random search
- MIPRO-style prompt search
- GEPA-style prompt mutation
- ACE/SIMBA style policy search on compact prompts

Because the program is effectively one step, this container is useful for
measuring pure prompt quality and optimizer overhead without confounding
multi-step execution effects.

## Initial container contract

- fixed train/test seed splits
- one canonical label inventory
- one canonical evaluation path
- one canonical baseline prompt
- no hidden retrieval or external tools beyond the classification interface

## Local source reference

The old implementation lives at:

- `research/old/benchmarks/langprobe/task_apps/banking77/banking77_task_app.py`

The old benchmark metadata lives at:

- `research/old/benchmarks/langprobe/README.md`
- `research/old/benchmarks/langprobe/run_benchmark.py`
