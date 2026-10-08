# Synth Public Cookbooks (v0.6 turnover)

Canonical, reviewed, agent-readable examples for the current Synth stack.

Website: `https://usesynth.ai/cookbooks/`  
Source base: `origin/main` at `1e524fa` (clean worktree; not the dirty v0.5 checkout).

Legacy recipes from that SHA live under [`old/cookbooks/`](old/cookbooks/). New cookbooks use one directory contract under [`cookbooks/`](cookbooks/).

## Launch set (containers × algorithms)

| Cookbook | Algorithm / product lane | Container / runtime | Task | Why this pair |
| --- | --- | --- | --- | --- |
| `gepa-banking77` | **GEPA** (Optimizers OSS) | **Banking77** HTTP task (`synth_containers` banking77 runtime; cookbook configures, does not fork) | 77-way intent classification | Fast exact-match hello-world for prompt evolution; existing reviewed evidence to migrate. |
| `evals-healthbench` | **Evals** (no optimizer) | **HealthBench** container + rubric grader | Professional medical QA | First-class eval lane; Groq/HealthBench material already in `old/`. |
| `mlx-sft-craftax` | **SFT** via MLX Tinker API | **Craftax** container for rollout/eval; MLX Tinker for local train | Long-horizon Craftax survival | Local Apple-silicon training on the same env as `/evals/craftax`; Tinker job/artifact contract. |
| `remote-cispo-gsm8k` | **CISPO** (hosted adapter) | GSM8K eval (hosted language-task proof) | Grade-school math | Hosted training lane; not the Craftax twin — different execution class and task on purpose. |

Deferred from the old tree (archive only until a second wave): Crafter *GEPA* (prompt ReAct, not SFT), MiniGrid, HotpotQA, TBLite, GELO, managed-research quickstart, blog chart packets. Those stay in `old/` so Frontend historical links can redirect.

## Layout

```text
cookbooks/<id>/{cookbook.yaml,README.md,AGENT.md,REPORT.md,configs,runbooks,fixtures,tests,evidence}
catalog/{catalog.v1.json,schema/,generated/}
old/cookbooks/   # frozen v0.5 material
```

## Tested against (2026-08-18)

In-process pytest: 8 passed against local `containers` (`f558140`, Banking77/HealthBench/Craftax HTTP) and `synth-mlx-rl` (`848752c`, FakeEngine SFT + `cispo_minimax`). Catalog pins also record Containers `origin/main` `a2a316b` and Optimizers `origin/main` `1c89092`. Paid GEPA search and live HealthBench grader runs are not claimed.

## Dirty checkout (do not use as release base)

The original checkout `/Users/joshuapurtell/Documents/GitHub/synth-cookbooks-public` remains on `v0.5/banking77-durable-rollouts` @ `7fc5b48`. Uncommitted Banking77 timeout/proxy-error changes and run traces were stashed as `v0.6-turnover: preserve dirty banking77 patch and run traces` (`stash` `aaf04b84`). Patch copy: `Documents/Codex/2026-08-18/re/outputs/public-cookbooks-v0.6-dirty-preserve/`.
