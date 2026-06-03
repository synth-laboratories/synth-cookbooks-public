# OSS Containers and GEPA — Reproduction Cookbook

Reproduction code for the charts in the **OSS Containers and GEPA**
blog post (June 2026).

Live post: <https://usesynth.ai/blog/introducing-gepa-platform> (link
goes live with the launch).

## What this folder contains

Every result chart in the post has a matching subfolder. Each launch subfolder
has its own README, producer script, run commands, and tracked source evidence
so the chart can be regenerated against the public container contract. Large
local rerun outputs live under ignored `runs/` directories; the launch figures
track compact evidence snapshots and checksums in committed JSON.

Hard rule for launch: every number, curve, table cell, and chart series in
the post must be produced by committed code in this folder. A chart folder
must contain the producer script that reads real run artifacts and emits the
data JSON consumed by the frontend. Hand-typed chart numbers, ad-hoc `/tmp`
scripts, invented curves, and unregenerable illustrative data do not ship.

## Charts

| Folder | Launch status | What the chart shows |
|---|---|---|
| [chart-a-head-to-head/](./chart-a-head-to-head/) | in post | Final same-container head-to-head: Synth GEPA vs gepa-ai across HealthBench Pro, Harvey Lab Tax, tau2-bench retail, and DungeonGrid. |
| [chart-c-use-case-coverage/](./chart-c-use-case-coverage/) | in post | Cumulative heldout coverage for the final four same-container tasks. |
| [chart-d-proposer-scaling/](./chart-d-proposer-scaling/) | in post | Proposer-model sweep across `gpt-5.4-nano`, `gpt-5.4-mini`, and `gpt-5.4` at fixed task container, policy model, splits, and budget. |
| [chart-g-dungeongrid/](./chart-g-dungeongrid/) | in post addendum | DungeonGrid candidate trajectory and progress-signal frequencies from final heldout evidence. |
| [chart-h-reward-diagnostics/](./chart-h-reward-diagnostics/) | in post | Harvey fractional row reward plus DungeonGrid reward and objective-recovery diagnostics from final heldout evidence. |
| [chart-b-prompt-diff/](./chart-b-prompt-diff/) | support/future | Qualitative prompt-diff extraction for selected candidate fields. Not a launch result chart unless embedded by the post. |
| [chart-e-policy-model-variation/](./chart-e-policy-model-variation/) | future | Cross-student-model transfer. Not used by the launch post. |
| [chart-f-program-stage-scaling/](./chart-f-program-stage-scaling/) | future | Program-stage scaling. Not used by the launch post. |

## Prerequisites

- `synth-optimizers` (Rust binary) — `cargo install synth-optimizers` or
  build from source.
- `synth-containers` package — `uv pip install synth-containers`.
- Public cookbook containers from
  [cookbooks/optimizers/gepa/](../../optimizers/gepa/).
- For head-to-head: gepa-ai installed locally
  (`pip install gepa` per their repo).
- Proposer auth: either Codex app server (for the default proposer) or
  the proposer backend you're sweeping.

## Container catalog

Containers organized by the shape of the program they expose.
Categories track the lever surface the optimizer gets to mutate, not the
task domain.

Status legend:

- **✓ public** — container shipped publicly today.
- **→ public** — container being flipped public with this release.
- **runner OSS · container private** — the `synth-optimizers` runner that
  executes these is open-source, but the container itself stays
  internal (typically because the underlying task is proprietary or
  service-coupled and impossible to reproduce DIY).
- **roadmap** — upstream task identified; container build in progress.

### Single / multistage programs (DSPy-style stage chains)

| Container | Upstream / dataset | Status |
|---|---|---|
| Banking77 | [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) | ✓ public ([container](../../optimizers/gepa/banking77_container/)) |
| Banking77 (MIPROv2-shaped) | [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) | → public |
| HotpotQA | [hotpotqa.github.io](https://hotpotqa.github.io/) | → public |
| HoVer | [hover-nlp.github.io](https://hover-nlp.github.io/) | → public |
| Heart Disease (UCI) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) | → public |

### Coding agent (agentic shell / code-edit)

| Container | Upstream / dataset | Status |
|---|---|---|
| TBLite | [Terminal-Bench](https://www.tbench.ai/) | ✓ public ([container](../../optimizers/gepa/tblite_container/)) — uses OpenAI API key auth (no Codex bundle required) |
| Code Review | (internal cookbook PR review task) | runner OSS · container private |
| NGO-style | (internal research target) | runner OSS · container private |

### ReAct environments (long-horizon, game / world)

| Container | Upstream / dataset | Status |
|---|---|---|
| Crafter | [danijar/crafter](https://github.com/danijar/crafter) | ✓ public ([container](../../optimizers/gepa/crafter_container/)) |
| MiniGrid | [Farama MiniGrid](https://minigrid.farama.org/) (via OpenEnv) | → public |
| tau2-bench retail | [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | ✓ public ([container](../../optimizers/gepa/tau2_retail_container/)) |
| DungeonGrid | [JoshuaPurtell/DungeonGrid](https://github.com/JoshuaPurtell/DungeonGrid) | ✓ public ([container](../../optimizers/gepa/dungeongrid_container/)) |

### Vertical / domain agent (real-world professional workflows)

| Container | Upstream / dataset | Status |
|---|---|---|
| Harvey Labs (legal) | [harveyai/harvey-labs](https://github.com/harveyai/harvey-labs) | ✓ public ([container](../../optimizers/gepa/harvey_lab_container/)) |
| Legal Apex Agents | [mercor/apex-agents](https://huggingface.co/datasets/mercor/apex-agents) | roadmap |
| HealthBench Professional (medical) | [openai/healthbench-professional](https://huggingface.co/datasets/openai/healthbench-professional) ([paper](https://cdn.openai.com/dd128428-0184-4e25-b155-3a7686c7d744/HealthBench-Professional.pdf)) | ✓ public ([container](../../optimizers/gepa/healthbench_container/)) |

### Tally

The launch evidence in the post uses four public containers:
HealthBench Professional, Harvey Labs Tax, tau2-bench retail, and DungeonGrid.
The broader catalog remains here to show the container-contract surface across
classification, QA, coding, environment-control, and professional-workflow tasks.

## Compute-parity ground rules

Every "vs gepa-ai" chart in this folder documents the matched conditions
it uses. Chart A matches the public HTTP container boundary for every row;
the final launch rows are HealthBench Pro, Harvey Lab Tax, tau2-bench retail,
and DungeonGrid. Each row pins the task container, train/heldout split,
coverage threshold, proposer/reflection model, and task-specific policy or
judge model in `cookbooks/optimizers/gepa/evals/evidence/benchmarks/*/`.
Other sweep charts should pin:

- Same `max_total_rollouts` budget per run.
- Same proposer model (default: `gpt-5.4-mini`, unless the chart is
  explicitly sweeping the proposer).
- Same student/policy model (default: the cookbook's fixture policy,
  unless the chart is explicitly sweeping policy).
- Same minibatch size.
- Same train/heldout seed splits.
- Same wall-clock budget where applicable.

Each chart's README documents the exact parity conditions used for the
numbers in the post.
