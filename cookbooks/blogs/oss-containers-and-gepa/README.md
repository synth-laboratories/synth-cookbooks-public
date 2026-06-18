# OSS Containers and GEPA — Reproduction Cookbook

Reproduction code for the charts in the **OSS Containers and GEPA**
blog post (May 2026).

Live post: <https://usesynth.ai/blog/introducing-gepa-platform> (link
goes live with the launch).

## What this folder contains

Every chart in the post has a matching subfolder. Each subfolder has its
own README, configs, run commands, and tracked source evidence so the
chart can be regenerated end-to-end against the public container
contract. Large local rerun outputs live under ignored `runs/`
directories; Chart A tracks the launch evidence compactly in
`chart-a-head-to-head/figures/source_evidence.json`.

## Charts

| Folder | What the chart shows |
|---|---|
| [chart-a-head-to-head/](./chart-a-head-to-head/) | Head-to-head: Synth GEPA vs gepa-ai on the public launch cookbooks. Non-public task rows are omitted from this cookbook release. |
| [chart-b-prompt-diff/](./chart-b-prompt-diff/) | Qualitative side-by-side of best candidate prompts from both implementations on Banking77, TBLite, and Crafter. |
| [chart-c-use-case-coverage/](./chart-c-use-case-coverage/) | Honest capability matrix: what each implementation can target today (incl. `optimize_anything` gap). |
| [chart-d-proposer-scaling/](./chart-d-proposer-scaling/) | Does optimizer quality ride the proposer-model curve? Sweep across `gpt-5` → `gpt-5.5`. |
| [chart-e-policy-model-variation/](./chart-e-policy-model-variation/) | Cross-student-model transfer: does a GEPA-discovered prompt carry to gpt-4.1-nano, gemini-3.1-flash-lite, claude-haiku, etc. |
| [chart-f-program-stage-scaling/](./chart-f-program-stage-scaling/) | LangProbe-analog: heldout reward vs program size (1/2/3 stages) on HotpotQA, HoVer, Banking77, Heart Disease. |

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
- **omitted** — not part of this public cookbook release.
- **roadmap** — upstream task identified; container build in progress.

### Single / multistage programs (DSPy-style stage chains)

| Container | Upstream / dataset | Status |
|---|---|---|
| Banking77 | [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) | ✓ public ([container](../../optimizers/gepa/banking77_container/)) |
| Banking77 multistage | [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) | → public |
| HotpotQA | [hotpotqa.github.io](https://hotpotqa.github.io/) | → public |
| HoVer | [hover-nlp.github.io](https://hover-nlp.github.io/) | → public |
| Heart Disease (UCI) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) | → public |

### Coding agent (agentic shell / code-edit)

| Container | Upstream / dataset | Status |
|---|---|---|
| TBLite coding micro-container | pytest-verified Python tasks | ✓ public ([container](../../optimizers/gepa/tblite_container/)) — uses OpenAI API key auth (no Codex bundle required) |
| Code Review | — | omitted |
| NGO-style | — | omitted |

### ReAct environments (long-horizon, game / world)

| Container | Upstream / dataset | Status |
|---|---|---|
| Crafter | [danijar/crafter](https://github.com/danijar/crafter) | ✓ public ([container](../../optimizers/gepa/crafter_container/)) |
| MiniGrid | [Farama MiniGrid](https://minigrid.farama.org/) (via OpenEnv) | → public |
| Tau-Bench 3 | [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench) | → public |

### Vertical / domain agent (real-world professional workflows)

| Container | Upstream / dataset | Status |
|---|---|---|
| Harvey Labs (legal) | [harveyai/harvey-labs](https://github.com/harveyai/harvey-labs) | roadmap |
| Legal Apex Agents | [mercor/apex-agents](https://huggingface.co/datasets/mercor/apex-agents) | roadmap |
| HealthBench Professional (medical) | [openai/healthbench-professional](https://huggingface.co/datasets/openai/healthbench-professional) ([paper](https://cdn.openai.com/dd128428-0184-4e25-b155-3a7686c7d744/HealthBench-Professional.pdf)) | roadmap |

### Tally

- **Public today:** 3 (Banking77, TBLite, Crafter).
- **Going public with this release:** 6 (Banking77 multistage, HotpotQA,
  HoVer, Heart Disease, MiniGrid, Tau-Bench 3).
- **Omitted from this public cookbook release:** 2 (Code Review, NGO-style).
- **Vertical roadmap:** 3 (Harvey Labs, Legal Apex Agents, HealthBench
  Professional).

That's **9 public containers** after the flip across four lever-shape
categories.

## Compute-parity ground rules

Every "vs gepa-ai" chart in this folder documents the matched conditions
it uses. Chart A matches the public HTTP container boundary for every row;
Banking77 is the broad fixed-budget parity case, while TBLite and Crafter
are smoke-scale public splits. Other sweep charts should pin:

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
