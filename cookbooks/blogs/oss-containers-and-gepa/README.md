# OSS Containers and GEPA — Reproduction Cookbook

Reproduction code for the charts in the **OSS Containers and GEPA**
blog post (June 2026).

Live post: <https://usesynth.ai/blog/introducing-gepa-platform> (link
goes live with the launch).

## Launch evidence scope

The June 2026 launch post cites only the launch-scope rows below. The broader
catalog entries are included as post-launch context only; they are not evidence
for the launch charts unless they are explicitly marked `launch A/C` or
`launch A/C/D`.

| Container | Archetype | Scope | Status | Container path |
|-----------|-----------|-------|--------|----------------|
| **HealthBench Professional** | vertical / medical QA | **launch A/C** | ✓ public | [`healthbench_container/`](../../optimizers/gepa/healthbench_container/) |
| **tau2-bench retail** | ReAct / tool workflow | **launch A/C/D** | ✓ public | [`tau2_retail_container/`](../../optimizers/gepa/tau2_retail_container/) |
| **Banking77** | single-step classification | **launch A/C** | ✓ public | [`banking77_container/`](../../optimizers/gepa/banking77_container/) |
| **HotpotQA** | multistage QA | **launch A/C** | ✓ public | [`hotpotqa_container/`](../../optimizers/gepa/hotpotqa_container/) |
| **Harvey Lab Tax** | vertical / legal agent | post-launch judge-eval roadmap | ✓ public | [`harvey_lab_container/`](../../optimizers/gepa/harvey_lab_container/) |
| **TaxCalcBench** | vertical / accounting | post-launch | runner OSS · container private | `taxcalcbench_container/` (E03) |
| **FinQA (codex)** | vertical / finance QA | post-launch | runner OSS · container private | `finqa_container/` (E14–E15) |
| **tau2-bench airline** | ReAct / tool workflow | post-launch extended | → public | `taubench_airline_container/` |
| **HoVer** | multistage QA | LangProbe addendum | → public | `hover_container/` |
| **Heart Disease (UCI)** | single-step classification | LangProbe addendum | → public | `heart_disease_container/` |
| **TBLite micro coding smoke** | coding agent | post-launch only after real rerun | ✓ public smoke, not launch evidence | [`tblite_container/`](../../optimizers/gepa/tblite_container/) |
| **Crafter** | ReAct / world env | parity / historical | ✓ public | [`crafter_container/`](../../optimizers/gepa/crafter_container/) |
| **MiniGrid** | ReAct / grid env | catalog | → public | `minigrid_container/` |
| **DungeonGrid** | ReAct / roguelike | **omit from launch** | ✓ public | [`dungeongrid_container/`](../../optimizers/gepa/dungeongrid_container/) |
| Banking77 (MIPROv2-shaped) | single-step | catalog | → public | — |
| Code Review | coding agent | catalog | runner OSS · private | — |
| NGO-style | coding agent | catalog | runner OSS · private | — |
| Legal Apex Agents | vertical / legal | catalog | roadmap | — |
| Finance Agent Benchmark | vertical / finance research | catalog | **not ready** | — |
| BankerToolBench | vertical / investment banking | catalog | roadmap | — |
| BlueFin | vertical / spreadsheets | catalog | roadmap | — |
| **PaperBench JudgeEval** | judge / verifier eval | judge eval (planned) | roadmap | — ([PaperBench](https://openai.com/index/paperbench/), [frontier-evals](https://github.com/openai/frontier-evals/tree/main/project/paperbench)) |
| **Harvey LAB judge eval** | judge / verifier eval | judge eval (planned) | roadmap | — ([LAB verifiers](https://www.langchain.com/blog/designing-efficient-verifiers-for-legal-agents), pairs with [`harvey_lab_container/`](../../optimizers/gepa/harvey_lab_container/) judge) |

**Scope legend:** **launch A/C** = current post Chart A head-to-head and Chart C
coverage rows · **launch A/C/D** = same plus Chart D proposer sweep ·
**LangProbe addendum** = reference parity tasks in `build_langprobe_addendum.py`
that are not active launch rows · **parity / historical** = earlier
head-to-head cookbooks (Crafter and the micro TBLite smoke) · **catalog** =
container-contract surface, not in launch evidence · **judge eval (planned)** =
containerized LLM-as-judge tasks that should use the same evidence and chart
contracts once runnable · **omit** = descoped (DungeonGrid).

Post-launch catalog rows stay below so the public container surface is
inspectable, but they must not be cited as launch coverage. In particular,
TBLite is a local micro coding smoke, not Terminal-Bench Lite/OpenThoughts
coverage, and Harvey/PaperBench judge-eval work is roadmap until a fresh
evidence packet exists.

## What this folder contains

This cookbook is **experiment-centric**. Eval runs and their evidence are the
source of truth; charts are derived views built from that data.

```text
oss-containers-and-gepa/
  README.md                 # this file — experiment grid + launch checklist
  blog_paths.py             # shared paths for producers and runbooks
  experiment_records/       # one folder per runnable cell (container × chart × setup)
  charts/                   # chart producer scripts + committed figure JSON
```

- **[`experiment_records/`](./experiment_records/)** — how each experiment was
  run, configs, committed evidence, and per-cell derived slices. Raw rollout
  artifacts stay in gitignored `runs/`; each cell README points at them.
- **[`charts/`](./charts/)** — producer scripts that read experiment data and
  emit `figures/*.json` (plus frontend mirrors). No chart script invents numbers;
  every series traces to experiment evidence.

Hard rule for launch: every number, curve, table cell, and chart series in the
post must be produced by committed code reading real experiment artifacts.
Hand-typed chart numbers, ad-hoc `/tmp` scripts, and unregenerable illustrative
data do not ship.

Eval harness (runs both stacks, posthoc heldout, builds summaries):
`cookbooks/optimizers/gepa/evals/`.

## Charts

Chart producers live under [`charts/`](./charts/). See the
[charts README](./charts/README.md) for the active registry, data-flow diagram,
and reproduce commands. The launch packet commits only the A/C/D chart surfaces;
ignored local archaeology under `charts/` is not evidence authority.

| Folder | Launch status | What the chart shows |
|---|---|---|
| [charts/chart-a-head-to-head/](./charts/chart-a-head-to-head/) | in post | Final same-container head-to-head: Synth GEPA vs gepa-ai across launch-scope tasks. |
| [charts/chart-c-use-case-coverage/](./charts/chart-c-use-case-coverage/) | in post | Cumulative heldout coverage for same-container parity runs. |
| [charts/chart-d-proposer-scaling/](./charts/chart-d-proposer-scaling/) | in post | Proposer-model sweep (`gpt-5.4-nano` / `mini` / `gpt-5.4`). |

Historical and future chart ideas are intentionally absent from the launch
packet. Re-add Chart B/E/F/G/H-style material only with a fresh evidence packet,
producer output, frontend mirror, and explicit MDX section in the same commit.

## Experiments

Every eval run, producer refresh, and frontend wiring task for the post is tracked
here. **Runnable cells** are container × chart × setup combinations with real
rollout data. **Work items** (E01–E16, W01–W05) are the broader launch checklist.

Per-cell runbooks (how we ran, raw run pointers, committed evidence, derived chart
JSON) live under [`experiment_records/`](./experiment_records/) — one folder per
runnable cell. These are compact provenance records that point at the shared eval
evidence, Chart D manifest snapshots, and each chart's
`figures/source_evidence.json`.

### Validity audit — updated 2026-06-05 (launch scope: 4 containers)

Launch charts are scoped to **HealthBench Pro, Banking77, HotpotQA, tau2 retail**.
FinQA, Harvey Lab Tax, and tau2 airline are **omitted from launch** (the broader
tables below keep them as roadmap).

**Data paths feeding the charts:**

- **Chart A scatter** reads
  `charts/chart-a-head-to-head/figures/head_to_head_data.json`, rebuilt from
  committed summaries for HealthBench, tau2 retail, Banking77, and HotpotQA.
- **Chart C coverage** reads the live aggregate
  `evals/evidence/{heldout_evaluations,train_evaluations,candidate_timeline}.jsonl`,
  with `BENCHES = [healthbench, tau2_retail, banking77, hotpotqa]`.
- **Chart D proposer sweep** reads compact proposer-sweep manifest snapshots
  under `charts/chart-d-proposer-scaling/figures/manifest_snapshots/` and
  reports observed optimization reward; these sweep manifests skipped heldout by
  design.

**Experiment-unit verdict:**

| Container | Chart A | Chart C | Chart D | Verdict |
|---|---|---|---|---|
| **HealthBench Pro** | ✅ finished | ✅ finished | ✅ finished | **FINISHED** |
| **Banking77** | ✅ finished | ✅ finished | n/a | **FINISHED** |
| **HotpotQA** | ✅ finished | ✅ finished | n/a | **FINISHED** |
| **tau2 retail** | ✅ finished | ✅ finished | ✅ finished | **FINISHED** |

`python3 -m experiment_unit status` now reports `summary: finished=6` for run
evidence and `publication packet: READY` for the committed A/C/D launch packet.
TBLite and wider GEPA workspace dirt remain warning-only quarantine surfaces, not
launch evidence. Banking77 was replumbed from the backup bundle into the live
aggregate. HotpotQA was rebuilt from the fresh same-container rerun. tau2 retail
was rerun on the Gemini-direct policy route with fresh A/C evidence that records
candidate and rollout counters for the launch parity checks.

**Observed head-to-head result:** P1 is mixed, not a clean win for the
pre-registered ordering. Synth GEPA is positive on HealthBench (+0.008 heldout),
tau2 retail (+0.030), and tied on Banking77 (+0.000), but negative on HotpotQA
(-0.042). tau2 is the largest positive gap, but HotpotQA falsifies the predicted
`healthbench ≈ hotpotqa > banking77` ordering.

**Observed Chart D result:** tau2 retail is monotonic with proposer size
(`0.600 → 0.600 → 0.667` observed train reward), while HealthBench is non-monotonic
(`0.347 → 0.314 → 0.339`). This supports the tau2 interaction but weakens any
blanket monotonic proposer-size claim.

### Pre-registered predictions — 2026-06-03 (judge the rerun against THESE)

Stated **before** the rerun so results are judged against expectations, not
rationalized after. The initial pre-registered thesis was **GEPA's returns scale
with the compute in the loop**, on two axes; the launch post now treats the
result as mixed same-container evidence with an honest negative.

**Mechanism (under the locked same-container protocol).** Both stacks record
candidate-count and rollout-count counters against fairness floors, so any Synth
GEPA edge is not from an order-of-magnitude exploration advantage. Prediction:
that quality edge compounds more when each rollout is expensive.

**Prediction 1 — head-to-head gap grows with policy compute (Charts A/C).**
Define `gap = Synth − gepa-ai` (best heldout, and coverage rows). Predicted
ordering by median policy tokens/rollout:

| Task | ~policy tok/rollout | predicted `gap` (heldout / coverage) |
|------|--------------------:|--------------------------------------|
| Banking77 | ~1k | ≈ 0 (within noise; both saturate) |
| HotpotQA | ~2k | ≈ 0 to slightly + |
| HealthBench Pro | ~2.3k | small + |
| tau2 retail | ~20k | **clearly +** (the load-bearing point) |

Predicted: `gap(tau2) > gap(healthbench) ≈ gap(hotpotqa) > gap(banking77) ≈ 0`,
i.e. **positive slope of gap vs log(policy tokens)**.
Current launch evidence is mixed rather than a clean slope law. tau2 has the
largest positive heldout gap (`+0.030`) and ties HealthBench on coverage gap
(`+1`), which supports the high-compute anchor; HotpotQA is negative
(`-0.042` heldout, `-3` coverage), and Banking77 ties on heldout while losing
coverage (`-5`). The blog should describe this as partial support with an honest
negative, not a smooth policy-compute scaling result.

**Prediction 2 — better proposer → better optimization, steeper where compute is
higher (Chart D).** Per task, best observed optimization reward is
non-decreasing in proposer
(`gpt-5.4-nano ≤ mini ≤ gpt-5.4`). Predicted interaction: the slope is **steeper
on tau2 than on HealthBench** (HealthBench may saturate / be near-flat).

**Falsification (any of these disproves the corresponding claim):**
- Synth does **not** beat gepa-ai on tau2 under the same-container rerun → headline (P1) fails.
- `gap(tau2) ≤ gap(banking77)` under the same-container rerun → policy-compute scaling fails.
- Chart D observed-reward curve flat or non-monotonic **on tau2** → proposer-size trend fails.
- HealthBench and tau2 proposer slopes equally steep → the interaction claim weakens.

**What we are NOT claiming (state these as caveats in the post):**
- Not a controlled compute law — policy compute is **confounded with task
  archetype** (high-compute = agentic, low = single-step). The claim is a
  *correlation across the suite*.
- Not a smooth curve — the x-axis is 3 low-compute points + **one** high-compute
  anchor (tau2). Re-adding tau2-airline would give a second high-compute point;
  until then, frame as low-vs-high contrast.

### Status legend

| Status | Meaning |
|--------|---------|
| **DONE** | Run completed; compact evidence is tracked in the public evidence packet |
| **PARTIAL** | Data exists but claim is weak (tiny N, unfair budget, or missing sibling rows) |
| **RERUN** | Must re-execute before citing in the post |
| **MISSING** | No runnable/public artifact yet |
| **PLANNED** | Scoped but not started |
| **UI** | Frontend/doc only — no new eval run |
| **OMIT** | Descoped from launch |

**Priority:** P0 = launch blocker · P1 = primary launch claim · P2 = addendum/diagnostic · P3 = post-launch

Publication checklist and model-parity locks:
`Jstack/.jstack/daily_notes/2026-06-03/gepa_blog_publication_failures.md`.

### Runnable cells — container × chart × setup

Each row is one experiment folder target under `experiment_records/`. Chart A and C
cells share the eval evidence pipeline
(`cookbooks/optimizers/gepa/evals/evidence/benchmarks/`). Chart D cells use the
self-contained sweep under `charts/chart-d-proposer-scaling/runs/final_20260603/`.

| Container | Chart | Setup | Status | Priority | Headline result / artifact | Backfill record folder |
|-----------|:-----:|-------|:------:|:--------:|----------------------------|---------------|
| HealthBench Pro | A/C | Synth GEPA | DONE | P0 | heldout 0.361; 17 candidates / 1240 rollouts | `experiment_records/healthbench_pro__chart_a__synth_gepa/` |
| HealthBench Pro | A/C | gepa-ai | DONE | P0 | heldout 0.353; 21 candidates / 1368 rollouts | `experiment_records/healthbench_pro__chart_a__gepa_ai/` |
| Banking77 | A/C | Synth GEPA | DONE | P0 | heldout 0.785; 17 candidates / 1140 rollouts | `experiment_records/banking77__chart_a__synth_gepa/` |
| Banking77 | A/C | gepa-ai | DONE | P0 | heldout 0.785; 20 candidates / 1358 rollouts | `experiment_records/banking77__chart_a__gepa_ai/` |
| HotpotQA | A/C | Synth GEPA | DONE | P0 | heldout 0.707; 9 candidates / 840 rollouts | `experiment_records/hotpotqa__chart_a__synth_gepa/` |
| HotpotQA | A/C | gepa-ai | DONE | P0 | heldout 0.748; 11 candidates / 796 rollouts | `experiment_records/hotpotqa__chart_a__gepa_ai/` |
| tau2 retail | A/C | Synth GEPA | DONE | P0 | heldout 0.430; 5 candidates / 154 rollouts | `experiment_records/tau2_retail__chart_a__synth_gepa/` |
| tau2 retail | A/C | gepa-ai | DONE | P0 | heldout 0.400; 4 candidates / 129 rollouts | `experiment_records/tau2_retail__chart_a__gepa_ai/` |
| HealthBench Pro | D | gpt-5.4-nano | DONE | P2 | observed reward 0.347 | `experiment_records/healthbench_pro__chart_d__gpt-5.4-nano/` |
| HealthBench Pro | D | gpt-5.4-mini | DONE | P2 | observed reward 0.314 | `experiment_records/healthbench_pro__chart_d__gpt-5.4-mini/` |
| HealthBench Pro | D | gpt-5.4 | DONE | P2 | observed reward 0.339 | `experiment_records/healthbench_pro__chart_d__gpt-5.4/` |
| tau2 retail | D | gpt-5.4-nano | DONE | P2 | observed reward 0.600 | `experiment_records/tau2_retail__chart_d__gpt-5.4-nano/` |
| tau2 retail | D | gpt-5.4-mini | DONE | P2 | observed reward 0.600 | `experiment_records/tau2_retail__chart_d__gpt-5.4-mini/` |
| tau2 retail | D | gpt-5.4 | DONE | P2 | observed reward 0.667 | `experiment_records/tau2_retail__chart_d__gpt-5.4/` |

Chart A/C shared evidence inputs (when DONE or PARTIAL):

```text
cookbooks/optimizers/gepa/evals/evidence/benchmarks/<task>/summary.json
cookbooks/optimizers/gepa/evals/evidence/benchmarks/<task>/run_manifest.json
```

Chart D publication input: compact manifest snapshots under
`charts/chart-d-proposer-scaling/figures/manifest_snapshots/`. Raw sweep outputs
under `charts/chart-d-proposer-scaling/runs/` are local/gitignored and are not
the public artifact.

### Work items — launch record and post-launch backlog

The P0/P1 DONE rows below are the launch evidence record. Planned, UI, and
omitted rows are tracked so future work does not get confused with launch chart
coverage.

| ID | Experiment | Type | Tasks | Status | Priority | Charts | Artifact | Next step |
|:--:|------------|:----:|-------|:------:|:--------:|--------|----------|-----------|
| E01 | Same-container Synth vs gepa-ai parity | EVAL | HB, Banking77, HotpotQA, tau2 | DONE | P0 | A, C, MDX | `evals/evidence/benchmarks/*/summary.json` | Launch-scope summaries rebuilt from live aggregate |
| E02 | Harvey heldout — larger split | EVAL | Harvey | OMIT | P3 | — | — | Omitted from launch scope; prior n=9 smoke cannot support a chart claim |
| E03 | TaxCalcBench — container + GEPA eval | EVAL | TaxCalc | OMIT | P3 | — | — | Omitted from launch scope; post-launch parity work |
| E04 | Head-to-head parity counters | EVAL | launch scope | DONE | P0 | A, MDX | `charts/chart-a-head-to-head/figures/head_to_head_data.json` | Candidate/rollout parity checks pass `experiment_unit status` |
| E05 | Proposer sweep — nano / mini / full | EVAL | HB, tau2 | DONE | P2 | D | `charts/chart-d-proposer-scaling/figures/manifest_snapshots/` | HB and tau2 sweeps rebuilt; chart reports observed reward from compact manifests |
| E06 | Proposer sweep — broader post-launch set | EVAL | HB, Harvey, tau2, TaxCalc, FinQA | PLANNED | P3 | — | — | Only if a future post claims broader proposer-sweep coverage |
| E07 | Proposer failure-mode table | CONTENT | HB, tau2 | PLANNED | P2 | D | — | Mine validity / mutation errors per cell before adding content |
| E08 | Policy-model variation sweep | EVAL | TBD | PLANNED | P3 | — | — | Define grid; run sweep |
| E09 | Program-stage scaling | EVAL | B77, HotpotQA, HoVer, Heart | PLANNED | P3 | — | — | Post-launch |
| E10 | Prompt diff extraction | PRODUCER | historical | OMIT | P3 | — | — | Fresh runs and MDX re-add required before any future use |
| E11 | Budget / protocol fairness table | CONTENT | launch scope | PLANNED | P3 | — | `head_to_head_data.json` fields | Optional follow-up table; current post caveats candidate counts, rollouts, and wall-clock variance in prose and chart data |
| E12 | Chart → producer provenance map | CONTENT | all wired | PARTIAL | P2 | — | chart READMEs + `figures/source_evidence.json` | One table: chart → producer → JSON → summary |
| E13 | Evidence pipeline refresh + sync | PRODUCER | launch scope | DONE | P1 | A, C, D | cookbook + frontend `data/*.json` | Producers rebuilt for Chart A/C/D |
| E14 | FinQA codex scaffold + container | IMPL | FinQA | OMIT | P3 | — | — | Omitted from launch scope; post-launch container work |
| E15 | FinQA GEPA parity (Synth vs gepa-ai) | EVAL | FinQA | OMIT | P3 | — | — | Omitted from launch scope; post-launch parity work |
| E16 | FinQA heldout scale (define N) | EVAL | FinQA | OMIT | P3 | — | — | Omitted from launch scope; define if FinQA is re-added |
| W01 | Head-to-head UI redesign | UI | — | UI | P0 | A | `head-to-head-results.tsx` | Readable budget columns |
| W02 | Coverage chart panel filter | UI | launch scope | UI | P1 | C | `pareto-coverage-chart.tsx` | Panels only when evidence exists |
| W03 | Chart D limitation copy | UI | HB, tau2 | UI | P2 | D | `index.mdx` addendum | "2-task run group" disclaimer |
| W04 | Reward diagnostics addendum | UI | historical | OMIT | P3 | — | — | Fresh evidence and MDX re-add required before any future use |
| W05 | Systems diagram narrative pass | UI | — | UI | P2 | React figure | `src/components/blog/posts/introducing-gepa-platform/systems-focus-modal.tsx` | Four-container launch framing |
| E17 | PaperBench JudgeEval container | EVAL | PaperBench judge | PLANNED | P3 | — | — | After current launch scope: wire SimpleJudge + JudgeEval rubrics from frontier-evals as a normal container row |
| E18 | Harvey LAB judge eval container | EVAL | Harvey LAB judge | PLANNED | P3 | — | — | Per-criterion vs batch verifier parity; LAB rubric contract |

### Launch-scope summary counts

These counts cover the active A/C/D launch packet only. Post-launch roadmap
rows still need real experiment work before they can appear in a future chart.

| Status | Count | IDs / scope |
|:------:|------:|-------------|
| DONE | 4 | E01, E04, E05, E13 |
| PARTIAL | 1 | E12 provenance map still needs a polished content table |
| RERUN | 0 | no active A/C/D launch-scope reruns remain |
| MISSING | 0 | — |
| PLANNED | 7 | E06–E09, E11, E17, E18 |
| UI | 4 | W01–W03, W05 |
| OMIT | 7 | E02, E03, E10, E14, E15, E16, W04 |

Launch-scope run data is present in the public evidence packet. Remaining
non-DONE rows are roadmap, optional content, or UI work; they are not launch
evidence blockers for the four-container post. The post still needs the
frontend launch commit and release checklist to cite the final public evidence
commit before it ships.

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

- **✓ public** — container ships publicly for the release.
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
| HotpotQA | [hotpotqa.github.io](https://hotpotqa.github.io/) | ✓ public ([container](../../optimizers/gepa/hotpotqa_container/)) |
| HoVer | [hover-nlp.github.io](https://hover-nlp.github.io/) | → public |
| Heart Disease (UCI) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) | → public |

### Coding agent (agentic shell / code-edit)

| Container | Upstream / dataset | Status |
|---|---|---|
| TBLite micro coding smoke | local pytest micro tasks | ✓ public ([container](../../optimizers/gepa/tblite_container/)) — not Terminal-Bench Lite and not launch evidence |
| Code Review | (internal cookbook PR review task) | runner OSS · container private |
| NGO-style | (internal research target) | runner OSS · container private |

### ReAct environments (long-horizon, game / world)

| Container | Upstream / dataset | Status |
|---|---|---|
| Crafter | [danijar/crafter](https://github.com/danijar/crafter) | ✓ public ([container](../../optimizers/gepa/crafter_container/)) |
| MiniGrid | [Farama MiniGrid](https://minigrid.farama.org/) (via OpenEnv) | → public |
| tau2-bench retail | [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | ✓ public ([container](../../optimizers/gepa/tau2_retail_container/)) |
| tau2-bench airline | [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | → public (`taubench_airline_container`) |
| DungeonGrid | [JoshuaPurtell/DungeonGrid](https://github.com/JoshuaPurtell/DungeonGrid) | ✓ public ([container](../../optimizers/gepa/dungeongrid_container/)) |

### Vertical / domain agent (real-world professional workflows)

| Container | Upstream / dataset | Status |
|---|---|---|
| Harvey Labs (legal) | [harveyai/harvey-labs](https://github.com/harveyai/harvey-labs) | ✓ public ([container](../../optimizers/gepa/harvey_lab_container/)) |
| Legal Apex Agents | [mercor/apex-agents](https://huggingface.co/datasets/mercor/apex-agents) | roadmap |
| HealthBench Professional (medical) | [openai/healthbench-professional](https://huggingface.co/datasets/openai/healthbench-professional) ([paper](https://cdn.openai.com/dd128428-0184-4e25-b155-3a7686c7d744/HealthBench-Professional.pdf)) | ✓ public ([container](../../optimizers/gepa/healthbench_container/)) |
| TaxCalcBench (accounting) | [column-tax/tax-calc-bench](https://github.com/column-tax/tax-calc-bench) — TY2024 return calculation + MeF-style XML grading | runner OSS · container private |
| Finance Agent Benchmark (finance research) | [paper](https://arxiv.org/abs/2508.00828) — 537 expert-authored SEC/financial-research tasks | **not ready** (private scaffold; missing Tavily/SEC/Tiingo keys + no E2E proof) |
| BankerToolBench (investment banking) | [paper](https://arxiv.org/abs/2604.11304) — end-to-end IB workflows (modeling, decks, reports) | roadmap |
| BlueFin (financial spreadsheets) | [paper](https://arxiv.org/abs/2605.30907) — 131 spreadsheet tasks, 3,225 eval criteria | roadmap |

### Judge / verifier evals (planned)

Containerized tasks for optimizing and evaluating grading models. These measure
judge accuracy, agreement, and cost rather than downstream policy reward, but
they should still be handled as normal GEPA containers: rows, candidates,
rollouts, rewards, traces, usage, and chart evidence all come through the same
contract.

When promoted into the post, PaperBench JudgeEval should be another container
tab/row beside HealthBench Pro, tau2 retail, Banking77, and HotpotQA. It should
not be framed as a separate PaperBench-only addendum unless the main container
charts cannot represent the evidence.

| Eval | Upstream / reference | What it measures | Status |
|---|---|---|---|
| **PaperBench JudgeEval** | [PaperBench](https://openai.com/index/paperbench/) ([paper](https://arxiv.org/abs/2504.01848), [frontier-evals](https://github.com/openai/frontier-evals/tree/main/project/paperbench)) | Accuracy of **SimpleJudge** (LLM rubric grader) vs human-graded leaf nodes on partial paper replications; binary classification metrics (F1) per rubric item | roadmap |
| **Harvey LAB judge eval** | [Designing Efficient Verifiers for Legal Agents](https://www.langchain.com/blog/designing-efficient-verifiers-for-legal-agents) (LangChain × Harvey) | Verifier agreement, false-pass/fail rates, and cost for **LAB**-style per-criterion vs batch rubric judging; planned reference must be revalidated before any chart claim | roadmap |

**PaperBench JudgeEval** — OpenAI's auxiliary benchmark inside the PaperBench
pipeline. Agents replicate ICML papers; SimpleJudge grades submissions against
author co-developed rubrics. JudgeEval validates the judge itself using manual
human grades as ground truth ([PaperBench §4.2](https://arxiv.org/abs/2504.01848)).

**Harvey LAB judge eval** — Harvey's open **LAB** benchmark scores legal-agent
deliverables with many pass/fail criteria per task. The LangChain × Harvey study
is useful for the verifier design shape (per-criterion vs batch, agreement,
false-pass/fail, and cost), but the model/reference choice is **not locked** for
this cookbook. Planned cookbook work: expose the same rubric-judge contract used
by [`harvey_lab_container/`](../../optimizers/gepa/harvey_lab_container/) as a
standalone judge-eval container, then choose an approved reference only after a
spot-check proves it is a credible legal judge.

### Tally

The launch evidence in the post uses four containers:
HealthBench Professional, tau2-bench retail, Banking77, and HotpotQA.
Harvey Labs Tax, tau2-bench airline, TaxCalcBench, FinQA, and DungeonGrid are
descoped from launch charts.
The public TBLite path is a micro coding smoke, not OpenThoughts/Terminal-Bench
Lite result coverage; add TBLite back only after a real Harbor/OpenThoughts
rerun has a meaningful heldout denominator.
The broader catalog remains here to show the container-contract surface across
classification, QA, coding, environment-control, and professional-workflow tasks.

## Compute-parity ground rules

Every "vs gepa-ai" chart in this folder documents the recorded comparison
conditions it uses. Chart A uses the public HTTP container boundary for every row.
Each row pins the task container, train/heldout split, coverage threshold,
proposer/reflection model, and task-specific policy or judge model in
`cookbooks/optimizers/gepa/evals/evidence/benchmarks/*/`.

### Launch parity source of truth

The current post has exactly four Chart A/C containers: HealthBench Pro,
tau2-bench retail, Banking77, and HotpotQA. Chart D is a two-task proposer sweep
on HealthBench Pro and tau2-bench retail only. The exact splits, models,
budgets, and evidence checksums live in the chart READMEs and generated
`figures/source_evidence.json` files.

Older Harvey, TaxCalcBench, FinQA, tau2-airline, LangProbe, TBLite, Crafter,
MiniGrid, and DungeonGrid planning notes are historical. They must not be cited
as launch coverage unless a fresh evidence packet and explicit MDX re-add land
in the same public commit.
