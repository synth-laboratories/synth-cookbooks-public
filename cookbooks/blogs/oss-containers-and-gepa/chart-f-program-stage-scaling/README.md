# Chart F — Program-Stage Scaling (LangProbe Analog)

Heldout reward as a function of program size (1 stage / 2 stages /
3 stages of composed LM calls) on four tasks, comparing Synth GEPA to
standard GEPA. Visual reference: LangProbe paper,
<https://arxiv.org/abs/2502.20315>.

The claim this chart lands: the value of the optimizer platform
*compounds with program complexity*. At program size 1, both
implementations should be close. As program size grows, the gap should
open up.

## Sweep grid

- **Program sizes** (X-axis): 1 stage · 2 stages · 3 stages.
- **Tasks** (small-multiple panels):
  - HotpotQA — multi-hop QA
  - HoVer — claim verification, multi-hop
  - Banking77 — intent classification, supervised (sanity check)
  - Heart Disease — UCI tabular classification (non-NLP sanity check)
- **Series**: Synth GEPA · standard GEPA (gepa-ai).
- **Held constant**:
  - `max_total_rollouts` = same per (task, size) cell
  - Proposer = `gpt-5.4-mini`
  - Student = same per task across both implementations
  - Train/heldout splits

## Why these four tasks

- **HotpotQA** — canonical multi-hop; everyone in this space reports on it.
- **HoVer** — verification with explicit reasoning chain; exercises
  stage-2 signal.
- **Banking77** — supervised classifier sanity check at the simple end.
- **Heart Disease** — tabular non-NLP sanity check that the program
  scaling story holds outside language-heavy tasks.

## Layout

```
chart-f-program-stage-scaling/
  README.md
  containers/                  # New cookbook containers for the LangProbe tasks
    hotpotqa/                  # size_1.py, size_2.py, size_3.py + gepa.toml each
    hover/
    banking77/                 # reuses existing cookbooks/optimizers/gepa/banking77_container/
    heart_disease/
  configs/
    synth_gepa/
      hotpotqa_size1.toml
      hotpotqa_size2.toml
      hotpotqa_size3.toml
      ...                      # 4 tasks × 3 sizes = 12 configs
    gepa_ai/
      hotpotqa_size1.yaml
      ...
  run_sweep.sh
  build_chart.py               # Emits figures/program_stage_scaling.svg
  runs/
    synth_gepa/
      hotpotqa_size1_<ts>/
      ...
    gepa_ai/
      hotpotqa_size1_<ts>/
      ...
  figures/
    program_stage_scaling.svg
```

## Reproduce

```bash
./run_sweep.sh                 # 24 runs (12 cells × 2 implementations)
python build_chart.py
```

## Status

- [ ] Size-1 / size-2 / size-3 containers built for all 4 tasks.
  (Banking77 size-1 already exists; need to build size-2/3 variants.
  HotpotQA / HoVer / Heart Disease need full container builds.)
- [ ] Synth GEPA configs generated.
- [ ] gepa-ai configs generated.
- [ ] Sweep launched at parity budget.
- [ ] Chart SVG rendered.
- [ ] Section in blog MDX embeds chart.

## Open questions

- Building 11 new containers (3 sizes × 4 tasks, minus the existing
  Banking77 size-1) is real work. Probably a follow-up chart, not a
  launch-tonight chart.
- Compute-parity budget per cell needs to be set explicitly in the
  caption and in `run_sweep.sh`.
