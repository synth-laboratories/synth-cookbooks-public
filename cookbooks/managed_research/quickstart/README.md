# Managed Research — SDK quickstart

Start one hosted research run and read back its evidence, with the
`synth-ai[research]` SDK. Faithful mirror of the
[Python SDK Quickstart](https://docs.usesynth.ai/managed-research/sdk-quickstart).

## Setup

```bash
uv add "synth-ai[research]==0.11.2"   # or: pip install "synth-ai[research]==0.11.2"
export SYNTH_API_KEY="sk_..."
```

## Run

```bash
python quickstart.py
```

`quickstart.py` starts a one-off `directed_effort` run on the `lite` runbook, waits
for it to finish, and prints the run's state, stop reason, and artifacts (the
reviewable evidence).

## Notes

- Objectives, experiments, and milestones are **managed by the runtime** — you start a
  run and read its evidence; you don't create or step them by hand.
- Upgrading from `0.11.1`? Several SDK methods were removed when that surface became
  runtime-managed — see
  [Upgrading to 0.11.2](https://docs.usesynth.ai/managed-research/upgrading).
