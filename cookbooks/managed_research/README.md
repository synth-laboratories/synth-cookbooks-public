# Managed Research cookbooks

Runnable examples for [Managed Research](https://docs.usesynth.ai/managed-research/intro)
— hosted research workers that turn a brief into reviewable evidence.

These mirror the docs so a copy-paste works end to end. Install the SDK with the
research extra and set your key:

```bash
uv add "synth-ai[research]==0.11.2"   # or: pip install "synth-ai[research]==0.11.2"
export SYNTH_API_KEY="sk_..."
```

| Cookbook | What it shows |
| --- | --- |
| [`quickstart/`](./quickstart) | Start one hosted run and read back its evidence. |
| [`research_factory_reflexion/`](./research_factory_reflexion) | Run one 24/7 Factory that builds a Reflexion implementation from scratch, preserves experiment/Wiki/git/artifact evidence, and graduates through Daytona, exe.dev, Railway, and a fail-closed blog release gate. |

> Objectives, experiments, and milestones are **managed by the runtime** — you start a
> run and read its evidence; you don't create or step them by hand. If you're upgrading
> from `0.11.1`, see [Upgrading to 0.11.2](https://docs.usesynth.ai/managed-research/upgrading).
