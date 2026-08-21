# Synth Public Packages and Cookbooks

Public-safe Synth cookbook recipes, examples, and package source.

This repo is the public monorepo for Synth cookbook content and first-class
public examples. Cookbook material should only land here after public-promotion
review confirms the recipe is reproducible without private credentials,
internal datasets, unreleased behavior, or unredacted logs/screenshots.

## Layout

- `packages/` - public package source, with independently releasable packages
  such as `synth-containers`.
- `cookbooks/` - public recipes, including GEPA optimizer examples and the
  GELO quickstart slice.
- `skills/` - portable agent skills that help users run or adapt the public
  packages and cookbooks.
- `assets/` - public-safe screenshots, diagrams, and generated media.

## Target Layout

This repo uses the public monorepo layout:

```text
synth-cookbooks-public/
  packages/
    synth-containers/
    synth-optimizers/
  cookbooks/
  skills/
  assets/
```

Keep recipe-first content in `cookbooks/` only after public promotion review;
move installable, versioned code under `packages/`.

## Packages

Packages are independently releasable. Each package should own its package
metadata, README, version, build command, and publish command.

- `packages/synth-containers/` - public container harnesses, service wrappers,
  and task adapters used by published recipes.
- `packages/synth-optimizers/` - prerelease public optimizer tooling with a
  Rust GEPA core and thin Python bindings.

## Optimizer Cookbooks

The public GEPA examples live under `cookbooks/optimizers/gepa/`. They use the
Rust `synth-optimizers` GEPA runtime with a Codex app-server proposer and
HTTP task containers. Current cookbook containers include Banking77, HotpotQA,
the TBLite coding micro-container, Crafter, and MiniGrid.

The public GELO quickstart lives under `cookbooks/optimizers/gelo/`. It is the
current public GELO cookbook path for launch, docs, and blog links.

Related reusable agent skills now live in the
[`synth-laboratories/optimizers`](https://github.com/synth-laboratories/optimizers)
repository.

Run GEPA from the container directory. Each container keeps its base
`gepa.toml`, profile TOMLs under `run_profiles/`, and a `run_fresh_gepa.sh`
helper that generates the concrete run config:

```bash
cd cookbooks/optimizers/gepa/banking77_container
bash run_fresh_gepa.sh --profile long
```

Equivalent local runs:

```bash
cd cookbooks/optimizers/gepa/hotpotqa_container && bash run_fresh_gepa.sh --profile long
cd cookbooks/optimizers/gepa/tblite_container && bash run_fresh_gepa.sh --profile long
cd cookbooks/optimizers/gepa/crafter_container && bash run_fresh_gepa.sh --profile long
cd cookbooks/optimizers/gepa/minigrid_container && bash run_fresh_gepa.sh --profile long
```

Use `bash run_fresh_gepa.sh --list` inside a container directory to inspect its
available profiles. Run artifacts are written under
`cookbooks/optimizers/gepa/runs/` and are intentionally ignored by git.

Public GEPA runs should use API-key auth for the proposer, not a developer's
local Codex login. Set `OPENAI_API_KEY` for the Codex proposer. Some shipped
policy profiles also need `OPENROUTER_API_KEY`; check the selected TOML profile.

A GEPA-compatible task container must expose the optimizer HTTP contract:
`/health`, `/metadata`, `/task_info`, `/program`, `/dataset`,
`/dataset/rows`, and `/rollout`. The `/task_info` route is important: it gives
the general proposer task context, objectives, output constraints, and
prompt-writing guidance so the same Rust GEPA loop can work across
classification, QA, coding, and agent-environment tasks.

For the full GEPA config schema, container contract, Codex auth settings,
artifact layout, and public-safety rules, read
`cookbooks/optimizers/gepa/README.md`.

## Development

See `DEVELOPERS.md` for contribution guidance, package validation commands, and
the conventions for keeping cookbook examples separate from reusable package
code.
