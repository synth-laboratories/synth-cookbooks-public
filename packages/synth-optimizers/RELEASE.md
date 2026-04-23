# Release: synth-optimizers

Release this package independently from the repository root and from other
packages in this monorepo.

## Build

Run from `packages/synth-optimizers/`:

```bash
uv run --group dev ruff check src
uv run --group dev ty check src
uv build
```

## Publish

After confirming the version, inspecting the generated artifacts, and confirming
that the required `synth-containers` version is already available on PyPI:

```bash
uv publish dist/*
```

Publish automation is intentionally TBD until the repository-level packaging
pipeline is chosen.
