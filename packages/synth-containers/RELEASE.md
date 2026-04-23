# Release: synth-containers

Release this package independently from the repository root and from other
packages in this monorepo.

## Build

Run from `packages/synth-containers/`:

```bash
python -m build
```

## Publish

After confirming the version and inspecting the generated artifacts:

```bash
python -m twine upload dist/*
```

Publish automation is intentionally TBD until the repository-level packaging
pipeline is chosen.
