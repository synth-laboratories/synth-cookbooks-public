# Synth Public Packages and Cookbooks

Public-safe Synth cookbook recipes, examples, and package source.

This repo is the public monorepo for Synth cookbook content and first-class
public packages. Cookbook material should only land here after review in
`synth-cookbooks-private` confirms the recipe is reproducible without private
credentials, internal datasets, unreleased behavior, or unredacted
logs/screenshots.

## Layout

- `packages/` - public package source, with independently releasable packages
  such as `synth-containers` and `synth-optimizers`.
- `cookbooks/` - published recipes, narrative tutorials, runnable
  walkthroughs, and cookbook-local example files.
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

Keep recipe-first content in `cookbooks/`; move installable, versioned code
under `packages/`.

## Packages

Packages are independently releasable. Each package should own its package
metadata, README, version, build command, and publish command.

- `packages/synth-containers/` - public container harnesses, service wrappers,
  and task adapters used by published recipes.
- `packages/synth-optimizers/` - public optimizer package source.

## Development

See `DEVELOPERS.md` for contribution guidance, package validation commands, and
the conventions for keeping cookbook examples separate from reusable package
code.
