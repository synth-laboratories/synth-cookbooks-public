# Synth Public Packages and Cookbooks

Public-safe Synth cookbook recipes, examples, and package source.

This repo is the public monorepo for Synth cookbook content and first-class
public packages. Cookbook material should only land here after review in
`synth-cookbooks-private` confirms the recipe is reproducible without private
credentials, internal datasets, unreleased behavior, or unredacted
logs/screenshots.

## Layout

- `packages/` - public package source, with independently releasable packages
  such as `synth-containers`.
- `cookbooks/` - public recipes, including the Banking77 GEPA optimizer slice.
  Active private drafts still live in `synth-cookbooks-private` until promoted.
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

## Development

See `DEVELOPERS.md` for contribution guidance, package validation commands, and
the conventions for keeping cookbook examples separate from reusable package
code.
