# Public Package Monorepo Handoff

Date: 2026-04-23

Audience: the next engineer setting up the public-facing Synth package layout for cookbooks, containers, and optimizers.

## Decision

Use `synth-cookbooks-public` as the single public monorepo for:

- published cookbook content
- first-class public package source for Synth packages

The goal is to reduce the number of public repos users need to understand while
still publishing proper versioned packages from a clean package-oriented layout.

## Chosen Target Layout

```text
synth-cookbooks-public/
  packages/
    synth-containers/
    synth-optimizers/
  cookbooks/
  assets/
  README.md
```

Optional later:

```text
  examples/
```

but the current preference is to keep examples inside cookbook folders unless a
top-level standalone examples directory becomes clearly necessary.

## Why This Layout

### `packages/`

This is the authoritative source for public publishable packages.

Initial package candidates:

- `synth-containers`
- `synth-optimizers`

These should be treated as first-class packages:

- their own `pyproject.toml`
- explicit versioning
- explicit release workflow
- package-local README / metadata
- package-specific build and publish commands

### `cookbooks/`

This is the home for the user-facing recipe content itself.

Use `cookbooks/` for:

- narrative tutorials
- end-to-end recipes
- runnable public-safe walkthroughs
- cookbook-local example files and configs when they belong to that recipe

### `assets/`

Keep as the public-safe media and diagram folder.

## Current State

Current public repo layout:

```text
synth-cookbooks-public/
  assets/
  containers/
  cookbooks/
```

Current meaning from the repo README:

- `cookbooks/` = published recipes
- `containers/` = public-safe benchmark harnesses, service wrappers, task adapters
- `assets/` = screenshots, diagrams, generated media

## Migration Direction

The intended direction is:

1. introduce `packages/`
2. move package-worthy public code under `packages/`
3. leave `cookbooks/` focused on published recipe content
4. keep `assets/` as-is

That means the long-term replacement for the current top-level `containers/`
folder is likely:

```text
packages/synth-containers/
```

and the optimizer package should live at:

```text
packages/synth-optimizers/
```

## Packaging Conventions

Use distinct distribution names on PyPI, not generic names:

- `synth-containers`
- `synth-optimizers`

Avoid trying to publish overly generic names like:

- `containers`
- `optimizers`
- `optimizer`

because they are either already taken or too collision-prone.

Python import namespaces do not need to match the repo folder exactly if there
is a compatibility reason to keep them stable, but the public package metadata
should be clear and consistent.

## Recommended Migration Order

1. Create `packages/` in this repo.
2. Move the current public container package/code into `packages/synth-containers/`.
3. Move or re-home the public optimizer package/code into `packages/synth-optimizers/`.
4. Update the top-level README to describe the repo as a public monorepo with
   published packages plus cookbook content.
5. Add package-local READMEs and release instructions.
6. Add CI/release jobs that build and publish package artifacts from the
   package subdirectories.

## Release Model

Treat each package as independently releasable.

That implies:

- package-local versioning
- package-local build artifacts
- explicit per-package publish commands

Do not force the cookbook docs to version-lock with package releases unless that
turns out to be helpful later.

## Non-Goals For The First Pass

- no need to split into more public repos
- no need to rename `cookbooks/`
- no need to add a top-level `examples/` directory yet
- no need to unify all package import namespaces immediately if that would break
  existing users

## Open Questions

1. Whether the current public repo name `synth-cookbooks-public` should stay
   forever if it becomes the canonical public package monorepo.
2. Whether any existing public package code should keep legacy import paths for
   compatibility while changing only the distribution/repo naming.
3. Whether package release automation should live in this repo or be triggered
   by an external release pipeline.

## Practical Guideline

If a directory exists mainly to ship installable code, it belongs under
`packages/`.

If a directory exists mainly to teach users how to do something, it belongs
under `cookbooks/`.

If a directory exists mainly to hold images or diagrams, it belongs under
`assets/`.
