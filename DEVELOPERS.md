# Developer Guide

This repository is the public home for Synth cookbooks and public Python
packages. Keep the repo easy to audit: public examples should be reproducible,
package code should be releaseable, and private Synth internals should not leak
into either surface.

## Repository Shape

- `cookbooks/` contains public recipes, runnable examples, narrative guides, and
  cookbook-local assets.
- `packages/` contains independently versioned Python packages published from
  this monorepo.
- `skills/` contains portable agent skills that help users work with the public
  packages and examples.
- `assets/` contains public-safe screenshots, diagrams, and generated media.

Use this rule of thumb: code that users should import belongs in `packages/`;
code that teaches a workflow or wires a concrete example belongs in
`cookbooks/`.

## Public-Safe Contributions

Before adding or moving content into this repository, check that it does not
depend on private credentials, unpublished services, internal datasets,
unredacted logs, private URLs, or screenshots with sensitive project data.

Cookbooks should be runnable from public instructions. If a recipe needs an
external service, document the required URL, token, data path, or dry-run mode
explicitly. Prefer small checked-in sample inputs over large opaque blobs.

## Package Development

Each package under `packages/` owns its own `pyproject.toml`, README, version,
build artifacts, and release notes. Do package work from the package directory:

```bash
cd packages/synth-containers
uv sync --group dev
uv run --group dev ruff check src
uv run --group dev ty check src
uv build
uv run --group dev twine check dist/*
```

Use the same shape for `packages/synth-optimizers`.

Do not publish from the repository root. Follow the package-local `RELEASE.md`
for manual release steps and keep package versions independent.

## synth-containers

`synth-containers` is the shared runtime and task contract substrate. Its core
surface should stay framework-neutral:

- Put public nouns, protocols, resource references, capability declarations, and
  HTTP contract helpers in `src/synth_containers/`.
- Put Harbor, OpenEnv, Archipelago, or other framework-specific translation
  code in `src/synth_containers/compat/`.
- Keep compat adapters thin. They should wrap caller-owned services, task
  registries, and rollout handlers; they should not own Docker lifecycle,
  native environment implementations, or private service internals.
- Use `ResourceKind` only for broad umbrellas. Put framework-specific detail in
  `ResourceRef.subtype`, `ResourceRef.labels`, and `ResourceRef.metadata`.
- Expose task resources through `resource_refs` on task definitions, task
  instances, task info, and task catalogs.
- Preserve stable HTTP behavior through `create_reference_app`; add optional
  runtime methods when a runtime needs richer behavior, such as
  `task_info_for_request(query)`.

Avoid compatibility fallbacks in new public interfaces unless they are required
for a documented migration path. Prefer explicit dataclasses, enums, protocols,
and typed payload conversion at the boundary.

## synth-optimizers

`synth-optimizers` is the public optimizer package. It should start from
Synth Lab-derived optimizer logic, not old `prompt_opt` scaffolding.

Guidelines:

- Keep the import surface under `synth_optimizers`.
- Keep optimizer-specific implementation under feature subpackages such as
  `synth_optimizers.miprov2`.
- Depend on `synth-containers` for shared task, resource, and runtime capability
  concepts instead of redefining them.
- Make optimizer requirements explicit. If an optimizer needs rollout,
  evaluation, checkpoint, trace, or token support, represent that requirement in
  code near the optimizer entrypoint.
- Keep cookbook examples as clients of the package rather than copies of package
  internals.

## Cookbook Development

Cookbooks are proofs that the packages are useful. They should favor clarity and
small runnable entrypoints over framework completeness.

For each cookbook:

- Include a README with prerequisites, inputs, environment variables, and an
  expected command.
- Use package APIs instead of duplicating shared runtime, task, optimizer, or
  HTTP scaffolding.
- Keep generated outputs under ignored paths such as `.out/`.
- Keep large datasets, build contexts, and runtime services referenced by path,
  URL, digest, or documented setup step rather than vendoring them.
- When adapting a framework, use the relevant `synth_containers.compat` module
  if one exists; add a new thin compat adapter before copying bespoke endpoint
  glue across examples.

## Validation

Run one focused validation pass after implementation and before committing.
Choose the checks that match the files you changed.

For package code:

```bash
cd packages/<package-name>
uv run --group dev ruff check src
uv run --group dev ty check src
uv build
uv run --group dev twine check dist/*
```

For cookbook entrypoints:

```bash
python -m py_compile path/to/entrypoint.py
```

For container contract changes, also smoke the public import path and
serialization shape from source or the built wheel. If a gate is skipped, record
which one and why in your handoff or commit notes.

## Style

Prefer boring, explicit interfaces:

- dataclasses for structured public values
- `StrEnum` for stable string vocabularies
- protocols for runtime capabilities
- typed conversion helpers at HTTP/service boundaries
- small subpackages with obvious ownership

Keep private Synth assumptions out of public names and docs. Public abstractions
should describe what a user can rely on, not how an internal system happened to
implement it first.

## Release Readiness

A package is release-ready when:

- its README explains the public surface and basic usage
- `pyproject.toml` has the intended distribution name, version, dependencies,
  license, and URLs
- lint, type checking, build, and `twine check` pass from the package directory
- cookbook examples import the package rather than vendored copies of the same
  logic
- release notes or package-local `RELEASE.md` describe the manual publish steps

Do not mix unrelated cookbook rewrites, package migrations, and release prep in
one commit unless the change is intentionally a migration commit and the commit
message says so.
