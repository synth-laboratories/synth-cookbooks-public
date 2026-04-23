# Packages

This directory contains public Synth packages published from this monorepo.

Each package should be treated as independently releasable:

- package-local metadata and versioning
- package-local README
- package-local build command
- package-local publish command

Current package slots:

- `synth-containers`
- `synth-optimizers`

Use distinct public distribution names such as `synth-containers` and
`synth-optimizers`; avoid generic names like `containers` or `optimizers`.

Package builds are checked by `.github/workflows/package-build.yml`. Publish
automation is intentionally not chosen yet; until that is decided, follow the
package-local `RELEASE.md` instructions from the package directory.
