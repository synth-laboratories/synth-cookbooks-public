# Packages

The public Synth packages used by these cookbooks now live in their own
canonical repositories and are published to PyPI. They are no longer vendored
here.

| Package | PyPI | Source repo |
| --- | --- | --- |
| `synth-containers` | <https://pypi.org/project/synth-containers/> | <https://github.com/synth-laboratories/containers> |
| `synth-optimizers` | <https://pypi.org/project/synth-optimizers/> | <https://github.com/synth-laboratories/optimizers> |

Cookbooks consume them from PyPI. For example, the GEPA runners invoke the
optimizer with:

```bash
uv run --no-project --with synth-optimizers==0.2.0 synth-optimizers gepa run --config <config>
```

and the container apps depend on `synth-containers>=0.2.0` in their per-container
`pyproject.toml`. To work on the package internals, clone the canonical repo and
follow its `RELEASE.md`.
