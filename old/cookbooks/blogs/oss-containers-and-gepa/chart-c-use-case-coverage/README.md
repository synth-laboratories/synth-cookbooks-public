# Chart C — Use Case Coverage

Honest capability matrix: which optimization use cases each
implementation can target today.

The single most important row is **`optimize_anything`**: gepa-ai
supports it, Synth GEPA does not (yet). This folder makes that gap
explicit instead of hiding it.

## Coverage matrix (current state)

| Use case | Synth GEPA | gepa-ai |
|---|---|---|
| Single-prompt supervised classification | ✓ | ✓ |
| Single-prompt agentic (tool-use) | ✓ | ✓ |
| Multi-module compound prompt programs | ✓ | ✓ |
| Long-horizon ReAct environments | ✓ | ✓ |
| `optimize_anything` (executable code, DSL configs, policy params) | ✗ | ✓ |
| Container-owned task boundary (HTTP contract) | ✓ | partial |
| Cached / readonly replay of public runs | ✓ | ✗ |
| Frozen TOML config, narrow env overrides | ✓ | ✗ |

(Update the matrix in this README and in `coverage.json` whenever
either implementation ships new surface area.)

## Layout

```
chart-c-use-case-coverage/
  README.md
  coverage.json                # Machine-readable matrix
  build_chart.py               # Reads coverage.json, emits figures/coverage.svg
  figures/
    coverage.svg
```

## Reproduce

```bash
python build_chart.py
```

## Status

- [ ] `coverage.json` populated for current state of both stacks.
- [ ] `build_chart.py` renders SVG.
- [ ] Section in blog MDX renders the matrix.
