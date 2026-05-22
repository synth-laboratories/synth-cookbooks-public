# Blog Reproduction Cookbooks

Every public Synth blog post that ships with quantitative claims has a
matching folder under this directory containing the reproduction code,
configs, and tracked evidence snapshots for each chart in the post.

The convention is:

```
cookbooks/blogs/<post-slug>/
  README.md                        # post-level overview, link to the live post
  chart-<letter>-<short-name>/     # one folder per chart
    README.md                      # what the chart shows + how to reproduce
    *.toml                         # configs (gepa.toml, parity sweeps, etc)
    *.sh                           # repro commands
    runs/                          # ignored local rerun outputs
    figures/                       # generated SVGs / PNGs used in the post
```

Each chart folder is self-contained: someone should be able to clone the
repo, `cd` into the chart folder, follow the README, and regenerate the
chart from scratch using the public container contract. When raw runs are
too large for the launch branch, the chart folder tracks compact source
evidence with hashes under `figures/`.

## Posts

- [oss-containers-and-gepa/](./oss-containers-and-gepa/) — May 2026 launch
  of `synth-optimizers` platform + container contract, GEPA as the first
  algorithm on top.
