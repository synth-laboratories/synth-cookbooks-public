# Synth Cookbook Skills

This directory contains portable agent skills for the public Synth cookbooks.

A skill is a small folder with a required `SKILL.md` file and optional
`references/`, `scripts/`, and `assets/` folders. The format follows the Agent
Skills shape used by Codex, Claude, and the backend SMR skill catalog:

```text
skills/
  miprov2/
    SKILL.md
  containers/
    SKILL.md
```

Use these skills as copyable, public cookbook assets. For native Codex
auto-discovery inside a checkout, copy or symlink the desired skill folder into
`.agents/skills/`.

## Authoring Rules

- Keep each skill focused on one repeatable workflow.
- Put trigger words and boundaries in the frontmatter `description`; agents use
  that field to decide when a skill applies.
- Keep run-specific evidence in cookbook `run_artifacts/`, not in the reusable
  skill body.
- Reference public-safe paths only. Do not include private endpoints, raw
  secrets, private traces, or unredacted local logs.
- Prefer instructions first. Add `scripts/` only when deterministic helper code
  is worth the extra surface area.

