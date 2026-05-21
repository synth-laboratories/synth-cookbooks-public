# OpenAI API Proposer

A `local_process_json` GEPA proposer that calls the OpenAI Responses
API (falling back to Chat Completions) using `OPENAI_API_KEY`. No
ChatGPT/Codex auth bundle required.

This is the default OSS proposer for the public cookbook. Any
container in `cookbooks/optimizers/gepa/` can wire it in by pointing
`[proposer].command` at `proposer.py`.

## Wire-up

```toml
[proposer]
backend = "local_process_json"
execution_mode = "local_process"
command = [
  "uv", "run", "--with", "openai>=1.0", "python3",
  "../_proposers/openai_api/proposer.py",
]
timeout_seconds = 120
model = "gpt-4.1-mini"   # any OpenAI model; gpt-5.x routes via Responses API
reasoning_effort = ""    # or "low" / "medium" / "high" for reasoning models
```

Paths in `[proposer].command` are resolved relative to the `cwd`
declared in `[container]`, which the cookbooks set to
`cookbooks/optimizers/gepa/` (the parent of the per-cookbook
container folder). The relative path above (`../_proposers/...`)
points at this file from any per-container folder.

## Required env

- `OPENAI_API_KEY` — your OpenAI key.
- `OPENAI_BASE_URL` (optional) — alternate endpoint (Azure, proxy, etc).

## Protocol

The proposer speaks the `local_process_json` protocol:

**Stdin (request):**

```json
{
  "generation": 0,
  "parent": { "payload": { "<module field>": "<current text>", ... }, ... },
  "program": { "modules": [...], "target_modules": [...] },
  "target_modules": ["<field>"],
  "proposal_count": 2,
  "model": "gpt-4.1-mini",
  "reasoning_effort": ""
}
```

**Stdout (response):**

```json
{
  "proposals": [
    { "candidate": { "<field>": "<new text>", ... }, "rationale": "..." }
  ],
  "usage": { "prompt_tokens": int, "completion_tokens": int, "total_tokens": int },
  "backend": "openai_api"
}
```

Errors are written to stderr; the platform surfaces them as
`ProposerError`. No silent failures.

## Behavior

- For each proposal in `proposal_count`, picks one mutable target
  field round-robin from `target_modules`.
- Builds a prompt that includes the program id, the target objective
  (from `program.target_modules[*].objective`), the current text of the
  field being mutated, and the read-only context of other modules in
  the same program.
- Calls OpenAI once per proposal. Tracks token usage across all calls.
- Outputs a candidate that copies the parent payload and replaces only
  the target field with the model's rewrite.

## Used by

- `tblite_container/gepa.toml` (default OSS path)
- Available to wire into any other cookbook container as needed.
