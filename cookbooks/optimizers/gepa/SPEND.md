# GEPA run spend log

Spend from the `run_gepa.sh --cfg` GEPA runs (banking77 + minigrid), 2026-05-27.
The optimizer logs token counts but does **not** compute $ cost (`cost=$0.0000`),
so authoritative spend is the provider dashboards.

## Authoritative account state (2026-05-27)

- **OpenRouter (policy models): total_usage $296.88 / $299.97 credits → ~$3 remaining.**
  ⚠ Nearly exhausted — top up before more OpenRouter policy runs. (Cumulative
  account usage, not just this session.)
- OpenAI (codex app-server proposer, gpt-5.4-mini/nano): separate billing; codex
  token usage is not surfaced in the run logs.

## This session's GEPA runs (tokens from runs/run_registry.jsonl)

Finalized runs (policy + reported usage):

| run_id | policy model | tokens |
|---|---|---|
| banking77_sanity_141018 | gemini-2.5-flash-lite (OR) | 0.764M |
| banking77_sanity_162308 | gemini-2.5-flash-lite (OR) | 1.336M |
| banking77_sanity_163558 | llama-3.2-3b-instruct (OR) | 1.080M |
| earlier small banking77 (134xxx–140116) | gpt-4.1-nano / gemini (OR) | ~0.05–0.25M each |
| failed/partial (142554, 142759, 143015, 145938) | gpt-oss-20b / gemini (OR) | not finalized (0 in registry); 143015 ran ~part of 4 gens before disk-death |
| minigrid_concurrent (empty3/hi) | gpt-5.4-mini (OpenAI) | ~0.15M each |

Rough session policy-token total ≈ **5–8M tokens** (OpenRouter), i.e. low
single-digit dollars at flash-lite/llama/nano rates. Proposer (OpenAI codex
gpt-5.4-mini/nano) adds an un-surfaced amount — codex app-server runs are
reasoning/workspace-heavy; estimate a few dollars.

## Notes
- Pricing is not wired into the runner; to record exact $, read OpenRouter
  `/api/v1/credits` (policy) and the OpenAI usage dashboard (proposer) before/after.
- Biggest single cost driver: multi-generation runs with large heldout (the 4-gen
  100/200 run hit 21 GB disk and ~1.3M+ tokens before the disk cap).
