# Harvey LAB (Tax) GEPA cookbook container

A public GEPA container for **Harvey LAB** — Harvey AI's open-source **Legal
Agent Benchmark** (MIT, `github.com/harveyai/harvey-labs`) — mirroring the
[`crafter_container`](../crafter_container) / [`dungeongrid_container`](../dungeongrid_container)
pattern.

- GEPA optimizes a single mutable module, `system_prompt` (the legal-associate
  guidance).
- Each rollout: the system prompt drives an LLM that reads a **Tax** matter's
  instructions + document text and writes a work product; an LLM **rubric judge**
  then scores every atomic PASS/FAIL criterion. Reward = fraction of criteria
  passed. No string matching, no fixtures.

**Simplification (disclosed):** this is the *text-in/text-out* version of LAB —
documents are provided as text in context rather than navigated in a sandboxed
file system, and the work product is text (no `.docx`/pandoc/Docker). Only the
**Tax** practice area is bundled.

## Run

```bash
# from cookbooks/optimizers/gepa/
python harvey_lab_container/prepare_dataset.py        # clone MIT benchmark + bundle Tax
synth-optimizers gepa run --config harvey_lab_container/gepa.toml
```

## Env

| Var | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `HARVEY_LAB_POLICY_MODEL` | `gpt-4.1-nano` | associate model |
| `HARVEY_LAB_JUDGE_MODEL` | `gpt-4.1-mini` | rubric-judge model |
| `HARVEY_LAB_MAX_DOC_CHARS` | `9000` | per-document context budget |

## Files
- `synth_service_app.py` — GEPA contract service (mirrors Crafter/DungeonGrid).
- `prepare_dataset.py` — clones the MIT benchmark, bundles the Tax practice area.
- `gepa.toml` — GEPA run config.
- `pyproject.toml` — light deps (`fastapi`, `uvicorn`, `openai`).
