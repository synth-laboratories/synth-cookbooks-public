#!/usr/bin/env python3
"""
OpenAI-API GEPA proposer for synth-optimizers `local_process_json`.

Reads the platform's proposer JSON request from stdin and writes a
proposer JSON response to stdout. Calls the OpenAI Responses API
(via the `openai` SDK) with `OPENAI_API_KEY` — no ChatGPT/Codex
auth bundle required.

Wire it via `gepa.toml`:

    [proposer]
    backend = "local_process_json"
    execution_mode = "local_process"
    command = ["uv", "run", "--with", "openai>=1.0", "python",
               "openai_api_proposer.py"]
    model = "gpt-5.4-mini"
    reasoning_effort = "medium"
    timeout_seconds = 300

Environment:

    OPENAI_API_KEY   — required.
    OPENAI_BASE_URL  — optional, for custom endpoints.

Stdin protocol:

    {
      "generation": int,
      "parent": { "payload": {<field>: <current text>, ...}, ... },
      "program": { "modules": [...], "target_modules": [...], ... },
      "target_modules": [<field name>, ...],
      "proposal_count": int,
      "model": str,
      "sandbox_mode": str,
      "approval_policy": str,
      "reasoning_effort": str
    }

Stdout protocol:

    {
      "proposals": [
        { "candidate": {<field>: <new text>, ...}, "rationale": str },
        ...
      ],
      "usage": { "prompt_tokens": int, "completion_tokens": int, "total_tokens": int },
      "backend": "openai_api"
    }

Errors go to stderr (platform surfaces them as ProposerError).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    sys.stderr.write(
        "openai_api_proposer: missing dependency `openai`. "
        "Install with `pip install openai>=1.0` or pass `--with openai>=1.0` to `uv run`.\n"
    )
    sys.exit(2)


SYSTEM_PROMPT = """You are GEPA, a prompt optimizer.

You are given the current text of one mutable prompt module from a
language-model program, plus the program's program-level context.
Your job: propose a single revised version of that module that will
score higher on the program's task.

Constraints:
- Return ONLY the new module text. No preamble, no explanation, no quotes.
- Preserve the module's intended role and output format.
- Do not invent new tool names, new output schemas, or new constraints
  that are not already implied by the program.
- Be concise. Long prompts are not better. Tight, evidence-grounded
  instructions outperform verbose scaffolding.
"""


def build_user_prompt(
    *,
    field_name: str,
    current_text: str,
    program: dict[str, Any],
    generation: int,
    proposal_index: int,
) -> str:
    program_id = program.get("program_id", "<unknown>")
    target_objectives = [
        t.get("objective", "")
        for t in program.get("target_modules", [])
        if t.get("candidate_field") == field_name
    ]
    objective_block = (
        "\nTarget objective for this module:\n"
        + "\n".join(f"- {o}" for o in target_objectives if o)
        if any(target_objectives)
        else ""
    )

    other_modules = [
        m
        for m in program.get("modules", [])
        if m.get("candidate_field") != field_name and m.get("content")
    ]
    other_block = ""
    if other_modules:
        formatted = "\n\n".join(
            f"### Module `{m.get('module_id', '?')}` (role: {m.get('role', '?')}):\n{m.get('content', '')}"
            for m in other_modules
        )
        other_block = (
            "\n\nOther modules in the same program (for context — do NOT rewrite):\n\n"
            + formatted
        )

    return f"""Program: `{program_id}`
Generation: {generation}, proposal index: {proposal_index}
{objective_block}

You are rewriting this mutable module:

### Module field: `{field_name}`

Current text:

\"\"\"
{current_text}
\"\"\"
{other_block}

Return ONLY the new text for `{field_name}`. No preamble, no quotes, no explanation.
"""


def call_openai(
    client: OpenAI,
    *,
    model: str,
    reasoning_effort: str,
    user_prompt: str,
) -> tuple[str, dict[str, int]]:
    """Call OpenAI; return (text, usage_dict)."""
    # Use Responses API for new models, fall back to Chat Completions otherwise.
    # We try Responses first since that's where gpt-5.x lives.
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        resp = client.responses.create(**kwargs)
        text = (resp.output_text or "").strip()
        usage = {
            "prompt_tokens": getattr(resp.usage, "input_tokens", 0) or 0,
            "completion_tokens": getattr(resp.usage, "output_tokens", 0) or 0,
            "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
        }
        return text, usage
    except Exception as resp_err:
        # Fallback to chat.completions for older models / API endpoints.
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
            }
            return text, usage
        except Exception as chat_err:
            raise RuntimeError(
                f"OpenAI call failed: responses_api={resp_err!r}; chat_completions={chat_err!r}"
            ) from chat_err


def main() -> int:
    if "OPENAI_API_KEY" not in os.environ:
        sys.stderr.write("openai_api_proposer: OPENAI_API_KEY not set.\n")
        return 2

    try:
        request = json.load(sys.stdin)
    except json.JSONDecodeError as err:
        sys.stderr.write(f"openai_api_proposer: invalid JSON on stdin: {err}\n")
        return 2

    generation = int(request.get("generation", 0))
    parent = request.get("parent") or {}
    parent_payload = parent.get("payload") or {}
    program = request.get("program") or {}
    target_modules = list(request.get("target_modules") or [])
    proposal_count = max(1, int(request.get("proposal_count", 1)))
    model = request.get("model") or "gpt-4.1-mini"
    reasoning_effort = (request.get("reasoning_effort") or "").strip()

    if not target_modules:
        sys.stderr.write("openai_api_proposer: target_modules is empty.\n")
        return 2

    client = OpenAI()  # picks up OPENAI_API_KEY + OPENAI_BASE_URL from env

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Each proposal is an independent OpenAI call; fan them out in parallel.
    # This is critical when proposal_count > 1 — sequential per-call latency
    # was the long pole in head-to-head wall-clock measurements.
    from concurrent.futures import ThreadPoolExecutor

    def _one(proposal_index: int):
        # Round-robin across target modules so each proposal mutates a different one
        # when proposal_count >= len(target_modules); otherwise mutate the first.
        field_name = target_modules[proposal_index % len(target_modules)]
        current_text = parent_payload.get(field_name)
        if not isinstance(current_text, str):
            current_text = ""
        user_prompt = build_user_prompt(
            field_name=field_name,
            current_text=current_text,
            program=program,
            generation=generation,
            proposal_index=proposal_index,
        )
        new_text, usage = call_openai(
            client,
            model=model,
            reasoning_effort=reasoning_effort,
            user_prompt=user_prompt,
        )
        return proposal_index, field_name, new_text, usage

    proposals_indexed: list[tuple[int, dict[str, Any]] | None] = [None] * proposal_count
    with ThreadPoolExecutor(max_workers=proposal_count) as pool:
        futures = [pool.submit(_one, i) for i in range(proposal_count)]
        for fut in futures:
            try:
                proposal_index, field_name, new_text, usage = fut.result()
            except Exception as err:
                sys.stderr.write(
                    f"openai_api_proposer: model call failed: {err}\n"
                )
                return 3
            if not new_text:
                sys.stderr.write(
                    f"openai_api_proposer: model returned empty text on "
                    f"proposal {proposal_index} (field={field_name!r}).\n"
                )
                return 3
            candidate: dict[str, Any] = dict(parent_payload)
            candidate[field_name] = new_text
            proposals_indexed[proposal_index] = (
                proposal_index,
                {
                    "candidate": candidate,
                    "rationale": (
                        f"openai_api proposer rewrote `{field_name}` "
                        f"via model={model} reasoning_effort={reasoning_effort or 'default'} "
                        f"at generation={generation}, proposal_index={proposal_index}."
                    ),
                },
            )
            for k in total_usage:
                total_usage[k] += usage.get(k, 0) or 0

    # Preserve deterministic order across runs.
    proposals = [entry[1] for entry in proposals_indexed if entry is not None]

    response = {
        "proposals": proposals,
        "usage": total_usage,
        "backend": "openai_api",
    }

    json.dump(response, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
