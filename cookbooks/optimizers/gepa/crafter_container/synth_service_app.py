"""
Crafter GEPA cookbook container (live Craftax env, OpenAI policy).

This container speaks the public synth-optimizers GEPA contract:
  GET  /metadata
  GET  /task_info
  GET  /program
  GET  /dataset
  POST /dataset/rows
  POST /rollout

Each rollout runs a real Craftax episode using the candidate's
`react_system_prompt` as the system prompt for an OpenAI-driven agent.
Reward = total environment reward for the episode (no string matching,
no fixture). Tools and tile vocabulary come from craftax directly.

Required env:
  OPENAI_API_KEY            — required.
  CRAFTER_POLICY_MODEL      — default: gpt-4.1-nano
  CRAFTER_MAX_TURNS         — default: 20  (per-episode hard cap)
  CRAFTER_MIN_BATCH         — default: 1   (min actions per LLM call)
  CRAFTER_MAX_BATCH         — default: 5   (max actions per LLM call)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request

try:
    from synth_containers import GEPA_OPTIMIZER_CONTRACT_VERSION
except Exception:
    GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v1"

try:
    from openai import OpenAI
except Exception as _openai_err:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = _openai_err
else:
    _OPENAI_IMPORT_ERROR = None


TASK_ID = "crafter.react_policy"
DATASET_ID = "crafter_public_episodes"
REACT_TOOL_NAME = "crafter_interact"

POLICY_MODEL = os.environ.get("CRAFTER_POLICY_MODEL", "gpt-4.1-nano")
MAX_TURNS = int(os.environ.get("CRAFTER_MAX_TURNS", "20"))
MIN_BATCH = int(os.environ.get("CRAFTER_MIN_BATCH", "1"))
MAX_BATCH = int(os.environ.get("CRAFTER_MAX_BATCH", "5"))

# Action vocabulary mirrored from craftax_text_env so the system prompt can
# reference the same names without a hard import at module-load time.
VALID_ACTIONS = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep", "place_stone", "place_table", "place_furnace",
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
    "make_iron_sword",
]
_ACTION_SET = set(VALID_ACTIONS)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

DEFAULT_REACT_SYSTEM_PROMPT = (
    "You are controlling a Crafter survival agent. Each turn you see a compact "
    "text observation (player stats, inventory, local map). Respond ONLY with a "
    "single <tool_call> block of the form:\n"
    f"<tool_call>{{\"name\":\"{REACT_TOOL_NAME}\",\"arguments\":{{\"actions_list\":[\"move_right\",\"do\"]}}}}</tool_call>\n\n"
    "Use 1-5 valid actions per call. Valid actions: " + ", ".join(VALID_ACTIONS) + "\n"
    "Prioritize collecting wood, placing a table, crafting tools, then collecting "
    "stone/coal/iron. Avoid lava."
)

# Episode seeds. Train seeds are used during GEPA's reflective loop; heldout
# seeds gate the final acceptance score.
ROWS = [
    {"seed": 11, "split": "train", "example_id": "ep_train_11"},
    {"seed": 13, "split": "train", "example_id": "ep_train_13"},
    {"seed": 17, "split": "train", "example_id": "ep_train_17"},
    {"seed": 19, "split": "train", "example_id": "ep_train_19"},
    {"seed": 101, "split": "test", "example_id": "ep_heldout_101"},
    {"seed": 103, "split": "test", "example_id": "ep_heldout_103"},
]


# --- OpenAI client (lazy) -----------------------------------------------------

_openai_client: Any = None


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    if OpenAI is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "openai package not installed; container deps in pyproject.toml. "
                f"Original import error: {_OPENAI_IMPORT_ERROR!r}"
            ),
        )
    if "OPENAI_API_KEY" not in os.environ:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY not set in container env; cannot serve live rollouts.",
        )
    _openai_client = OpenAI()
    return _openai_client


# --- Agent / env loop ---------------------------------------------------------


def _parse_tool_actions(raw_text: str, raw_tool_calls: list[dict] | None) -> list[str]:
    """Parse `actions_list` out of either a native tool call or a <tool_call> XML block."""
    # Native tool calls (OpenAI function calling)
    if raw_tool_calls:
        try:
            args = json.loads(raw_tool_calls[0]["function"]["arguments"])
            actions = args.get("actions_list", [])
            return [a for a in actions if a in _ACTION_SET][:MAX_BATCH]
        except Exception:
            pass
    # XML-style fallback (model emits it as plain text)
    if raw_text:
        match = _TOOL_CALL_RE.search(raw_text)
        if match:
            try:
                obj = json.loads(match.group(1))
                actions = obj.get("arguments", {}).get("actions_list", [])
                return [a for a in actions if a in _ACTION_SET][:MAX_BATCH]
            except Exception:
                pass
        # Last resort: pick the first VALID_ACTION token found in the text
        for token in raw_text.split():
            tok = token.strip(",.![](){}\"'").lower()
            if tok in _ACTION_SET:
                return [tok]
    return []


def _llm_step(client: Any, system_prompt: str, observation_text: str, step: int) -> tuple[list[str], dict[str, int]]:
    user_content = (
        f"Step {step + 1}. Current observation:\n\n{observation_text}\n\n"
        f"Call {REACT_TOOL_NAME} with actions_list containing {MIN_BATCH}-{MAX_BATCH} valid actions."
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": REACT_TOOL_NAME,
                "description": "Submit a batch of Craftax actions to execute in order.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions_list": {
                            "type": "array",
                            "items": {"type": "string", "enum": VALID_ACTIONS},
                            "minItems": MIN_BATCH,
                            "maxItems": MAX_BATCH,
                        }
                    },
                    "required": ["actions_list"],
                },
            },
        }
    ]
    # Use chat.completions for tool calling (Responses API tool calling support varies by model).
    resp = client.chat.completions.create(
        model=POLICY_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        tools=tools,
        tool_choice="auto",
    )
    msg = resp.choices[0].message
    text = msg.content or ""
    tool_calls = []
    if getattr(msg, "tool_calls", None):
        tool_calls = [
            {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    actions = _parse_tool_actions(text, tool_calls)
    usage = {
        "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
    }
    return actions, usage


def _run_episode(seed: int, system_prompt: str) -> dict[str, Any]:
    """One real Craftax episode driven by an OpenAI agent."""
    # Lazy import so the FastAPI app can boot for /health without jax.
    from crafter_text_env import CrafterTextEnv

    client = _get_openai_client()
    env = CrafterTextEnv()
    _, text = env.reset(seed)

    total_reward = 0.0
    all_actions: list[str] = []
    all_achievements: list[str] = []
    step = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    done = False

    while step < MAX_TURNS:
        actions, usage = _llm_step(client, system_prompt, text, step)
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)
        if not actions:
            # Model didn't emit a valid action — count the turn but don't step.
            step += 1
            continue
        for action in actions:
            if step >= MAX_TURNS:
                break
            _, text, reward, done, info = env.step(action)
            total_reward += float(reward)
            all_actions.append(action)
            all_achievements.extend(info.get("achievements") or [])
            step += 1
            if done:
                break
        if done:
            break

    env.close()
    return {
        "seed": seed,
        "n_steps": step,
        "total_reward": total_reward,
        "done": done,
        "actions": all_actions,
        "achievements": all_achievements,
        "usage": total_usage,
    }


# --- FastAPI app --------------------------------------------------------------

app = FastAPI(title="crafter-gepa-container")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metadata")
@app.get("/info")
async def metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": "crafter_gepa_live",
            "name": "Crafter GEPA (live Craftax env, OpenAI policy)",
            "description": "Public ReAct prompt-optimizer cookbook running real Craftax episodes with an OpenAI-driven agent.",
        },
        "capabilities": {
            "contract_version": "container_contract.v1",
            "rollout_modes": ["blocking"],
            "metadata": {
                "trace_schema": "prompt_calls.llm_request.messages.v1",
            },
        },
        "metadata": {
            "optimizer_contracts": {
                "gepa": {
                    "version": GEPA_OPTIMIZER_CONTRACT_VERSION,
                    "program_route": "/program",
                    "dataset_route": "/dataset",
                    "dataset_rows_route": "/dataset/rows",
                    "rollout_route": "/rollout",
                }
            }
        },
    }


@app.get("/task_info")
async def task_info() -> dict[str, Any]:
    return {
        "task": {
            "task_id": TASK_ID,
            "name": "Crafter ReAct policy",
            "description": "Optimize a ReAct system prompt for a Craftax survival agent.",
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "visible_splits": ["train", "test"],
            "default_split": "train",
            "row_count": len(ROWS),
        },
        "metadata": {
            "policy_model": POLICY_MODEL,
            "max_turns": MAX_TURNS,
            "tool_name": REACT_TOOL_NAME,
            "trace_schema": "prompt_calls.llm_request.messages.v1",
        },
    }


@app.get("/program")
async def program() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "crafter_react_prompt_gepa",
        "modules": [
            {
                "module_id": "react_system_prompt",
                "role": "system",
                "content": DEFAULT_REACT_SYSTEM_PROMPT,
                "mutable": True,
                "candidate_field": "react_system_prompt",
                "template_variables": [],
                "metadata": {
                    "surface": "react_system_prompt",
                    "tool_name": REACT_TOOL_NAME,
                },
            }
        ],
        "target_modules": [
            {
                "module_id": "react_system_prompt",
                "candidate_field": "react_system_prompt",
                "objective": "total_episode_reward",
            }
        ],
        "seed_candidate": {"react_system_prompt": DEFAULT_REACT_SYSTEM_PROMPT},
        "rollout_overlay_schema": {"candidate_fields": ["react_system_prompt"]},
        "metadata": {
            "task_id": TASK_ID,
            "dataset_id": DATASET_ID,
            "tool_name": REACT_TOOL_NAME,
            "trace_schema": "prompt_calls.llm_request.messages.v1",
        },
    }


@app.get("/dataset")
async def dataset() -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "splits": {
            "train": sum(1 for row in ROWS if row["split"] == "train"),
            "test": sum(1 for row in ROWS if row["split"] == "test"),
        },
        "source": "crafter_public_episode_seeds",
    }


@app.post("/dataset/rows")
async def dataset_rows(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    seeds = [int(seed) for seed in payload.get("seeds") or []]
    return {"rows": [_row_for_seed(split=split, seed=seed) for seed in seeds]}


@app.post("/rollout")
@app.post("/rollouts")
async def rollout(request: Request) -> dict[str, Any]:
    payload = await request.json()
    row = payload.get("dataset_row") if isinstance(payload.get("dataset_row"), dict) else None
    if not row:
        row = _row_for_seed(
            split=str(payload.get("split") or "train"),
            seed=int(payload.get("seed") or 11),
        )
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    system_prompt = str(candidate.get("react_system_prompt") or DEFAULT_REACT_SYSTEM_PROMPT)

    seed = int(row.get("seed") or 0)
    episode = _run_episode(seed=seed, system_prompt=system_prompt)
    reward = float(episode["total_reward"])

    rollout_id = str(payload.get("rollout_id") or f"rollout_{uuid.uuid4().hex[:12]}")
    now = _now()
    return {
        "rollout_id": rollout_id,
        "status": "completed",
        "success_status": "succeeded" if reward > 0 else "failed",
        "task_id": TASK_ID,
        "seed": seed,
        "reward_info": {
            "outcome_reward": reward,
            "event_rewards": [reward],
            "details": {
                "example_id": row.get("example_id"),
                "n_steps": episode["n_steps"],
                "achievements": episode["achievements"],
                "policy_model": POLICY_MODEL,
                "max_turns": MAX_TURNS,
                "tool_name": REACT_TOOL_NAME,
            },
        },
        "summary": {
            "outcome_reward": reward,
            "example_id": row.get("example_id"),
            "n_steps": episode["n_steps"],
            "achievements_unlocked": episode["achievements"],
            "actions_taken": episode["actions"],
        },
        "usage": {**episode["usage"], "cost_usd": 0.0},
        "trace": {
            "event_history": [
                {
                    "type": "episode_complete",
                    "seed": seed,
                    "total_reward": reward,
                    "n_steps": episode["n_steps"],
                    "achievements": episode["achievements"],
                }
            ],
            "metadata": {
                "example_id": row.get("example_id"),
                "call_site_id": "crafter.react_policy",
            },
        },
        "metadata": {"candidate": candidate},
        "created_at": now,
        "updated_at": now,
    }


def _row_for_seed(*, split: str, seed: int) -> dict[str, Any]:
    normalized_split = "test" if split in {"heldout", "test", "validation", "val"} else "train"
    rows = [row for row in ROWS if row["split"] == normalized_split]
    if not rows:
        rows = list(ROWS)
    match = next((row for row in rows if int(row["seed"]) == int(seed)), None)
    row = match or rows[int(seed) % len(rows)]
    return dict(row)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
