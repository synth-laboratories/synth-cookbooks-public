"""
DungeonGrid GEPA cookbook container (live DungeonGrid env, LLM policy).

Mirrors `crafter_container` and speaks the public synth-optimizers GEPA contract:
  GET  /metadata
  GET  /task_info
  GET  /program
  GET  /dataset
  POST /dataset/rows
  POST /rollout

Each rollout runs a real DungeonGrid episode using the candidate's
`react_system_prompt` as the system prompt for an LLM-driven party of heroes.
Reward = total environment reward for the episode (achievement bonuses included,
no string matching, no fixture). Actions and achievements come from the
`dungeongrid` package directly.

Required env:
  OPENAI_API_KEY            — required.
  OPENAI_BASE_URL           — optional; point at a gemini OpenAI-compatible
                              gateway to run the default gemini policy.
  OPENROUTER_API_KEY        — optional fallback when OPENAI_BASE_URL routes to OpenRouter.
  DUNGEONGRID_POLICY_MODEL  — default: gemini-3.1-flash-lite
  DUNGEONGRID_QUEST         — default: lantern_crypt
  DUNGEONGRID_NUM_HEROES    — default: 2
  DUNGEONGRID_MAX_TURNS     — default: 30  (per-episode policy-call cap)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request

from dungeongrid_text_env import DIRECTIONS, HERO_ACTIONS, build_action

try:
    from synth_containers import GEPA_OPTIMIZER_CONTRACT_VERSION
except Exception:
    GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"
else:
    GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"

try:
    from openai import OpenAI
except Exception as _openai_err:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = _openai_err
else:
    _OPENAI_IMPORT_ERROR = None


TASK_ID = "dungeongrid.react_policy"
DATASET_ID = "dungeongrid_public_episodes"
REACT_TOOL_NAME = "dungeongrid_act"

POLICY_MODEL = os.environ.get("DUNGEONGRID_POLICY_MODEL", "gemini-3.1-flash-lite")
QUEST_ID = os.environ.get("DUNGEONGRID_QUEST", "lantern_crypt")
NUM_HEROES = int(os.environ.get("DUNGEONGRID_NUM_HEROES", "2"))
MAX_TURNS = int(os.environ.get("DUNGEONGRID_MAX_TURNS", "30"))
_MAX_STEPS = MAX_TURNS * 6  # hard safety cap incl. warden turns

DEFAULT_REACT_SYSTEM_PROMPT = (
    "You control a party of heroes exploring a dangerous DungeonGrid dungeon. "
    "Each turn you see a text observation for the active hero (its stats, "
    "position, the visible map, nearby entities and objects, and the objective). "
    f"Respond with a single {REACT_TOOL_NAME} tool call choosing ONE action for "
    "the active hero.\n"
    "Valid action_type values: " + ", ".join(HERO_ACTIONS) + ".\n"
    "Use `direction` (north/south/west/east) for move. Use `target` for the "
    "id of an entity/object, or \"x,y\" coordinates for inspect_tile. Use `text` "
    "for message content.\n"
    "Strategy: reveal rooms and map tiles, open doors, search for traps and "
    "treasure, defeat monsters that block progress, recover the objective, and "
    "extract the party. Coordinate heroes and avoid wasting turns."
)

# Episode seeds. Train seeds drive GEPA's reflective loop; heldout seeds gate
# the final acceptance score. DungeonGrid reset(seed) is deterministic.
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
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if base_url and "openrouter.ai" in base_url and os.environ.get("OPENROUTER_API_KEY"):
        api_key = os.environ["OPENROUTER_API_KEY"]
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY not set in container env; cannot serve live rollouts.",
        )
    _openai_client = OpenAI(api_key=api_key, base_url=base_url)
    return _openai_client


# --- Agent / env loop ---------------------------------------------------------


def _action_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": REACT_TOOL_NAME,
            "description": "Submit one DungeonGrid action for the active hero.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "enum": HERO_ACTIONS},
                    "direction": {"type": "string", "enum": DIRECTIONS},
                    "target": {
                        "type": "string",
                        "description": "Entity/object id, or \"x,y\" tile for inspect_tile.",
                    },
                    "text": {"type": "string", "description": "Message text."},
                },
                "required": ["action_type"],
            },
        },
    }


def _parse_action(raw_text: str, raw_tool_calls: list[dict] | None) -> dict[str, Any] | None:
    """Parse one action out of a native tool call (or a JSON blob fallback)."""
    if raw_tool_calls:
        try:
            args = json.loads(raw_tool_calls[0]["function"]["arguments"])
            if isinstance(args, dict) and args.get("action_type") in HERO_ACTIONS:
                return build_action(args)
        except Exception:
            pass
    if raw_text:
        try:
            obj = json.loads(raw_text)
            if isinstance(obj, dict) and obj.get("action_type") in HERO_ACTIONS:
                return build_action(obj)
        except Exception:
            pass
    return None


def _llm_step(
    client: Any, system_prompt: str, observation_text: str, step: int
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    user_content = (
        f"Turn {step + 1}. Active hero observation:\n\n{observation_text}\n\n"
        f"Call {REACT_TOOL_NAME} with one action for the active hero."
    )
    resp = client.chat.completions.create(
        model=POLICY_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        tools=[_action_tool()],
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
    action = _parse_action(text, tool_calls)
    usage = {
        "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
    }
    return action, usage


def _run_episode(seed: int, system_prompt: str) -> dict[str, Any]:
    """One real DungeonGrid episode driven by an LLM-controlled hero party."""
    from dungeongrid_text_env import DungeonGridTextEnv

    client = _get_openai_client()
    env = DungeonGridTextEnv(quest_id=QUEST_ID, num_heroes=NUM_HEROES)
    _, text = env.reset(seed)

    total_reward = 0.0
    all_achievements: list[str] = []
    policy_calls = 0
    total_steps = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    done = False

    while policy_calls < MAX_TURNS and total_steps < _MAX_STEPS:
        if env.active_agent.startswith("warden"):
            # Environment-controlled adversary turn.
            _, text, reward, done, info = env.step({"type": "warden_auto"})
        else:
            action, usage = _llm_step(client, system_prompt, text, policy_calls)
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)
            policy_calls += 1
            if not action:
                # Model failed to emit a valid action — end this hero's turn.
                action = {"type": "end_turn"}
            _, text, reward, done, info = env.step(action)
        total_reward += float(reward)
        all_achievements.extend(info.get("achievements") or [])
        total_steps += 1
        if done:
            break

    return {
        "seed": seed,
        "n_steps": total_steps,
        "policy_calls": policy_calls,
        "total_reward": total_reward,
        "done": done,
        "achievements": all_achievements,
        "usage": total_usage,
    }


# --- FastAPI app --------------------------------------------------------------

app = FastAPI(title="dungeongrid-gepa-container")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metadata")
@app.get("/info")
async def metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": "dungeongrid_gepa_live",
            "name": "DungeonGrid GEPA (live DungeonGrid env, LLM policy)",
            "description": "Public ReAct prompt-optimizer cookbook running real DungeonGrid episodes with an LLM-driven hero party.",
        },
        "capabilities": {
            "contract_version": "container_contract.v1",
            "rollout_modes": ["blocking"],
            "metadata": {
                "policy_ready": True,
                "trace_schema": "prompt_calls.llm_request.messages.v1",
            },
        },
        "metadata": {
            "optimizer_contracts": {
                "gepa": {
                    "version": GEPA_OPTIMIZER_CONTRACT_VERSION,
                    "program_route": "/program",
                    "taskset_route": "/taskset",
                    "taskset_tasks_route": "/taskset/tasks",
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
            "name": "DungeonGrid ReAct policy",
            "description": (
                "Optimize a ReAct system prompt for an LLM-controlled DungeonGrid "
                "hero party. Each rollout is a live environment episode with typed "
                "tool-call actions."
            ),
            "objective": "Maximize total episode reward and achievements unlocked before the turn cap.",
            "domain": "turn-based multi-hero dungeon exploration with combat, search, items, and an objective extraction",
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "visible_splits": ["train", "test"],
            "default_split": "train",
            "row_count": len(ROWS),
            "seed_semantics": (
                "Rows are generated from requested episode seeds. The same seed is "
                "deterministic for the DungeonGrid environment."
            ),
        },
        "prompt_program": {
            "mutable_modules": ["react_system_prompt"],
            "candidate_field": "react_system_prompt",
            "output_contract": (
                "Every policy call must return one dungeongrid_act tool call with a "
                "valid action_type for the active hero."
            ),
        },
        "environment": {
            "valid_actions": HERO_ACTIONS,
            "directions": DIRECTIONS,
            "tool_name": REACT_TOOL_NAME,
            "quest_id": QUEST_ID,
            "num_heroes": NUM_HEROES,
        },
        "evaluation": {
            "primary_metric": "outcome_reward",
            "success_status": "succeeded when total reward is positive",
            "rollout_trace_contains": ["episode_complete", "policy_calls", "achievements"],
        },
        "proposal_guidance": {
            "premises": [
                "The policy sees a text observation for one active hero each turn.",
                "Heroes act through typed actions (move, search, attack, interact, message), not free-form text.",
                "Reward comes from achievements: mapping, opening doors, defeating monsters, recovering the objective, and extracting the party.",
            ],
            "constraints": [
                "Do not output prose; the response must be a single dungeongrid_act tool call.",
                "Do not invent action_type values outside the valid action list.",
                "Keep the prompt concise enough to run every turn.",
            ],
            "high_leverage_heuristics": [
                "Reveal rooms and map tiles early; open doors to expand the explorable area.",
                "Search for traps before moving into unknown tiles; disarm what you find.",
                "Engage monsters that block the objective; avoid unnecessary fights.",
                "Recover the objective and extract the full party for the largest bonuses.",
                "Use message actions to coordinate heroes when they separate.",
            ],
            "anti_patterns": [
                "Generic dungeon-crawler advice with no valid action vocabulary.",
                "Long deliberation instructions that do not change the emitted action.",
                "Ignoring the active hero's position, AP, or the visible objective.",
            ],
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
        "program_id": "dungeongrid_react_prompt_gepa",
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
        "source": "dungeongrid_public_episode_seeds",
    }


@app.post("/dataset/rows")
async def dataset_rows(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    seeds = [int(seed) for seed in payload.get("seeds") or []]
    return {"rows": [_row_for_seed(split=split, seed=seed) for seed in seeds]}


@app.get("/taskset")
async def taskset() -> dict[str, Any]:
    return {
        "taskset_id": DATASET_ID,
        "name": "DungeonGrid",
        "splits": {
            "train": {"num_tasks": sum(1 for row in ROWS if row["split"] == "train")},
            "test": {"num_tasks": sum(1 for row in ROWS if row["split"] == "test")},
        },
    }


def _seed_from_task_id(task_id: Any) -> int:
    return int(str(task_id).rsplit(":", 1)[-1])


def _task_for_id(split: str, task_id: Any) -> dict[str, Any]:
    seed = _seed_from_task_id(task_id)
    row = _row_for_seed(split=split, seed=seed)
    return {"task_id": str(task_id), **row}


@app.post("/taskset/tasks")
async def taskset_tasks(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    task_ids = payload.get("task_ids") or []
    tasks = [_task_for_id(split, task_id) for task_id in task_ids]
    return {"tasks": tasks, "metadata": {"split": split, "count": len(tasks)}}


@app.post("/rollout")
@app.post("/rollouts")
def rollout(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    row = payload.get("dataset_row") if isinstance(payload.get("dataset_row"), dict) else None
    if not row and isinstance(payload.get("task"), dict):
        row = payload["task"]
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
                "policy_calls": episode["policy_calls"],
                "achievements": episode["achievements"],
                "policy_model": POLICY_MODEL,
                "quest_id": QUEST_ID,
                "num_heroes": NUM_HEROES,
                "max_turns": MAX_TURNS,
                "tool_name": REACT_TOOL_NAME,
            },
        },
        "summary": {
            "outcome_reward": reward,
            "example_id": row.get("example_id"),
            "n_steps": episode["n_steps"],
            "achievements_unlocked": episode["achievements"],
        },
        "usage": {**episode["usage"], "cost_usd": 0.0},
        "trace": {
            "event_history": [
                {
                    "type": "episode_complete",
                    "seed": seed,
                    "total_reward": reward,
                    "n_steps": episode["n_steps"],
                    "policy_calls": episode["policy_calls"],
                    "achievements": episode["achievements"],
                }
            ],
            "metadata": {
                "example_id": row.get("example_id"),
                "call_site_id": "dungeongrid.react_policy",
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
    if match:
        return dict(match)
    return {
        "seed": int(seed),
        "split": normalized_split,
        "example_id": f"ep_{normalized_split}_{int(seed)}",
    }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
