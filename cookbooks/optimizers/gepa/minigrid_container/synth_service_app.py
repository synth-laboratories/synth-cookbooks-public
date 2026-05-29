"""
MiniGrid GEPA cookbook container (live gymnasium env, OpenAI policy).

Speaks the public synth-optimizers GEPA contract:
  GET  /metadata
  GET  /task_info
  GET  /program
  GET  /dataset
  POST /dataset/rows
  POST /rollout

Each rollout instantiates a real MiniGrid env, drives it for up to N steps
with an OpenAI-driven agent using the candidate's `system_prompt`, and
returns the actual env reward.

Required env:
  OPENAI_API_KEY              — required when rollout.policy.credential_mode=byok.
  MINIGRID_MAX_STEPS          — default: 48 (per-episode hard cap)
  MINIGRID_ENV_ID             — default: MiniGrid-DoorKey-5x5-v0
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

try:
    from synth_containers import GEPA_OPTIMIZER_CONTRACT_VERSION
except Exception:
    GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"

try:
    from openai import OpenAI
except Exception as _openai_err:
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = _openai_err
else:
    _OPENAI_IMPORT_ERROR = None


TASK_ID = "minigrid.gridworld_policy"
DATASET_ID = "minigrid_public_episodes"

MAX_STEPS = int(os.environ.get("MINIGRID_MAX_STEPS", "48"))
ENV_ID = os.environ.get("MINIGRID_ENV_ID", "MiniGrid-DoorKey-5x5-v0")

# Standard MiniGrid 7 actions, ordered by gymnasium Action enum.
ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]

DEFAULT_SYSTEM_PROMPT = (
    "You are a MiniGrid agent. Each turn you see a compact text description of "
    "the gridworld (mission, your position/direction, what you're carrying, "
    "visible objects, admissible actions). Respond ONLY with strict JSON of the "
    "form: {\"action\": \"<one admissible action name>\"}. "
    "Choose actions that make real progress toward the mission. Pick up keys "
    "before trying to open locked doors. Use toggle to open doors. Use forward "
    "to move in your current direction."
)

# Episode seeds. Train seeds drive GEPA's reflective loop; heldout gates final score.
ROWS = [
    {"seed": 1, "split": "train", "example_id": "ep_train_1"},
    {"seed": 2, "split": "train", "example_id": "ep_train_2"},
    {"seed": 3, "split": "train", "example_id": "ep_train_3"},
    {"seed": 4, "split": "train", "example_id": "ep_train_4"},
    {"seed": 100, "split": "test", "example_id": "ep_heldout_100"},
    {"seed": 101, "split": "test", "example_id": "ep_heldout_101"},
]


_openai_clients: dict[tuple[str, str, str], Any] = {}
_RAW_CREDENTIAL_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "openai_api_key",
    "openrouter_api_key",
    "secret_key",
}


def _find_raw_credential_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for raw_key, raw_value in value.items():
            normalized = str(raw_key).strip().lower().replace("-", "_")
            if normalized in _RAW_CREDENTIAL_KEYS or normalized.endswith("_api_key"):
                return str(raw_key)
            nested = _find_raw_credential_key(raw_value)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _find_raw_credential_key(item)
            if nested is not None:
                return nested
    return None


def _normalize_policy_enum(value: Any, default: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return text or default


def _strip_openai_endpoint_suffix(url: str) -> str:
    normalized = url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _require_policy(payload: dict[str, Any]) -> dict[str, Any]:
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        raise HTTPException(
            status_code=422,
            detail="rollout.policy is required for GEPA optimizer contract v2.",
        )
    raw_key = _find_raw_credential_key(policy.get("config", {}))
    if raw_key is not None:
        raise HTTPException(
            status_code=422,
            detail=f"rollout.policy.config must not carry raw credential field {raw_key!r}.",
        )
    provider = str(policy.get("provider") or "").strip()
    model = str(policy.get("model") or "").strip()
    if not provider or not model:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.provider and rollout.policy.model are required.",
        )
    api_family = _normalize_policy_enum(policy.get("api_family"), "chat_completions")
    if api_family != "chat_completions":
        raise HTTPException(
            status_code=422,
            detail=f"{TASK_ID} supports rollout.policy.api_family='chat_completions'; got {api_family!r}.",
        )
    credential_mode = _normalize_policy_enum(policy.get("credential_mode"), "byok")
    if credential_mode not in {"byok", "proxy"}:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported rollout.policy.credential_mode: {credential_mode!r}",
        )
    raw_base_url = (
        str(policy.get("inference_url") or "").strip()
        if credential_mode == "proxy"
        else str(policy.get("base_url") or "").strip()
    )
    if credential_mode == "proxy" and not raw_base_url:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.inference_url is required when credential_mode=proxy.",
        )
    if provider.lower() == "openrouter" and credential_mode == "byok" and not raw_base_url:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.base_url is required for provider=openrouter.",
        )
    max_tokens = policy.get("max_tokens")
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail="rollout.policy.max_tokens must be an integer when set.",
            ) from exc
        if max_tokens <= 0:
            raise HTTPException(
                status_code=422,
                detail="rollout.policy.max_tokens must be positive when set.",
            )
    return {
        "provider": provider,
        "model": model,
        "base_url": _strip_openai_endpoint_suffix(raw_base_url) if raw_base_url else None,
        "credential_mode": credential_mode,
        "max_tokens": max_tokens,
    }


def _policy_api_key(policy: dict[str, Any]) -> str:
    if policy["credential_mode"] == "proxy":
        return "proxy"
    env_name = "OPENROUTER_API_KEY" if policy["provider"].lower() == "openrouter" else "OPENAI_API_KEY"
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    raise HTTPException(
        status_code=503,
        detail=f"{env_name} is not set; rollout.policy credential_mode=byok requires a container env credential.",
    )


def _get_openai_client(policy: dict[str, Any]) -> Any:
    if OpenAI is None:
        raise HTTPException(
            status_code=503,
            detail=f"openai package not installed; container deps in pyproject.toml. {_OPENAI_IMPORT_ERROR!r}",
        )
    base_url = policy.get("base_url")
    key = (policy["provider"].lower(), policy["credential_mode"], str(base_url or ""))
    client = _openai_clients.get(key)
    if client is None:
        client_kwargs = {"api_key": _policy_api_key(policy)}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        _openai_clients[key] = client
    return client


# --- Env helpers --------------------------------------------------------------


def _make_env(seed: int):
    import gymnasium as gym
    import minigrid  # noqa: F401 — registers envs as side effect
    from minigrid.wrappers import FullyObsWrapper

    env = gym.make(ENV_ID, render_mode=None)
    env = FullyObsWrapper(env)
    obs, _ = env.reset(seed=int(seed))
    if hasattr(env.unwrapped, "max_steps"):
        env.unwrapped.max_steps = MAX_STEPS
    return env, obs


_DIRECTION_NAMES = {0: "right", 1: "down", 2: "left", 3: "up"}


def _render_observation_text(env, obs) -> str:
    mission = str(obs.get("mission") or getattr(env.unwrapped, "mission", "") or "")
    agent_pos = list(getattr(env.unwrapped, "agent_pos", []) or [])
    agent_dir = int(getattr(env.unwrapped, "agent_dir", 0) or 0)
    direction_name = _DIRECTION_NAMES.get(agent_dir, "?")
    carrying = getattr(env.unwrapped, "carrying", None)
    carrying_str = (
        f"{getattr(carrying, 'color', '')} {getattr(carrying, 'type', '')}".strip()
        if carrying is not None
        else "nothing"
    )

    grid = getattr(env.unwrapped, "grid", None)
    width = int(getattr(grid, "width", 0) or 0)
    height = int(getattr(grid, "height", 0) or 0)
    visible: list[str] = []
    if grid is not None:
        for x in range(width):
            for y in range(height):
                obj = grid.get(x, y)
                if obj is None:
                    continue
                obj_type = str(getattr(obj, "type", "") or "")
                obj_color = str(getattr(obj, "color", "") or "")
                is_locked = bool(getattr(obj, "is_locked", False))
                is_open = bool(getattr(obj, "is_open", False))
                label_bits = [obj_color, obj_type]
                if is_locked:
                    label_bits.append("(locked)")
                if is_open:
                    label_bits.append("(open)")
                visible.append(f"  ({x},{y}): {' '.join(b for b in label_bits if b)}")

    visible_block = "\n".join(visible[:25]) if visible else "  (none)"
    return (
        f"mission: {mission}\n"
        f"agent_position: {agent_pos}  facing: {direction_name}\n"
        f"carrying: {carrying_str}\n"
        f"admissible_actions: {ACTION_NAMES}\n"
        f"visible_objects:\n{visible_block}"
    )


def _parse_action(raw_text: str) -> str | None:
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            cand = parsed.get("action")
            if isinstance(cand, str) and cand.lower() in ACTION_NAMES:
                return cand.lower()
        if isinstance(parsed, str) and parsed.lower() in ACTION_NAMES:
            return parsed.lower()
    except json.JSONDecodeError:
        pass
    lowered = text.lower()
    if lowered in ACTION_NAMES:
        return lowered
    for action in ACTION_NAMES:
        if action in lowered:
            return action
    return None


def _llm_step(
    client: Any,
    policy: dict[str, Any],
    system_prompt: str,
    observation_text: str,
    step: int,
) -> tuple[str | None, dict[str, int]]:
    user_content = (
        f"Step {step + 1}. Current state:\n\n{observation_text}\n\n"
        'Reply with strict JSON: {"action": "<one admissible action name>"}'
    )
    request_kwargs = {
        "model": policy["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
    }
    if policy["max_tokens"] is not None:
        request_kwargs["max_tokens"] = policy["max_tokens"]
    resp = client.chat.completions.create(**request_kwargs)
    text = (resp.choices[0].message.content or "").strip()
    action = _parse_action(text)
    usage = {
        "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
    }
    return action, usage


def _run_episode(seed: int, system_prompt: str, policy: dict[str, Any]) -> dict[str, Any]:
    client = _get_openai_client(policy)
    env, obs = _make_env(seed)

    total_reward = 0.0
    all_actions: list[str] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    done = False
    step = 0

    try:
        while step < MAX_STEPS:
            obs_text = _render_observation_text(env, obs)
            action_name, usage = _llm_step(client, policy, system_prompt, obs_text, step)
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)
            if action_name is None:
                # Invalid response: count the turn, try again next step.
                step += 1
                continue
            action_idx = ACTION_NAMES.index(action_name)
            obs, reward, terminated, truncated, _info = env.step(action_idx)
            total_reward += float(reward)
            all_actions.append(action_name)
            step += 1
            done = bool(terminated or truncated)
            if done:
                break
    finally:
        env.close()

    return {
        "seed": seed,
        "n_steps": step,
        "total_reward": total_reward,
        "done": done,
        "solved": total_reward > 0.0,  # MiniGrid success → positive reward only on goal
        "actions": all_actions,
        "usage": total_usage,
    }


# --- FastAPI app --------------------------------------------------------------

app = FastAPI(title="minigrid-gepa-container")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metadata")
@app.get("/info")
async def metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": "minigrid_gepa_live",
            "name": "MiniGrid GEPA (live gymnasium env, OpenAI policy)",
            "description": "Public prompt-optimizer cookbook running real MiniGrid episodes with an OpenAI-driven agent.",
        },
        "capabilities": {
            "contract_version": "container_contract.v1",
            "rollout_modes": ["blocking"],
            "metadata": {"trace_schema": "prompt_calls.llm_request.messages.v1"},
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
            "name": f"MiniGrid policy ({ENV_ID})",
            "description": (
                "Optimize a system prompt for an OpenAI-controlled MiniGrid agent. "
                "Each rollout is a live gymnasium episode, not a fixture replay."
            ),
            "objective": "Maximize solved episodes and total environment reward before the step cap.",
            "domain": "partially observable gridworld control with text observations and discrete actions",
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "visible_splits": ["train", "test"],
            "default_split": "train",
            "row_count": len(ROWS),
            "seed_semantics": (
                "Rows are generated from requested episode seeds. The same seed is deterministic "
                "for a given MiniGrid env id."
            ),
        },
        "prompt_program": {
            "mutable_modules": ["system_prompt"],
            "candidate_field": "system_prompt",
            "output_contract": "Every policy call must return strict JSON: {\"action\": \"<admissible action>\"}.",
        },
        "evaluation": {
            "primary_metric": "outcome_reward",
            "success_status": "succeeded when the episode reaches the mission goal",
            "rollout_trace_contains": ["episode_complete", "actions_taken", "n_steps", "solved"],
        },
        "proposal_guidance": {
            "premises": [
                "The agent receives mission, position, direction, carried object, visible objects, and admissible actions each turn.",
                "MiniGrid tasks often require short action plans: orient, move, pick up keys, toggle doors, then reach the goal.",
                "Invalid JSON or invalid action names waste the step budget and usually fail the episode.",
            ],
            "constraints": [
                "Do not ask for chain-of-thought or verbose plans in the final response.",
                "Do not introduce actions outside the seven MiniGrid action names.",
                "Keep the system prompt operational and compact enough to run on every step.",
            ],
            "high_leverage_heuristics": [
                "Prioritize mission progress over exploration once the goal object or door is visible.",
                "Use explicit door/key rules: pick up matching keys, face doors before toggle, avoid repeated no-op toggles.",
                "Add recovery behavior for blocked forward moves and loops.",
                "Make JSON compliance non-negotiable.",
            ],
            "anti_patterns": [
                "Generic assistant persona text.",
                "Long reflective reasoning instructions that increase latency without changing actions.",
                "Rules that ignore admissible actions or the current facing direction.",
            ],
        },
        "metadata": {
            "policy_model_source": "rollout.policy.model",
            "env_id": ENV_ID,
            "max_steps": MAX_STEPS,
            "trace_schema": "prompt_calls.llm_request.messages.v1",
        },
    }


@app.get("/program")
async def program() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "minigrid_system_prompt_gepa",
        "modules": [
            {
                "module_id": "system_prompt",
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
                "mutable": True,
                "candidate_field": "system_prompt",
                "template_variables": [],
                "metadata": {"env_id": ENV_ID},
            }
        ],
        "target_modules": [
            {
                "module_id": "system_prompt",
                "candidate_field": "system_prompt",
                "objective": "task_success_rate",
            }
        ],
        "seed_candidate": {"system_prompt": DEFAULT_SYSTEM_PROMPT},
        "rollout_overlay_schema": {"candidate_fields": ["system_prompt"]},
        "metadata": {
            "task_id": TASK_ID,
            "dataset_id": DATASET_ID,
            "env_id": ENV_ID,
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
        "source": "minigrid_public_episode_seeds",
    }


@app.post("/dataset/rows")
async def dataset_rows(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    seeds = [int(seed) for seed in payload.get("seeds") or []]
    return {"rows": [_row_for_seed(split=split, seed=seed) for seed in seeds]}


@app.post("/rollout")
@app.post("/rollouts")
def rollout(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    policy = _require_policy(payload)
    row = payload.get("dataset_row") if isinstance(payload.get("dataset_row"), dict) else None
    if not row:
        row = _row_for_seed(
            split=str(payload.get("split") or "train"),
            seed=int(payload.get("seed") or 1),
        )
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    system_prompt = str(candidate.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
    seed = int(row.get("seed") or 0)

    episode = _run_episode(seed=seed, system_prompt=system_prompt, policy=policy)
    reward = float(episode["total_reward"])

    rollout_id = str(payload.get("rollout_id") or f"rollout_{uuid.uuid4().hex[:12]}")
    now = _now()
    return {
        "rollout_id": rollout_id,
        "status": "completed",
        "success_status": "succeeded" if episode["solved"] else "failed",
        "task_id": TASK_ID,
        "seed": seed,
        "reward_info": {
            "outcome_reward": reward,
            "event_rewards": [reward],
            "details": {
                "example_id": row.get("example_id"),
                "env_id": ENV_ID,
                "n_steps": episode["n_steps"],
                "done": episode["done"],
                "solved": episode["solved"],
                "policy_model": policy["model"],
            },
        },
        "summary": {
            "outcome_reward": reward,
            "example_id": row.get("example_id"),
            "n_steps": episode["n_steps"],
            "actions_taken": episode["actions"],
        },
        "usage": {**episode["usage"], "cost_usd": 0.0},
        "trace": {
            "event_history": [
                {
                    "type": "episode_complete",
                    "seed": seed,
                    "env_id": ENV_ID,
                    "total_reward": reward,
                    "n_steps": episode["n_steps"],
                    "solved": episode["solved"],
                }
            ],
            "metadata": {
                "example_id": row.get("example_id"),
                "call_site_id": "minigrid.gridworld_policy",
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
    row = {
        "seed": int(seed),
        "split": normalized_split,
        "example_id": f"ep_{normalized_split}_{int(seed)}",
    }
    return dict(row)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
