from __future__ import annotations

import asyncio
import copy
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import gymnasium as gym
    import numpy as np
    from minigrid.wrappers import FullyObsWrapper

    MINIGRID_AVAILABLE = True
    MINIGRID_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - runtime dependency path
    gym = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    MINIGRID_AVAILABLE = False
    MINIGRID_IMPORT_ERROR = str(exc)


CONTAINER_ROOT = Path(__file__).resolve().parent
TASK_REGISTRY_PATH = CONTAINER_ROOT / "task_registry.json"
DEFAULT_INFERENCE_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_MAX_STEPS = 48
MILESTONE_BONUS_WEIGHTS = {
    "key_acquired": 0.15,
    "door_unlocked": 0.2,
    "new_room_entered": 0.1,
    "goal_visible": 0.05,
    "goal_reached": 1.0,
}
DEFAULT_SYSTEM_PROMPT = (
    "You are a MiniGrid planning agent. Return strict JSON only as "
    '{"action":"<one admissible action name>"} using exactly one action from admissible_actions. '
    "Prefer real task progress over commentary. Do not explain anything outside the JSON object."
)
SUPPORTED_FAMILIES = {"DoorKey", "UnlockPickup", "KeyCorridor", "MultiRoom"}


class RolloutEnvSpec(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None


class RolloutPolicySpec(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)


class RolloutRequest(BaseModel):
    trace_correlation_id: str
    trial_id: str | None = None
    rollout_id: str | None = None
    task_id: str | None = None
    task_instance_id: str | None = None
    task_metadata: dict[str, Any] = Field(default_factory=dict)
    env: RolloutEnvSpec = Field(default_factory=RolloutEnvSpec)
    policy: RolloutPolicySpec = Field(default_factory=RolloutPolicySpec)
    params: dict[str, Any] = Field(default_factory=dict)
    terminator: dict[str, Any] = Field(default_factory=dict)
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    submission_mode: str = "sync"


class CheckpointRequest(BaseModel):
    checkpoint_id: str | None = None
    label: str | None = None
    source: str | None = None
    actor_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)


class ResumeOverrides(BaseModel):
    env: dict[str, Any] = Field(default_factory=dict)
    env_config: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)
    policy_config: dict[str, Any] = Field(default_factory=dict)
    segment_steps: int | None = None
    continue_steps: int | None = None
    task_id: str | None = None
    task_instance_id: str | None = None


class ResumeRequest(BaseModel):
    rollout_id: str | None = None
    checkpoint_id: str | None = None
    target_rollout_id: str | None = None
    mode: str = "new_rollout"
    submission_mode: str = "sync"
    overrides: ResumeOverrides = Field(default_factory=ResumeOverrides)


class PauseBody(BaseModel):
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TerminateBody(BaseModel):
    reason: str | None = None


@dataclass(slots=True)
class RegistryEntry:
    task_instance_id: str
    registry_task_id: str
    seed: int
    split_group: str
    family: str
    env_id: str
    label: str


app = FastAPI(title="evals-minigrid-goex")
_STORE_LOCK = asyncio.Lock()
_ROLLOUTS: dict[str, dict[str, Any]] = {}
_CHECKPOINTS: dict[str, dict[str, Any]] = {}
_ROLLOUT_TASKS: dict[str, asyncio.Task[None]] = {}
_REGISTRY_CACHE: dict[str, Any] | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _ensure_runtime_available() -> None:
    if not MINIGRID_AVAILABLE:
        raise HTTPException(status_code=503, detail=f"minigrid_runtime_unavailable:{MINIGRID_IMPORT_ERROR}")


def _load_registry() -> dict[str, Any]:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = json.loads(TASK_REGISTRY_PATH.read_text(encoding="utf-8"))
    return dict(_REGISTRY_CACHE)


def _registry_entries() -> list[RegistryEntry]:
    payload = _load_registry()
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("minigrid task registry missing entries")
    out: list[RegistryEntry] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        out.append(
            RegistryEntry(
                task_instance_id=str(item.get("task_instance_id") or "").strip(),
                registry_task_id=str(item.get("registry_task_id") or "").strip(),
                seed=int(item.get("seed") or 0),
                split_group=str(item.get("split_group") or "").strip(),
                family=str(item.get("family") or "").strip(),
                env_id=str(item.get("env_id") or "").strip(),
                label=str(item.get("label") or "").strip(),
            )
        )
    return out


def _entry_by_identity(
    *,
    seed: int | None = None,
    task_instance_id: str | None = None,
    task_id: str | None = None,
    split_group: str | None = None,
    family: str | None = None,
) -> RegistryEntry:
    entries = _registry_entries()
    if task_instance_id:
        for entry in entries:
            if entry.task_instance_id == task_instance_id:
                return entry
        raise HTTPException(status_code=404, detail=f"unknown_task_instance_id:{task_instance_id}")
    if task_id:
        for entry in entries:
            if entry.registry_task_id == task_id:
                return entry
        raise HTTPException(status_code=404, detail=f"unknown_task_id:{task_id}")
    filtered = entries
    if split_group:
        filtered = [entry for entry in filtered if entry.split_group == split_group]
    if family:
        filtered = [entry for entry in filtered if entry.family == family]
    if not filtered:
        raise HTTPException(status_code=404, detail="no_minigrid_registry_entries_match_selector")
    ordered = sorted(filtered, key=lambda entry: entry.task_instance_id)
    if seed is None:
        return ordered[0]
    for entry in ordered:
        if entry.seed == int(seed):
            return entry
    return ordered[int(seed) % len(ordered)]


def _policy_config(payload: dict[str, Any]) -> dict[str, Any]:
    policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
    return dict(policy.get("config") or {}) if isinstance(policy, dict) else {}


def _compose_system_prompt(policy_config: dict[str, Any]) -> str:
    base = str(policy_config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT).strip()
    suffix = str(policy_config.get("system_prompt_suffix") or "").strip()
    if suffix:
        return f"{base}\n\nAdditional policy guidance:\n{suffix}"
    return base


def _json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _make_env(entry: RegistryEntry, *, max_steps: int, seed: int) -> Any:
    _ensure_runtime_available()
    env = gym.make(entry.env_id, render_mode=None)  # type: ignore[union-attr]
    env = FullyObsWrapper(env)
    env.reset(seed=int(seed))
    if hasattr(env, "max_steps"):
        env.max_steps = max_steps
    if hasattr(env.unwrapped, "max_steps"):
        env.unwrapped.max_steps = max_steps
    return env


def _action_names(env: Any) -> list[str]:
    actions = getattr(env.unwrapped, "actions", None)
    if actions is None:
        return ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
    names: list[str] = []
    for item in list(actions):
        name = getattr(item, "name", None)
        names.append(str(name or item).lower())
    return names


def _carrying_summary(env: Any) -> dict[str, Any] | None:
    carrying = getattr(env.unwrapped, "carrying", None)
    if carrying is None:
        return None
    return {
        "type": str(getattr(carrying, "type", "") or ""),
        "color": str(getattr(carrying, "color", "") or ""),
    }


def _grid_encoding(obs: dict[str, Any]) -> list[list[list[int]]]:
    image = obs.get("image")
    if image is None:
        return []
    if hasattr(image, "tolist"):
        return image.tolist()
    return list(image)


def _visible_objects(env: Any) -> list[dict[str, Any]]:
    grid = getattr(env.unwrapped, "grid", None)
    width = int(getattr(grid, "width", 0) or 0)
    height = int(getattr(grid, "height", 0) or 0)
    objects: list[dict[str, Any]] = []
    for x in range(width):
        for y in range(height):
            obj = grid.get(x, y)
            if obj is None:
                continue
            objects.append(
                {
                    "x": x,
                    "y": y,
                    "type": str(getattr(obj, "type", "") or ""),
                    "color": str(getattr(obj, "color", "") or ""),
                    "is_locked": bool(getattr(obj, "is_locked", False)),
                    "is_open": bool(getattr(obj, "is_open", False)),
                }
            )
    return objects


def _env_summary(env: Any, obs: dict[str, Any]) -> dict[str, Any]:
    mission = str(obs.get("mission") or getattr(env.unwrapped, "mission", "") or "")
    objects = _visible_objects(env)
    return _json_safe({
        "grid": _grid_encoding(obs),
        "agent_pos": list(getattr(env.unwrapped, "agent_pos", []) or []),
        "agent_dir": int(getattr(env.unwrapped, "agent_dir", 0) or 0),
        "carrying": _carrying_summary(env),
        "mission": mission,
        "admissible_actions": _action_names(env),
        "visible_objects": objects,
    })


def _parse_action(raw_text: str, admissible_actions: list[str]) -> tuple[str, bool]:
    text = raw_text.strip()
    invalid_parse = False
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            candidate = parsed.get("action")
            if isinstance(candidate, str) and candidate.lower() in admissible_actions:
                return candidate.lower(), invalid_parse
            if isinstance(candidate, int):
                return admissible_actions[int(candidate) % len(admissible_actions)], invalid_parse
        if isinstance(parsed, str) and parsed.lower() in admissible_actions:
            return parsed.lower(), invalid_parse
    except json.JSONDecodeError:
        invalid_parse = True
    normalized = text.lower()
    if normalized in admissible_actions:
        return normalized, invalid_parse
    for action in admissible_actions:
        if action in normalized:
            return action, True
    return admissible_actions[0], True


async def _send_inference_request(
    *,
    inference_url: str,
    api_key: str | None,
    model: str,
    temperature: float,
    max_tokens: int,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if "api.openai.com" in inference_url:
        body["max_completion_tokens"] = max_tokens
    else:
        body["max_tokens"] = max_tokens
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(inference_url, headers=headers, json=body)
        response.raise_for_status()
        payload = response.json()
    choice = payload["choices"][0]["message"]
    content = choice.get("content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return {
        "assistant_text": str(content),
        "usage": payload.get("usage", {}),
        "request_id": payload.get("id"),
    }


def _format_messages(
    *,
    system_prompt: str,
    entry: RegistryEntry,
    env_summary: dict[str, Any],
    action_history: list[str],
) -> list[dict[str, str]]:
    user_payload = {
        "family": entry.family,
        "mission": env_summary.get("mission"),
        "agent_pos": env_summary.get("agent_pos"),
        "agent_dir": env_summary.get("agent_dir"),
        "carrying": env_summary.get("carrying"),
        "admissible_actions": env_summary.get("admissible_actions"),
        "visible_objects": env_summary.get("visible_objects"),
        "grid": env_summary.get("grid"),
        "action_history": action_history[-8:],
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(_json_safe(user_payload), sort_keys=True)},
    ]


def _scan_milestones(current_obs: dict[str, Any], previous: dict[str, Any] | None) -> list[str]:
    labels: list[str] = []
    carrying = current_obs.get("carrying") or {}
    if isinstance(carrying, dict) and str(carrying.get("type") or "") == "key":
        labels.append("key_acquired")
    objects = current_obs.get("visible_objects") or []
    if any(obj.get("type") == "door" and not obj.get("is_locked", False) for obj in objects):
        labels.append("door_unlocked")
    if any(obj.get("type") == "goal" for obj in objects):
        labels.append("goal_visible")
    if previous is not None:
        prev_positions = {tuple(item) for item in previous.get("visited_positions", []) if isinstance(item, list)}
        curr_pos = tuple(current_obs.get("agent_pos") or [])
        if curr_pos and curr_pos not in prev_positions:
            labels.append("new_room_entered")
    return list(dict.fromkeys(labels))


def _milestone_bonus(new_labels: list[str]) -> float:
    return float(sum(float(MILESTONE_BONUS_WEIGHTS.get(label, 0.0)) for label in new_labels))


def _checkpoint_descriptor(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_id": str(record["checkpoint_id"]),
        "rollout_id": str(record["rollout_id"]),
        "checkpoint_uri": str(record["checkpoint_uri"]),
        "created_at": str(record["created_at"]),
        "checkpoint_version": "minigrid.true_env_snapshot.v1",
        "restore_eligible": True,
        "restore_semantics": "true_environment_snapshot",
        "true_environment_snapshot": True,
        "supports_branching": True,
        "label": record.get("label"),
        "labels": list(record.get("labels") or []),
        "source": "minigrid_goex_container",
        "actor_ids": list(record.get("actor_ids") or []),
        "metadata": dict(record.get("metadata") or {}),
        "annotations": dict(record.get("annotations") or {}),
    }


def _state_payload(session: dict[str, Any]) -> dict[str, Any]:
    checkpoints = [
        _checkpoint_descriptor(item)
        for item in _CHECKPOINTS.values()
        if str(item.get("rollout_id") or "") == str(session["rollout_id"])
    ]
    env_summary = dict(session.get("env_summary") or {})
    env_summary["action_history"] = list(session.get("action_history") or [])
    return {
        "rollout_id": str(session["rollout_id"]),
        "trace_correlation_id": str(session["trace_correlation_id"]),
        "status": str(session["status"]),
        "success_status": str(session["success_status"]),
        "status_detail": str(session.get("status_detail") or ""),
        "created_at": str(session["created_at"]),
        "updated_at": str(session.get("updated_at") or session["created_at"]),
        "reward": float(session.get("total_reward") or 0.0),
        "agent_turn_count": int(len(session.get("turns") or [])),
        "env_action_count": int(len(session.get("event_history") or [])),
        "checkpoint_count": len(checkpoints),
        "milestone_labels": sorted(session.get("milestones") or []),
        "observation": env_summary,
        "checkpoints": checkpoints,
        "resume_semantics": {
            "restore_semantics": "true_environment_snapshot",
            "true_environment_snapshot": True,
            "supports_branching": True,
        },
    }


def _response_payload(session: dict[str, Any]) -> dict[str, Any]:
    turns = copy.deepcopy(session.get("turns") or [])
    running_return = 0.0
    for turn in reversed(turns):
        running_return += float(turn.get("decision_reward") or 0.0)
        turn["return_to_go"] = running_return
    event_history = copy.deepcopy(session.get("event_history") or [])
    checkpoint_descriptors = [
        _checkpoint_descriptor(item)
        for item in _CHECKPOINTS.values()
        if str(item.get("rollout_id") or "") == str(session["rollout_id"])
    ]
    return {
        "trace_correlation_id": str(session["trace_correlation_id"]),
        "rollout_id": str(session["rollout_id"]),
        "trial_id": str(session["trial_id"]),
        "status": str(session["status"]),
        "success_status": str(session["success_status"]),
        "status_detail": str(session.get("status_detail") or ""),
        "reward": float(session.get("total_reward") or 0.0),
        "reward_info": {
            "outcome_reward": float(session.get("total_reward") or 0.0),
            "details": {
                "reward_type": "minigrid_env_progress_reward",
                "family": str(session["entry"].family),
                "env_id": str(session["entry"].env_id),
                "checkpoint_count": len(checkpoint_descriptors),
                "env_reward_total": float(session.get("env_reward_total") or 0.0),
                "milestone_bonus_total": float(session.get("milestone_bonus_total") or 0.0),
                "milestone_labels": sorted(session.get("milestones") or []),
            },
        },
        "summary": {
            "family": str(session["entry"].family),
            "env_id": str(session["entry"].env_id),
            "total_reward": float(session.get("total_reward") or 0.0),
            "checkpoint_count": len(checkpoint_descriptors),
            "agent_turn_count": int(len(turns)),
            "env_action_count": int(len(event_history)),
            "milestone_labels": sorted(session.get("milestones") or []),
        },
        "artifact": [
            {"artifact_type": "turns", "turns": turns},
            {"artifact_type": "event_history", "events": event_history},
        ],
        "trace": {
            "task": {
                "task_instance_id": str(session["entry"].task_instance_id),
                "registry_task_id": str(session["entry"].registry_task_id),
                "family": str(session["entry"].family),
                "seed": int(session["entry"].seed),
                "split_group": str(session["entry"].split_group),
                "env_id": str(session["entry"].env_id),
            },
            "event_history": event_history,
            "inference": {"turns": copy.deepcopy(session.get("inference_turns") or [])},
            "milestone_labels": sorted(session.get("milestones") or []),
        },
        "checkpoints": checkpoint_descriptors,
        "restore_semantics": "true_environment_snapshot",
        "true_environment_snapshot": True,
        "supports_branching": True,
    }


def _make_snapshot(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "env": copy.deepcopy(session.get("env")),
        "obs": copy.deepcopy(session.get("obs") or {}),
        "env_summary": copy.deepcopy(session.get("env_summary") or {}),
        "total_reward": float(session.get("total_reward") or 0.0),
        "env_reward_total": float(session.get("env_reward_total") or 0.0),
        "milestone_bonus_total": float(session.get("milestone_bonus_total") or 0.0),
        "turns": copy.deepcopy(session.get("turns") or []),
        "event_history": copy.deepcopy(session.get("event_history") or []),
        "inference_turns": copy.deepcopy(session.get("inference_turns") or []),
        "action_history": list(session.get("action_history") or []),
        "milestones": sorted(session.get("milestones") or []),
        "visited_positions": copy.deepcopy(session.get("visited_positions") or []),
        "policy_config": copy.deepcopy(session.get("policy_config") or {}),
    }


def _create_checkpoint_locked(
    session: dict[str, Any],
    *,
    checkpoint_id: str | None,
    label: str | None,
    source: str | None,
    actor_ids: list[str] | None,
    metadata: dict[str, Any] | None,
    annotations: dict[str, Any] | None,
) -> dict[str, Any]:
    cid = checkpoint_id or f"minigrid_ckpt_{uuid.uuid4().hex[:10]}"
    record = {
        "checkpoint_id": cid,
        "rollout_id": str(session["rollout_id"]),
        "checkpoint_uri": f"minigrid://checkpoints/{cid}",
        "created_at": _utc_now_iso(),
        "label": label,
        "labels": [label] if label else [],
        "source": source or "go_explore",
        "actor_ids": list(actor_ids or []),
        "metadata": {
            "checkpoint_restore_semantics": "true_environment_snapshot",
            "task_instance_id": str(session["entry"].task_instance_id),
            "supports_branching": True,
            "true_environment_snapshot": True,
            **dict(metadata or {}),
        },
        "annotations": dict(annotations or {}),
        "_snapshot": _make_snapshot(session),
    }
    _CHECKPOINTS[cid] = record
    session["updated_at"] = _utc_now_iso()
    return record


async def _run_rollout(rollout_id: str) -> None:
    while True:
        async with _STORE_LOCK:
            session = _ROLLOUTS.get(rollout_id)
            if session is None:
                return
            if session["status"] not in {"queued", "running"}:
                return
            if session["status"] == "queued":
                session["status"] = "running"
                session["success_status"] = "running"
                session["status_detail"] = "rollout_started"
                session["updated_at"] = _utc_now_iso()
            if session.get("terminate_requested"):
                session["status"] = "cancelled"
                session["success_status"] = "cancelled"
                session["status_detail"] = "terminated_by_request"
                session["response_payload"] = _response_payload(session)
                return
            if session.get("pause_requested"):
                _create_checkpoint_locked(
                    session,
                    checkpoint_id=None,
                    label="paused_boundary",
                    source="pause_endpoint",
                    actor_ids=[],
                    metadata={"pause_reason": str(session.get("pause_reason") or "")},
                    annotations={},
                )
                session["status"] = "paused"
                session["success_status"] = "paused"
                session["status_detail"] = "paused_at_control_boundary"
                session["response_payload"] = _response_payload(session)
                return
            env_summary = dict(session.get("env_summary") or {})
            admissible_actions = [str(item) for item in env_summary.get("admissible_actions") or []]
            if not admissible_actions:
                session["status"] = "completed"
                session["success_status"] = "success"
                session["status_detail"] = "no_more_actions"
                session["response_payload"] = _response_payload(session)
                return
            policy_cfg = dict(session["policy_config"])
            messages = _format_messages(
                system_prompt=_compose_system_prompt(policy_cfg),
                entry=session["entry"],
                env_summary=env_summary,
                action_history=list(session.get("action_history") or []),
            )
            turn_index = len(session.get("turns") or [])
        try:
            inference = await _send_inference_request(
                inference_url=str(policy_cfg.get("inference_url") or DEFAULT_INFERENCE_URL),
                api_key=str(policy_cfg.get("api_key") or "").strip() or os.environ.get(str(policy_cfg.get("api_key_env") or "OPENAI_API_KEY")),
                model=str(policy_cfg.get("model") or DEFAULT_MODEL),
                temperature=float(policy_cfg.get("temperature") or 0.0),
                max_tokens=max(int(policy_cfg.get("max_tokens") or 128), 32),
                messages=messages,
            )
        except Exception as exc:
            inference = {
                "assistant_text": json.dumps({"action": admissible_actions[0]}),
                "usage": {},
                "request_id": None,
                "error": str(exc),
            }
        action_text = str(inference.get("assistant_text") or "")
        action_name, invalid_parse = _parse_action(action_text, admissible_actions)
        action_index = admissible_actions.index(action_name)
        async with _STORE_LOCK:
            session = _ROLLOUTS.get(rollout_id)
            if session is None or session["status"] not in {"running", "queued"}:
                return
            env = session.get("env")
            obs, reward, terminated, truncated, _info = env.step(action_index)
            done = bool(terminated or truncated)
            env_summary = _env_summary(env, obs)
            previous_positions = copy.deepcopy(session.get("visited_positions") or [])
            curr_pos = list(env_summary.get("agent_pos") or [])
            if curr_pos:
                previous_positions.append(curr_pos)
            session["visited_positions"] = previous_positions
            session["obs"] = copy.deepcopy(obs)
            session["env_summary"] = env_summary
            env_reward = float(reward or 0.0)
            session.setdefault("action_history", []).append(action_name)
            previous_milestones = set(session.get("milestones") or [])
            new_labels = _scan_milestones(env_summary, {"visited_positions": previous_positions[:-1]})
            session.setdefault("milestones", set()).update(new_labels)
            session.setdefault("turns", []).append(
                {
                    "turn_index": turn_index,
                    "prompt_messages": messages,
                    "assistant_text": action_text,
                    "reasoning_text": None,
                    "actions": [action_name],
                    "decision_reward": 0.0,
                    "reward_before": 0.0,
                    "reward_after": 0.0,
                    "episode_return": 0.0,
                    "return_to_go": 0.0,
                    "trainable": True,
                    "invalid_parse": invalid_parse,
                    "behavior_version": "v000001",
                    "behavior_model": str(policy_cfg.get("model") or DEFAULT_MODEL),
                    "route": str(policy_cfg.get("inference_url") or DEFAULT_INFERENCE_URL),
                    "request_id": inference.get("request_id"),
                    "usage": inference.get("usage", {}),
                    "metadata": {
                        "family": session["entry"].family,
                        "env_id": session["entry"].env_id,
                    },
                }
            )
            session.setdefault("event_history", []).append(
                {
                    "step_idx": turn_index,
                    "action": action_name,
                    "reward": env_reward,
                    "done": done,
                    "agent_pos": curr_pos,
                    "carrying": env_summary.get("carrying"),
                }
            )
            session.setdefault("inference_turns", []).append(
                {
                    "turn_index": turn_index,
                    "request_id": inference.get("request_id"),
                    "assistant_text": action_text,
                    "action": action_name,
                    "invalid_parse": invalid_parse,
                    "error": inference.get("error"),
                }
            )
            if done and float(reward or 0.0) > 0.0:
                session.setdefault("milestones", set()).add("goal_reached")
            milestone_labels = set(session.get("milestones") or [])
            milestone_delta = sorted(milestone_labels - previous_milestones)
            milestone_bonus = _milestone_bonus(milestone_delta)
            decision_reward = env_reward + milestone_bonus
            session["env_reward_total"] = float(session.get("env_reward_total") or 0.0) + env_reward
            session["milestone_bonus_total"] = float(session.get("milestone_bonus_total") or 0.0) + milestone_bonus
            reward_before = float(session.get("total_reward") or 0.0)
            session["total_reward"] = reward_before + decision_reward
            turn = session["turns"][-1]
            turn["decision_reward"] = decision_reward
            turn["reward_before"] = reward_before
            turn["reward_after"] = float(session["total_reward"])
            turn["episode_return"] = float(session["total_reward"])
            turn["metadata"]["env_reward"] = env_reward
            turn["metadata"]["milestone_bonus"] = milestone_bonus
            turn["metadata"]["new_milestone_labels"] = milestone_delta
            session["updated_at"] = _utc_now_iso()
            if done or len(session.get("turns") or []) >= int(session.get("max_steps") or DEFAULT_MAX_STEPS):
                session["status"] = "completed"
                session["success_status"] = "success"
                session["status_detail"] = "completed" if done else "max_steps_exhausted"
                session["response_payload"] = _response_payload(session)
                return


async def _schedule_rollout(rollout_id: str) -> None:
    task = asyncio.create_task(_run_rollout(rollout_id))
    async with _STORE_LOCK:
        _ROLLOUT_TASKS[rollout_id] = task


def _build_session(
    *,
    request: RolloutRequest,
    entry: RegistryEntry,
    env: Any,
    obs: dict[str, Any],
) -> dict[str, Any]:
    env_config = dict(request.env.config or {})
    max_steps = max(int(env_config.get("max_steps") or env_config.get("segment_steps") or DEFAULT_MAX_STEPS), 1)
    rollout_id = str(request.rollout_id or f"minigrid_{uuid.uuid4().hex[:10]}")
    return {
        "rollout_id": rollout_id,
        "trace_correlation_id": str(request.trace_correlation_id),
        "trial_id": str(request.trial_id or f"minigrid_trial_{entry.seed}"),
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "status": "queued",
        "success_status": "pending",
        "status_detail": "queued_for_execution",
        "request_payload": _dump_model(request),
        "policy_config": _policy_config(_dump_model(request)),
        "entry": entry,
        "env": env,
        "obs": copy.deepcopy(obs),
        "env_summary": _env_summary(env, obs),
        "visited_positions": [list(getattr(env.unwrapped, "agent_pos", []) or [])],
        "max_steps": max_steps,
        "total_reward": 0.0,
        "env_reward_total": 0.0,
        "milestone_bonus_total": 0.0,
        "turns": [],
        "event_history": [],
        "inference_turns": [],
        "action_history": [],
        "milestones": set(),
        "pause_requested": False,
        "terminate_requested": False,
        "pause_reason": "",
        "response_payload": {},
    }


def _restore_session_from_checkpoint(
    *,
    checkpoint: dict[str, Any],
    request: ResumeRequest,
    entry: RegistryEntry,
    rollout_id: str,
) -> dict[str, Any]:
    snapshot = dict(checkpoint.get("_snapshot") or {})
    policy_cfg = copy.deepcopy(snapshot.get("policy_config") or {})
    policy_cfg.update(request.overrides.policy_config or {})
    return {
        "rollout_id": rollout_id,
        "trace_correlation_id": f"resume_{rollout_id}",
        "trial_id": f"minigrid_trial_{entry.seed}",
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "status": "queued",
        "success_status": "pending",
        "status_detail": "queued_from_checkpoint",
        "request_payload": _dump_model(request),
        "policy_config": policy_cfg,
        "entry": entry,
        "env": copy.deepcopy(snapshot.get("env")),
        "obs": copy.deepcopy(snapshot.get("obs") or {}),
        "env_summary": copy.deepcopy(snapshot.get("env_summary") or {}),
        "visited_positions": copy.deepcopy(snapshot.get("visited_positions") or []),
        "max_steps": int(request.overrides.segment_steps or request.overrides.continue_steps or DEFAULT_MAX_STEPS),
        "total_reward": float(snapshot.get("total_reward") or 0.0),
        "env_reward_total": float(snapshot.get("env_reward_total") or 0.0),
        "milestone_bonus_total": float(snapshot.get("milestone_bonus_total") or 0.0),
        "turns": copy.deepcopy(snapshot.get("turns") or []),
        "event_history": copy.deepcopy(snapshot.get("event_history") or []),
        "inference_turns": copy.deepcopy(snapshot.get("inference_turns") or []),
        "action_history": list(snapshot.get("action_history") or []),
        "milestones": set(snapshot.get("milestones") or []),
        "pause_requested": False,
        "terminate_requested": False,
        "pause_reason": "",
        "response_payload": {},
        "parent_rollout_id": checkpoint.get("rollout_id"),
        "parent_checkpoint_id": checkpoint.get("checkpoint_id"),
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "evals_minigrid_goex",
        "runtime_available": MINIGRID_AVAILABLE,
        "runtime_import_error": MINIGRID_IMPORT_ERROR,
    }


@app.get("/task_info")
async def task_info(
    seed: int | None = None,
    split_group: str | None = None,
    family: str | None = None,
    task_instance_id: str | None = None,
) -> dict[str, Any]:
    entry = _entry_by_identity(
        seed=seed,
        task_instance_id=task_instance_id,
        split_group=split_group,
        family=family,
    )
    registry = _load_registry()
    dataset = dict(registry.get("dataset") or {})
    return {
        "status": "ok",
        "task_family": "minigrid",
        "reward_type": "env_progress_reward",
        "supports_rollout": True,
        "restore_semantics": "true_environment_snapshot",
        "true_environment_snapshot": True,
        "supports_branching": True,
        "task": {
            "id": "minigrid_prompt_opt",
            "name": "MiniGrid Prompt Optimization",
            "version": "v1",
        },
        "dataset": {
            **dataset,
            "task_registry": registry,
            "task_registry_path": str(TASK_REGISTRY_PATH),
            "train_seed_manifest": [item.seed for item in _registry_entries() if item.split_group == "train"],
            "heldout_seed_manifest": [item.seed for item in _registry_entries() if item.split_group == "heldout"],
        },
        "task_preview": {
            "task_instance_id": entry.task_instance_id,
            "registry_task_id": entry.registry_task_id,
            "seed": entry.seed,
            "split_group": entry.split_group,
            "family": entry.family,
            "env_id": entry.env_id,
            "label": entry.label,
        },
        "capabilities": {
            "checkpoint_support": True,
            "resume_support": True,
            "state_support": True,
            "pause_support": True,
            "terminate_support": True,
            "trace_support": True,
            "fork_support": True,
            "supports_branching": True,
            "true_environment_snapshot": True,
            "restore_semantics": "true_environment_snapshot",
            "checkpoint_semantics": "true_environment_snapshot",
            "control_boundary": "policy_turn",
        },
    }


@app.post("/rollout")
async def rollout(request: RolloutRequest) -> dict[str, Any]:
    _ensure_runtime_available()
    env_config = dict(request.env.config or {})
    entry = _entry_by_identity(
        seed=request.env.seed,
        task_instance_id=request.task_instance_id or str(request.task_metadata.get("task_instance_id") or "").strip() or None,
        task_id=request.task_id or str(request.task_metadata.get("task_id") or "").strip() or None,
        split_group=str(env_config.get("split_group") or "").strip() or None,
        family=str(env_config.get("family") or "").strip() or None,
    )
    env = _make_env(
        entry,
        max_steps=max(int(env_config.get("max_steps") or env_config.get("segment_steps") or DEFAULT_MAX_STEPS), 1),
        seed=int(request.env.seed if request.env.seed is not None else entry.seed),
    )
    obs, _info = env.reset(seed=int(request.env.seed if request.env.seed is not None else entry.seed))
    session = _build_session(request=request, entry=entry, env=env, obs=obs)
    async with _STORE_LOCK:
        _ROLLOUTS[session["rollout_id"]] = session
    if str(request.submission_mode or "sync").strip().lower() == "async":
        await _schedule_rollout(session["rollout_id"])
        async with _STORE_LOCK:
            return _state_payload(_ROLLOUTS[session["rollout_id"]])
    await _run_rollout(session["rollout_id"])
    async with _STORE_LOCK:
        stored = _ROLLOUTS[session["rollout_id"]]
        return dict(stored.get("response_payload") or _response_payload(stored))


@app.get("/rollouts/{rollout_id}/state")
async def rollout_state(rollout_id: str) -> dict[str, Any]:
    async with _STORE_LOCK:
        session = _ROLLOUTS.get(rollout_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _state_payload(session)


@app.get("/rollouts/{rollout_id}/trace")
async def rollout_trace(rollout_id: str) -> dict[str, Any]:
    async with _STORE_LOCK:
        session = _ROLLOUTS.get(rollout_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return {"trace": (_response_payload(session)).get("trace", {})}


@app.post("/rollouts/{rollout_id}/pause")
async def pause_rollout(rollout_id: str, body: PauseBody) -> dict[str, Any]:
    async with _STORE_LOCK:
        session = _ROLLOUTS.get(rollout_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        if session["status"] not in {"queued", "running"}:
            return _state_payload(session)
        session["pause_requested"] = True
        session["pause_reason"] = str(body.reason or "")
        session["updated_at"] = _utc_now_iso()
        return _state_payload(session)


@app.post("/rollouts/{rollout_id}/terminate")
async def terminate_rollout(rollout_id: str, body: TerminateBody) -> dict[str, Any]:
    async with _STORE_LOCK:
        session = _ROLLOUTS.get(rollout_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        if session["status"] in {"completed", "failed", "cancelled"}:
            return _state_payload(session)
        session["terminate_requested"] = True
        session["status_detail"] = str(body.reason or "terminate_requested")
        session["updated_at"] = _utc_now_iso()
        return _state_payload(session)


@app.post("/rollouts/{rollout_id}/checkpoints")
async def create_checkpoint(rollout_id: str, body: CheckpointRequest) -> dict[str, Any]:
    async with _STORE_LOCK:
        session = _ROLLOUTS.get(rollout_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        record = _create_checkpoint_locked(
            session,
            checkpoint_id=body.checkpoint_id,
            label=body.label,
            source=body.source,
            actor_ids=body.actor_ids,
            metadata=body.metadata,
            annotations=body.annotations,
        )
        session["response_payload"] = _response_payload(session)
        return _checkpoint_descriptor(record)


@app.get("/checkpoints/{checkpoint_id}")
async def checkpoint_info(checkpoint_id: str) -> dict[str, Any]:
    async with _STORE_LOCK:
        record = _CHECKPOINTS.get(checkpoint_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown_checkpoint:{checkpoint_id}")
        return _checkpoint_descriptor(record)


@app.post("/rollouts/{rollout_id}/resume")
@app.post("/rollouts/{rollout_id}/fork")
async def resume_rollout(rollout_id: str, request: ResumeRequest) -> dict[str, Any]:
    _ensure_runtime_available()
    checkpoint_id = str(request.checkpoint_id or "").strip()
    if not checkpoint_id:
        raise HTTPException(status_code=400, detail="checkpoint_id is required for minigrid resume")
    async with _STORE_LOCK:
        checkpoint = _CHECKPOINTS.get(checkpoint_id)
        if checkpoint is None:
            raise HTTPException(status_code=404, detail=f"unknown_checkpoint:{checkpoint_id}")
    entry = _entry_by_identity(
        task_instance_id=request.overrides.task_instance_id or None,
        task_id=request.overrides.task_id or None,
    )
    if not request.overrides.task_instance_id and not request.overrides.task_id:
        async with _STORE_LOCK:
            source_rollout = _ROLLOUTS.get(str(checkpoint.get("rollout_id") or ""))
        if source_rollout is not None:
            entry = source_rollout["entry"]
    target_rollout_id = str(request.target_rollout_id or f"minigrid_resume_{uuid.uuid4().hex[:10]}")
    session = _restore_session_from_checkpoint(
        checkpoint=checkpoint,
        request=request,
        entry=entry,
        rollout_id=target_rollout_id,
    )
    async with _STORE_LOCK:
        _ROLLOUTS[target_rollout_id] = session
    if str(request.submission_mode or "sync").strip().lower() == "async":
        await _schedule_rollout(target_rollout_id)
        async with _STORE_LOCK:
            return _state_payload(_ROLLOUTS[target_rollout_id])
    await _run_rollout(target_rollout_id)
    async with _STORE_LOCK:
        stored = _ROLLOUTS[target_rollout_id]
        return dict(stored.get("response_payload") or _response_payload(stored))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8922"))
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)
