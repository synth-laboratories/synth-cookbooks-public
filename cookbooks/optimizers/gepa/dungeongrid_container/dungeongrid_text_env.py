"""
Text wrapper around the DungeonGrid environment.

Mirrors the simple interface the Crafter cookbook container uses
(`reset(seed) -> (None, text)`, `step(action) -> (None, text, reward, done, info)`)
so the GEPA service app stays parallel to `crafter_container`.

DungeonGrid is a turn-based, multi-hero dungeon. Each `step` advances the
currently active agent (`active_agent`). Heroes are driven by the LLM policy;
the Warden is environment-controlled, so when the active agent is the warden the
service app steps it with a `warden_auto` action instead of calling the model.
"""
from __future__ import annotations

from typing import Any

# Hero actions the LLM policy may emit (the warden actions are env-controlled).
HERO_ACTIONS = [
    "move", "open_door", "attack_melee", "attack_ranged", "cast",
    "inspect_tile", "inspect_room", "search_traps", "search_secrets",
    "search_treasure", "search_furniture", "attack_object", "disarm",
    "interact", "use_item", "equip_item", "give_item", "message", "guard",
    "end_turn",
]
DIRECTIONS = ["north", "south", "west", "east"]
_TILE_TARGET_ACTIONS = {"inspect_tile"}


def build_action(args: dict[str, Any]) -> dict[str, Any]:
    """Assemble a DungeonGrid action dict from flat tool-call arguments.

    The env validates the result; malformed actions come back as `info.invalid`
    rather than raising, so this stays permissive on purpose.
    """
    action_type = str(args.get("action_type") or "end_turn")
    action: dict[str, Any] = {"type": action_type}

    direction = args.get("direction")
    if direction in DIRECTIONS:
        action["direction"] = direction

    target = args.get("target")
    if target not in (None, ""):
        if action_type in _TILE_TARGET_ACTIONS and isinstance(target, str) and "," in target:
            try:
                x, y = (int(p.strip()) for p in target.split(",", 1))
                action["target"] = [x, y]
            except Exception:
                action["target"] = target
        else:
            action["target"] = target

    text = args.get("text")
    if action_type == "message":
        action.setdefault("target", target or "party")
        action["payload"] = {"text": str(text or "")}
    elif text not in (None, ""):
        action["payload"] = {"text": str(text)}

    return action


class DungeonGridTextEnv:
    """One LLM-driven DungeonGrid episode, rendered as text per turn."""

    def __init__(self, quest_id: str, num_heroes: int) -> None:
        self._quest_id = quest_id
        self._num_heroes = num_heroes
        self._env: Any = None
        self._obs: Any = None

    @property
    def active_agent(self) -> str:
        return str(getattr(self._obs, "active_agent", "") or "")

    def reset(self, seed: int) -> tuple[None, str]:
        from dungeongrid import DungeonGridEnvironment

        self._env = DungeonGridEnvironment()
        self._obs = self._env.reset(
            quest_id=self._quest_id, num_heroes=self._num_heroes, seed=seed
        )
        return None, str(self._obs.text)

    def step(self, action: dict[str, Any]) -> tuple[None, str, float, bool, dict]:
        result = self._env.step(action)
        self._obs = result.observation
        info = result.info or {}
        achievements = [
            a.get("id")
            for a in (info.get("new_achievements") or [])
            if isinstance(a, dict)
        ]
        return (
            None,
            str(self._obs.text),
            float(result.reward),
            bool(result.done),
            {"achievements": achievements, "invalid": bool(info.get("invalid"))},
        )
