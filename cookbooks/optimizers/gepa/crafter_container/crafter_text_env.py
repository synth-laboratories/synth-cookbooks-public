"""
Text-aware Craftax wrapper.

Returns BOTH a pixel frame (for the CNN critic) and a compact text rendering
(for a text-only LM policy). Reads attributes off the Craftax `EnvState` so we
don't need craftaxlm as a dep.
"""
from __future__ import annotations

from typing import Any

import numpy as np

VALID_ACTIONS = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep", "place_stone", "place_table", "place_furnace",
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
    "make_iron_sword",
]
_ACTION_IDX = {a: i for i, a in enumerate(VALID_ACTIONS)}

_DIRECTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}

# Inventory item slots in Craftax (full). Order matches state.inventory namedtuple.
_INVENTORY_FIELDS = [
    "wood", "stone", "coal", "iron", "diamond", "sapling", "wood_pickaxe",
    "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword",
]


def _safe_int(x: Any) -> int:
    try:
        import jax
        return int(jax.device_get(x))
    except Exception:
        try:
            return int(x)
        except Exception:
            return 0


def _render_inventory(state: Any) -> str:
    inv = getattr(state, "inventory", None)
    if inv is None:
        return "(unknown)"
    parts = []
    for field in _INVENTORY_FIELDS:
        v = getattr(inv, field, None)
        if v is None:
            continue
        n = _safe_int(v)
        if n > 0:
            parts.append(f"{field}={n}")
    return ", ".join(parts) if parts else "empty"


def _render_local_map(state: Any, radius: int = 4) -> str:
    """Render a (2*radius+1) ASCII view centered on the player."""
    try:
        import jax
        m = jax.device_get(state.map)  # (H, W) int
        pos = jax.device_get(state.player_position)  # (2,) int
    except Exception:
        return "(no map)"
    py, px = int(pos[0]), int(pos[1])
    H, W = m.shape[:2]
    rows = []
    for dy in range(-radius, radius + 1):
        row = []
        for dx in range(-radius, radius + 1):
            y, x = py + dy, px + dx
            if not (0 <= y < H and 0 <= x < W):
                row.append("#")  # off-map
                continue
            if dy == 0 and dx == 0:
                row.append("@")  # player
                continue
            tile = int(m[y, x])
            row.append(_TILE_GLYPHS.get(tile, "?"))
        rows.append("".join(row))
    return "\n".join(rows)


# Minimal tile glyphs covering the most common Craftax tile ids.
# Extra ids fall back to '?' which is fine — model learns from context.
_TILE_GLYPHS = {
    0: ".",   # invalid / out-of-map
    1: ".",   # grass
    2: "T",   # tree
    3: "S",   # stone
    4: "~",   # water
    5: "C",   # coal
    6: "I",   # iron
    7: "D",   # diamond
    8: "P",   # crafting table
    9: "F",   # furnace
    10: "p",  # plant/sapling growing
    11: "L",  # lava
    12: "W",  # wood placed
    13: "s",  # placed stone
    14: ".",  # path / sand
}


def render_state_text(state: Any, step_count: int, recent_achievements: list[str]) -> str:
    """Compact, fixed-format text suitable for a small text LM."""
    health = _safe_int(getattr(state, "player_health", 0))
    food = _safe_int(getattr(state, "player_food", 0))
    water = _safe_int(getattr(state, "player_water", 0))
    energy = _safe_int(getattr(state, "player_energy", 0))
    direction = _DIRECTION_NAMES.get(_safe_int(getattr(state, "player_direction", 0)), "?")
    pos = getattr(state, "player_position", None)
    if pos is not None:
        try:
            import jax
            pos_arr = jax.device_get(pos)
            pos_str = f"({int(pos_arr[0])},{int(pos_arr[1])})"
        except Exception:
            pos_str = "?"
    else:
        pos_str = "?"

    inv = _render_inventory(state)
    local_map = _render_local_map(state)

    return (
        f"step={step_count}\n"
        f"health={health} food={food} water={water} energy={energy} facing={direction} pos={pos_str}\n"
        f"inventory: {inv}\n"
        f"recent_achievements: {recent_achievements if recent_achievements else 'none'}\n"
        f"map (9x9, @=you, T=tree, S=stone, ~=water, C=coal, I=iron, D=diamond, P=table, F=furnace, L=lava, .=open):\n"
        f"{local_map}"
    )


class CrafterTextEnv:
    """Returns (frame, text) per step. Frame goes to CNN critic; text goes to LM policy."""

    def __init__(self) -> None:
        self._env = None
        self._env_params = None
        self._state = None
        self._key = None
        self._step_count = 0

    def reset(self, seed: int) -> tuple[np.ndarray, str]:
        import jax
        from craftax.craftax_env import make_craftax_env_from_name

        if self._env is None:
            self._env = make_craftax_env_from_name("Craftax-Pixels-v1", auto_reset=False)
            self._env_params = self._env.default_params

        self._key = jax.random.PRNGKey(seed)
        self._key, subkey = jax.random.split(self._key)
        obs, self._state = self._env.reset(subkey, self._env_params)
        self._step_count = 0
        return self._to_frame(obs), render_state_text(self._state, self._step_count, [])

    def step(self, action: str) -> tuple[np.ndarray, str, float, bool, dict]:
        import jax
        import jax.numpy as jnp

        if action not in _ACTION_IDX:
            raise ValueError(f"invalid action {action!r}; must be one of {VALID_ACTIONS}")

        prev_ach = jax.device_get(self._state.achievements)
        self._key, subkey = jax.random.split(self._key)
        obs, self._state, reward, done, info = self._env.step(
            subkey, self._state, jnp.int32(_ACTION_IDX[action]), self._env_params
        )
        self._step_count += 1

        curr_ach = jax.device_get(self._state.achievements)
        new_indices = [i for i, (p, c) in enumerate(zip(prev_ach, curr_ach)) if not p and c]
        new_achievements: list[str] = []
        if new_indices:
            try:
                from craftax.craftax.constants import Achievement
                new_achievements = [Achievement(i).name.lower() for i in new_indices]
            except Exception:
                new_achievements = [f"achievement_{i}" for i in new_indices]

        text = render_state_text(self._state, self._step_count, new_achievements)
        return (
            self._to_frame(obs),
            text,
            float(jax.device_get(reward)),
            bool(jax.device_get(done)),
            {"achievements": new_achievements},
        )

    def close(self) -> None:
        pass

    def _to_frame(self, obs) -> np.ndarray:
        import jax
        arr = jax.device_get(obs)
        if arr.dtype != np.uint8:
            arr = (np.array(arr) * 255).clip(0, 255).astype(np.uint8)
        return arr


class BatchedCrafterTextEnv:
    """N envs in lockstep via jax.vmap. One JAX call per step processes all N envs.

    API:
      reset(seeds)        -> frames, texts                    # lists of length N
      step(actions)       -> frames, texts, rewards, dones, infos
      step_idx(action_idx) -> same, takes int32 array directly (skip ACTION_IDX lookup)

    `frames` is np.ndarray (N, H, W, 3) uint8.
    """

    def __init__(self, n_envs: int) -> None:
        import jax
        from craftax.craftax_env import make_craftax_env_from_name

        self.n = n_envs
        self._env = make_craftax_env_from_name("Craftax-Pixels-v1", auto_reset=False)
        self._params = self._env.default_params
        # Close over params to avoid JIT hashing the tracable EnvParams as a static arg
        _reset = self._env.reset
        _step = self._env.step
        _p = self._params

        def _reset_one(k):
            return _reset(k, _p)

        def _step_one(k, s, a):
            return _step(k, s, a, _p)

        self._batched_reset = jax.jit(jax.vmap(_reset_one))
        self._batched_step = jax.jit(jax.vmap(_step_one))
        self._state = None
        self._keys = None
        self._step_count = 0

    def reset(self, seeds: list[int]) -> tuple[np.ndarray, list[str]]:
        import jax
        import jax.numpy as jnp
        assert len(seeds) == self.n, f"expected {self.n} seeds, got {len(seeds)}"
        keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
        # vmap split: (N, 2) -> (N, 2, 2). first half = reset keys, second half = step keys
        pairs = jax.vmap(jax.random.split)(keys)
        reset_keys = pairs[:, 0]
        self._keys = pairs[:, 1]
        obs, self._state = self._batched_reset(reset_keys)
        self._step_count = 0
        return self._frames_and_texts(obs, prev_ach_per_env=None)

    def step(self, actions: list[str]) -> tuple[np.ndarray, list[str], list[float], list[bool], list[dict]]:
        import jax.numpy as jnp
        action_idx = jnp.array(
            [_ACTION_IDX.get(a, _ACTION_IDX["noop"]) for a in actions], dtype=jnp.int32
        )
        return self.step_idx(action_idx)

    def step_idx(self, action_idx) -> tuple[np.ndarray, list[str], list[float], list[bool], list[dict]]:
        import jax
        import jax.numpy as jnp

        prev_ach = jax.device_get(self._state.achievements)  # (N, A)
        # fresh subkeys for each env's step
        new_pairs = jax.vmap(jax.random.split)(self._keys)
        step_keys = new_pairs[:, 0]
        self._keys = new_pairs[:, 1]

        obs, self._state, rewards, dones, info = self._batched_step(
            step_keys, self._state, action_idx
        )
        self._step_count += 1
        rewards_np = jax.device_get(rewards).tolist()
        dones_np = [bool(d) for d in jax.device_get(dones).tolist()]
        return self._frames_and_texts(obs, prev_ach_per_env=prev_ach, rewards=rewards_np, dones=dones_np)

    def _frames_and_texts(
        self,
        obs,
        prev_ach_per_env=None,
        rewards: list[float] | None = None,
        dones: list[bool] | None = None,
    ):
        """Pull state to host once, slice per env, render text + frame for each."""
        import jax
        obs_arr = jax.device_get(obs)
        if obs_arr.dtype != np.uint8:
            obs_arr = (obs_arr * 255).clip(0, 255).astype(np.uint8)
        # One device_get for the whole state (numpy after this)
        state_host = jax.device_get(self._state)
        curr_ach_host = state_host.achievements  # (N, A)

        # Build per-env achievement diffs
        new_ach_per_env: list[list[str]] = []
        if prev_ach_per_env is not None:
            try:
                from craftax.craftax.constants import Achievement
                _name_for = lambda j: Achievement(j).name.lower()
            except Exception:
                _name_for = lambda j: f"achievement_{j}"
            for i in range(self.n):
                idx = [
                    j for j, (p, c) in enumerate(zip(prev_ach_per_env[i], curr_ach_host[i]))
                    if not p and c
                ]
                new_ach_per_env.append([_name_for(j) for j in idx])
        else:
            new_ach_per_env = [[] for _ in range(self.n)]

        frames = np.asarray(obs_arr)  # (N, H, W, 3)
        texts: list[str] = []
        for i in range(self.n):
            single_state = jax.tree_util.tree_map(lambda x: x[i], state_host)
            texts.append(render_state_text(single_state, self._step_count, new_ach_per_env[i]))

        if rewards is None:
            return frames, texts
        infos = [{"achievements": ach} for ach in new_ach_per_env]
        return frames, texts, rewards, dones, infos
