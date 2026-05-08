"""Crafter env wrapper — constructs `crafter.Env()` directly and exposes a
gymnasium-style API for PRISM.

Crafter 1.8.x doesn't register itself with gymnasium any more; you call
`crafter.Env(reward=True)` and wrap it yourself. The native env returns
old-style 4-tuples from step (obs, reward, done, info) and a single obs
from reset(), which we adapt to gymnasium's 5-tuple / (obs, info) here.

We:
  - Cast obs from (64, 64, 3) uint8 to (3, 64, 64) float32 in [0, 1]
  - Track per-episode achievement set so the eval can score by the
    standard geometric-mean formula
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

STATE_DIM = 12  # dims in get_game_state() output

# Crafter exposes 22 named achievements. Keeping them here so the eval
# script doesn't need to import crafter just for the names.
CRAFTER_ACHIEVEMENTS: tuple[str, ...] = (
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
)


def _encode_rgb(obs_hwc_uint8: np.ndarray) -> np.ndarray:
    """64x64x3 uint8 → 3x64x64 float32 in [0, 1]."""
    chw = np.transpose(obs_hwc_uint8, (2, 0, 1))
    return chw.astype(np.float32) / 255.0


class CrafterPrismEnv:
    """Plain wrapper around `crafter.Env()` exposing a gymnasium-style API.

    Not a `gym.Wrapper` subclass — Crafter's env is old-style gym.Env and
    gymnasium.Wrapper makes assumptions (Wrapper.reset signature, etc.)
    that don't hold. Composing rather than inheriting keeps the surface
    small and predictable.
    """

    def __init__(self, env: Any):
        self._env = env
        self._unlocked: set[str] = set()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32
        )
        # Crafter's action_space is a gym.spaces.Discrete which exposes .n
        self.action_space = gym.spaces.Discrete(getattr(env.action_space, "n", 17))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._unlocked = set()
        # Crafter.Env.reset() takes no args. Older versions silently
        # accept seed=; newer ones don't. We try with seed first, fall
        # back to plain reset() on TypeError.
        try:
            obs = self._env.reset(seed=seed) if seed is not None else self._env.reset()
        except TypeError:
            obs = self._env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return _encode_rgb(obs), {}

    def step(self, action: int):
        out = self._env.step(int(action))
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False
        ach = info.get("achievements", {}) or {}
        for name, on in ach.items():
            if on:
                self._unlocked.add(name)
        info["achievements_unlocked"] = set(self._unlocked)
        return _encode_rgb(obs), float(reward), bool(terminated), bool(truncated), info

    def get_game_state(self) -> np.ndarray:
        """Extract 12-dim structured game state as float32 numpy array.

        dims 0-5:  inventory wood/stone/coal/iron/diamond/sapling ÷ 9.0
        dims 6-9:  health/food/drink/energy ÷ 9.0
        dims 10-11: facing dx, dy each in {-1,0,1} → (x+1)/2
        """
        try:
            p = self._env._player
            inv = p.inventory
            out = np.zeros(12, dtype=np.float32)
            out[0] = inv.get("wood",    0) / 9.0
            out[1] = inv.get("stone",   0) / 9.0
            out[2] = inv.get("coal",    0) / 9.0
            out[3] = inv.get("iron",    0) / 9.0
            out[4] = inv.get("diamond", 0) / 9.0
            out[5] = inv.get("sapling", 0) / 9.0
            out[6] = p.health / 9.0
            out[7] = p.food   / 9.0
            out[8] = p.drink  / 9.0
            out[9] = p.energy / 9.0
            dx, dy  = p.facing
            out[10] = (dx + 1) / 2.0
            out[11] = (dy + 1) / 2.0
            return out
        except Exception:
            return np.zeros(12, dtype=np.float32)

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()


def make_crafter_env(reward_mode: str = "reward", seed: int | None = None):
    """Construct a Crafter env wrapped for PRISM.

    Args:
        reward_mode: "reward" (dense achievement reward, the standard
                     benchmark mode) or "noreward" (unsupervised setting).
        seed: optional seed forwarded to the underlying crafter.Env if
              that version supports it.
    """
    import crafter  # lazy: keeps PRISM importable on hosts without crafter

    use_reward = reward_mode == "reward"
    # Different crafter versions accept different ctor kwargs. Build
    # incrementally so we work on 1.6 → 1.8.
    kwargs: dict[str, Any] = {"reward": use_reward}
    if seed is not None:
        kwargs["seed"] = seed
    try:
        env = crafter.Env(**kwargs)
    except TypeError:
        # Older versions take only reward=. Newer might not accept seed=.
        env = crafter.Env(reward=use_reward)
    return CrafterPrismEnv(env)
