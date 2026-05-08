"""Crafter env wrapper — gym.make + obs preprocessing for PRISM.

Crafter exposes a 64×64×3 uint8 RGB image observation and a Discrete(17)
action space. We:
  - Cast obs to float32 in [0, 1]
  - Transpose HWC → CHW (PyTorch convention)
  - Track per-episode achievement set so the eval can score by the standard
    geometric-mean formula

The wrapper is gymnasium-compatible (reset → (obs, info), step →
(obs, reward, terminated, truncated, info)).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

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


class CrafterPrismWrapper(gym.Wrapper):
    """Thin wrapper that:
      - Encodes obs to (3, 64, 64) float32 in [0, 1]
      - Returns gymnasium-style (obs, info) and (obs, r, term, trunc, info)
        tuples (Crafter's native API is older gym style)
      - Tracks unlocked achievements in info["achievements_unlocked"]
        (cumulative set across the episode)
    """

    def __init__(self, env: Any):
        super().__init__(env)
        self._unlocked: set[str] = set()
        # Crafter's underlying obs is (64, 64, 3) uint8. After our wrapping,
        # gym thinks it's (3, 64, 64) float32 — declare that.
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._unlocked = set()
        # Crafter's reset signature is just reset() — older gym style.
        # If the seed kwarg is unsupported we just ignore it.
        try:
            obs = self.env.reset()
        except TypeError:
            obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return _encode_rgb(obs), {}

    def step(self, action: int):
        out = self.env.step(int(action))
        # Crafter returns (obs, reward, done, info) (4-tuple, old style).
        # Newer gymnasium-compatible builds may return 5-tuple.
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False
        # Track newly-unlocked achievements (info["achievements"] is a
        # name → bool dict).
        ach = info.get("achievements", {}) or {}
        for name, on in ach.items():
            if on:
                self._unlocked.add(name)
        info["achievements_unlocked"] = set(self._unlocked)
        return _encode_rgb(obs), float(reward), bool(terminated), bool(truncated), info


def make_crafter_env(reward_mode: str = "reward", seed: int | None = None):
    """Construct a Crafter env wrapped for PRISM.

    Args:
        reward_mode: "reward" (Crafter-Reward-v1, dense achievement reward)
                     or "noreward" (Crafter-NoReward-v1, unsupervised setting).
        seed: optional seed for np.random / Crafter's procedural generator.
    """
    # Lazy import — we want the module importable on machines without crafter
    # so the smoke test can detect the missing dependency cleanly.
    import crafter  # noqa: F401  (registers the gym envs)

    env_id = "CrafterReward-v1" if reward_mode == "reward" else "CrafterNoReward-v1"
    env = gym.make(env_id)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            pass
    return CrafterPrismWrapper(env)
