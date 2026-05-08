"""BabyAI / MiniGrid environment factory.

The Farama-maintained `minigrid` package ships the BabyAI levels as
`BabyAI-*-v0` envs. We wrap them with a small obs adapter that:

  1. Extracts the partial-view (HWC) and rearranges to (C, H, W).
  2. Normalizes by the per-channel max code (NOT by 255 — these are symbolic
     codes, not pixels: ch0=object type 0–10, ch1=color 0–5, ch2=state 0–3).
     Dividing by 255 squashes everything into [0, 0.04] and PPO can't learn.
  3. Keeps the natural-language `mission` string accessible alongside.
  4. Optionally fixes the seed for deterministic eval.

This wrapper is intentionally thin — Phase 0 only needs to load + step.
Encoding `mission` into instruction-conditioning vectors happens in Phase 4.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401  (registers BabyAI-* envs)
import numpy as np
from gymnasium import spaces


# Per-channel max values for MiniGrid's symbolic image obs.
# ch0: object type — wall/floor/door/key/ball/box/goal/lava/agent → up to 10.
# ch1: color id (red, green, blue, purple, yellow, grey)         → up to 5.
# ch2: state (door open/closed/locked, etc.)                     → up to 3.
# Use a small safety margin so we don't clip on edge cases.
_CHANNEL_MAX = np.array([11.0, 6.0, 4.0], dtype=np.float32).reshape(3, 1, 1)


def _encode_image(img_hwc: np.ndarray) -> np.ndarray:
    """HWC uint8 (symbolic codes) -> CHW float32 in roughly [0, 1]."""
    chw = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32)
    return chw / _CHANNEL_MAX


class PrismImageObsWrapper(gym.ObservationWrapper):
    """Convert BabyAI dict obs → {image: (C,H,W) float32, mission: str, direction: int}."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        img_space = env.observation_space["image"]
        H, W, C = img_space.shape  # noqa: N806
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0.0, high=1.0, shape=(C, H, W), dtype=np.float32),
                "direction": spaces.Discrete(4),
                # Mission stays as a plain string; SB3 won't index this directly,
                # so for SB3 PPO baselines we provide a separate text-free env via
                # `make_babyai_env(include_mission=False)`.
                "mission": spaces.Text(max_length=256),
            }
        )

    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "image": _encode_image(obs["image"]),
            "direction": int(obs["direction"]),
            "mission": str(obs.get("mission", "")),
        }


class PrismImageOnlyWrapper(gym.ObservationWrapper):
    """Image-only obs (CHW float32). Used for the Phase 0 PPO sanity baseline,
    which doesn't yet condition on mission text."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        img_space = env.observation_space["image"]
        H, W, C = img_space.shape  # noqa: N806
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(C, H, W), dtype=np.float32
        )

    def observation(self, obs: dict[str, Any]) -> np.ndarray:
        return _encode_image(obs["image"])


def make_babyai_env(
    env_id: str = "BabyAI-GoToLocal-v0",
    *,
    seed: int | None = None,
    include_mission: bool = True,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a BabyAI env with PRISM's standard wrappers.

    Args:
        env_id: any minigrid `BabyAI-*-v0` id. Default GoToLocal is the simplest
                Phase-0 sanity level (single-room, single instruction).
        seed: deterministic eval seed (`None` to skip).
        include_mission: keep the mission string in obs (needed Phase 2+).
                         Set False for the SB3 image-only PPO baseline.
        render_mode: e.g. "rgb_array" for video logging.
    """
    env = gym.make(env_id, render_mode=render_mode)
    if include_mission:
        env = PrismImageObsWrapper(env)
    else:
        env = PrismImageOnlyWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env
