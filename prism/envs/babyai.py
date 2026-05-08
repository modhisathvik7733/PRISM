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

# Suppress minigrid's noisy "Sampling rejected: unreachable object at (X, Y)"
# stdout lines emitted during BabyAI level resets. These can be 20-30 lines
# per cohort eval and bury our actual logs. Idempotent — installing twice
# is a no-op. See prism/utils/log_filter.py for details.
from prism.utils.log_filter import install_minigrid_noise_filter
install_minigrid_noise_filter()


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
    max_steps: int | None = None,
) -> gym.Env:
    """Create a BabyAI env with PRISM's standard wrappers.

    Args:
        env_id: any minigrid `BabyAI-*-v0` id. Default GoToLocal is the simplest
                Phase-0 sanity level (single-room, single instruction).
        seed: deterministic eval seed (`None` to skip).
        include_mission: keep the mission string in obs (needed Phase 2+).
                         Set False for the SB3 image-only PPO baseline.
        render_mode: e.g. "rgb_array" for video logging.
        max_steps: override the env's internal max-step budget. Default (None)
                   leaves the level's built-in cap (64 for GoToLocal). A larger
                   value gives the agent more time to solve hard spawns AND
                   raises per-episode reward via BabyAI's
                   `1 − 0.9 × (steps/max_steps)` formula (same step count
                   becomes a smaller fraction of a longer budget). We apply
                   the override on `env.unwrapped.max_steps` after gym.make
                   AND on the spec's max_episode_steps so the gymnasium
                   TimeLimit wrapper picks it up.
    """
    env = gym.make(env_id, render_mode=render_mode)
    if max_steps is not None:
        # MiniGrid stores its own truncation counter on the unwrapped env,
        # gymnasium adds a TimeLimit wrapper that reads spec.max_episode_steps.
        # Set both so neither truncates earlier than the other.
        try:
            env.unwrapped.max_steps = max_steps
        except AttributeError:
            pass
        if env.spec is not None:
            env.spec.max_episode_steps = max_steps
    if include_mission:
        env = PrismImageObsWrapper(env)
    else:
        env = PrismImageOnlyWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def set_max_steps(env: gym.Env, max_steps: int) -> None:
    """Override max_steps on an already-constructed env.

    Three layers can truncate episodes:
      1. MiniGrid's internal `step_count >= max_steps` on the unwrapped env.
         We update `env.unwrapped.max_steps`.
      2. gymnasium's `TimeLimit` wrapper holds `_max_episode_steps` set at
         construction. We walk the wrapper chain and patch any wrapper
         that exposes that attribute.
      3. The level-class itself may also override max_steps in its `_gen_grid`
         method or `__init__`. The post-construction mutation handles 1 and 2.

    For *new* env construction, callers should also pass
    `gym.make(env_id, max_episode_steps=N)` — that's the documented gymnasium
    API and the most reliable way to set the cap. This helper is a
    best-effort retrofit for envs already constructed.
    """
    try:
        env.unwrapped.max_steps = max_steps
    except AttributeError:
        pass
    e = env
    seen_ids = set()
    while e is not None and id(e) not in seen_ids:
        seen_ids.add(id(e))
        if hasattr(e, "_max_episode_steps"):
            try:
                e._max_episode_steps = max_steps
            except AttributeError:
                pass
        spec = getattr(e, "spec", None)
        if spec is not None:
            try:
                spec.max_episode_steps = max_steps
            except AttributeError:
                pass
        e = getattr(e, "env", None)


def make_env_with_max_steps(env_id: str, max_steps: int) -> gym.Env:
    """Construct a BabyAI env with the requested max_steps using gymnasium's
    documented `max_episode_steps` kwarg AND the post-construction patch as
    belt-and-suspenders. Use this in scripts that need raw dict obs (run_agent,
    eval_agent_cohorts, ppo_train) — make_babyai_env is for image-only flows.

    Prints a diagnostic line so it's obvious in logs whether the override
    actually took effect."""
    env = gym.make(env_id, max_episode_steps=max_steps)
    set_max_steps(env, max_steps)
    # Diagnostic — print the effective caps so the operator can verify.
    unwrapped_cap = getattr(env.unwrapped, "max_steps", "?")
    spec_cap = getattr(env.spec, "max_episode_steps", "?") if env.spec else "?"
    print(f"[env] {env_id}: unwrapped.max_steps={unwrapped_cap} "
          f"spec.max_episode_steps={spec_cap}")
    return env
