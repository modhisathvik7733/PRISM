"""Synthetic Unity-nav gym env for continual fine-tuning.

Mirrors the Unity demo bridge exactly:
  * Continuous 2D plane (±plane_extent in x, z).
  * Action space: 7 discrete (BabyAI conventions:
    turn_L=0, turn_R=1, forward=2, pickup=3, drop=4, toggle=5, done=6).
    Only 0/1/2 actually move the agent here; 3-6 are no-ops (matches our
    inference-time mask).
  * Observation: rendered via Unity2DAdapter.render_obs_multi — same code
    path the inference server uses, so train/deploy match exactly.
  * One target + N distractors per episode, positions randomized.
  * Reward: +1 on touching the green target (within reach_threshold),
    small step penalty otherwise.
  * Mission: fixed per episode (one of (color, type) configurations).

The env is self-contained so PPO can collect many rollouts fast without
any Unity round-trip.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from prism.adapters.unity_2d import (
    Unity2DAdapter,
    _FORWARD_VEC,  # Reused so heading semantics match the inference server.
)
from prism.perception.slots import (
    COLOR_NAME_TO_IDX,
    OBJECT_NAME_TO_TYPE,
)


_OBJECT_TYPE_NAMES = ("door", "key", "ball", "box")
_COLOR_NAMES = ("red", "green", "blue", "purple", "yellow", "grey")


class UnityNavEnv(gym.Env):
    """Single-target nav with optional same-/different-color distractors.

    Designed for continual fine-tuning: training distribution matches the
    Unity demo adapter's runtime observations, so weights transfer
    one-to-one.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target_color: str = "green",
        target_type: str = "ball",
        distractor_specs: list[tuple[str, str]] | None = None,
        max_steps: int = 100,
        forward_step: float = 0.5,
        reach_threshold: float = 1.0,
        plane_extent: float = 4.5,
        obs_scale: float = 2.0,
        step_penalty: float = 0.005,
        randomize_target_color: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if distractor_specs is None:
            # Default: one red ball distractor (matches Unity demo).
            distractor_specs = [("red", "ball")]
        for c, t in distractor_specs:
            if c not in COLOR_NAME_TO_IDX:
                raise ValueError(f"unknown distractor color {c!r}")
            if t not in OBJECT_NAME_TO_TYPE:
                raise ValueError(f"unknown distractor type {t!r}")
        self._distractor_specs = distractor_specs

        self._max_steps = max_steps
        self._forward_step = forward_step
        self._reach_threshold = reach_threshold
        self._plane_extent = plane_extent
        self._step_penalty = step_penalty
        self._randomize_target_color = randomize_target_color
        self._target_color = target_color
        self._target_type = target_type

        self._rng = np.random.default_rng(seed)

        self._obs_scale = obs_scale
        # Adapter does the egocentric rendering. We rebuild it every reset
        # because target color is allowed to vary across episodes.
        self._adapter: Unity2DAdapter = self._make_adapter(target_color, target_type)

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0.0, high=1.0, shape=(3, 7, 7), dtype=np.float32),
                "direction": spaces.Discrete(4),
                "mission": spaces.Text(max_length=64),
            }
        )

        # Episode state — initialized in reset().
        self._agent_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self._target_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self._distractor_positions: list[np.ndarray] = []
        self._distractor_type_ids: list[int] = []
        self._distractor_color_ids: list[int] = []
        self._step_count = 0
        self._mission_str = ""

    # ------------------------------------------------------------------
    # Adapter management
    # ------------------------------------------------------------------
    def _make_adapter(self, color: str, ttype: str) -> Unity2DAdapter:
        return Unity2DAdapter(
            target_color=color,
            target_type=ttype,
            obs_scale=self._obs_scale,
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Optionally randomize target color so the model can't just memorize
        # one color slot. For Phase 1 of continual learning we keep the
        # target fixed; PR2 of this fine-tune can flip this on.
        if self._randomize_target_color:
            target_color = _COLOR_NAMES[int(self._rng.integers(0, len(_COLOR_NAMES)))]
        else:
            target_color = self._target_color

        self._adapter = self._make_adapter(target_color, self._target_type)
        self._adapter.heading = 0
        self._mission_str = f"go to the {target_color} {self._target_type}"

        # Agent at origin.
        self._agent_pos = np.array([0.0, 0.0], dtype=np.float32)

        # Target at random position, away from origin.
        self._target_pos = self._sample_position(min_dist_from=[self._agent_pos], min_dist=1.5)

        # Distractors: random colors and positions, away from target.
        self._distractor_positions = []
        self._distractor_type_ids = []
        self._distractor_color_ids = []
        forbidden = [self._agent_pos, self._target_pos]
        for c, t in self._distractor_specs:
            # If target was randomized to match a distractor's color, skip
            # that distractor for this episode (would be ambiguous).
            if c == target_color and t == self._target_type:
                continue
            pos = self._sample_position(min_dist_from=forbidden, min_dist=1.5)
            self._distractor_positions.append(pos)
            self._distractor_type_ids.append(OBJECT_NAME_TO_TYPE[t])
            self._distractor_color_ids.append(COLOR_NAME_TO_IDX[c])
            forbidden.append(pos)

        self._step_count = 0
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        action = int(action)
        # 0 = turn_left, 1 = turn_right, 2 = forward; 3-6 = no-op.
        if action == 0:
            self._adapter.heading = (self._adapter.heading - 1) % 4
        elif action == 1:
            self._adapter.heading = (self._adapter.heading + 1) % 4
        elif action == 2:
            fwd = _FORWARD_VEC[self._adapter.heading]
            self._agent_pos = self._agent_pos + fwd * self._forward_step
            self._agent_pos = np.clip(
                self._agent_pos, -self._plane_extent, self._plane_extent
            ).astype(np.float32)

        self._step_count += 1

        dist_to_target = float(np.linalg.norm(self._agent_pos - self._target_pos))
        reached_target = dist_to_target < self._reach_threshold

        # Anti-distractor shaping: small penalty if agent gets too close
        # to any wrong-colored object. Encourages discrimination during
        # fine-tuning. (Removed/zeroed for the final demo eval pass.)
        distractor_penalty = 0.0
        approached_distractor = False
        for d_pos in self._distractor_positions:
            if float(np.linalg.norm(self._agent_pos - d_pos)) < self._reach_threshold:
                approached_distractor = True
                distractor_penalty -= 0.25
                break

        if reached_target:
            reward = 1.0
            terminated = True
        else:
            reward = -self._step_penalty + distractor_penalty
            terminated = False
        truncated = self._step_count >= self._max_steps

        info = self._info()
        info["reached_target"] = reached_target
        info["approached_distractor"] = approached_distractor
        info["dist_to_target"] = dist_to_target

        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _sample_position(
        self, min_dist_from: list[np.ndarray], min_dist: float
    ) -> np.ndarray:
        for _ in range(50):
            pos = self._rng.uniform(
                -self._plane_extent + 0.5, self._plane_extent - 0.5, size=2
            ).astype(np.float32)
            if all(float(np.linalg.norm(pos - p)) >= min_dist for p in min_dist_from):
                return pos
        # Fallback after 50 rejections — return the last sample.
        return pos

    def _obs(self) -> dict:
        scene: list[tuple[int, int, tuple[float, float]]] = [
            (
                self._adapter.target_type_id,
                self._adapter.target_color_id,
                (float(self._target_pos[0]), float(self._target_pos[1])),
            )
        ]
        for type_id, color_id, pos in zip(
            self._distractor_type_ids,
            self._distractor_color_ids,
            self._distractor_positions,
        ):
            scene.append((type_id, color_id, (float(pos[0]), float(pos[1]))))
        image = self._adapter.render_obs_multi(
            (float(self._agent_pos[0]), float(self._agent_pos[1])),
            scene,
        )
        return {
            "image": image.astype(np.float32),
            "direction": int(self._adapter.heading),
            "mission": self._mission_str,
        }

    def _info(self) -> dict:
        return {
            "agent_pos": self._agent_pos.copy(),
            "target_pos": self._target_pos.copy(),
            "distractor_positions": [p.copy() for p in self._distractor_positions],
            "heading": int(self._adapter.heading),
            "step": self._step_count,
            "mission": self._mission_str,
        }
