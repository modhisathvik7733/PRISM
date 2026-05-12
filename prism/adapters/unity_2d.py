"""Unity 2D nav adapter — domain interface for the Unity demo (Day 2).

Bridges Unity's continuous (x, z) world to the BabyAI-shaped substrate
input: 7x7x3 partial-view obs + 24-d (type, color) mission one-hot +
7-action discrete output. The substrate weights stay unchanged from
BabyAI training; only this adapter layer differs from the BabyAI domain.

Per-episode state:
  - virtual heading ∈ {N=0, E=1, S=2, W=3}: updated by the substrate's
    turn_left / turn_right actions. Unity has no heading concept, so it
    is purely internal to this adapter.

Action remap (substrate 7-action → Unity 5-action):
  0 turn_left  → heading -= 1, send Unity stay (0)
  1 turn_right → heading += 1, send Unity stay (0)
  2 forward    → translate current heading → N/S/E/W Unity action
  3..6         → Unity stay (action masking should prevent these)

Unity action codes (from unity_demo/prism_server.py:14):
  0 stay, 1 N(+z), 2 S(-z), 3 E(+x), 4 W(-x)
"""

from __future__ import annotations

import numpy as np
import torch

from prism.perception.predicates import type_color_index
from prism.perception.slots import (
    AGENT_POS,
    AGENT_VIEW_SIZE,
    COLOR_NAME_TO_IDX,
    NUM_COLORS,
    NUM_TYPES,
    OBJECT_NAME_TO_TYPE,
)

# MiniGrid OBJECT_TO_IDX (the values we need).
_MG_EMPTY = 1
_MG_AGENT = 10

# Per-channel max for JEPA-normalized obs. Mirrors prism/envs/babyai.py:39
# (kept here rather than imported to avoid a circular import via gym).
_CHANNEL_MAX = np.array([11.0, 6.0, 4.0], dtype=np.float32).reshape(3, 1, 1)

# Heading 0..3 → (x, z) unit forward vector in Unity world.
_FORWARD_VEC = np.array(
    [
        [0.0, 1.0],   # N → +z
        [1.0, 0.0],   # E → +x
        [0.0, -1.0],  # S → -z
        [-1.0, 0.0],  # W → -x
    ],
    dtype=np.float32,
)
# Heading 0..3 → right vector (90° clockwise from forward, looking down).
_RIGHT_VEC = np.array(
    [
        [1.0, 0.0],   # N → right = +x
        [0.0, -1.0],  # E → right = -z
        [-1.0, 0.0],  # S → right = -x
        [0.0, 1.0],   # W → right = +z
    ],
    dtype=np.float32,
)

# heading → Unity action int when substrate says "forward".
_HEADING_TO_UNITY = {0: 1, 1: 3, 2: 2, 3: 4}  # N, E, S, W

# Substrate (BabyAI) action ids.
_ACT_LEFT, _ACT_RIGHT, _ACT_FORWARD = 0, 1, 2

# Allowed actions for the "at" predicate (matches
# BabyAIAdapter.MISSION_ALLOWED_ACTIONS["at"]).
_ALLOWED_ACTIONS: tuple[int, ...] = (0, 1, 2)


class Unity2DAdapter:
    """Tiny inference-time adapter for the Unity 2D nav demo.

    Not a full `DomainAdapter` Protocol implementation — it doesn't own
    the JEPA encoder (the inference server reuses `BabyAIAdapter` for
    that). This class just handles:

      1. Obs synthesis: (agent_pos, target_pos, heading) → fake 7x7x3 obs.
      2. Mission one-hot for the fixed "go to the <color> <type>" goal.
      3. Action remap with heading bookkeeping.
      4. Logit masking to the "at"-predicate's allowed actions.

    One instance per Unity WebSocket connection. Call `reset()` at the
    start of each episode (Unity's `episode_done=true` flag).
    """

    def __init__(
        self,
        target_color: str = "green",
        target_type: str = "ball",
        n_actions: int = 7,
        view_size: int = AGENT_VIEW_SIZE,
        obs_scale: float = 2.0,
    ) -> None:
        if target_color not in COLOR_NAME_TO_IDX:
            raise ValueError(
                f"unknown target_color={target_color!r}; expected one of "
                f"{sorted(COLOR_NAME_TO_IDX)}"
            )
        if target_type not in OBJECT_NAME_TO_TYPE:
            raise ValueError(
                f"unknown target_type={target_type!r}; expected one of "
                f"{sorted(OBJECT_NAME_TO_TYPE)}"
            )
        if obs_scale <= 0:
            raise ValueError(f"obs_scale must be > 0; got {obs_scale}")
        self.target_color_id = COLOR_NAME_TO_IDX[target_color]
        self.target_type_id = OBJECT_NAME_TO_TYPE[target_type]
        self.n_actions = n_actions
        self.mission_dim = NUM_TYPES * NUM_COLORS  # 24
        self.view_size = view_size
        # Unity units per BabyAI grid cell. >1 = compressed view (target
        # stays in 7x7 window even at long distances; cost: lower spatial
        # resolution per cell).
        self.obs_scale = float(obs_scale)
        self.heading: int = 0

    def reset(self) -> None:
        """Call on episode boundary (Unity's `episode_done=true`)."""
        self.heading = 0

    # ------------------------------------------------------------------
    # Observation synthesis
    # ------------------------------------------------------------------
    def render_obs(
        self,
        agent_pos_xz: tuple[float, float],
        target_pos_xz: tuple[float, float],
    ) -> np.ndarray:
        """Single-object obs (backward-compatible wrapper)."""
        return self.render_obs_multi(
            agent_pos_xz,
            [(self.target_type_id, self.target_color_id, target_pos_xz)],
        )

    def render_obs_multi(
        self,
        agent_pos_xz: tuple[float, float],
        scene_objects: list[tuple[int, int, tuple[float, float]]],
    ) -> np.ndarray:
        """Build a (3, 7, 7) float32 obs in JEPA-normalized space.

        scene_objects: list of (type_id, color_id, (x, z)) entries. All
        objects whose grid coords fall inside the 7x7 egocentric view
        get rendered; off-view objects are silently dropped (matches
        BabyAI partial-observability).
        """
        view = np.zeros((self.view_size, self.view_size, 3), dtype=np.float32)
        view[..., 0] = _MG_EMPTY  # empty floor everywhere

        ax, ay = AGENT_POS
        view[ay, ax] = (_MG_AGENT, 0.0, 0.0)

        agent_world = np.asarray(agent_pos_xz, dtype=np.float32)
        fwd = _FORWARD_VEC[self.heading]
        right = _RIGHT_VEC[self.heading]

        for type_id, color_id, pos in scene_objects:
            delta = np.asarray(pos, dtype=np.float32) - agent_world
            # Compress world distance into grid cells via obs_scale so the
            # target stays in the 7x7 window even at long Unity distances.
            forward_dist = float(delta @ fwd) / self.obs_scale
            right_dist = float(delta @ right) / self.obs_scale
            gx = ax + int(round(right_dist))
            gy = ay - int(round(forward_dist))
            if 0 <= gx < self.view_size and 0 <= gy < self.view_size:
                # Don't overwrite the agent cell.
                if (gx, gy) == (ax, ay):
                    continue
                view[gy, gx] = (float(type_id), float(color_id), 0.0)

        chw = np.transpose(view, (2, 0, 1))
        return chw / _CHANNEL_MAX

    # ------------------------------------------------------------------
    # Mission
    # ------------------------------------------------------------------
    def mission_onehot_np(self) -> np.ndarray:
        v = np.zeros(self.mission_dim, dtype=np.float32)
        v[type_color_index(self.target_type_id, self.target_color_id)] = 1.0
        return v

    def mission_onehot(self, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(self.mission_onehot_np()).to(device)

    # ------------------------------------------------------------------
    # Action masking + remap
    # ------------------------------------------------------------------
    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask substrate logits to the 'at'-predicate's allowed actions.

        logits: (..., n_actions). Disallowed indices get -inf.
        """
        mask = torch.full_like(logits, float("-inf"))
        for a in _ALLOWED_ACTIONS:
            mask[..., a] = 0.0
        return logits + mask

    def map_action(self, substrate_action: int) -> int:
        """Substrate action [0..6] → Unity action [0..4]; mutates heading."""
        if substrate_action == _ACT_LEFT:
            self.heading = (self.heading - 1) % 4
            return 0
        if substrate_action == _ACT_RIGHT:
            self.heading = (self.heading + 1) % 4
            return 0
        if substrate_action == _ACT_FORWARD:
            return _HEADING_TO_UNITY[self.heading]
        return 0
