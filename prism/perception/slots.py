"""Extract typed object slots from BabyAI's symbolic partial-view obs.

BabyAI's image obs is shape (H, W, 3) where each cell encodes
  channel 0: object type id   (per minigrid OBJECT_TO_IDX)
  channel 1: color id         (per minigrid COLOR_TO_IDX)
  channel 2: state            (door open/closed/locked, etc.)

We are wrapping the obs in `_encode_image` (CHW + per-channel normalization),
so on raw BabyAI obs the values are integers; on our wrapped obs they are
floats in roughly [0, 1] after dividing by per-channel max. This module
operates on the *raw* uint8 obs from the underlying env, NOT the wrapped one.
For the wrapped obs use `extract_slots_from_normalized`.

Why slots:
  Operators need addressable objects. "go to the red ball" must resolve to a
  specific object. Slot extraction turns the cell grid into a list of
  (type, color, position) tuples that the planner / probes can index.

The agent itself is implicit: it always sits at (W//2, H-1) in the partial
view and faces up (so smaller y = "in front of the agent").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# minigrid OBJECT_TO_IDX (selected — only the four BabyAI manipulables + door).
# Source: minigrid/core/constants.py
TYPE_DOOR = 4
TYPE_KEY = 5
TYPE_BALL = 6
TYPE_BOX = 7
# We exclude wall/floor/empty — operators don't act on them as targets.
OBJECT_TYPES: tuple[int, ...] = (TYPE_DOOR, TYPE_KEY, TYPE_BALL, TYPE_BOX)
OBJECT_TYPE_NAMES = {
    TYPE_DOOR: "door",
    TYPE_KEY: "key",
    TYPE_BALL: "ball",
    TYPE_BOX: "box",
}
OBJECT_NAME_TO_TYPE = {v: k for k, v in OBJECT_TYPE_NAMES.items()}

# minigrid COLOR_TO_IDX
COLOR_NAMES = {
    0: "red",
    1: "green",
    2: "blue",
    3: "purple",
    4: "yellow",
    5: "grey",
}
COLOR_NAME_TO_IDX = {v: k for k, v in COLOR_NAMES.items()}
NUM_COLORS = 6
NUM_TYPES = len(OBJECT_TYPES)

# Agent's fixed position in the 7x7 partial view (BabyAI default).
AGENT_VIEW_SIZE = 7
AGENT_POS: tuple[int, int] = (AGENT_VIEW_SIZE // 2, AGENT_VIEW_SIZE - 1)  # (x=3, y=6)


@dataclass(frozen=True)
class Slot:
    """A single visible object in the partial view."""

    type_id: int    # minigrid object type id
    color_id: int   # minigrid color id
    x: int          # column in partial view, 0..6
    y: int          # row in partial view, 0..6 — y=6 is agent row, y=0 is far ahead

    @property
    def type_name(self) -> str:
        return OBJECT_TYPE_NAMES.get(self.type_id, f"type{self.type_id}")

    @property
    def color_name(self) -> str:
        return COLOR_NAMES.get(self.color_id, f"color{self.color_id}")

    @property
    def label(self) -> str:
        return f"{self.color_name} {self.type_name}"


def extract_slots(image_hwc_uint8: np.ndarray) -> list[Slot]:
    """Parse raw uint8 BabyAI image (H, W, 3) → list of Slot.

    Args:
        image_hwc_uint8: shape (H, W, 3) with integer codes in each channel.
                         (i.e. the un-normalized obs from MiniGrid before our
                         _encode_image wrapper.)
    """
    if image_hwc_uint8.ndim != 3 or image_hwc_uint8.shape[2] != 3:
        raise ValueError(
            f"expected (H, W, 3) raw obs, got shape {image_hwc_uint8.shape}"
        )
    H, W, _ = image_hwc_uint8.shape
    slots: list[Slot] = []
    for y in range(H):
        for x in range(W):
            t = int(image_hwc_uint8[y, x, 0])
            if t in OBJECT_TYPES:
                c = int(image_hwc_uint8[y, x, 1])
                slots.append(Slot(type_id=t, color_id=c, x=x, y=y))
    return slots


def extract_slots_from_normalized(image_chw_float: np.ndarray,
                                  channel_max: tuple[float, ...] = (11.0, 6.0, 4.0)
                                  ) -> list[Slot]:
    """Parse our wrapped (C, H, W) normalized obs back to slots.

    Inverts `_encode_image` from prism.envs.babyai. Useful when we only have
    the wrapped obs around (e.g. inside a vec-env replay buffer).
    """
    if image_chw_float.ndim != 3 or image_chw_float.shape[0] != 3:
        raise ValueError(
            f"expected (3, H, W) normalized obs, got shape {image_chw_float.shape}"
        )
    # Undo per-channel normalization and round to nearest integer.
    chw = np.asarray(image_chw_float, dtype=np.float32)
    cmax = np.array(channel_max, dtype=np.float32).reshape(3, 1, 1)
    raw = np.rint(chw * cmax).astype(np.int64)
    hwc = np.transpose(raw, (1, 2, 0))
    return extract_slots(hwc)
