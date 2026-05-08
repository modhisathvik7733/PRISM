"""Compute ground-truth predicates from extracted slots.

Per-(type, color) predicates we care about for Phase 2:

  * visible(type, color)  — at least one (type, color) object in partial view
  * near(type, color)     — there's a (type, color) within manhattan dist ≤ 2
  * facing(type, color)   — there's a (type, color) in the agent's forward
                            arc (y < agent_y AND |dx| ≤ y-distance)
  * adjacent(type, color) — there's a (type, color) at manhattan dist == 1

Output is a flat float32 vector of length:
  NUM_TYPES * NUM_COLORS * NUM_PREDICATES = 4 * 6 * 4 = 96
indexed by `predicate_index(predicate, type, color)`.

These ground-truth labels train the predicate probe (linear head on JEPA
latent). At eval, probe accuracy tells us whether the JEPA latent linearly
encodes object-structured information — the prerequisite for grounded
operators.
"""

from __future__ import annotations

import numpy as np

from prism.perception.slots import (
    AGENT_POS,
    NUM_COLORS,
    NUM_TYPES,
    OBJECT_TYPES,
    Slot,
)

PREDICATE_NAMES: tuple[str, ...] = ("visible", "near", "facing", "adjacent")
NUM_PREDICATES = len(PREDICATE_NAMES)
NUM_TYPE_COLOR_PAIRS = NUM_TYPES * NUM_COLORS  # 24
PREDICATE_VECTOR_DIM = NUM_TYPE_COLOR_PAIRS * NUM_PREDICATES  # 96

# Continuous per-(type, color) distance vector concatenated after the binary
# predicates. Provides a smooth gradient for the agent ("forward gets me 1
# cell closer") that the binary predicates cannot — the binary set only flips
# at threshold crossings (visible/near/adjacent), so for "target visible but
# 5 cells away" no predicate changes per forward step and the agent has no
# reason to prefer forward over turn.
DISTANCE_VECTOR_DIM = NUM_TYPE_COLOR_PAIRS  # 24
AUGMENTED_VECTOR_DIM = PREDICATE_VECTOR_DIM + DISTANCE_VECTOR_DIM  # 120
# Max manhattan distance from any cell in 7x7 view to the agent at (3, 6):
# corner (6, 0) → |6-3| + |0-6| = 9. Use as the normalizer so distance ∈ [0, 1].
MAX_VIEW_MANHATTAN = 9.0

NEAR_THRESHOLD = 2
ADJACENT_THRESHOLD = 1


def type_color_index(type_id: int, color_id: int) -> int:
    """(type_id, color_id) → flat index into the (type, color) vocabulary."""
    try:
        ti = OBJECT_TYPES.index(type_id)
    except ValueError as e:
        raise ValueError(f"unknown type_id {type_id} (want one of {OBJECT_TYPES})") from e
    if not (0 <= color_id < NUM_COLORS):
        raise ValueError(f"color_id {color_id} out of range")
    return ti * NUM_COLORS + color_id


def predicate_index(predicate: str, type_id: int, color_id: int) -> int:
    """(predicate_name, type_id, color_id) → flat index into the 96-d vector."""
    try:
        pi = PREDICATE_NAMES.index(predicate)
    except ValueError as e:
        raise ValueError(
            f"unknown predicate {predicate!r} (want one of {PREDICATE_NAMES})"
        ) from e
    return pi * NUM_TYPE_COLOR_PAIRS + type_color_index(type_id, color_id)


def distance_index(type_id: int, color_id: int) -> int:
    """(type_id, color_id) → flat index into the 24-d distance block of the
    augmented 120-d vector. The distance block sits *after* the 96 binary
    predicates: idx = PREDICATE_VECTOR_DIM + type_color_index.
    """
    return PREDICATE_VECTOR_DIM + type_color_index(type_id, color_id)


def _is_facing(slot: Slot) -> bool:
    """Is the slot in the agent's forward arc (in agent-centric partial view)?

    Agent is at (3, 6) facing up → forward arc is y < 6 with widening x range.
    """
    ax, ay = AGENT_POS
    if slot.y >= ay:  # behind or beside agent
        return False
    forward_distance = ay - slot.y  # 1..6
    lateral = abs(slot.x - ax)
    return lateral <= forward_distance


def compute_predicates(slots: list[Slot]) -> np.ndarray:
    """Compute the 96-d binary predicate vector from a list of slots."""
    out = np.zeros(PREDICATE_VECTOR_DIM, dtype=np.float32)
    ax, ay = AGENT_POS

    for s in slots:
        if s.type_id not in OBJECT_TYPES:
            continue
        manhattan = abs(s.x - ax) + abs(s.y - ay)

        # visible (any slot we extracted is by definition visible)
        out[predicate_index("visible", s.type_id, s.color_id)] = 1.0

        # near: manhattan ≤ NEAR_THRESHOLD
        if manhattan <= NEAR_THRESHOLD:
            out[predicate_index("near", s.type_id, s.color_id)] = 1.0

        # adjacent: manhattan == ADJACENT_THRESHOLD
        if manhattan == ADJACENT_THRESHOLD:
            out[predicate_index("adjacent", s.type_id, s.color_id)] = 1.0

        # facing: in forward arc
        if _is_facing(s):
            out[predicate_index("facing", s.type_id, s.color_id)] = 1.0

    return out


def compute_distances(slots: list[Slot]) -> np.ndarray:
    """Per-(type, color) min normalized manhattan distance, 1.0 if not visible.

    Returns a flat (24,) float32 vector. Distance is normalized by
    `MAX_VIEW_MANHATTAN` so all values are in [0, 1]. Missing pairs default
    to 1.0 (max). The agent minimizes this for the goal pair.
    """
    out = np.ones(NUM_TYPE_COLOR_PAIRS, dtype=np.float32)
    ax, ay = AGENT_POS
    for s in slots:
        if s.type_id not in OBJECT_TYPES:
            continue
        d = abs(s.x - ax) + abs(s.y - ay)
        idx = type_color_index(s.type_id, s.color_id)
        d_norm = min(1.0, d / MAX_VIEW_MANHATTAN)
        if d_norm < out[idx]:
            out[idx] = d_norm
    return out


def compute_augmented_predicates(slots: list[Slot]) -> np.ndarray:
    """Concatenated [96 binary predicates ‖ 24 normalized distances] = (120,).

    This is the JEPA's aux-head training target when distance-augmented
    training is enabled. The first 96 dims supervise the binary head with
    BCE; the last 24 supervise the distance head with MSE.
    """
    return np.concatenate([compute_predicates(slots), compute_distances(slots)])


def predicate_summary(vec: np.ndarray) -> list[tuple[str, str, str]]:
    """Decode a (possibly-noisy) 96-d vector into a list of (predicate, color, type)
    triples for predicates above 0.5. Useful for debugging / qualitative output.
    """
    from prism.perception.slots import COLOR_NAMES, OBJECT_TYPE_NAMES
    out = []
    for pi, pname in enumerate(PREDICATE_NAMES):
        for ti_idx, type_id in enumerate(OBJECT_TYPES):
            for color_id in range(NUM_COLORS):
                idx = pi * NUM_TYPE_COLOR_PAIRS + ti_idx * NUM_COLORS + color_id
                if vec[idx] > 0.5:
                    out.append(
                        (pname, COLOR_NAMES[color_id], OBJECT_TYPE_NAMES[type_id])
                    )
    return out
