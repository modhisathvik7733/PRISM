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
