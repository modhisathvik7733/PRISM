"""Extended mission parser for the generalization fork.

Wraps `prism.language.mission_parser.parse_mission` (existing, untouched)
and adds tolerant patterns for the BabyAI-Open-v0 mission family. Phase 0
zero-shot eval showed 75% of Open episodes failed to parse — the original
regex only matches "open [the] [color] door". Real Open-v0 missions look
like:

  "open a door"                        ← article "a"/"an" not handled
  "open the locked door"               ← extra adjective "locked"
  "open the red door"                  ← already handled
  "open the door"                      ← already handled (color is optional)
  "open the X door, then open the Y"   ← compositional; we take the first
                                          target — partial credit but lets
                                          the agent at least navigate

We deliberately keep the v1.3 parser bit-for-bit identical (the v1.3
GoToLocal capstone numbers must remain reproducible) and add a fallback
chain here.
"""

from __future__ import annotations

import re

from prism.agents.grounded_agent import (
    GOAL_PREDICATE_WEIGHTS,
    WeightedGoal,
)
from prism.language.mission_parser import GoalSpec, parse_mission as _v1_parse
from prism.perception.predicates import predicate_index
from prism.perception.slots import COLOR_NAME_TO_IDX, OBJECT_NAME_TO_TYPE


# Tolerant Open patterns. Matched IN ORDER, only after the v1 parser fails.
# All three match a single door target — compositional missions get the
# FIRST clause (so the agent at least makes progress).
_COLOR = r"red|green|blue|purple|yellow|grey"
_OPEN_FALLBACKS: tuple[re.Pattern, ...] = (
    # "open the X door[,. then ...]"
    re.compile(
        rf"^open\s+(?:the|a|an)\s+(?:locked\s+)?(?P<color>{_COLOR})\s+door\b",
        re.IGNORECASE,
    ),
    # "open a/the door[,. then ...]"
    re.compile(
        r"^open\s+(?:the|a|an)\s+(?:locked\s+)?door\b",
        re.IGNORECASE,
    ),
    # bare "open <color> door"
    re.compile(
        rf"^open\s+(?P<color>{_COLOR})\s+door\b",
        re.IGNORECASE,
    ),
    # bare "open door"
    re.compile(
        r"^open\s+door\b",
        re.IGNORECASE,
    ),
)


def parse_mission_ext(mission: str) -> GoalSpec | None:
    """Try the v1 parser first; on miss, fall back to widened Open patterns.

    Returns None only when neither matches — same contract as v1, so callers
    can drop this in without changing their None-handling code paths.
    """
    spec = _v1_parse(mission)
    if spec is not None:
        return spec

    text = mission.strip().lower()
    for pat in _OPEN_FALLBACKS:
        m = pat.match(text)
        if not m:
            continue
        gd = m.groupdict()
        color_name = gd.get("color")
        return GoalSpec(
            predicate="open",
            color_id=COLOR_NAME_TO_IDX[color_name] if color_name else None,
            type_id=OBJECT_NAME_TO_TYPE["door"],
            raw=mission,
        )
    return None


def goal_predicates_for_mission_ext(
    mission: str,
) -> tuple[list[WeightedGoal], GoalSpec] | None:
    """Same contract as `prism.agents.goal_predicates_for_mission` but uses
    the extended parser. Replicates the v1 weight-curriculum construction
    so downstream code (memory teacher, BC collector) can drop this in
    without further changes.
    """
    spec = parse_mission_ext(mission)
    if spec is None:
        return None

    color = spec.color_id
    if color is None:
        # Match v1's fallback choice — keeps the predicate computation
        # deterministic across runs.
        color = COLOR_NAME_TO_IDX["red"]

    type_id = spec.type_id
    goals = [
        WeightedGoal(
            name=name,
            type_id=type_id,
            color_id=color,
            weight=w,
            flat_index=predicate_index(name, type_id, color),
        )
        for name, w in GOAL_PREDICATE_WEIGHTS.items()
    ]
    return goals, spec
