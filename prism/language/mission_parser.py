"""Parse BabyAI mission strings into structured goal specs.

This is intentionally a tiny rule-based parser, not a learned text encoder.
The thesis behind PRISM is that language understanding emerges from
*operators bound to predicted state transitions*, not from imitating a
corpus. So at this layer the parser is dumb on purpose: its only job is to
turn the small, regular vocabulary BabyAI uses into a goal predicate that
the planner can target.

BabyAI mission templates we care about (Phases 0–4 walk through these):

  "go to (the|a) <color> <object>"               → at(agent, <color> <object>)
  "pick up (the|a) <color> <object>"             → holding(agent, <color> <object>)
  "open (the) <color> door"                      → open(<color> door)
  "put (the|a) <color> <object> next to ..."     → near(<obj1>, <obj2>)  [Phase 4+]

For Phase 2 v0 we only need the first two — they're enough to test
operator-grounded action selection on GoTo and PickUp levels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from prism.perception.slots import COLOR_NAME_TO_IDX, OBJECT_NAME_TO_TYPE


# Each template captures (predicate, color, object_type) groups.
# Order matters: more specific patterns first.
_TEMPLATES: tuple[tuple[str, re.Pattern], ...] = (
    (
        "holding",
        re.compile(
            r"^pick up (?:the|a|an)?\s*(?P<color>red|green|blue|purple|yellow|grey)?\s*"
            r"(?P<obj>door|key|ball|box)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "open",
        re.compile(
            r"^open (?:the)?\s*(?P<color>red|green|blue|purple|yellow|grey)?\s*"
            r"door\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "at",
        re.compile(
            r"^go to (?:the|a|an)?\s*(?P<color>red|green|blue|purple|yellow|grey)?\s*"
            r"(?P<obj>door|key|ball|box)\s*$",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class GoalSpec:
    """A parsed BabyAI mission.

    Fields:
        predicate:  one of "at" / "holding" / "open"
        color_id:   0..5 or None (None = "any color")
        type_id:    minigrid object type id (door=4, key=5, ball=6, box=7),
                    fixed for "open" goals (door).
        raw:        the original mission string, kept for debugging.
    """

    predicate: str
    color_id: int | None
    type_id: int
    raw: str


def parse_mission(mission: str) -> GoalSpec | None:
    """Return a GoalSpec or None if the mission doesn't match a known template.

    Returning None is preferable to raising — we want the upstream agent loop
    to be able to fall back gracefully (e.g. on Phase 4+ compositional levels
    where the full template set is needed).
    """
    text = mission.strip().lower()
    for predicate, pattern in _TEMPLATES:
        m = pattern.match(text)
        if not m:
            continue
        color = m.group("color") if "color" in m.groupdict() else None
        if predicate == "open":
            obj = "door"
        else:
            obj = m.group("obj")
        return GoalSpec(
            predicate=predicate,
            color_id=COLOR_NAME_TO_IDX[color] if color else None,
            type_id=OBJECT_NAME_TO_TYPE[obj],
            raw=mission,
        )
    return None
