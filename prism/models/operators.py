"""Seeded operator library — Phase 2.

Per the roadmap: do *not* attempt unsupervised operator discovery from day 1.
Start with 10–20 hand-defined primitives chosen to span:
  - the BabyAI verb space (`go to`, `pick up`, `open`, `put X near Y`),
  - and the physical-interaction core (move, touch, push, contain, transfer,
    break, increase, decrease).

Each operator has a typed signature: preconditions and effects expressed in
predicate space. The predicate space is a small, fixed set of relational
predicates over a typed object vocabulary.

Refinement / merging / discovery is layered ON TOP of this scaffold in later
sub-phases — it does not replace it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Object types the agent can reason about. BabyAI uses {key, ball, box, door}
# plus the agent itself; we keep it minimal but extensible.
OBJECT_TYPES: tuple[str, ...] = ("agent", "key", "ball", "box", "door", "wall", "loc")


# Relational predicates. `_(x, y)` means "predicate holds between x and y".
# `_(x)` means a unary predicate over x. `at(x, loc)` is the workhorse.
PREDICATES: tuple[str, ...] = (
    "at",        # at(obj, loc)
    "near",      # near(obj, obj)
    "holding",   # holding(agent, obj)
    "inside",    # inside(obj, container)
    "open",      # open(door)  — unary
    "broken",    # broken(obj) — unary
    "touching",  # touching(obj, obj)
)


@dataclass(frozen=True)
class OperatorSignature:
    """Symbolic operator: name, typed parameters, preconditions, effects.

    Preconditions and effects are stored as tuples of (predicate, *params)
    where each param is the *index* of a parameter in `params`, or a literal
    (e.g. `"agent"` to bind `self`).

    Effects use a leading sign: `("+", "at", 0, 1)` adds `at(p0, p1)`,
                                `("-", "at", 0, 1)` retracts it.
    """

    name: str
    params: tuple[str, ...]                                 # ordered param types
    preconditions: tuple[tuple, ...] = field(default_factory=tuple)
    effects: tuple[tuple, ...] = field(default_factory=tuple)


# ------------------------------------------------------------ seeded primitives

SEED_OPERATORS: tuple[OperatorSignature, ...] = (
    OperatorSignature(
        name="move",
        params=("agent", "loc"),
        preconditions=(),  # always available; failure handled by env
        effects=(("+", "at", 0, 1),),
    ),
    OperatorSignature(
        name="touch",
        params=("agent", "obj"),
        preconditions=(("at", 0, "near_p1"),),
        effects=(("+", "touching", 0, 1),),
    ),
    OperatorSignature(
        name="push",
        params=("agent", "obj", "loc"),
        preconditions=(("touching", 0, 1),),
        effects=(("+", "at", 1, 2),),
    ),
    OperatorSignature(
        name="pickup",
        params=("agent", "obj"),
        preconditions=(("touching", 0, 1), ("-", "holding", 0, "_any")),
        effects=(("+", "holding", 0, 1), ("-", "at", 1, "_anywhere")),
    ),
    OperatorSignature(
        name="drop",
        params=("agent", "obj", "loc"),
        preconditions=(("holding", 0, 1), ("at", 0, 2)),
        effects=(("-", "holding", 0, 1), ("+", "at", 1, 2)),
    ),
    OperatorSignature(
        name="open",
        params=("agent", "door"),
        preconditions=(("touching", 0, 1),),
        effects=(("+", "open", 1),),
    ),
    OperatorSignature(
        name="close",
        params=("agent", "door"),
        preconditions=(("touching", 0, 1), ("open", 1)),
        effects=(("-", "open", 1),),
    ),
    OperatorSignature(
        name="contain",
        params=("container", "obj"),
        # Static structural assertion: used at perception time, not as an action.
        preconditions=(("at", 0, "_anywhere"), ("at", 1, "_same_as_p0")),
        effects=(("+", "inside", 1, 0),),
    ),
    OperatorSignature(
        name="transfer",
        params=("agent", "obj", "from_container", "to_container"),
        preconditions=(("inside", 1, 2), ("touching", 0, 2), ("touching", 0, 3)),
        effects=(("-", "inside", 1, 2), ("+", "inside", 1, 3)),
    ),
    OperatorSignature(
        name="break",
        params=("agent", "obj"),
        preconditions=(("touching", 0, 1),),
        effects=(("+", "broken", 1),),
    ),
    OperatorSignature(
        name="increase",
        # Abstract operator: increases a scalar attribute (placeholder, used in
        # later phases when we extend env to non-symbolic state). Kept here so
        # the seed library spans more than spatial verbs.
        params=("attr",),
        effects=(),
    ),
    OperatorSignature(
        name="decrease",
        params=("attr",),
        effects=(),
    ),
)


# ------------------------------------------------------- BabyAI instruction map

# Phase 2 binds BabyAI's `mission` strings to operator templates. The mapping
# is intentionally crude — Phase 4's curriculum walk will extend it.
INSTRUCTION_TO_OPERATOR: dict[str, str] = {
    "go to": "move",
    "pick up": "pickup",
    "open": "open",
    "put": "drop",
    "put next to": "drop",
}


def operator_by_name(name: str) -> OperatorSignature:
    for op in SEED_OPERATORS:
        if op.name == name:
            return op
    raise KeyError(f"unknown operator: {name}")
