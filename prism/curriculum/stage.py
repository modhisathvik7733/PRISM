"""Stage dataclass — one entry in a developmental curriculum.

A Stage declares its environment(s), success criteria, training budget,
and the freeze policy applied at stage exit. The Stage does not contain
training code itself — the CurriculumEngine drives training; Stages are
declarative.

Stages are constructed by experiment scripts or adapters (since stage
content is domain-specific). The Stage type itself is domain-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Stage:
    """One developmental curriculum stage.

    Fields:
        name: short identifier, used in logging and as part of
            substrate_config_hash provenance.
        env_factory: zero-arg callable returning a fresh env instance.
            Engine calls this once per parallel worker on stage entry.
        max_env_steps: hard cap on environment steps for this stage.
            Used both as a training budget and as a denominator for the
            activation_freeze_threshold computation.
        success_criterion: callable that takes recent training metrics
            and returns True when the stage's goal is met. If it never
            returns True, the stage runs until max_env_steps is hit.
        freeze_after: if True, the engine computes the activation-based
            freeze set at stage exit and applies it to the bank. If
            False, no freezing happens — used for the final stage or
            for ablation arms.
        expand_slots_before: number of new slots to add to the bank
            BEFORE this stage begins. 0 means use existing capacity.
            Slots are added with near-inactive init and masked-softmax
            warmup (resolution 5 in plan).
        extra: arbitrary stage-specific config the experiment may need
            (e.g. held-out (color, type) combos for E1 ordering ablation).
    """

    name: str
    env_factory: Callable[[], Any]
    max_env_steps: int
    success_criterion: Callable[[dict], bool] = field(
        default=lambda _metrics: False,
    )
    freeze_after: bool = True
    expand_slots_before: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
