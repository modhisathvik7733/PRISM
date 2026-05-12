"""CurriculumEngine — sequences Stages with activation-based freezing.

NOTE: this is a stub. PR-6 implements the actual engine. PR-1 (this file)
provides the interface so other modules can reference it without circular
imports.

The engine's responsibilities (per plan resolutions 4, 5, 6):
  * At Stage 0 init: collect probe set with a random policy, persist to
    disk, hash into substrate_config_hash.
  * Before each Stage: call bank.expand(stage.expand_slots_before) with
    near-inactive init.
  * During each Stage: training driver accumulates per-slot activation
    history in the bank.
  * At each Stage transition: read bank.slot_activation_history(),
    compute freeze set as {slots with activation_mass/steps_in_stage >
    threshold}, call bank.freeze_slots(freeze_set).
  * Stage transitions are SYNCHRONOUS; no path through ConceptManager
    or async naming gates the freeze decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism.curriculum.stage import Stage


@dataclass
class CurriculumEngineConfig:
    """Substrate-locked configuration for the curriculum engine.

    These values are part of substrate_config_hash; changing them across
    domains is forbidden by the locked-hyperparameter invariant.
    """

    activation_freeze_threshold: float = 0.005
    """Slots with cumulative activation_mass / steps_in_stage above this
    enter the freeze set at stage transition. Default is provisional;
    Phase C must sweep {0.001, 0.005, 0.01, 0.05} and lock the result."""

    warmup_steps: int = 5000
    """Gradient steps during which newly-expanded slots are excluded from
    the retrieval softmax denominator (masked-softmax expansion)."""

    cold_anneal_steps: int = 5000
    """Gradient steps over which new slots' temperature β anneals from
    0.1 to 1.0 after warmup completes. Linear schedule."""

    probe_set_size: int = 5000
    """Number of (obs, mission) pairs collected with a random policy at
    Stage 0 and persisted as the immutable E4 probe set."""


class CurriculumEngine:
    """Sequencer for developmental Stages with activation-based freezing.

    PR-1: skeleton only. PR-6 implements the actual engine.
    """

    def __init__(
        self,
        stages: list["Stage"],
        config: CurriculumEngineConfig | None = None,
    ):
        self.stages = stages
        self.config = config or CurriculumEngineConfig()
        self._current_stage_idx: int = 0

    @property
    def current_stage(self) -> "Stage":
        return self.stages[self._current_stage_idx]

    def advance_stage(self) -> bool:
        """Advance to the next stage. Returns False if no more stages.

        NOT IMPLEMENTED in PR-1. PR-6 wires the activation-history read
        and the bank.freeze_slots call.
        """
        raise NotImplementedError(
            "CurriculumEngine.advance_stage is implemented in PR-6"
        )

    def collect_probe_set(self, *args, **kwargs) -> None:
        """Collect and persist the E4 probe set at Stage 0 init.

        NOT IMPLEMENTED in PR-1. PR-6 implements random-policy rollouts
        and persistence to runs/<run_name>/probe_set.pt with hash.
        """
        raise NotImplementedError(
            "CurriculumEngine.collect_probe_set is implemented in PR-6"
        )
