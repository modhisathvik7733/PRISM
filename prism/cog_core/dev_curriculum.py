"""DevelopmentalCurriculum — strict child→adult stage progression for
JEPA training.

Implements your principle: "treat the model as a person who knows
nothing, decide what it should learn FIRST, then add concepts that
build on it." Each developmental stage trains the JEPA on a SPECIFIC
environment that exercises ONE concept; we only graduate to the next
stage when competence on the current stage is demonstrated.

Default stages (BabyAI-only — can be extended):

    0a   BabyAI-OneRoomS8-v0    movement + walls in tiny empty room
                                 → learns: action consequences, position
    0b   BabyAI-GoToObj-v0      single salient object in small room
                                 → learns: object permanence, persistence
    0c   BabyAI-GoToLocal-v0    few objects in small room
                                 → learns: object discrimination, multi-entity scenes
    0d   BabyAI-OneRoomS16-v0   large room (more spatial complexity)
                                 → learns: longer-range structure, scaling

Transition gate: a stage is "passed" when the JEPA's 1-step latent
prediction cosine similarity on held-out rollouts of THAT stage's env
exceeds the stage's `transition_cos` threshold AND we've trained for
at least `min_steps`. We force-advance after `max_steps` even if
under-converged so a single stuck stage doesn't block the whole run.

Why these specific envs / order:
- Each stage adds ONE element of complexity over the prior
  (movement → 1 object → many objects → larger world)
- The JEPA carries weights forward stage-to-stage (transfer, not restart)
- Transition gates are objective and measurable, not hand-picked
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DevStage:
    """One developmental stage in the curriculum."""
    name: str                            # "0a", "0b", "0c", "0d"
    env_id: str                          # gym env id used for both training data and eval
    description: str                     # human-readable concept being taught
    min_steps: int                       # don't transition before this many JEPA optimizer steps
    max_steps: int                       # force-transition after this many even if under-converged
    transition_cos: float                # 1-step latent cosine sim threshold to graduate

    def __str__(self) -> str:
        return f"Stage {self.name} ({self.env_id}): {self.description}"


DEFAULT_STAGES: list[DevStage] = [
    DevStage(
        name="0a",
        env_id="BabyAI-OneRoomS8-v0",
        description="basic movement + walls in 8×8 empty single-room",
        min_steps=2000,
        max_steps=10000,
        transition_cos=0.90,
    ),
    DevStage(
        name="0b",
        env_id="BabyAI-GoToObj-v0",
        description="single-object permanence in small room",
        min_steps=5000,
        max_steps=30000,
        transition_cos=0.92,
    ),
    DevStage(
        name="0c",
        env_id="BabyAI-GoToLocal-v0",
        description="multiple objects + selection in small room",
        min_steps=10000,
        max_steps=50000,
        transition_cos=0.93,
    ),
    DevStage(
        name="0d",
        env_id="BabyAI-OneRoomS16-v0",
        description="larger room (16×16) with objects",
        min_steps=15000,
        max_steps=80000,
        transition_cos=0.93,
    ),
]


@dataclass
class StageTransition:
    """Recorded when the curriculum advances to the next stage."""
    from_stage: str
    to_stage: str
    at_step: int
    cosine_at_transition: float
    reason: str                          # "competence_reached" or "max_steps_hit"


@dataclass
class DevelopmentalCurriculum:
    """Tracks current stage + decides when to advance.

    Usage in training loop:
        curr = DevelopmentalCurriculum(DEFAULT_STAGES)
        for step in range(total_steps):
            stage = curr.current_stage()
            # collect data + train JEPA on stage.env_id ...
            if step % EVAL_EVERY == 0:
                cos = measure_cosine_on_held_out(stage.env_id, jepa)
                advanced = curr.maybe_advance(step, cos)
                if advanced and curr.is_done():
                    break
    """

    stages: list[DevStage]
    current_idx: int = 0
    transitions: list[StageTransition] = field(default_factory=list)
    # Per-stage tally of how many optimizer steps the JEPA has spent here.
    stage_step_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for s in self.stages:
            self.stage_step_counts.setdefault(s.name, 0)

    def current_stage(self) -> DevStage:
        return self.stages[self.current_idx]

    def is_done(self) -> bool:
        """True after we have advanced past the last stage."""
        return self.current_idx >= len(self.stages)

    def increment_step(self) -> None:
        """Call once per optimizer step to track per-stage budget."""
        if self.is_done():
            return
        s = self.current_stage()
        self.stage_step_counts[s.name] = self.stage_step_counts.get(s.name, 0) + 1

    def maybe_advance(self, global_step: int, recent_cosine: float) -> Optional[StageTransition]:
        """Decide whether to graduate to the next stage. Returns the
        StageTransition if advanced, None otherwise."""
        if self.is_done():
            return None
        s = self.current_stage()
        steps_in_stage = self.stage_step_counts.get(s.name, 0)

        if steps_in_stage < s.min_steps:
            return None

        passed_competence = recent_cosine >= s.transition_cos
        forced_by_budget = steps_in_stage >= s.max_steps

        if not (passed_competence or forced_by_budget):
            return None

        reason = "competence_reached" if passed_competence else "max_steps_hit"
        from_name = s.name
        self.current_idx += 1
        to_name = self.stages[self.current_idx].name if not self.is_done() else "DONE"
        t = StageTransition(
            from_stage=from_name,
            to_stage=to_name,
            at_step=global_step,
            cosine_at_transition=float(recent_cosine),
            reason=reason,
        )
        self.transitions.append(t)
        return t

    def summary(self) -> dict:
        """Snapshot for logging / saving in checkpoints."""
        return {
            "current_idx": self.current_idx,
            "current_stage_name": (self.current_stage().name
                                   if not self.is_done() else "DONE"),
            "stage_step_counts": dict(self.stage_step_counts),
            "transitions": [
                {
                    "from": t.from_stage, "to": t.to_stage,
                    "at_step": t.at_step,
                    "cosine_at_transition": t.cosine_at_transition,
                    "reason": t.reason,
                }
                for t in self.transitions
            ],
            "stages_total": len(self.stages),
            "stages_completed": len(self.transitions),
        }
