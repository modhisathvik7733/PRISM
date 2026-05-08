"""The first end-to-end PRISM agent.

Pipeline at every timestep:

    obs                       (raw 7x7x3 partial view)
      │
      ▼
    encode(obs) ──────►  z_t  (frozen JEPA encoder)
      │
      ▼  for each candidate action a in {0..n_actions-1}:
    jepa.predict(z_t, a) ──►  ẑ_{t+1}     (latent dynamics)
      │
      ▼
    probe(ẑ_{t+1})            ──►  predicate logits over (predicate, type, color)
      │
      ▼
    score(a) = Σ wᵢ · σ(logitᵢ) for goal-relevant predicates i
      │
      ▼
    a* = argmax score                     (action selection)

No learned policy. No reward. The agent acts because rolling the world
forward in its own latent and reading out predicates says "this action gets
me closer to the goal predicate." That's the whole point of the architecture.

Goal-predicate weighting (curriculum-style):
  For "go to <color> <type>" the agent doesn't care equally about all
  predicates — there's a natural ordering of how-close-to-done you are:
    visible  < facing  < near  < adjacent
  We weight them in increasing order so the score has a coarse-to-fine
  gradient: out-of-view → in-view → in-front → close → done. This gives
  signal even when the agent is far from the goal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prism.language.mission_parser import GoalSpec, parse_mission
from prism.models.jepa import JepaWorldModel
from prism.perception.predicates import predicate_index
from prism.perception.slots import COLOR_NAME_TO_IDX, OBJECT_NAME_TO_TYPE


# Default weights: each finer-grained predicate is worth more than the
# previous coarser one. Magnitudes matter less than the ordering.
GOAL_PREDICATE_WEIGHTS: dict[str, float] = {
    "visible": 1.0,
    "facing": 2.0,
    "near": 4.0,
    "adjacent": 8.0,
}


@dataclass(frozen=True)
class WeightedGoal:
    """A goal predicate with its weight in the action-selection score."""

    name: str          # one of the keys of GOAL_PREDICATE_WEIGHTS
    type_id: int       # minigrid object type id
    color_id: int      # minigrid color id
    weight: float
    flat_index: int    # position in the 96-d predicate vector


def goal_predicates_for_mission(mission: str) -> list[WeightedGoal] | None:
    """Map a BabyAI mission string to the set of weighted goal predicates the
    agent should try to maximize.

    Returns None if the mission can't be parsed (e.g. compositional templates
    we don't yet support). Caller can fall back to random or no-op behavior.
    """
    spec: GoalSpec | None = parse_mission(mission)
    if spec is None:
        return None

    # GoToLocal / PickUp / Open all reduce to "be near / adjacent to the
    # target object" at the agent-action level. The probe doesn't (yet)
    # readout `holding` or `door-open`, so we treat all of them with the
    # same spatial-progress curriculum for v0.
    color = spec.color_id
    if color is None:
        # "any color" — for v0 we pick red as a placeholder. Compositional
        # mission handling lives in Phase 4.
        color = COLOR_NAME_TO_IDX["red"]

    type_id = spec.type_id
    return [
        WeightedGoal(
            name=name,
            type_id=type_id,
            color_id=color,
            weight=w,
            flat_index=predicate_index(name, type_id, color),
        )
        for name, w in GOAL_PREDICATE_WEIGHTS.items()
    ]


class GroundedAgent:
    """JEPA + predicate-probe action selector.

    The probe is the JEPA's own `aux_predicate_head` (trained jointly during
    JEPA training). If your checkpoint was trained without aux loss, pass an
    external probe via `probe`.
    """

    def __init__(
        self,
        jepa: JepaWorldModel,
        device: torch.device,
        *,
        probe: torch.nn.Module | None = None,
        horizon: int = 1,
    ):
        self.jepa = jepa
        self.device = device
        self.horizon = horizon

        # Resolve which probe to use.
        if probe is not None:
            self.probe = probe
        elif jepa.aux_predicate_head is not None:
            self.probe = jepa.aux_predicate_head
        else:
            raise ValueError(
                "no probe available — pass `probe=...` or train the JEPA with "
                "aux_predicate_weight > 0 so it has an internal predicate head"
            )

        self.jepa.eval()
        self.probe.eval()
        self.n_actions = jepa.cfg.n_actions

    @torch.no_grad()
    def select_action(
        self,
        obs: torch.Tensor,            # (3, 7, 7) float32 normalized
        goal_preds: list[WeightedGoal],
    ) -> tuple[int, dict[str, float]]:
        """Pick an action by rolling each candidate forward and scoring its
        predicted next-latent under the goal predicates.

        Returns (action, info) where info has the per-action scores for
        debugging / logging.
        """
        obs_b = obs.unsqueeze(0).to(self.device)      # (1, 3, 7, 7)
        z_t = self.jepa.encode(obs_b)                 # (1, embed_dim)

        # Replicate z_t once per candidate action.
        z_t_rep = z_t.expand(self.n_actions, -1)
        actions = torch.arange(self.n_actions, device=self.device, dtype=torch.long)

        z_next = self.jepa.predict(z_t_rep, actions)  # (n_actions, embed_dim)

        # Optional multi-step rollout under random follow-up actions.
        # horizon=1 (default) just scores ẑ_{t+1} directly.
        if self.horizon > 1:
            for _ in range(self.horizon - 1):
                rand_a = torch.randint(
                    0, self.n_actions, (self.n_actions,), device=self.device
                )
                z_next = self.jepa.predict(z_next, rand_a)

        pred_logits = self.probe(z_next)              # (n_actions, 96)
        pred_probs = torch.sigmoid(pred_logits)

        # Score each candidate action.
        scores = torch.zeros(self.n_actions, device=self.device)
        for g in goal_preds:
            scores = scores + g.weight * pred_probs[:, g.flat_index]

        action = int(scores.argmax().item())
        info = {f"score_a{i}": float(scores[i].item()) for i in range(self.n_actions)}
        info["chosen"] = action
        return action, info
