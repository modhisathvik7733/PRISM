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
from prism.perception.predicates import predicate_index, type_color_index
from prism.perception.slots import COLOR_NAME_TO_IDX


# MiniGrid action ids:
#   0 = turn left, 1 = turn right, 2 = forward,
#   3 = pickup,    4 = drop,        5 = toggle,
#   6 = done
#
# For each mission predicate type we know which actions are operationally
# relevant. The other actions are no-ops in this env and would otherwise
# dominate the agent's score because no-ops keep the predicted latent close
# to the current latent (preserving the probe's noise floor) while real
# actions can push predicates either up or down.
MISSION_ALLOWED_ACTIONS: dict[str, tuple[int, ...]] = {
    "at":      (0, 1, 2),         # "go to <X>" — only navigation
    "holding": (0, 1, 2, 3),      # "pick up <X>" — navigation + pickup
    "open":    (0, 1, 2, 5),      # "open <door>" — navigation + toggle
}


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


def goal_predicates_for_mission(
    mission: str,
) -> tuple[list[WeightedGoal], GoalSpec] | None:
    """Map a BabyAI mission string to the weighted goal predicates the agent
    should try to maximize, plus the parsed GoalSpec (so the caller can derive
    which actions are operationally relevant).

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


def allowed_actions_for_spec(spec: GoalSpec, n_actions: int) -> tuple[int, ...]:
    """Resolve a parsed mission spec to the action ids the agent is allowed
    to take. Falls back to all actions if we don't have a mapping.
    """
    return MISSION_ALLOWED_ACTIONS.get(spec.predicate, tuple(range(n_actions)))


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
        horizon: int = 4,
        n_samples: int = 8,
        scoring_mode: str = "magnitude",
    ):
        """Args:
            horizon:    multi-step rollout depth. Each candidate first action
                        is followed by `horizon - 1` random actions in the
                        imagined latent space; predicates are read from the
                        final latent. horizon=4 is the default — at horizon=1
                        the agent only sees what each action does *immediately*,
                        which doesn't help when the target is multiple steps
                        away.
            n_samples:  how many random follow-up sequences to draw per first
                        action. The scores are averaged across samples to
                        reduce variance from the random follow-ups.
        """
        self.jepa = jepa
        self.device = device
        self.horizon = horizon
        self.n_samples = max(1, n_samples)
        if scoring_mode not in ("magnitude", "binary", "distance"):
            raise ValueError(f"unknown scoring_mode={scoring_mode!r}")
        self.scoring_mode = scoring_mode
        if scoring_mode == "distance":
            d = getattr(jepa.cfg, "aux_distance_dim", 0)
            if d <= 0:
                raise ValueError(
                    "scoring_mode='distance' requires JEPA trained with "
                    "aux_distance_dim > 0; this checkpoint has aux_distance_dim=0"
                )

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
        *,
        allowed_actions: tuple[int, ...] | None = None,
        exploration_threshold: float = 1e-2,
    ) -> tuple[int, dict[str, float]]:
        """Pick an action by rolling each candidate forward and scoring the
        IMPROVEMENT of each goal predicate relative to the current state.

        Why improvement, not absolute probability:
          Even when no goal predicate is true, the probe outputs a small
          baseline ~0.003 per slot (sigmoid noise floor). With our weighting
          {visible=1, facing=2, near=4, adjacent=8} summing to 15, no-op
          actions like pickup/drop/toggle/done preserve this 0.05 of "weighted
          baseline noise" — while informative actions like turn-right that
          push visible from 0.003 → 0.02 only get a score of ~0.04.
          The agent then prefers no-ops over progress. (This is exactly the
          Phase 2 capstone failure we observed.)

          Subtracting baseline = scoring "advantage": no-ops get 0, useful
          actions get the real magnitude of predicate change.

        Exploration fallback:
          If the maximum advantage is below `exploration_threshold`, the
          model thinks no action makes meaningful progress — usually because
          the target is so far away it doesn't appear in any 1-step rollout.
          We pick a random action so the agent isn't stuck.

        Returns (action, info) with per-action scores for diagnostics.
        """
        obs_b = obs.unsqueeze(0).to(self.device)      # (1, 3, 7, 7)
        z_t = self.jepa.encode(obs_b)                 # (1, embed_dim)

        # Baseline predicate probabilities at the CURRENT state.
        baseline_logits = self.probe(z_t)             # (1, 96)
        baseline_probs = torch.sigmoid(baseline_logits).squeeze(0)  # (96,)

        # Multi-sample rollouts: for each candidate first action, simulate
        # `n_samples` independent random follow-up trajectories of length
        # `horizon - 1`. Score by AVERAGING the final-state predicate probs
        # across samples — this gives a low-variance estimate of "this first
        # action tends to lead to good futures" that beats single-rollout noise.
        nA = self.n_actions
        nS = self.n_samples
        # Replicate (z_t, action) for every (action_id, sample_idx) pair.
        # Shape-agnostic: works for flat (1, D) and spatial (1, C, H, W) latents.
        expand_shape = (nA * nS,) + (-1,) * (z_t.ndim - 1)
        z_t_rep = z_t.expand(*expand_shape)
        first_actions = torch.arange(nA, device=self.device, dtype=torch.long)
        first_actions = first_actions.repeat_interleave(nS)  # [0,0,..,0, 1,1,..,1, ...]
        z_next = self.jepa.predict(z_t_rep, first_actions)

        if self.horizon > 1:
            for _ in range(self.horizon - 1):
                rand_a = torch.randint(0, nA, (nA * nS,), device=self.device)
                z_next = self.jepa.predict(z_next, rand_a)

        next_probs = torch.sigmoid(self.probe(z_next))                 # (nA*nS, 96)
        next_probs_mean = next_probs.view(nA, nS, -1).mean(dim=1)      # (nA, 96)

        # Advantage: per-action improvement over baseline, averaged across samples.
        # `magnitude` uses raw probability difference — vulnerable to noise on
        # low-base-rate predicates (adjacent positive_rate=0.007 means a 0.4
        # turn-noise prediction × weight 8 = +3.2 fake score, which dominates
        # forward's truthful +0). `binary` thresholds both predictions at 0.5
        # and scores only predicate FLIPS, which silences the noise floor.
        # `distance` reads the continuous distance dim for the goal (type, color)
        # — score = base_dist − next_dist (positive = closer). This gives a
        # smooth gradient where binary predicates are flat (target visible
        # 5 cells away → no flip per forward step → no signal in binary mode).
        if self.scoring_mode == "distance":
            g = goal_preds[0]  # all goals share (type_id, color_id)
            P = self.jepa.cfg.aux_predicate_dim
            dist_idx = P + type_color_index(g.type_id, g.color_id)
            base_d = baseline_probs[dist_idx]                    # scalar
            next_d = next_probs_mean[:, dist_idx]                # (nA,)
            scores = base_d - next_d                              # positive = closer
            improvement = None  # unused below; we set scores directly
        elif self.scoring_mode == "binary":
            next_bin = (next_probs_mean > 0.5).float()
            base_bin = (baseline_probs > 0.5).float().unsqueeze(0)
            improvement = next_bin - base_bin
        else:
            improvement = next_probs_mean - baseline_probs.unsqueeze(0)
        if self.scoring_mode != "distance":
            scores = torch.zeros(nA, device=self.device)
            for g in goal_preds:
                scores = scores + g.weight * improvement[:, g.flat_index]

        # Mask disallowed actions to -inf so argmax skips them. Also defines
        # the candidate pool for the exploration fallback. For "go to X"
        # missions this restricts the agent to {turn_left, turn_right, forward}
        # — pickup/drop/toggle/done are no-ops and would otherwise dominate
        # the score by preserving the probe's noise floor.
        if allowed_actions is None:
            allowed = tuple(range(self.n_actions))
        else:
            allowed = allowed_actions
        allowed_t = torch.tensor(allowed, device=self.device, dtype=torch.long)
        mask = torch.zeros(self.n_actions, dtype=torch.bool, device=self.device)
        mask[allowed_t] = True
        scores = torch.where(mask, scores, torch.full_like(scores, float("-inf")))

        # Exploration fallback when no allowed action shows progress —
        # uniformly sample over the allowed set, not all actions.
        explored = False
        max_score = float(scores.max().item())
        if max_score < exploration_threshold:
            pick = int(torch.randint(0, len(allowed_t), (1,), device=self.device).item())
            action = int(allowed_t[pick].item())
            explored = True
        else:
            action = int(scores.argmax().item())

        info = {f"score_a{i}": float(scores[i].item()) for i in range(self.n_actions)}
        info["chosen"] = action
        info["explored"] = float(explored)
        info["max_score"] = max_score
        info["baseline_visible"] = float(baseline_probs.max().item())
        return action, info
