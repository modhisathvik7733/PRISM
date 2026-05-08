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
from prism.perception.slots import (
    AGENT_POS,
    COLOR_NAME_TO_IDX,
    extract_slots_from_normalized,
)


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
        if scoring_mode not in (
            "magnitude", "binary", "distance", "curriculum", "memory", "recurrent",
        ):
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
        # Episode-local exploration step counter — used by curriculum mode to
        # alternate forward/turn when the target is hidden so the agent doesn't
        # walk into a wall and stall.
        self._explore_step = 0
        # ----- Memory-mode state (Phase 3) ---------------------------------
        # Memoryless agents cannot exceed ~0.4 mean reward in BabyAI partial-view
        # because they re-visit the same cells when the target is hidden at t=0
        # (55% of episodes). We track a relative-pose frame and the set of
        # cells already visited so the policy can drive toward unexplored
        # frontier cells. World-frame pose (relative to episode start):
        #   facing 0 = initial-forward, 1 = right, 2 = back, 3 = left
        self._mem_x = 0
        self._mem_y = 0
        self._mem_facing = 0
        self._mem_visited: set[tuple[int, int]] = {(0, 0)}
        self._mem_blocked: set[tuple[int, int]] = set()
        self._mem_last_action: int | None = None
        self._mem_last_obs: torch.Tensor | None = None
        # Object-instance memory: (type_id, color_id) → last-seen world (x, y).
        # Lets the agent navigate back to a target it saw and lost sight of
        # rather than falling into frontier exploration.
        self._mem_objects: dict[tuple[int, int], tuple[int, int]] = {}
        # ----- Recurrent-policy state (Phase 3 step 2) ---------------------
        # When scoring_mode='recurrent' is active, these get populated with a
        # trained RecurrentPolicy and its hidden state h_t.
        self._rec_policy = None
        self._rec_hidden: torch.Tensor | None = None
        self._rec_prev_action: int = -1
        self._rec_mission: torch.Tensor | None = None
        # Path B — optional pose tracker; only built when the attached policy
        # has mem_feat_dim > 0.
        self._rec_pose_tracker = None

    def reset(self) -> None:
        """Call at the start of each episode so curriculum-mode exploration
        cycle starts fresh. Calling this from run_agent's run_episode keeps
        exploration deterministic across episodes for reproducibility."""
        self._explore_step = 0
        # Reset memory-mode state
        self._mem_x = 0
        self._mem_y = 0
        self._mem_facing = 0
        self._mem_visited = {(0, 0)}
        self._mem_blocked = set()
        self._mem_last_action = None
        self._mem_last_obs = None
        self._mem_objects = {}
        # Reset recurrent state — fresh GRU hidden + 'no previous action'.
        self._rec_hidden = None
        self._rec_prev_action = -1
        self._rec_mission = None
        # The PoseTracker (if any) gets re-initialized inside
        # attach_recurrent_policy() once the goal pair is known.
        self._rec_pose_tracker = None

    @torch.no_grad()
    def _curriculum_select(
        self,
        z_t: torch.Tensor,
        baseline_probs: torch.Tensor,
        goal_preds: list[WeightedGoal],
        allowed_actions: tuple[int, ...] | None,
    ) -> tuple[int, dict[str, float]]:
        """Predicate-curriculum policy: state recognition by current predicates,
        dynamics used only for the binary 'which turn' decision.

        Decision tree (preserved with weights so we can score-mask later):
          adjacent(goal) → forward (terminates the episode)
          facing(goal)   → forward (approach)
          visible(goal)  → turn toward higher predicted facing-prob
          else           → forward, with a turn every K steps to scan the room
        """
        nA = self.n_actions
        if allowed_actions is None:
            allowed = tuple(range(nA))
        else:
            allowed = allowed_actions

        # Find the goal-targeted predicate indices. goal_preds is the 4-tuple
        # for (visible/facing/near/adjacent) on the goal (type, color).
        idx_visible = idx_facing = idx_adjacent = None
        for g in goal_preds:
            if g.name == "visible":
                idx_visible = g.flat_index
            elif g.name == "facing":
                idx_facing = g.flat_index
            elif g.name == "adjacent":
                idx_adjacent = g.flat_index

        p_visible = float(baseline_probs[idx_visible].item()) if idx_visible is not None else 0.0
        p_facing = float(baseline_probs[idx_facing].item()) if idx_facing is not None else 0.0
        p_adjacent = float(baseline_probs[idx_adjacent].item()) if idx_adjacent is not None else 0.0

        info = {
            "p_visible": p_visible,
            "p_facing": p_facing,
            "p_adjacent": p_adjacent,
            "explored": 0.0,
            "max_score": 0.0,
        }

        FORWARD = 2
        TURN_LEFT = 0
        TURN_RIGHT = 1

        def _allowed_or_first(a: int) -> int:
            return a if a in allowed else allowed[0]

        # 1. adjacent: step in (env terminates)
        if p_adjacent > 0.5 and FORWARD in allowed:
            info["chosen"] = FORWARD
            info["branch"] = "adjacent"
            return FORWARD, info

        # 2. facing the goal: forward
        if p_facing > 0.5 and FORWARD in allowed:
            info["chosen"] = FORWARD
            info["branch"] = "facing"
            return FORWARD, info

        # 3. visible but not facing: pick the turn that maximizes predicted
        # facing-prob. Uses 1-step dynamics for a binary choice — much more
        # robust than scoring continuous magnitudes.
        if p_visible > 0.5 and idx_facing is not None and (TURN_LEFT in allowed or TURN_RIGHT in allowed):
            actions = torch.tensor([TURN_LEFT, TURN_RIGHT], device=self.device, dtype=torch.long)
            expand_shape = (2,) + (-1,) * (z_t.ndim - 1)
            z_rep = z_t.expand(*expand_shape)
            z_next = self.jepa.predict(z_rep, actions)
            next_probs = torch.sigmoid(self.probe(z_next))  # (2, P+D)
            p_face_left = float(next_probs[0, idx_facing].item())
            p_face_right = float(next_probs[1, idx_facing].item())
            info["p_face_after_left"] = p_face_left
            info["p_face_after_right"] = p_face_right
            chosen = TURN_LEFT if p_face_left >= p_face_right else TURN_RIGHT
            info["chosen"] = _allowed_or_first(chosen)
            info["branch"] = "visible_rotate"
            return info["chosen"], info

        # 4. target hidden: explore. Forward most steps, turn occasionally so
        # we scan the room. Mostly-forward exploration matches what works in
        # BabyAI partial-view envs — turning rotates view but doesn't move you.
        EXPLORE_TURN_EVERY = 4
        self._explore_step += 1
        if self._explore_step % EXPLORE_TURN_EVERY == 0 and TURN_RIGHT in allowed:
            info["chosen"] = TURN_RIGHT
            info["branch"] = "explore_turn"
            return TURN_RIGHT, info
        if FORWARD in allowed:
            info["chosen"] = FORWARD
            info["branch"] = "explore_forward"
            return FORWARD, info
        # Fallback if forward isn't allowed
        info["chosen"] = allowed[0]
        info["branch"] = "explore_fallback"
        return allowed[0], info

    @staticmethod
    def _step_xy(x: int, y: int, facing: int) -> tuple[int, int]:
        """Return (x, y) one cell in front of the given pose.
        Convention: facing 0 = +y (initial forward), 1 = +x (right of initial),
        2 = -y (back), 3 = -x (left). Right-handed when seen from above."""
        if facing == 0:
            return x, y + 1
        if facing == 1:
            return x + 1, y
        if facing == 2:
            return x, y - 1
        return x - 1, y  # facing == 3

    def _slot_to_world(self, sx: int, sy: int) -> tuple[int, int]:
        """Convert agent-frame slot coords (BabyAI partial view at agent pos
        (3, 6) facing-up) to world-frame coords using the agent's current pose.

        rel_forward = how many cells in front of the agent the slot is (>=0).
        rel_right   = how many cells to the agent's right (>0 right, <0 left).
        These are then rotated by the agent's facing into the world frame."""
        ax, ay = AGENT_POS  # (3, 6)
        rel_forward = ay - sy
        rel_right = sx - ax
        f = self._mem_facing
        if f == 0:    # +y world is "ahead" of the agent
            wdx, wdy = rel_right, rel_forward
        elif f == 1:  # +x world is "ahead"
            wdx, wdy = rel_forward, -rel_right
        elif f == 2:  # -y world is "ahead"
            wdx, wdy = -rel_right, -rel_forward
        else:         # f == 3, -x world is "ahead"
            wdx, wdy = -rel_forward, rel_right
        return (self._mem_x + wdx, self._mem_y + wdy)

    @staticmethod
    def _goal_in_slots(slots_list, weighted_goal) -> bool:
        """True if the goal (type, color) appears in the current view's slots."""
        gt, gc = weighted_goal.type_id, weighted_goal.color_id
        return any(s.type_id == gt and s.color_id == gc for s in slots_list)

    def _slot_rotate_fallback(self, slots_list, weighted_goal,
                              allowed: tuple[int, ...],
                              forward_blocked_now: bool) -> int:
        """When the JEPA's visibility predicate dips below 0.5 while the goal
        is still in the slot list, rotate using slot-based geometry instead of
        the noisy 1-step prediction. Slots are perception, not learning — fair
        to use as a backup so we don't fall through to cache when the target
        is actually right in view."""
        FORWARD, TURN_LEFT, TURN_RIGHT = 2, 0, 1
        ax, ay = AGENT_POS
        gt, gc = weighted_goal.type_id, weighted_goal.color_id
        cands = [s for s in slots_list if s.type_id == gt and s.color_id == gc]
        target = min(cands, key=lambda s: abs(s.x - ax) + abs(s.y - ay))
        if target.x < ax and TURN_LEFT in allowed:
            return TURN_LEFT
        if target.x > ax and TURN_RIGHT in allowed:
            return TURN_RIGHT
        # target.x == ax: in front of agent — go forward if not blocked.
        if FORWARD in allowed and not forward_blocked_now:
            return FORWARD
        return TURN_RIGHT if TURN_RIGHT in allowed else allowed[0]

    def _navigate_toward(self, target: tuple[int, int], allowed: tuple[int, ...],
                         forward_blocked_now: bool) -> int:
        """Greedy single-step toward a known world-frame target.

        Picks the cardinal direction that closes the larger of |dx|, |dy|.
        Turns toward that direction one step at a time, forwards when aligned.
        Falls through to a turn if forward is currently blocked.
        """
        FORWARD, TURN_LEFT, TURN_RIGHT = 2, 0, 1
        tx, ty = target
        dx = tx - self._mem_x
        dy = ty - self._mem_y
        if dx == 0 and dy == 0:
            # Already on the cell — turn to scan (target may be in an adjacent
            # tile; this nudges the camera).
            return TURN_RIGHT if TURN_RIGHT in allowed else allowed[0]
        # Pick the desired facing (0=+y, 1=+x, 2=-y, 3=-x).
        if abs(dy) >= abs(dx):
            desired = 0 if dy > 0 else 2
        else:
            desired = 1 if dx > 0 else 3
        diff = (desired - self._mem_facing) % 4
        if diff == 0:
            if FORWARD in allowed and not forward_blocked_now:
                return FORWARD
            # Aligned but blocked: turn right to look for a sidestep
            return TURN_RIGHT if TURN_RIGHT in allowed else allowed[0]
        if diff == 1:
            return TURN_RIGHT if TURN_RIGHT in allowed else allowed[0]
        if diff == 3:
            return TURN_LEFT if TURN_LEFT in allowed else allowed[0]
        # diff == 2 (180°): rotate one step at a time toward target
        return TURN_RIGHT if TURN_RIGHT in allowed else allowed[0]

    @torch.no_grad()
    def _memory_select(
        self,
        obs: torch.Tensor,
        z_t: torch.Tensor,
        baseline_probs: torch.Tensor,
        goal_preds: list[WeightedGoal],
        allowed_actions: tuple[int, ...] | None,
    ) -> tuple[int, dict[str, float]]:
        """Memory-equipped policy: pose tracking + frontier exploration.

        The world model is used as in curriculum mode (current-state predicates
        choose the action class; 1-step prediction breaks the binary turn-
        direction tie when target is visible). The new bit is exploration:
        when the target is hidden, the agent uses its tracked pose and the
        set of visited cells to drive toward unexplored frontier cells, so it
        doesn't re-visit the same area until timeout.

        Pose update is delayed by one step: we only mark a forward as
        successful (and update pose / visited) if the next observation
        differs from the previous one. If the obs is unchanged, the front
        cell is treated as blocked.
        """
        FORWARD, TURN_LEFT, TURN_RIGHT = 2, 0, 1

        # 1. Reconcile pose from previous step.
        last_a = self._mem_last_action
        last_o = self._mem_last_obs
        forward_blocked_now = False
        if last_a == FORWARD and last_o is not None:
            if torch.equal(obs, last_o):
                # Forward last step did NOT move us — front cell at the time
                # of last decision is blocked. We record that *world-frame*
                # cell as blocked.
                fx, fy = self._step_xy(self._mem_x, self._mem_y, self._mem_facing)
                self._mem_blocked.add((fx, fy))
                forward_blocked_now = True
            else:
                # Forward succeeded — update position.
                self._mem_x, self._mem_y = self._step_xy(
                    self._mem_x, self._mem_y, self._mem_facing
                )
                self._mem_visited.add((self._mem_x, self._mem_y))

        # 1b. Extract slots once. We use them for two things:
        #   (a) update the object-instance cache from currently-visible objects
        #   (b) gate cache use to "goal is genuinely not in current view" so the
        #       cache doesn't fire spuriously when the model's visibility
        #       predicate dips below 0.5 while the slot is still there.
        slots_list: list = []
        try:
            obs_np = obs.detach().cpu().numpy()
            slots_list = list(extract_slots_from_normalized(obs_np))
            for s in slots_list:
                self._mem_objects[(s.type_id, s.color_id)] = self._slot_to_world(s.x, s.y)
        except Exception:
            pass

        # 2. Read goal predicates from current state.
        idx_visible = idx_facing = idx_adjacent = None
        for g in goal_preds:
            if g.name == "visible":
                idx_visible = g.flat_index
            elif g.name == "facing":
                idx_facing = g.flat_index
            elif g.name == "adjacent":
                idx_adjacent = g.flat_index
        p_visible = float(baseline_probs[idx_visible].item()) if idx_visible is not None else 0.0
        p_facing = float(baseline_probs[idx_facing].item()) if idx_facing is not None else 0.0
        p_adjacent = float(baseline_probs[idx_adjacent].item()) if idx_adjacent is not None else 0.0

        if allowed_actions is None:
            allowed = tuple(range(self.n_actions))
        else:
            allowed = allowed_actions

        info = {
            "p_visible": p_visible,
            "p_facing": p_facing,
            "p_adjacent": p_adjacent,
            "pose_x": float(self._mem_x),
            "pose_y": float(self._mem_y),
            "pose_facing": float(self._mem_facing),
            "n_visited": float(len(self._mem_visited)),
            "n_blocked": float(len(self._mem_blocked)),
            "explored": 0.0,
            "max_score": 0.0,
        }

        action = None
        branch = ""

        # 3. Action selection — curriculum hierarchy + frontier fallback.
        # 3a. adjacent: forward
        if p_adjacent > 0.5 and FORWARD in allowed and not forward_blocked_now:
            action = FORWARD
            branch = "adjacent"
        # 3b. facing the goal AND not blocked: forward
        elif p_facing > 0.5 and FORWARD in allowed and not forward_blocked_now:
            action = FORWARD
            branch = "facing"
        # 3c. visible (but not facing / forward blocked): turn toward
        elif p_visible > 0.5 and idx_facing is not None and (TURN_LEFT in allowed or TURN_RIGHT in allowed):
            actions_t = torch.tensor([TURN_LEFT, TURN_RIGHT], device=self.device, dtype=torch.long)
            expand_shape = (2,) + (-1,) * (z_t.ndim - 1)
            z_rep = z_t.expand(*expand_shape)
            z_next = self.jepa.predict(z_rep, actions_t)
            next_probs = torch.sigmoid(self.probe(z_next))
            p_face_left = float(next_probs[0, idx_facing].item())
            p_face_right = float(next_probs[1, idx_facing].item())
            action = TURN_LEFT if p_face_left >= p_face_right else TURN_RIGHT
            branch = "rotate_to_face"
        # 3d. predicate dipped < 0.5 but the goal is actually in the current
        # slot extraction — fall back to slot-based rotate so we don't redirect
        # via cache to a position we're already next to.
        elif goal_preds and self._goal_in_slots(slots_list, goal_preds[0]):
            action = self._slot_rotate_fallback(
                slots_list, goal_preds[0], allowed, forward_blocked_now
            )
            branch = "slot_fallback"
        # 3e. cached_navigate is currently DISABLED. Empirically, navigating
        # back to a cached world-frame target costs more than it saves in
        # BabyAI-GoToLocal-v0: random spawns + partial-view mean the agent has
        # often walked past the target by the time it loses sight, so going
        # "back" to the cached position requires a 180° turn + retrace,
        # whereas frontier exploration keeps moving forward and re-acquires
        # the target via natural sweep. We keep _slot_to_world / _navigate_toward
        # / _mem_objects in place for future experiments (e.g. using the cache
        # as a *bias* on frontier choice rather than a hard target). The cache
        # update still happens above so future approaches can use it.
        # 3f. target hidden — frontier exploration.
        else:
            action = self._frontier_action(allowed, forward_blocked_now)
            branch = "frontier"

        info["branch"] = branch
        info["chosen"] = action

        # 4. Local pose update for turn actions; forward update is deferred
        # to next call's reconciliation step (we don't yet know if it succeeds).
        if action == TURN_LEFT:
            self._mem_facing = (self._mem_facing - 1) % 4
        elif action == TURN_RIGHT:
            self._mem_facing = (self._mem_facing + 1) % 4

        # 5. Record for next-step reconciliation. Detach + clone to avoid
        # holding GPU memory across steps.
        self._mem_last_action = action
        self._mem_last_obs = obs.detach().clone()
        return action, info

    def _frontier_action(self, allowed: tuple[int, ...], forward_blocked_now: bool) -> int:
        """Pick action that drives the agent toward unvisited cells.

        Strategy (cheap and local):
          1. If the cell directly in front is unvisited and not known-blocked,
             go forward.
          2. Otherwise scan the 4 cardinal directions in {right, left, back}
             order and turn toward the first one whose target cell is unvisited
             and not known-blocked. Turning back = one turn now, the next call
             will continue the rotation.
          3. If every direction is visited or blocked, fall through to forward
             (or, if forward is blocked, turn right) so we don't deadlock.
        """
        FORWARD, TURN_LEFT, TURN_RIGHT = 2, 0, 1
        x, y, facing = self._mem_x, self._mem_y, self._mem_facing

        # Helper: is (cell) a fresh frontier?
        def fresh(cell: tuple[int, int]) -> bool:
            return cell not in self._mem_visited and cell not in self._mem_blocked

        front = self._step_xy(x, y, facing)
        # 1. forward into a fresh cell
        if FORWARD in allowed and not forward_blocked_now and fresh(front):
            return FORWARD

        # 2. turn toward a fresh-cell direction
        for dir_offset, turn in ((1, TURN_RIGHT), (-1, TURN_LEFT), (2, TURN_RIGHT)):
            new_facing = (facing + dir_offset) % 4
            target = self._step_xy(x, y, new_facing)
            if fresh(target) and turn in allowed:
                return turn

        # 3. all known options exhausted — break out of the local trap.
        if FORWARD in allowed and not forward_blocked_now:
            return FORWARD
        if TURN_RIGHT in allowed:
            return TURN_RIGHT
        return allowed[0]

    def attach_recurrent_policy(
        self,
        policy: torch.nn.Module,
        mission: torch.Tensor,
        *,
        goal_type: int | None = None,
        goal_color: int | None = None,
    ) -> None:
        """Attach a trained RecurrentPolicy and the mission one-hot for the
        current episode. The mission is a (mission_dim,) tensor; we add a
        batch dim internally. Must be called after agent.reset() and before
        the first select_action call when scoring_mode='recurrent'.

        When the policy was trained with mem_feat_dim > 0 (Path B), pass
        goal_type/goal_color so a PoseTracker is created and 5-d memory
        features are fed alongside the latent at every step.
        """
        self._rec_policy = policy.to(self.device).eval()
        self._rec_mission = mission.to(self.device).unsqueeze(0).float()
        self._rec_hidden = None
        self._rec_prev_action = -1
        # Path B — only run pose tracking if the policy actually consumes it
        # (mem_proj exists). Avoids wasted slot extraction for legacy ckpts.
        mem_dim = int(getattr(policy, "mem_feat_dim", 0) or 0)
        if mem_dim > 0:
            from prism.agents.pose_tracker import PoseTracker
            self._rec_pose_tracker = PoseTracker()
            self._rec_pose_tracker.reset(goal_type, goal_color)
        else:
            self._rec_pose_tracker = None

    @torch.no_grad()
    def _recurrent_select(
        self,
        z_t: torch.Tensor,
        allowed_actions: tuple[int, ...] | None,
        obs: torch.Tensor | None = None,
    ) -> tuple[int, dict[str, float]]:
        """One step of the trained recurrent policy.

        Maintains h_t and prev_action across calls. Mission is fixed per
        episode (set via attach_recurrent_policy). The frozen JEPA encoder
        already produced z_t — we just feed it through the GRU + head. When a
        PoseTracker is attached (Path B), `obs` must be the normalized (3,7,7)
        view so the tracker can update pose / goal cache before features are
        read.
        """
        if self._rec_policy is None or self._rec_mission is None:
            raise RuntimeError(
                "scoring_mode='recurrent' requires attach_recurrent_policy(...) "
                "to be called after reset() and before select_action()"
            )
        if self._rec_hidden is None:
            self._rec_hidden = self._rec_policy.init_hidden(1, self.device)
        prev_a = torch.tensor([self._rec_prev_action], device=self.device, dtype=torch.long)
        mem_feat = None
        if self._rec_pose_tracker is not None and obs is not None:
            obs_np = obs.detach().cpu().numpy()
            self._rec_pose_tracker.observe(obs_np)
            mem_np = self._rec_pose_tracker.features()
            mem_feat = torch.from_numpy(mem_np).unsqueeze(0).to(self.device)
        logits, h_next = self._rec_policy.step(
            z_t, prev_a, self._rec_mission, self._rec_hidden, mem_feat=mem_feat
        )
        if allowed_actions is not None:
            mask = torch.full_like(logits, float("-inf"))
            for a in allowed_actions:
                mask[0, a] = 0.0
            logits = logits + mask
        action = int(logits.argmax(dim=-1).item())
        if self._rec_pose_tracker is not None:
            self._rec_pose_tracker.commit(action)
        self._rec_hidden = h_next.detach()
        self._rec_prev_action = action
        info = {"chosen": action, "branch": "recurrent", "explored": 0.0, "max_score": 0.0}
        for i in range(self.n_actions):
            info[f"logit_a{i}"] = float(logits[0, i].item())
        return action, info

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

        # ----- CURRICULUM SCORING ---------------------------------------------
        # The advantage-based scoring approaches (magnitude/binary/distance) have
        # all collapsed to uniform-random over allowed actions because turn-action
        # noise (F1=0.57, distance MAE=0.17 when visible) dominates the 1-cell
        # signal differences. Curriculum bypasses that by using only the most
        # reliable signal — the current-state predicate readout, F1=0.998 — to
        # pick the action class, and reduces dynamics use to a 1-bit decision
        # (which way to turn when target visible but not facing).
        if self.scoring_mode == "curriculum":
            return self._curriculum_select(z_t, baseline_probs, goal_preds, allowed_actions)
        if self.scoring_mode == "memory":
            return self._memory_select(obs, z_t, baseline_probs, goal_preds, allowed_actions)
        if self.scoring_mode == "recurrent":
            return self._recurrent_select(z_t, allowed_actions, obs=obs)
        # ----------------------------------------------------------------------

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
