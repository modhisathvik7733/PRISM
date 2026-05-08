"""Lightweight pose + memory tracker for the recurrent policy.

Path B (Component 3): the GRU's hidden state implicitly carries memory across
steps, but learning to encode "I saw the goal at relative (+2, -1)" purely
from a sequence of 7x7 views is hard — the curriculum's adjacent-cohort
weakness is direct evidence (49 mean steps for 1-cell-away targets where the
hand-coded memory agent takes ~10).

This tracker exposes the same pose/visited/goal-cache state the hand-coded
memory mode maintains in `GroundedAgent._memory_select`, but as a 5-d
feature vector the policy reads directly. The features are:

  0. n_visited / 30   (clamped to [0, 1]) — exploration progress
  1. n_blocked / 10   (clamped to [0, 1]) — obstacle density seen so far
  2. goal_seen flag    (0 or 1)             — have we ever spotted the goal?
  3. goal_fwd  / 8    (clamped to [-1, 1]) — last-known goal in agent frame
  4. goal_right / 8   (clamped to [-1, 1]) — (forward, right) of agent now

The tracker is reset per-episode and updated in two phases:
  - observe(obs)  : reconciles previous FORWARD's effect (moved or blocked),
                    updates the goal cache from the current view
  - commit(action): rotates facing if turn; records action for next reconcile

Both are pure Python / numpy and run on CPU per env worker.
"""

from __future__ import annotations

import numpy as np

from prism.perception.slots import AGENT_POS, extract_slots_from_normalized


MEM_FEAT_DIM = 5
TURN_LEFT, TURN_RIGHT, FORWARD = 0, 1, 2


def _step_xy(x: int, y: int, facing: int) -> tuple[int, int]:
    if facing == 0:
        return x, y + 1
    if facing == 1:
        return x + 1, y
    if facing == 2:
        return x, y - 1
    return x - 1, y


class PoseTracker:
    def __init__(self) -> None:
        self.goal_type: int | None = None
        self.goal_color: int | None = None
        self.x = 0
        self.y = 0
        self.facing = 0
        self.visited: set[tuple[int, int]] = {(0, 0)}
        self.blocked: set[tuple[int, int]] = set()
        self.goal_seen = False
        self.goal_world: tuple[int, int] = (0, 0)
        self.last_action: int | None = None
        self.last_obs: np.ndarray | None = None

    def reset(self, goal_type: int | None, goal_color: int | None) -> None:
        self.goal_type = goal_type
        self.goal_color = goal_color
        self.x = 0
        self.y = 0
        self.facing = 0
        self.visited = {(0, 0)}
        self.blocked = set()
        self.goal_seen = False
        self.goal_world = (0, 0)
        self.last_action = None
        self.last_obs = None

    def _slot_to_world(self, sx: int, sy: int) -> tuple[int, int]:
        ax, ay = AGENT_POS
        rel_forward = ay - sy
        rel_right = sx - ax
        f = self.facing
        if f == 0:
            wdx, wdy = rel_right, rel_forward
        elif f == 1:
            wdx, wdy = rel_forward, -rel_right
        elif f == 2:
            wdx, wdy = -rel_right, -rel_forward
        else:
            wdx, wdy = -rel_forward, rel_right
        return (self.x + wdx, self.y + wdy)

    def observe(self, obs_chw_normalized: np.ndarray) -> None:
        if self.last_action == FORWARD and self.last_obs is not None:
            if np.array_equal(obs_chw_normalized, self.last_obs):
                fx, fy = _step_xy(self.x, self.y, self.facing)
                self.blocked.add((fx, fy))
            else:
                self.x, self.y = _step_xy(self.x, self.y, self.facing)
                self.visited.add((self.x, self.y))

        if self.goal_type is not None:
            try:
                slots = extract_slots_from_normalized(obs_chw_normalized)
                for s in slots:
                    if s.type_id == self.goal_type and s.color_id == self.goal_color:
                        self.goal_world = self._slot_to_world(s.x, s.y)
                        self.goal_seen = True
                        break
            except Exception:
                pass

        self.last_obs = np.array(obs_chw_normalized, copy=True)

    def commit(self, action: int) -> None:
        if action == TURN_LEFT:
            self.facing = (self.facing - 1) % 4
        elif action == TURN_RIGHT:
            self.facing = (self.facing + 1) % 4
        self.last_action = int(action)

    def features(self) -> np.ndarray:
        if self.goal_seen:
            dx = self.goal_world[0] - self.x
            dy = self.goal_world[1] - self.y
            f = self.facing
            if f == 0:
                fwd, rgt = dy, dx
            elif f == 1:
                fwd, rgt = dx, -dy
            elif f == 2:
                fwd, rgt = -dy, -dx
            else:
                fwd, rgt = -dx, dy
        else:
            fwd, rgt = 0, 0
        return np.array(
            [
                min(len(self.visited) / 30.0, 1.0),
                min(len(self.blocked) / 10.0, 1.0),
                1.0 if self.goal_seen else 0.0,
                float(np.clip(fwd / 8.0, -1.0, 1.0)),
                float(np.clip(rgt / 8.0, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )
