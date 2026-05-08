"""PoseTrackerV2 — same logic as v1 PoseTracker, but the feature
normalizations are constructor args.

Why v2: phase 0 zero-shot showed GoTo-v0's hidden cohort failing at 11.7%
with mean_steps=113.9. Navigation cohorts (visible/facing/adjacent) all
hit 100%, so the bottleneck is exploration in the larger GoTo grid. The v1
tracker normalizes `n_visited / 30.0` and clamps to [0, 1] — once the
agent visits >30 cells, the feature saturates at 1.0 and the policy loses
the "how much exploration remains" signal it needs to push into unvisited
quadrants.

For BabyAI's larger envs (GoTo, Open with multi-room layouts), 80 visited
cells is more representative. The 5-d feature layout is unchanged so
downstream policies / checkpoints stay drop-in compatible.
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


class PoseTrackerV2:
    def __init__(
        self,
        *,
        visit_norm: float = 80.0,
        blocked_norm: float = 30.0,
        goal_norm: float = 12.0,
    ) -> None:
        """All three normalizations should be roughly the *95th-percentile*
        value of the corresponding raw count across the env distribution.
        Setting them too low saturates the feature (loses signal late in
        episode); too high makes the early-episode signal too small.

        Defaults target BabyAI-GoTo / -Open / multi-room envs.
        For pure GoToLocal-v0 (small single room), v1's 30/10/8 fits better.
        """
        self.visit_norm = float(visit_norm)
        self.blocked_norm = float(blocked_norm)
        self.goal_norm = float(goal_norm)
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
                min(len(self.visited) / self.visit_norm, 1.0),
                min(len(self.blocked) / self.blocked_norm, 1.0),
                1.0 if self.goal_seen else 0.0,
                float(np.clip(fwd / self.goal_norm, -1.0, 1.0)),
                float(np.clip(rgt / self.goal_norm, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )
