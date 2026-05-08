"""Heuristic baseline — bypass the JEPA, score actions on ground-truth predicates.

Purpose: establish the env's achievable ceiling for our predicate-driven action
selection setup. If a hand-coded policy with perfect predicate access can't
clear ~0.85 mean reward, then our 0.55 capstone target is genuinely near the
ceiling. If it does clear it, the bottleneck is purely model→policy translation.

Policy (with obstacle handling):
  1. parse mission → goal (type, color)
  2. inspect raw partial obs:
     - cell directly ahead = obs[y=5, x=3]
     - front cell empty/floor (type 1)         → "front_passable"
     - front cell == goal target               → "front_is_goal"
     - front cell wall/distractor/closed door  → "front_blocked"
  3. decision tree:
     adjacent(goal)              → forward (terminates)
     facing(goal) AND passable   → forward
     facing(goal) AND blocked    → turn toward open side (sidestep)
     visible(goal)               → turn toward target (slot.x vs agent.x)
     else                        → forward when passable, turn when blocked
  4. anti-stall: if we tried forward last step and didn't move (env blocked),
     do a random turn before retrying.

Usage:
    python -m scripts.run_heuristic_baseline --episodes 50 --seed 0
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

from prism.agents import goal_predicates_for_mission
from prism.agents.grounded_agent import allowed_actions_for_spec
from prism.perception import compute_predicates, extract_slots
from prism.perception.predicates import predicate_index
from prism.perception.slots import AGENT_POS

# MiniGrid OBJECT_TO_IDX (relevant entries):
#   0 = unseen, 1 = empty, 2 = wall, 3 = floor, 4 = door, 5 = key,
#   6 = ball, 7 = box, 8 = goal, 9 = lava, 10 = agent
PASSABLE_TYPES = {1, 3, 8}  # empty, floor, goal-tile
TURN_LEFT, TURN_RIGHT, FORWARD = 0, 1, 2


def front_cell(raw_obs):
    """Return (type_id, color_id) of the cell directly in front of the agent.
    Agent at (x=3, y=6) facing up, so front is (x=3, y=5)."""
    return int(raw_obs[5, 3, 0]), int(raw_obs[5, 3, 1])


def heuristic_action(
    raw_obs,
    goal_type: int,
    goal_color: int,
    allowed: tuple[int, ...],
    rng: np.random.Generator,
    last_action_was_forward_blocked: bool,
):
    """Pick an action with obstacle awareness."""
    slots = extract_slots(raw_obs)
    preds = compute_predicates(slots)

    adj = preds[predicate_index("adjacent", goal_type, goal_color)] > 0.5
    fac = preds[predicate_index("facing", goal_type, goal_color)] > 0.5
    vis = preds[predicate_index("visible", goal_type, goal_color)] > 0.5

    front_t, front_c = front_cell(raw_obs)
    front_is_goal = (front_t == goal_type and front_c == goal_color)
    front_passable = front_t in PASSABLE_TYPES or front_is_goal

    # Anti-stall: random turn after a blocked-forward.
    if last_action_was_forward_blocked:
        return TURN_LEFT if rng.random() < 0.5 else TURN_RIGHT

    # 1. adjacent: forward (env terminates on stepping into goal cell)
    if adj and FORWARD in allowed and front_is_goal:
        return FORWARD

    # 2. facing the goal AND path clear: forward
    if fac and front_passable and FORWARD in allowed:
        return FORWARD

    # 3. facing but blocked: rotate to sidestep
    if fac and not front_passable:
        return TURN_LEFT if rng.random() < 0.5 else TURN_RIGHT

    # 4. visible (but not facing): turn toward the target
    if vis:
        ax, ay = AGENT_POS
        candidates = [s for s in slots if s.type_id == goal_type and s.color_id == goal_color]
        if candidates:
            target = min(candidates, key=lambda s: abs(s.x - ax) + abs(s.y - ay))
            if target.x < ax and TURN_LEFT in allowed:
                return TURN_LEFT
            if target.x > ax and TURN_RIGHT in allowed:
                return TURN_RIGHT
            # target straight ahead but not "facing" (rare due to predicate definition);
            # rotate randomly.
            return TURN_LEFT if rng.random() < 0.5 else TURN_RIGHT

    # 5. not visible: forward when passable, turn when blocked
    if FORWARD in allowed and front_passable:
        return FORWARD
    return TURN_LEFT if rng.random() < 0.5 else TURN_RIGHT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = gym.make(args.env_id)
    rewards = []
    successes = 0
    rng = np.random.default_rng(args.seed)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep * 7919)
        mission = obs["mission"]
        parsed = goal_predicates_for_mission(mission)
        if parsed is None:
            print(f"[ep {ep:02d}] mission={mission!r} (UNPARSED)")
            rewards.append(0.0)
            continue
        goal_preds, spec = parsed
        allowed = allowed_actions_for_spec(spec, env.action_space.n)
        goal_type = goal_preds[0].type_id
        goal_color = goal_preds[0].color_id

        ep_reward = 0.0
        steps = 0
        last_was_forward_blocked = False

        for _ in range(args.max_steps):
            a = heuristic_action(
                obs["image"], goal_type, goal_color, allowed, rng,
                last_was_forward_blocked,
            )
            prev_image = obs["image"].copy()
            obs, r, term, trunc, _ = env.step(a)
            ep_reward += float(r)
            steps += 1
            # Detect "forward didn't move us": same partial view as before.
            last_was_forward_blocked = (
                a == FORWARD and not term and np.array_equal(prev_image, obs["image"])
            )
            if term or trunc:
                break

        rewards.append(ep_reward)
        if ep_reward > 0:
            successes += 1
        print(
            f"[ep {ep:02d}] mission={mission!r:60s} "
            f"steps={steps:3d} reward={ep_reward:.3f}"
        )

    mean_r = float(np.mean(rewards))
    print("\n=== heuristic baseline summary ===")
    print(f"  episodes              : {args.episodes}")
    print(f"  reward > 0            : {successes}/{args.episodes}")
    print(f"  mean reward           : {mean_r:.3f}")
    print(f"\n  This is the env's achievable ceiling with GT predicate access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
