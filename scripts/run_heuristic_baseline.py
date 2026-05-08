"""Heuristic baseline — bypass the JEPA, score actions on ground-truth predicates.

Purpose: establish the env's achievable ceiling for our predicate-driven action
selection setup. If a hand-coded policy that has perfect predicate access can't
clear ~0.85 mean reward, then our 0.55 capstone target is genuinely near the
ceiling and we shouldn't expect much from improving the model. If it clears
0.85+ easily, the bottleneck is purely model→policy translation and we know
to invest in better predicate signals (distance, value head, etc.).

Policy:
    1. parse mission → goal (type, color)
    2. compute GT predicates from raw obs
    3. if adjacent[goal]: forward          (terminates the episode)
       elif facing[goal]:  forward          (approach)
       elif visible[goal]: turn toward it   (use slot.x < 3 or > 3)
       else:               forward          (advance until target appears)

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


def heuristic_action(raw_obs, goal_type, goal_color, allowed):
    """Pick an action using GT slot info. Returns int action id."""
    slots = extract_slots(raw_obs)
    preds = compute_predicates(slots)

    adj = preds[predicate_index("adjacent", goal_type, goal_color)] > 0.5
    fac = preds[predicate_index("facing", goal_type, goal_color)] > 0.5
    vis = preds[predicate_index("visible", goal_type, goal_color)] > 0.5

    # 1. adjacent → step into the target (terminal)
    if adj and 2 in allowed:
        return 2
    # 2. facing → forward to approach
    if fac and 2 in allowed:
        return 2
    # 3. visible → turn to bring the target into the forward arc
    if vis:
        # Find the closest matching slot
        ax, _ = AGENT_POS
        candidates = [
            s for s in slots
            if s.type_id == goal_type and s.color_id == goal_color
        ]
        if candidates:
            # closest by manhattan
            target = min(candidates, key=lambda s: abs(s.x - ax) + abs(s.y - AGENT_POS[1]))
            if target.x < ax and 0 in allowed:
                return 0  # turn left
            if target.x > ax and 1 in allowed:
                return 1  # turn right
            # x == ax but not facing? It's behind. Turn around (right twice).
            if 1 in allowed:
                return 1
    # 4. not visible → advance
    if 2 in allowed:
        return 2
    # fallback: any allowed
    return allowed[0]


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
        for _ in range(args.max_steps):
            a = heuristic_action(obs["image"], goal_type, goal_color, allowed)
            obs, r, term, trunc, _ = env.step(a)
            ep_reward += float(r)
            steps += 1
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
    print(f"  If our model-driven agent is much below this, the bottleneck is")
    print(f"  signal quality (predicates → action), not the world model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
