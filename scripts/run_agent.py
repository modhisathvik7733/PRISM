"""Run the first end-to-end PRISM agent on BabyAI.

This is the Phase 2 capstone. The agent has zero learned policy. It acts by:
  1. Encoding the obs with the frozen JEPA encoder.
  2. Imagining the next latent under each candidate action.
  3. Reading out predicates from each imagined latent (frozen probe).
  4. Picking the action whose imagined predicates best match the parsed
     mission's goal.

Success criterion (Phase 2 capstone falsifier):
  Mean reward on BabyAI-GoToLocal-v0 over 50 episodes should clear ~0.6.
  Vanilla mission-blind PPO clocked 0.48 — beating that with NO learned
  policy is the proof that grounded-prediction-driven action selection
  works end-to-end.

Usage:
    python -m scripts.run_agent \
        --jepa-checkpoint runs/jepa_categorical_aux1_BabyAI-GoToLocal-v0_seed0/jepa_final.pt \
        --episodes 50 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  (registers BabyAI envs)
import numpy as np
import torch

from prism.agents import GroundedAgent, goal_predicates_for_mission
from prism.envs.babyai import _encode_image
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.utils.seed import set_global_seed


def run_episode(
    env: gym.Env,
    agent: GroundedAgent,
    *,
    seed: int,
    max_steps: int = 64,
    verbose: bool = False,
) -> dict:
    obs, _ = env.reset(seed=seed)
    mission = obs["mission"]
    goal_preds = goal_predicates_for_mission(mission)
    if goal_preds is None:
        # Fallback: random policy. Phase 4+ will handle compositional missions.
        n_actions = env.action_space.n
        rng = np.random.default_rng(seed)
        chosen_actions = []
        ep_reward = 0.0
        for _ in range(max_steps):
            a = int(rng.integers(n_actions))
            obs, r, term, trunc, _ = env.step(a)
            ep_reward += float(r)
            chosen_actions.append(a)
            if term or trunc:
                break
        return {
            "mission": mission,
            "parsed": False,
            "reward": ep_reward,
            "steps": len(chosen_actions),
            "actions": chosen_actions,
        }

    chosen_actions = []
    ep_reward = 0.0
    for step in range(max_steps):
        encoded = _encode_image(obs["image"])  # (3, 7, 7) float32 normalized
        obs_t = torch.from_numpy(encoded).float()
        action, info = agent.select_action(obs_t, goal_preds)
        if verbose:
            print(f"  step {step:2d} action={action} scores={[round(info[f'score_a{i}'], 2) for i in range(env.action_space.n)]}")
        obs, r, term, trunc, _ = env.step(action)
        ep_reward += float(r)
        chosen_actions.append(action)
        if term or trunc:
            break

    return {
        "mission": mission,
        "parsed": True,
        "reward": ep_reward,
        "steps": len(chosen_actions),
        "actions": chosen_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--probe-checkpoint", default=None,
                        help="Optional standalone probe ckpt. If omitted, the JEPA's "
                             "internal aux_predicate_head is used (which requires the "
                             "JEPA to have been trained with aux_predicate_weight>0).")
    parser.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=1,
                        help="latent rollout horizon for action scoring (>=1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true",
                        help="print per-step action scores for the first episode")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # ------------------------------------------------------ load JEPA
    ckpt = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
    cfg: JepaConfig = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    encoder_type = getattr(cfg, "encoder_type", "flat")
    aux_w = getattr(cfg, "aux_predicate_weight", 0.0)
    print(f"[agent] loaded JEPA: encoder={encoder_type} aux_predicate_weight={aux_w}")

    # ------------------------------------------------------ load probe (optional)
    external_probe = None
    if args.probe_checkpoint is not None:
        probe_ckpt = torch.load(args.probe_checkpoint, map_location=device, weights_only=False)
        from prism.models.predicate_probe import PredicateProbe
        external_probe = PredicateProbe(embed_dim=probe_ckpt["embed_dim"]).to(device)
        external_probe.load_state_dict(probe_ckpt["probe"])
        external_probe.eval()
        print(f"[agent] loaded external probe from {args.probe_checkpoint}")

    agent = GroundedAgent(jepa, device, probe=external_probe, horizon=args.horizon)
    print(f"[agent] horizon={args.horizon} n_actions={agent.n_actions}")

    # ------------------------------------------------------ env + run
    env = gym.make(args.env_id)
    rewards = []
    parsed_count = 0
    successes = 0  # episodes with reward > 0
    for ep in range(args.episodes):
        result = run_episode(
            env, agent,
            seed=args.seed + ep * 7919,  # spread seeds
            max_steps=args.max_steps,
            verbose=args.verbose and ep == 0,
        )
        rewards.append(result["reward"])
        if result["parsed"]:
            parsed_count += 1
        if result["reward"] > 0:
            successes += 1
        print(
            f"[ep {ep:02d}] mission={result['mission']!r:60s} "
            f"steps={result['steps']:3d} reward={result['reward']:.3f}"
            + ("" if result["parsed"] else "  (UNPARSED — fell back to random)")
        )

    mean_reward = float(np.mean(rewards))
    print("\n=== summary ===")
    print(f"  episodes              : {args.episodes}")
    print(f"  parsed missions       : {parsed_count}/{args.episodes}")
    print(f"  reward > 0            : {successes}/{args.episodes}")
    print(f"  mean reward           : {mean_reward:.3f}")

    # Phase 2 capstone falsifier: beat the mission-blind PPO baseline (0.48).
    pass_capstone = mean_reward > 0.55
    print(
        f"\n  Phase 2 capstone (mean_reward > 0.55): "
        f"{'PASS — grounded action selection works end-to-end' if pass_capstone else 'FAIL — diagnose before Phase 3'}"
    )
    return 0 if pass_capstone else 2


if __name__ == "__main__":
    raise SystemExit(main())
