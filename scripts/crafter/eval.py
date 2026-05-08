"""Evaluate a CrafterPolicy checkpoint and report the geometric-mean
achievement score (the standard Crafter benchmark metric).

Score = exp(1/N · Σ ln(1 + s_i)) - 1, where s_i is the success rate (%)
of achievement i across episodes. Reported in percent.

Usage:
    python -m scripts.crafter.eval \
        --policy-checkpoint runs/crafter_ppo_baseline/policy_final.pt \
        --episodes 100 --device cuda
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from prism.crafter.env_worker import CrafterEnvWorker, aggregate_achievement_score
from prism.crafter.env_wrapper import CRAFTER_ACHIEVEMENTS
from prism.crafter.policy import CrafterPolicy
from prism.utils.seed import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=10_000,
                        help="Crafter's native max episode length is 10000; "
                             "policies that don't die just run that long.")
    parser.add_argument("--seed", type=int, default=4242,
                        help="kept disjoint from training seeds.")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    pckpt = torch.load(args.policy_checkpoint, map_location=device, weights_only=False)
    policy = CrafterPolicy(
        n_actions=pckpt["n_actions"],
        embed_dim=pckpt["embed_dim"],
        hidden_dim=pckpt["hidden_dim"],
    ).to(device)
    sd = pckpt["policy_state_dict"]
    sd = {k.replace("cnn.fc.1.", "cnn.head.", 1) if k.startswith("cnn.fc.1.") else k: v
          for k, v in sd.items()}
    policy.load_state_dict(sd)
    policy.eval()
    print(f"[eval] policy loaded from {args.policy_checkpoint}")
    print(f"[eval] running {args.episodes} episodes…")

    per_episode_unlocks: list[set[str]] = []
    per_episode_R: list[float] = []
    per_episode_steps: list[int] = []
    per_episode_n_ach: list[int] = []

    # We use one CrafterEnvWorker for stepping but reset between
    # episodes via its built-in reset path (each episode bumps episode_idx
    # so seeds don't repeat). Single env keeps eval simple — Crafter's
    # episode lengths aren't long enough to need vec eval.
    worker = CrafterEnvWorker(args.seed, 0, reward_mode="reward")

    h = policy.init_hidden(1, device)
    prev_action = torch.tensor([-1], device=device, dtype=torch.long)

    for ep in range(args.episodes):
        # The worker just finished an episode (or was just constructed) —
        # its self.obs is the t=0 obs of the next one. Reset GRU state.
        h = policy.init_hidden(1, device)
        prev_action = torch.tensor([-1], device=device, dtype=torch.long)

        ep_reward = 0.0
        steps = 0
        unlocked: set[str] = set()
        for _ in range(args.max_steps):
            obs_t = torch.from_numpy(worker.obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _value, h = policy.step_with_value(obs_t, prev_action, h)
            action = int(logits.argmax(dim=-1).item())
            prev_action = torch.tensor([action], device=device, dtype=torch.long)

            _, r, done, info = worker.step(action)
            ep_reward += r
            steps += 1
            if done:
                unlocked = info["achievements"]
                break

        per_episode_unlocks.append(unlocked)
        per_episode_R.append(ep_reward)
        per_episode_steps.append(steps)
        per_episode_n_ach.append(len(unlocked))
        if (ep + 1) % 10 == 0 or ep == args.episodes - 1:
            recent_score, _ = aggregate_achievement_score(per_episode_unlocks)
            print(
                f"  [{ep+1:4d}/{args.episodes}] mean_R={np.mean(per_episode_R):.2f} "
                f"mean_steps={np.mean(per_episode_steps):.0f} "
                f"mean_n_ach={np.mean(per_episode_n_ach):.2f} "
                f"score={recent_score:.2f}%"
            )

    score, rates = aggregate_achievement_score(per_episode_unlocks)
    print()
    print(f"=== Crafter eval — {args.episodes} episodes ===")
    print(f"  mean_R              : {np.mean(per_episode_R):.2f}")
    print(f"  mean_steps          : {np.mean(per_episode_steps):.0f}")
    print(f"  mean_n_achievements : {np.mean(per_episode_n_ach):.2f}")
    print(f"  geometric-mean score: {score:.2f}%   "
          f"(random ~1, PPO paper ~5, DreamerV3 ~12)")
    print()
    print("  per-achievement success rate:")
    for ach in CRAFTER_ACHIEVEMENTS:
        bar = "#" * int(rates[ach] / 2.5)
        print(f"    {ach:25s} {rates[ach]:5.1f}%  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
