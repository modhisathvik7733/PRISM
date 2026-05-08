"""Reward-free latent planning with the JEPA world model on Crafter.

Two experiments:
  goal      — encode a goal observation, beam-search toward it in psi-space
               via receding-horizon replanning (no reward, no PPO gradient)
  curiosity — no goal; score each beam by novelty from visited psi-states,
               pure latent-space exploration

Usage:
    python -m scripts.crafter.plan_jepa \\
        --jepa-checkpoint runs/crafter_jepa/jepa_final.pt \\
        --ppo-checkpoint  runs/crafter_ppo_jepa/policy_final.pt \\
        --experiment both \\
        --horizon 15 --beam-k 10 --exec-steps 5 \\
        --trials 5 --seed 42 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from prism.crafter.cnn_encoder import CrafterCNN
from prism.crafter.env_worker import CrafterEnvWorker
from prism.crafter.jepa_crafter import CrafterJepaConfig, _LatentDynamics
from prism.crafter.latent_planner import LatentPlanner
from prism.crafter.policy_jepa import CrafterPolicyJepa
from prism.utils.seed import set_global_seed


# ── checkpoint helpers ────────────────────────────────────────────────────────

def load_jepa(path: str, device: torch.device):
    """Load frozen encoder + dynamics from jepa_final.pt."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg: CrafterJepaConfig = ckpt["cfg"]

    encoder = CrafterCNN(embed_dim=cfg.embed_dim).to(device)
    encoder.load_state_dict(ckpt["online_encoder_state"])
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    dynamics = _LatentDynamics(cfg.embed_dim, cfg.n_actions, cfg.dynamics_hidden).to(device)
    dynamics.load_state_dict(ckpt["dynamics_state"])
    for p in dynamics.parameters():
        p.requires_grad_(False)
    dynamics.eval()

    print(f"[jepa] loaded from {path}  embed_dim={cfg.embed_dim}")
    return encoder, dynamics, cfg


def collect_goal_obs(
    jepa_checkpoint: str,
    ppo_checkpoint: Optional[str],
    trigger_achievement: str = "place_table",
    max_episodes: int = 100,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> Optional[np.ndarray]:
    """Roll out a policy until trigger_achievement fires; return that obs.

    If ppo_checkpoint exists, uses the trained PPO policy.
    Otherwise falls back to a random policy (more episodes needed).
    """
    use_ppo = ppo_checkpoint is not None and Path(ppo_checkpoint).exists()
    if use_ppo:
        policy = CrafterPolicyJepa(
            jepa_checkpoint=jepa_checkpoint,
            n_actions=17,
            device=device,
        ).to(device)
        ckpt = torch.load(ppo_checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.eval()
        h           = policy.init_hidden(1, device)
        prev_action = torch.full((1,), -1, dtype=torch.long, device=device)
        print(f"[goal] collecting with PPO policy from {ppo_checkpoint}")
    else:
        policy = None
        print(f"[goal] PPO checkpoint not found — using random policy "
              f"(max_episodes raised to {max_episodes})")

    worker   = CrafterEnvWorker(seed, worker_id=0, reward_mode="reward")
    episodes = 0

    while episodes < max_episodes:
        obs_np     = worker.obs.copy()
        ach_before = set(worker.achievements)

        if use_ppo:
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _, h_next = policy.step_with_value(obs_t, prev_action, h)
            action_int = int(torch.distributions.Categorical(logits=logits).sample().item())
        else:
            action_int = int(np.random.randint(17))
            h_next     = None

        _, _, done, info = worker.step(action_int)

        if not done:
            new_ach = worker.achievements - ach_before
            if trigger_achievement in new_ach:
                print(f"[goal] '{trigger_achievement}' fired (ep {episodes}, non-terminal)")
                return worker.obs.copy()
        else:
            if trigger_achievement in info.get("achievements", set()):
                print(f"[goal] '{trigger_achievement}' fired (ep {episodes}, terminal)")
                return obs_np
            episodes += 1
            if use_ppo:
                h           = policy.init_hidden(1, device)
                prev_action = torch.full((1,), -1, dtype=torch.long, device=device)
            if episodes % 20 == 0:
                print(f"[goal] {episodes}/{max_episodes} episodes — trigger not yet fired")
            continue

        if use_ppo:
            h           = h_next
            prev_action = torch.tensor([action_int], dtype=torch.long, device=device)

    print(f"[goal] WARNING: '{trigger_achievement}' not fired in {max_episodes} episodes")
    return None


# ── experiments ───────────────────────────────────────────────────────────────

def run_goal_experiment(
    planner: LatentPlanner,
    goal_obs: np.ndarray,
    n_trials: int = 5,
    horizon: int = 15,
    exec_steps: int = 5,
    max_total_steps: int = 200,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Receding-horizon goal-directed planning. No reward signal."""
    z_goal   = planner.encode_obs(goal_obs)                    # (E,)
    z_goal_n = F.normalize(z_goal, dim=-1)

    print("\n" + "=" * 60)
    print("EXPERIMENT A: Goal-directed latent planning")
    print(f"  horizon={horizon}  exec_steps={exec_steps}  max_steps={max_total_steps}")
    print("=" * 60)

    all_final_dists: list[float] = []
    all_achievements: list[set[str]] = []

    for trial in range(n_trials):
        worker = CrafterEnvWorker(seed + trial * 1_000_003, worker_id=0, reward_mode="reward")
        step = 0
        dist_trace: list[float] = []
        trial_achievements: set[str] = set()

        while step < max_total_steps:
            z_cur   = planner.encode_obs(worker.obs)            # (E,)
            z_cur_n = F.normalize(z_cur, dim=-1)
            cos_dist = 1.0 - float((z_cur_n * z_goal_n).sum())
            dist_trace.append(cos_dist)

            actions = planner.beam_search(z_cur, z_goal, horizon=horizon)

            for a in actions[:exec_steps]:
                if step >= max_total_steps:
                    break
                _, _, done, info = worker.step(a)
                step += 1
                if info:
                    trial_achievements |= info.get("achievements", set())
                if done:
                    break  # replan from new episode's first obs

        z_final_n  = F.normalize(planner.encode_obs(worker.obs), dim=-1)
        final_dist = 1.0 - float((z_final_n * z_goal_n).sum())
        all_final_dists.append(final_dist)
        all_achievements.append(trial_achievements)

        trace_str = "  ".join(f"{d:.3f}" for d in dist_trace[:8])
        print(f"[trial {trial+1}/{n_trials}]  replans={len(dist_trace)}"
              f"  dist=[{trace_str}{'...' if len(dist_trace)>8 else ''}]"
              f"  final={final_dist:.3f}"
              f"  ach={sorted(trial_achievements)}")

    print(f"\n[goal] mean final cos_dist = {np.mean(all_final_dists):.3f}  "
          f"(lower = closer to goal in psi-space; random baseline ~1.0)")
    all_ach = set().union(*all_achievements)
    print(f"[goal] achievements across all trials: {sorted(all_ach)}")


def run_curiosity_experiment(
    planner: LatentPlanner,
    n_trials: int = 3,
    horizon: int = 15,
    exec_steps: int = 5,
    max_total_steps: int = 300,
    max_memory: int = 500,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Curiosity-driven exploration. No goal, no reward."""
    E = 256  # embed_dim

    print("\n" + "=" * 60)
    print("EXPERIMENT B: Curiosity-driven latent exploration")
    print(f"  horizon={horizon}  exec_steps={exec_steps}  max_steps={max_total_steps}"
          f"  max_memory={max_memory}")
    print("=" * 60)

    all_achievements: list[set[str]] = []

    for trial in range(n_trials):
        worker   = CrafterEnvWorker(seed + trial * 1_000_003, worker_id=0, reward_mode="reward")
        z_memory = torch.zeros(0, E, device=device)
        step     = 0
        trial_achievements: set[str] = set()

        while step < max_total_steps:
            z_cur   = planner.encode_obs(worker.obs)
            actions = planner.novelty_plan(z_cur, z_memory, horizon=horizon)

            for a in actions[:exec_steps]:
                if step >= max_total_steps:
                    break
                obs_visited = worker.obs.copy()
                _, _, done, info = worker.step(a)
                step += 1
                if info:
                    trial_achievements |= info.get("achievements", set())

                if len(z_memory) < max_memory:
                    z_vis = planner.encode_obs(obs_visited)
                    z_memory = torch.cat([z_memory, z_vis.unsqueeze(0)], dim=0)

                if done:
                    break

        # Memory diversity: mean pairwise cosine distance (sampled 100 pairs)
        if len(z_memory) >= 2:
            idx    = torch.randperm(len(z_memory), device=device)[:min(100, len(z_memory))]
            sample = F.normalize(z_memory[idx], dim=-1)
            sim    = sample @ sample.T
            n      = len(sample)
            mask   = ~torch.eye(n, dtype=torch.bool, device=device)
            mean_cos_dist = float(1.0 - sim[mask].mean())
        else:
            mean_cos_dist = 0.0

        all_achievements.append(trial_achievements)
        print(f"[trial {trial+1}/{n_trials}]  steps={step}"
              f"  memory={len(z_memory)}"
              f"  diversity={mean_cos_dist:.3f}"
              f"  ach={sorted(trial_achievements)}")

    all_ach = set().union(*all_achievements)
    print(f"\n[curiosity] achievements across all trials: {sorted(all_ach)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jepa-checkpoint",     default="runs/crafter_jepa/jepa_final.pt")
    p.add_argument("--ppo-checkpoint",      default="runs/crafter_ppo_jepa/policy_final.pt")
    p.add_argument("--experiment",          default="both",
                   choices=["goal", "curiosity", "both"])
    p.add_argument("--trigger-achievement", default="place_table")
    p.add_argument("--horizon",             type=int,   default=15)
    p.add_argument("--beam-k",              type=int,   default=10)
    p.add_argument("--exec-steps",         type=int,   default=5)
    p.add_argument("--trials",              type=int,   default=5)
    p.add_argument("--max-steps",           type=int,   default=200)
    p.add_argument("--max-memory",          type=int,   default=500)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device(args.device)

    encoder, dynamics, cfg = load_jepa(args.jepa_checkpoint, device)
    planner = LatentPlanner(encoder, dynamics, n_actions=cfg.n_actions, device=device)

    if args.experiment in ("goal", "both"):
        goal_obs = collect_goal_obs(
            args.jepa_checkpoint,
            args.ppo_checkpoint,
            trigger_achievement=args.trigger_achievement,
            seed=args.seed,
            device=device,
        )
        if goal_obs is None:
            print("[goal] could not collect goal obs — skipping")
        else:
            run_goal_experiment(
                planner, goal_obs,
                n_trials=args.trials,
                horizon=args.horizon,
                exec_steps=args.exec_steps,
                max_total_steps=args.max_steps,
                seed=args.seed + 1,
                device=device,
            )

    if args.experiment in ("curiosity", "both"):
        n_curiosity_trials = max(1, args.trials // 2) if args.experiment == "both" else args.trials
        run_curiosity_experiment(
            planner,
            n_trials=n_curiosity_trials,
            horizon=args.horizon,
            exec_steps=args.exec_steps,
            max_total_steps=args.max_steps + 100,
            max_memory=args.max_memory,
            seed=args.seed + 2,
            device=device,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
