"""RL fine-tune the recurrent BC policy with REINFORCE on env reward.

Phase 3 step 3 — close the BC distribution-shift gap. The BC student
(`v0.9-phase3-recurrent`) achieves 0.502 mean reward by perfectly imitating
the memory-mode teacher's actions on training trajectories, but compounding
error pulls reward down on novel rollouts. Here we fine-tune the policy
end-to-end on actual env reward, with the BC weights as initialization.

Algorithm (single-env REINFORCE with episode batching):
  for batch in batches:
    rollout K episodes:
      reset env, h_0 = 0, prev_a = -1, mission = one_hot(goal)
      for each step:
        z = jepa.encode(obs)              # frozen
        logits, h = policy.step(z, prev_a, mission, h)
        action ~ Categorical(softmax(logits))
        log_prob_action = log P(action | logits)
        env.step(action), collect reward
      compute G_t = sum_{k>=t} gamma^{k-t} r_k for the episode
    advantages = G_t − running_baseline
    loss = − mean(log_prob_action * advantage) − entropy_coef * entropy
    optimizer.step()  (only RecurrentPolicy params; JEPA frozen)

Mission encoding matches collect_bc_data.py — one-hot of (type, color)
slot index. Action masking: disallowed actions (per allowed_actions_for_spec)
get logit -inf before sampling, so the policy can never pick them.

Usage:
    python -m scripts.rl_finetune_policy \
        --jepa-checkpoint runs/<...>/jepa_final.pt \
        --bc-checkpoint runs/bc_recurrent_v0.9b/policy_final.pt \
        --episodes 4000 --batch-episodes 16 --device cuda
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch

from prism.agents import goal_predicates_for_mission
from prism.agents.grounded_agent import allowed_actions_for_spec
from prism.envs.babyai import _encode_image
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.models.recurrent_policy import RecurrentPolicy
from prism.perception.predicates import type_color_index
from prism.perception.slots import NUM_COLORS, OBJECT_TYPES
from prism.utils.seed import set_global_seed


def latent_dim_for_cfg(cfg: JepaConfig) -> int:
    enc = getattr(cfg, "encoder_type", "flat")
    if enc == "categorical_spatial":
        C = getattr(cfg, "spatial_channels", 64)
        return C * cfg.obs_h * cfg.obs_w
    return cfg.embed_dim


def build_mission_one_hot(goal_preds, device):
    tc_idx = type_color_index(goal_preds[0].type_id, goal_preds[0].color_id)
    out = torch.zeros(len(OBJECT_TYPES) * NUM_COLORS, device=device)
    out[tc_idx] = 1.0
    return out


def rollout_episode(
    env: gym.Env,
    policy: RecurrentPolicy,
    jepa: JepaWorldModel,
    *,
    device: torch.device,
    seed: int,
    max_steps: int,
    sample: bool = True,
) -> tuple[list[torch.Tensor], list[float], float, int]:
    """Run one episode under the (sampling) policy. Returns:
      log_probs:     list of action log-probs (gradient-tracking tensors)
      rewards:       list of per-step env rewards
      ep_return:     summed episode reward
      n_steps:       length of trajectory
    Skips parsed-mission failures by returning empty trajectory.
    """
    obs, _ = env.reset(seed=seed)
    mission = obs["mission"]
    parsed = goal_predicates_for_mission(mission)
    if parsed is None:
        return [], [], 0.0, 0
    goal_preds, spec = parsed
    allowed = allowed_actions_for_spec(spec, env.action_space.n)
    allowed_set = set(allowed)
    mission_oh = build_mission_one_hot(goal_preds, device).unsqueeze(0)

    h = policy.init_hidden(1, device)
    prev_a = torch.tensor([-1], device=device, dtype=torch.long)

    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    ep_return = 0.0

    n_actions = jepa.cfg.n_actions
    mask_vec = torch.full((1, n_actions), float("-inf"), device=device)
    for a in allowed:
        mask_vec[0, a] = 0.0

    for _ in range(max_steps):
        encoded = _encode_image(obs["image"])
        obs_t = torch.from_numpy(encoded).float().unsqueeze(0).to(device)
        with torch.no_grad():
            z = jepa.encode(obs_t)
        logits, h = policy.step(z, prev_a, mission_oh, h)
        masked_logits = logits + mask_vec  # disallowed → -inf

        if sample:
            dist = torch.distributions.Categorical(logits=masked_logits)
            a = dist.sample()
            log_probs.append(dist.log_prob(a))
        else:
            a = masked_logits.argmax(dim=-1)
            # We don't append a log_prob in eval mode — caller should set sample=True for training.

        action_int = int(a.item())
        if action_int not in allowed_set:
            # mask_vec made these -inf so this should never fire; defensive.
            action_int = allowed[0]
        obs, r, term, trunc, _ = env.step(action_int)
        rewards.append(float(r))
        ep_return += float(r)
        prev_a = a
        if term or trunc:
            break

    return log_probs, rewards, ep_return, len(rewards)


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """G_t = r_t + gamma * G_{t+1}, computed backwards."""
    n = len(rewards)
    out = [0.0] * n
    running = 0.0
    for t in reversed(range(n)):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--bc-checkpoint", required=True,
                        help="path to RecurrentPolicy .pt to initialize from")
    parser.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    parser.add_argument("--episodes", type=int, default=4000,
                        help="total episodes of env interaction during training")
    parser.add_argument("--batch-episodes", type=int, default=16,
                        help="episodes per gradient step")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--baseline-window", type=int, default=200,
                        help="running-mean window over recent episode returns")
    parser.add_argument("--seed", type=int, default=1234567,
                        help="training seed; chosen large to avoid eval-seed overlap")
    parser.add_argument("--run-name", default="rl_finetune_v1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=500,
                        help="save policy every N episodes")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # ---------- load frozen JEPA ----------
    ckpt = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
    cfg: JepaConfig = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    n_actions = cfg.n_actions
    latent_dim = latent_dim_for_cfg(cfg)
    print(f"[rl] frozen JEPA: encoder={cfg.encoder_type} latent_dim={latent_dim} n_actions={n_actions}")

    # ---------- load BC policy as init ----------
    bc = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
    policy = RecurrentPolicy(
        latent_in_dim=bc["latent_in_dim"],
        n_actions=bc["n_actions"],
        mission_dim=bc["mission_dim"],
        hidden_dim=bc["hidden_dim"],
        latent_proj_dim=bc["latent_proj_dim"],
    ).to(device)
    policy.load_state_dict(bc["policy_state_dict"])
    policy.train()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[rl] BC policy loaded from {args.bc_checkpoint} ({n_params:,} params)")
    if bc["latent_in_dim"] != latent_dim:
        raise SystemExit(
            f"BC policy expects latent_dim={bc['latent_in_dim']} but JEPA produces {latent_dim}; "
            "checkpoint mismatch."
        )

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rl] writing to {out_dir}")

    env = gym.make(args.env_id)

    # ---------- training loop ----------
    return_window = deque(maxlen=args.baseline_window)
    ep_idx = 0
    grad_step = 0
    last_grad_mean_R = float("nan")

    while ep_idx < args.episodes:
        # Collect a batch of episodes.
        batch_log_probs = []
        batch_advantages = []
        batch_entropies = []
        batch_returns_total = []

        for _ in range(args.batch_episodes):
            if ep_idx >= args.episodes:
                break
            seed = args.seed + ep_idx * 7919
            log_probs, rewards, ep_return, n_steps = rollout_episode(
                env, policy, jepa,
                device=device, seed=seed, max_steps=args.max_steps, sample=True,
            )
            ep_idx += 1
            if n_steps == 0:
                continue
            # Update running baseline AFTER this episode's data is committed.
            return_window.append(ep_return)
            baseline = float(np.mean(return_window))
            G = discounted_returns(rewards, args.gamma)
            G_t = torch.tensor(G, device=device, dtype=torch.float32)
            adv = G_t - baseline
            # Normalize advantages within episode for stability.
            if adv.shape[0] > 1 and adv.std().item() > 1e-6:
                adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            log_p = torch.cat([lp.view(-1) for lp in log_probs])
            # Entropy bonus from the SAMPLED log_probs distribution is approximate;
            # we approximate by — log_p mean (true entropy needs the full distribution
            # which we'd have to recompute. We use the simpler proxy here.)
            entropy_proxy = -log_p.mean()
            batch_log_probs.append(log_p)
            batch_advantages.append(adv)
            batch_entropies.append(entropy_proxy)
            batch_returns_total.append(ep_return)

        if not batch_log_probs:
            continue

        # ---------- gradient step ----------
        all_log_probs = torch.cat(batch_log_probs)
        all_advantages = torch.cat(batch_advantages)
        policy_loss = -(all_log_probs * all_advantages).mean()
        entropy_term = torch.stack(batch_entropies).mean()
        loss = policy_loss - args.entropy_coef * entropy_term

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        grad_step += 1
        last_grad_mean_R = float(np.mean(batch_returns_total))

        if grad_step % 5 == 0 or ep_idx == args.episodes:
            recent = float(np.mean(return_window)) if return_window else float("nan")
            print(
                f"[grad {grad_step:5d}] ep={ep_idx:5d}/{args.episodes} "
                f"batch_R={last_grad_mean_R:.3f} window_R={recent:.3f} "
                f"loss={loss.item():+.4f} "
                f"pi={policy_loss.item():+.4f} H={entropy_term.item():+.4f}"
            )

        if ep_idx % args.save_every == 0 or ep_idx == args.episodes:
            ckpt_path = out_dir / f"policy_ep{ep_idx}.pt"
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "latent_in_dim": bc["latent_in_dim"],
                "n_actions": bc["n_actions"],
                "mission_dim": bc["mission_dim"],
                "hidden_dim": bc["hidden_dim"],
                "latent_proj_dim": bc["latent_proj_dim"],
                "jepa_checkpoint": args.jepa_checkpoint,
                "bc_checkpoint": args.bc_checkpoint,
                "ep_idx": ep_idx,
                "window_mean_reward": float(np.mean(return_window)) if return_window else 0.0,
            }, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

    final_path = out_dir / "policy_final.pt"
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "latent_in_dim": bc["latent_in_dim"],
        "n_actions": bc["n_actions"],
        "mission_dim": bc["mission_dim"],
        "hidden_dim": bc["hidden_dim"],
        "latent_proj_dim": bc["latent_proj_dim"],
        "jepa_checkpoint": args.jepa_checkpoint,
        "bc_checkpoint": args.bc_checkpoint,
        "ep_idx": args.episodes,
        "window_mean_reward": float(np.mean(return_window)) if return_window else 0.0,
    }, final_path)
    print(f"[done] saved {final_path}")
    print(f"[done] final window_mean_R = {float(np.mean(return_window)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
