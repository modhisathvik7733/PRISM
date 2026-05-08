"""PPO from scratch on Crafter — the baseline for commit 2 of the port.

End-to-end CNN + GRU + actor-critic, trained with PPO. No JEPA, no BC
warmstart. Mirrors the BabyAI `scripts/ppo_train.py` PPO loop with the
Crafter-specific bits swapped in:
  - CrafterPolicy (CNN encoder fused into the policy)
  - CrafterEnvWorker (no mission, no allowed-actions mask)
  - Raw obs stored in the rollout buffer (re-encoded during update so
    PPO gradients flow through the CNN)

Usage:
    python -m scripts.crafter.ppo_train_baseline \
        --total-steps 250000 --n-envs 8 --rollout-steps 256 \
        --run-name crafter_ppo_baseline --device cuda

Reference numbers:
  random policy:        ~1.0% achievement score
  Crafter paper PPO:    ~5.0%
  DreamerV3:            ~12.1%
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prism.crafter.env_worker import CrafterEnvWorker, aggregate_achievement_score
from prism.crafter.policy import CrafterPolicy
from prism.utils.seed import set_global_seed


def compute_gae(
    rewards: torch.Tensor,    # (T, B)
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard GAE; same as scripts.ppo_train.compute_gae."""
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_adv = delta + gamma * lam * nonterminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=250_000)
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Crafter rendering is slow; 8 is a sensible "
                             "default. Bump to 16 if your CPU can keep up.")
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=3,
                        help="Crafter has dense reward → fewer epochs are fine.")
    parser.add_argument("--n-minibatches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-decay", action="store_true", default=True)
    parser.add_argument("--ent-coef-start", type=float, default=0.01)
    parser.add_argument("--ent-coef-end", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--embed-dim", type=int, default=256,
                        help="CNN output dim; also feeds the GRU.")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2_100_000)
    parser.add_argument("--run-name", default="crafter_ppo_baseline")
    parser.add_argument("--save-every-iters", type=int, default=10)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    n_actions = 17
    policy = CrafterPolicy(
        n_actions=n_actions,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[crafter-ppo] policy params: {n_params:,}  "
          f"(CNN end-to-end, no JEPA)")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[crafter-ppo] writing to {out_dir}")

    workers = [CrafterEnvWorker(args.seed, i, reward_mode="reward")
               for i in range(args.n_envs)]
    print(f"[crafter-ppo] n_envs={args.n_envs} reward_mode=reward")

    n_iterations = args.total_steps // (args.rollout_steps * args.n_envs)
    print(f"[crafter-ppo] target {args.total_steps} env steps "
          f"= {n_iterations} iterations @ {args.rollout_steps}*{args.n_envs}")

    h = policy.init_hidden(args.n_envs, device)
    prev_actions = torch.full((args.n_envs,), -1, device=device, dtype=torch.long)

    ep_R_window = deque(maxlen=100)
    ep_steps_window = deque(maxlen=100)
    ep_n_ach_window = deque(maxlen=100)
    ep_unlocks_window: deque[set[str]] = deque(maxlen=100)
    total_env_steps = 0

    for it in range(n_iterations):
        T, B = args.rollout_steps, args.n_envs
        # We store raw obs (3, 64, 64) so PPO gradients flow through the
        # CNN during the update. Memory cost: T*B*3*64*64*4 bytes ≈
        # 256*8*49152 = ~96 MB at the defaults — fine on a single GPU.
        buf_obs = torch.zeros(T, B, 3, 64, 64, device=device)
        buf_actions = torch.zeros(T, B, dtype=torch.long, device=device)
        buf_log_probs = torch.zeros(T, B, device=device)
        buf_rewards = torch.zeros(T, B, device=device)
        buf_values = torch.zeros(T, B, device=device)
        buf_dones = torch.zeros(T, B, device=device)
        buf_h_init = torch.zeros(T, B, policy.hidden_dim, device=device)
        buf_prev_actions = torch.zeros(T, B, dtype=torch.long, device=device)

        with torch.no_grad():
            for t in range(T):
                obs_np = np.stack([w.obs for w in workers], axis=0)
                obs_t = torch.from_numpy(obs_np).to(device)
                buf_obs[t] = obs_t
                buf_h_init[t] = h
                buf_prev_actions[t] = prev_actions

                logits, value, h_next = policy.step_with_value(
                    obs_t, prev_actions, h
                )
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                buf_actions[t] = action
                buf_log_probs[t] = log_prob
                buf_values[t] = value

                action_cpu = action.cpu().tolist()
                rewards = []
                dones = []
                for i, w in enumerate(workers):
                    _obs, r, d, info = w.step(action_cpu[i])
                    rewards.append(r)
                    dones.append(1.0 if d else 0.0)
                    if d and info:
                        ep_R_window.append(info["ep_reward"])
                        ep_steps_window.append(info["ep_steps"])
                        ep_n_ach_window.append(info["n_achievements"])
                        ep_unlocks_window.append(info["achievements"])
                buf_rewards[t] = torch.tensor(rewards, device=device)
                buf_dones[t] = torch.tensor(dones, device=device)

                done_t = buf_dones[t].bool()
                h = torch.where(done_t.unsqueeze(1),
                                policy.init_hidden(B, device), h_next)
                prev_actions = torch.where(done_t,
                                           torch.full_like(action, -1),
                                           action)

            # Bootstrap value for the last state.
            obs_np = np.stack([w.obs for w in workers], axis=0)
            obs_t = torch.from_numpy(obs_np).to(device)
            _, last_value, _ = policy.step_with_value(obs_t, prev_actions, h)

        total_env_steps += T * B

        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones, last_value,
            gamma=args.gamma, lam=args.lam,
        )
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp(min=1e-8)
        advantages_norm = (advantages - adv_mean) / adv_std

        progress = it / max(n_iterations - 1, 1)
        if args.lr_decay:
            for g in opt.param_groups:
                g["lr"] = args.lr * (1.0 - progress)
        ent_coef = args.ent_coef_start + (args.ent_coef_end - args.ent_coef_start) * progress

        env_indices = np.arange(B)
        mb_size = max(B // args.n_minibatches, 1)

        last_pi_loss = last_v_loss = last_ent = last_kl = 0.0
        for _epoch in range(args.ppo_epochs):
            np.random.shuffle(env_indices)
            for mb_start in range(0, B, mb_size):
                mb_envs = env_indices[mb_start:mb_start + mb_size]
                mb_envs_t = torch.from_numpy(mb_envs).to(device)

                mb_obs = buf_obs[:, mb_envs_t]
                mb_prev = buf_prev_actions[:, mb_envs_t]
                mb_actions = buf_actions[:, mb_envs_t]
                mb_old_logp = buf_log_probs[:, mb_envs_t]
                mb_returns = returns[:, mb_envs_t]
                mb_adv = advantages_norm[:, mb_envs_t]
                mb_dones = buf_dones[:, mb_envs_t]

                h_run = buf_h_init[0, mb_envs_t]
                logits_seq = []
                values_seq = []
                for t in range(T):
                    logits_t, value_t, h_run = policy.step_with_value(
                        mb_obs[t], mb_prev[t], h_run
                    )
                    logits_seq.append(logits_t)
                    values_seq.append(value_t)
                    done_t = mb_dones[t].bool()
                    h_run = torch.where(
                        done_t.unsqueeze(1),
                        policy.init_hidden(mb_obs[t].shape[0], device),
                        h_run,
                    )
                logits_all = torch.stack(logits_seq, dim=0)
                values_all = torch.stack(values_seq, dim=0)

                dist = torch.distributions.Categorical(logits=logits_all)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_all, mb_returns)
                entropy_term = entropy.mean()
                loss = policy_loss + args.value_coef * value_loss - ent_coef * entropy_term

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                opt.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - (new_logp - mb_old_logp)).mean()
                last_pi_loss = float(policy_loss.item())
                last_v_loss = float(value_loss.item())
                last_ent = float(entropy_term.item())
                last_kl = float(approx_kl.item())

        # Logging — Crafter's per-episode reward AND geometric-mean
        # achievement score over the recent window.
        if (it + 1) % 5 == 0 or it == 0 or it == n_iterations - 1:
            mean_R = float(np.mean(ep_R_window)) if ep_R_window else float("nan")
            mean_steps = float(np.mean(ep_steps_window)) if ep_steps_window else float("nan")
            mean_ach = float(np.mean(ep_n_ach_window)) if ep_n_ach_window else float("nan")
            score, _ = aggregate_achievement_score(list(ep_unlocks_window))
            print(
                f"[iter {it+1:4d}/{n_iterations}] env_steps={total_env_steps:>7d} "
                f"window_R={mean_R:.2f} ep_steps={mean_steps:.0f} "
                f"n_ach={mean_ach:.1f} score={score:.2f} "
                f"pi={last_pi_loss:+.4f} v={last_v_loss:.4f} H={last_ent:.3f} "
                f"KL={last_kl:.4f} lr={opt.param_groups[0]['lr']:.2e} "
                f"ent_coef={ent_coef:.4f}"
            )

        if (it + 1) % args.save_every_iters == 0 or it == n_iterations - 1:
            ckpt_path = out_dir / f"policy_iter{it+1}.pt"
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "n_actions": n_actions,
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
                "iteration": it + 1,
                "env_steps": total_env_steps,
                "window_R": float(np.mean(ep_R_window)) if ep_R_window else 0.0,
            }, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

    final_path = out_dir / "policy_final.pt"
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "n_actions": n_actions,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "env_steps": total_env_steps,
        "window_R": float(np.mean(ep_R_window)) if ep_R_window else 0.0,
    }, final_path)
    score, rates = aggregate_achievement_score(list(ep_unlocks_window))
    print(f"[done] saved {final_path}")
    print(f"[done] final window_R={float(np.mean(ep_R_window)):.2f} "
          f"score={score:.2f}% across {len(ep_unlocks_window)} recent episodes")
    print("[done] per-achievement unlock rates (recent window):")
    for ach, rate in sorted(rates.items(), key=lambda kv: -kv[1]):
        if rate > 0:
            print(f"    {ach:25s} {rate:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
