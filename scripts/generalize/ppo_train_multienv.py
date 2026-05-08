"""Multi-env PPO — train one policy across BabyAI levels in parallel.

Each of the N parallel workers is permanently bound to one env from
`--envs` (round-robin assignment), so a single training run sees mixed
transitions from all target levels. Per AMAGO-2 / BabyAI++ findings, this
is the recipe that actually generalizes — per-env PPO catastrophically
collapses on out-of-distribution levels.

Wraps `scripts.ppo_train`'s helpers (EnvWorker, GAE, action mask) and
the rollout/update loop logic — the only meaningful difference is worker
construction. No edits to `scripts/ppo_train.py`.

Usage:
    python -m scripts.generalize.ppo_train_multienv \
        --jepa-checkpoint runs/v2_jepa_universal/jepa_final.pt \
        --bc-checkpoint runs/v2_bc_multienv/policy_final.pt \
        --envs BabyAI-GoToLocal-v0 BabyAI-Pickup-v0 BabyAI-GoTo-v0 BabyAI-Open-v0 \
        --mem-feat-dim 5 \
        --max-steps 128 --shaping-coef 0.1 \
        --total-steps 2000000 --run-name v2_ppo_multienv --device cuda
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prism.generalize.pose_tracker_v2 import MEM_FEAT_DIM
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.models.recurrent_policy import RecurrentPolicy
from prism.perception.slots import NUM_COLORS, OBJECT_TYPES
from prism.utils.seed import set_global_seed
from scripts.ppo_train import (
    EnvWorker,
    compute_gae,
    latent_dim_for_cfg,
    make_action_mask,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--bc-checkpoint", required=True)
    parser.add_argument("--envs", nargs="+",
                        default=["BabyAI-GoToLocal-v0", "BabyAI-Pickup-v0",
                                 "BabyAI-GoTo-v0", "BabyAI-Open-v0"],
                        help="round-robin distributed across --n-envs workers. "
                             "With 16 workers and 4 envs, each env gets 4 workers.")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
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
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--shaping-coef", type=float, default=0.0,
                        help="potential-based shaping; safe across envs since "
                             "the shaping function only fires when the goal "
                             "object is in the current view (no goal -> 1.0).")
    parser.add_argument("--mem-feat-dim", type=int, default=0,
                        help="Path B memory features. 5 = enable. Existing "
                             "EnvWorker uses the v1 PoseTracker — for the "
                             "larger GoTo / Open rooms, the v1 thresholds "
                             "(visited/30, blocked/10) are the documented "
                             "limit. The mem_proj layer is zero-init from BC.")
    parser.add_argument("--seed", type=int, default=2_000_001)
    parser.add_argument("--run-name", default="ppo_multienv_v1")
    parser.add_argument("--save-every-iters", type=int, default=20)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # ---------- frozen JEPA ----------
    ckpt = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
    cfg: JepaConfig = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    n_actions = cfg.n_actions
    latent_dim = latent_dim_for_cfg(cfg)
    mission_dim = len(OBJECT_TYPES) * NUM_COLORS
    print(f"[ppo-multi] frozen JEPA: encoder={cfg.encoder_type} "
          f"latent_dim={latent_dim} n_actions={n_actions}")
    print(f"[ppo-multi] envs ({len(args.envs)}): {args.envs}")

    # ---------- recurrent policy ----------
    bc = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
    mem_feat_dim = max(int(args.mem_feat_dim), int(bc.get("mem_feat_dim", 0) or 0))
    policy = RecurrentPolicy(
        latent_in_dim=bc["latent_in_dim"],
        n_actions=bc["n_actions"],
        mission_dim=bc["mission_dim"],
        hidden_dim=bc["hidden_dim"],
        latent_proj_dim=bc["latent_proj_dim"],
        mem_feat_dim=mem_feat_dim,
    ).to(device)
    missing, unexpected = policy.load_state_dict(bc["policy_state_dict"], strict=False)
    print(f"[ppo-multi] BC weights loaded: missing={missing} unexpected={unexpected}")
    if bc["latent_in_dim"] != latent_dim:
        raise SystemExit("BC policy / JEPA latent_dim mismatch")
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[ppo-multi] policy params: {n_params:,}  (value head random-init)")
    print(f"[ppo-multi] mem_feat_dim={mem_feat_dim}  "
          f"(mem_proj {'enabled (zero-init)' if mem_feat_dim > 0 else 'disabled'})")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ppo-multi] writing to {out_dir}")

    # ---------- vectorized envs (sync), round-robin env assignment ----------
    use_pose_tracker = mem_feat_dim > 0
    if use_pose_tracker and mem_feat_dim != MEM_FEAT_DIM:
        raise SystemExit(
            f"--mem-feat-dim must be {MEM_FEAT_DIM} or 0; got {mem_feat_dim}"
        )
    workers = []
    env_assignments = []
    for i in range(args.n_envs):
        env_id = args.envs[i % len(args.envs)]
        env_assignments.append(env_id)
        workers.append(EnvWorker(
            env_id, args.seed, i, mission_dim, n_actions,
            max_steps=args.max_steps, shaping_coef=args.shaping_coef,
            use_pose_tracker=use_pose_tracker,
        ))
    print(
        f"[ppo-multi] worker → env: "
        + ", ".join(f"w{i}={e.replace('BabyAI-', '').replace('-v0', '')}"
                    for i, e in enumerate(env_assignments))
    )

    # ---------- training loop ----------
    n_iterations = args.total_steps // (args.rollout_steps * args.n_envs)
    print(f"[ppo-multi] target {args.total_steps} env steps "
          f"= {n_iterations} iterations @ {args.rollout_steps}*{args.n_envs}")

    h = policy.init_hidden(args.n_envs, device)
    prev_actions = torch.full((args.n_envs,), -1, device=device, dtype=torch.long)

    # Per-env reward windows so logs show whether all envs are improving.
    per_env_R = {e: deque(maxlen=100) for e in args.envs}
    per_env_steps = {e: deque(maxlen=100) for e in args.envs}
    total_env_steps = 0

    for it in range(n_iterations):
        T, B = args.rollout_steps, args.n_envs
        buf_z = torch.zeros(T, B, latent_dim, device=device)
        buf_actions = torch.zeros(T, B, dtype=torch.long, device=device)
        buf_log_probs = torch.zeros(T, B, device=device)
        buf_rewards = torch.zeros(T, B, device=device)
        buf_values = torch.zeros(T, B, device=device)
        buf_dones = torch.zeros(T, B, device=device)
        buf_h_init = torch.zeros(T, B, policy.hidden_dim, device=device)
        buf_prev_actions = torch.zeros(T, B, dtype=torch.long, device=device)
        buf_missions = torch.zeros(T, B, mission_dim, device=device)
        buf_action_mask = torch.zeros(T, B, n_actions, device=device)
        buf_mem = (
            torch.zeros(T, B, mem_feat_dim, device=device)
            if use_pose_tracker else None
        )

        with torch.no_grad():
            for t in range(T):
                obs_batch_np = np.stack([w.obs_encoded for w in workers], axis=0)
                obs_batch = torch.from_numpy(obs_batch_np).float().to(device)
                missions_np = np.stack([w.mission_oh for w in workers])
                missions = torch.from_numpy(missions_np).float().to(device)
                allowed_per_env = [w.allowed for w in workers]
                mask = make_action_mask(allowed_per_env, n_actions, device)

                z = jepa.encode(obs_batch)
                z_flat = z.flatten(start_dim=1)
                buf_h_init[t] = h
                buf_prev_actions[t] = prev_actions
                buf_missions[t] = missions
                buf_action_mask[t] = mask
                if use_pose_tracker:
                    mem_np = np.stack([w.mem_feat for w in workers], axis=0)
                    mem_batch = torch.from_numpy(mem_np).float().to(device)
                    buf_mem[t] = mem_batch
                else:
                    mem_batch = None
                logits, value, h_next = policy.step_with_value(
                    z, prev_actions, missions, h, mem_feat=mem_batch
                )
                masked = logits + mask
                dist = torch.distributions.Categorical(logits=masked)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                buf_z[t] = z_flat
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
                        per_env_R[env_assignments[i]].append(info["ep_reward"])
                        per_env_steps[env_assignments[i]].append(info["ep_steps"])
                buf_rewards[t] = torch.tensor(rewards, device=device)
                buf_dones[t] = torch.tensor(dones, device=device)
                done_t = buf_dones[t].bool()
                h = torch.where(done_t.unsqueeze(1), policy.init_hidden(B, device), h_next)
                prev_actions = torch.where(done_t, torch.full_like(action, -1), action)

            obs_batch_np = np.stack([w.obs_encoded for w in workers], axis=0)
            obs_batch = torch.from_numpy(obs_batch_np).float().to(device)
            missions_np = np.stack([w.mission_oh for w in workers])
            missions = torch.from_numpy(missions_np).float().to(device)
            z = jepa.encode(obs_batch)
            if use_pose_tracker:
                mem_np = np.stack([w.mem_feat for w in workers], axis=0)
                mem_last = torch.from_numpy(mem_np).float().to(device)
            else:
                mem_last = None
            _, last_value, _ = policy.step_with_value(
                z, prev_actions, missions, h, mem_feat=mem_last
            )

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

                mb_z = buf_z[:, mb_envs_t]
                mb_prev = buf_prev_actions[:, mb_envs_t]
                mb_missions = buf_missions[:, mb_envs_t]
                mb_mask = buf_action_mask[:, mb_envs_t]
                mb_actions = buf_actions[:, mb_envs_t]
                mb_old_logp = buf_log_probs[:, mb_envs_t]
                mb_returns = returns[:, mb_envs_t]
                mb_adv = advantages_norm[:, mb_envs_t]
                mb_dones = buf_dones[:, mb_envs_t]
                mb_mem = buf_mem[:, mb_envs_t] if buf_mem is not None else None

                h_run = buf_h_init[0, mb_envs_t]
                logits_seq = []
                values_seq = []
                for t in range(T):
                    mem_t = mb_mem[t] if mb_mem is not None else None
                    logits_t, value_t, h_run = policy.step_with_value(
                        mb_z[t], mb_prev[t], mb_missions[t], h_run, mem_feat=mem_t
                    )
                    logits_t = logits_t + mb_mask[t]
                    logits_seq.append(logits_t)
                    values_seq.append(value_t)
                    done_t = mb_dones[t].bool()
                    h_run = torch.where(
                        done_t.unsqueeze(1),
                        policy.init_hidden(mb_z[t].shape[0], device),
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

        if (it + 1) % 5 == 0 or it == 0 or it == n_iterations - 1:
            per_env_str = " ".join(
                f"{e.replace('BabyAI-', '').replace('-v0', '')}={float(np.mean(per_env_R[e])):.2f}"
                if per_env_R[e] else f"{e.replace('BabyAI-', '').replace('-v0', '')}=---"
                for e in args.envs
            )
            print(
                f"[iter {it+1:4d}/{n_iterations}] env_steps={total_env_steps:>7d} "
                f"per_env_R[{per_env_str}] "
                f"pi={last_pi_loss:+.4f} v={last_v_loss:.4f} H={last_ent:.3f} "
                f"KL={last_kl:.4f} lr={opt.param_groups[0]['lr']:.2e} ent_coef={ent_coef:.4f}"
            )

        if (it + 1) % args.save_every_iters == 0 or it == n_iterations - 1:
            ckpt_path = out_dir / f"policy_iter{it+1}.pt"
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "latent_in_dim": bc["latent_in_dim"],
                "n_actions": bc["n_actions"],
                "mission_dim": bc["mission_dim"],
                "hidden_dim": bc["hidden_dim"],
                "latent_proj_dim": bc["latent_proj_dim"],
                "mem_feat_dim": mem_feat_dim,
                "jepa_checkpoint": args.jepa_checkpoint,
                "bc_checkpoint": args.bc_checkpoint,
                "envs": args.envs,
                "iteration": it + 1,
                "env_steps": total_env_steps,
                "per_env_window_R": {e: float(np.mean(v)) if v else 0.0
                                     for e, v in per_env_R.items()},
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
        "mem_feat_dim": mem_feat_dim,
        "jepa_checkpoint": args.jepa_checkpoint,
        "bc_checkpoint": args.bc_checkpoint,
        "envs": args.envs,
        "env_steps": total_env_steps,
        "per_env_window_R": {e: float(np.mean(v)) if v else 0.0
                             for e, v in per_env_R.items()},
    }, final_path)
    print(f"[done] saved {final_path}")
    for e in args.envs:
        r = float(np.mean(per_env_R[e])) if per_env_R[e] else 0.0
        s = float(np.mean(per_env_steps[e])) if per_env_steps[e] else 0.0
        print(f"  {e:30s} window_R={r:.3f} ep_steps={s:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
