"""PPO training for the recurrent policy on top of frozen JEPA.

Phase 4 — closes the BC plateau. Hand-coded memory mode hits 0.601, BC
plateau is ~0.50, REINFORCE got us to 0.55 but no further. PPO with proper
components (value function, GAE, clipped surrogate, vec-env, K epochs)
is the standard recipe BabyAI papers use to hit 0.95+.

Algorithm: vectorized PPO.
  for iteration in range(N):
    rollout T steps × B parallel envs:
      z = jepa.encode(obs_batch)            # frozen
      logits, value, h = policy.step_with_value(z, prev_a, mission, h)
      action ~ Categorical(masked_logits)   # disallowed actions → -inf
      env.step(action) → next_obs, reward, done
      store (obs, action, log_prob, reward, value, done)
      reset h on done (per-env)
    compute GAE advantages from stored (rewards, values, dones, last_value)
    returns = advantages + values
    for K epochs:
      shuffle into mini-batches, re-run policy with stored hidden states
      clipped surrogate loss + value loss + entropy bonus
      optimizer.step (only RecurrentPolicy params; JEPA frozen)

Vec-env: SyncVectorEnv with 16 parallel BabyAI envs. Each env runs its
own episode independently; resets and `agent.reset` are per-env.

Mission encoding matches collect_bc_data.py — one-hot of (type, color)
slot index. Action masking: per-env disallowed actions get -inf logits.

Usage:
    python -m scripts.ppo_train \
        --jepa-checkpoint runs/<...>/jepa_final.pt \
        --bc-checkpoint runs/bc_recurrent_v0.9b/policy_final.pt \
        --total-steps 500_000 --device cuda
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F

from prism.agents import goal_predicates_for_mission
from prism.agents.grounded_agent import allowed_actions_for_spec
from prism.envs.babyai import _encode_image, set_max_steps
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.models.recurrent_policy import RecurrentPolicy
from prism.perception import compute_distances, extract_slots
from prism.perception.predicates import type_color_index
from prism.perception.slots import NUM_COLORS, OBJECT_TYPES
from prism.utils.seed import set_global_seed


def _goal_distance_from_raw_obs(raw_image_hwc, goal_pair) -> float:
    """Compute normalized manhattan distance to the closest goal slot in the
    current view. Returns 1.0 (max) when goal not in view. Used by reward
    shaping. raw_image_hwc is the un-normalized BabyAI image obs (H, W, 3)."""
    if goal_pair is None:
        return 1.0
    gt, gc = goal_pair
    slots = extract_slots(raw_image_hwc)
    cands = [s for s in slots if s.type_id == gt and s.color_id == gc]
    if not cands:
        return 1.0
    # Reuse the existing distance computation: returns (24,) where index i is
    # min normalized manhattan dist to (type, color)_i, 1.0 if absent.
    dists = compute_distances(slots)
    return float(dists[type_color_index(gt, gc)])


def latent_dim_for_cfg(cfg: JepaConfig) -> int:
    enc = getattr(cfg, "encoder_type", "flat")
    if enc == "categorical_spatial":
        C = getattr(cfg, "spatial_channels", 64)
        return C * cfg.obs_h * cfg.obs_w
    return cfg.embed_dim


# ----------------------------------------------------------------------
# Per-env state — wraps a BabyAI env and holds the per-env recurrent state
# the agent needs to maintain across timesteps.
# ----------------------------------------------------------------------
class EnvWorker:
    """Wraps one BabyAI env + per-env recurrent state. Sync vec-env steps
    each worker by calling .step(action) and reading .obs_encoded etc."""

    def __init__(self, env_id: str, base_seed: int, worker_id: int,
                 mission_dim: int, n_actions: int,
                 max_steps: int = 64, shaping_coef: float = 0.0):
        self.env = gym.make(env_id)
        if max_steps != 64:
            set_max_steps(self.env, max_steps)
        self.base_seed = base_seed
        self.worker_id = worker_id
        self.episode_idx = 0
        self.n_actions = n_actions
        self.mission_dim = mission_dim
        self.shaping_coef = shaping_coef
        # Hold raw HWC image to compute goal-distance for reward shaping.
        # The encoded version stored in self.obs_encoded is for the policy.
        self.raw_image = None
        self.prev_goal_dist = 1.0
        self._reset_episode()

    def _reset_episode(self):
        seed = self.base_seed + self.worker_id * 1_000_003 + self.episode_idx * 7919
        self.episode_idx += 1
        obs, _ = self.env.reset(seed=seed)
        # Re-seed loop until we get a parseable mission (rare to fail).
        for _ in range(5):
            parsed = goal_predicates_for_mission(obs["mission"])
            if parsed is not None:
                break
            seed += 13
            obs, _ = self.env.reset(seed=seed)
            self.episode_idx += 1
        if parsed is None:
            self.allowed = (0, 1, 2)  # fallback for go-to-style
            self.mission_oh = np.zeros(self.mission_dim, dtype=np.float32)
            self.goal_pair = None
        else:
            goal_preds, spec = parsed
            self.allowed = allowed_actions_for_spec(spec, self.n_actions)
            tc_idx = type_color_index(goal_preds[0].type_id, goal_preds[0].color_id)
            self.mission_oh = np.zeros(self.mission_dim, dtype=np.float32)
            self.mission_oh[tc_idx] = 1.0
            self.goal_pair = (goal_preds[0].type_id, goal_preds[0].color_id)
        self.raw_image = obs["image"]
        self.obs_encoded = _encode_image(obs["image"])
        self.prev_goal_dist = _goal_distance_from_raw_obs(self.raw_image, self.goal_pair)
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.prev_action = -1
        # h_prev is owned by the trainer and reset externally on done; we
        # don't carry it on the worker.

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        # Force action into allowed set (defensive — the masked sample
        # should already respect this).
        if action not in self.allowed:
            action = self.allowed[0]
        next_obs, env_reward, term, trunc, info = self.env.step(action)
        done = bool(term or trunc)

        # Component 2 — potential-based shaping.
        # Compute goal-distance BEFORE potentially resetting, on the new
        # observation. Bonus = shaping_coef * (prev_dist - cur_dist):
        #   - positive when the agent moved closer (or brought goal into view)
        #   - negative when it moved further / lost sight of the goal
        # Per Ng et al. 1999, this is potential-based and preserves the
        # optimal policy under terminal reward, but provides dense gradient
        # signal that PPO needs to converge fast on this sparse env.
        shaping_bonus = 0.0
        if self.shaping_coef != 0.0:
            cur_goal_dist = _goal_distance_from_raw_obs(next_obs["image"], self.goal_pair)
            shaping_bonus = self.shaping_coef * (self.prev_goal_dist - cur_goal_dist)
            self.prev_goal_dist = cur_goal_dist

        # Total reward seen by PPO; episode-reward summary uses the env
        # reward only so logged window_R stays comparable across runs.
        total_reward = float(env_reward) + float(shaping_bonus)
        self.episode_reward += float(env_reward)  # log unshaped for honesty
        self.episode_steps += 1
        self.prev_action = action

        if done:
            ep_summary = {
                "ep_reward": self.episode_reward,
                "ep_steps": self.episode_steps,
            }
            self._reset_episode()
            return self.obs_encoded, total_reward, True, ep_summary
        else:
            self.raw_image = next_obs["image"]
            self.obs_encoded = _encode_image(next_obs["image"])
            return self.obs_encoded, total_reward, False, {}


def make_action_mask(allowed_per_env, n_actions: int, device: torch.device):
    """Returns (B, n_actions) tensor of 0 for allowed, -inf for disallowed."""
    B = len(allowed_per_env)
    mask = torch.full((B, n_actions), float("-inf"), device=device)
    for i, allowed in enumerate(allowed_per_env):
        for a in allowed:
            mask[i, a] = 0.0
    return mask


def compute_gae(
    rewards: torch.Tensor,        # (T, B)
    values: torch.Tensor,         # (T, B)
    dones: torch.Tensor,          # (T, B)
    last_value: torch.Tensor,     # (B,)
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard GAE: A_t = delta_t + (gamma*lam)*(1-done_t) * A_{t+1}.
    Returns (advantages, returns) each of shape (T, B). returns = adv + values."""
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_adv = delta + gamma * lam * nonterminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--bc-checkpoint", required=True,
                        help="path to RecurrentPolicy .pt to initialize from")
    parser.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    parser.add_argument("--total-steps", type=int, default=500_000,
                        help="total env steps across all workers")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=128,
                        help="T per iteration; total transitions = T * n_envs")
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--n-minibatches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-decay", action="store_true", default=True,
                        help="linearly decay lr from --lr to 0 over training")
    parser.add_argument("--ent-coef-start", type=float, default=0.01)
    parser.add_argument("--ent-coef-end", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=64,
                        help="env's truncation budget (Component 1). 64 = "
                             "BabyAI default, 128 = extended budget for harder spawns. "
                             "Affects per-episode reward via 1 - 0.9*(steps/max_steps).")
    parser.add_argument("--shaping-coef", type=float, default=0.0,
                        help="Component 2 reward shaping coefficient. 0.0 = "
                             "disabled (env reward only). 0.1 = recommended; "
                             "shaping_bonus = coef * (prev_dist - cur_dist) where "
                             "dist is normalized manhattan to closest goal slot, "
                             "1.0 if goal not in view. Potential-based per Ng 1999, "
                             "preserves optimal policy.")
    parser.add_argument("--seed", type=int, default=2_000_000,
                        help="training seed; large to avoid eval-seed overlap")
    parser.add_argument("--run-name", default="ppo_v1")
    parser.add_argument("--save-every-iters", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"[ppo] frozen JEPA: encoder={cfg.encoder_type} latent_dim={latent_dim} n_actions={n_actions}")

    # ---------- recurrent policy ----------
    bc = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
    policy = RecurrentPolicy(
        latent_in_dim=bc["latent_in_dim"],
        n_actions=bc["n_actions"],
        mission_dim=bc["mission_dim"],
        hidden_dim=bc["hidden_dim"],
        latent_proj_dim=bc["latent_proj_dim"],
    ).to(device)
    # strict=False so the value_head (newly added) loads with random init.
    missing, unexpected = policy.load_state_dict(bc["policy_state_dict"], strict=False)
    print(f"[ppo] BC weights loaded: missing={missing} unexpected={unexpected}")
    if bc["latent_in_dim"] != latent_dim:
        raise SystemExit("BC policy / JEPA latent_dim mismatch")
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[ppo] policy params: {n_params:,}  (value head random-init)")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ppo] writing to {out_dir}")

    # ---------- vectorized envs (sync) ----------
    workers = [
        EnvWorker(
            args.env_id, args.seed, i, mission_dim, n_actions,
            max_steps=args.max_steps, shaping_coef=args.shaping_coef,
        )
        for i in range(args.n_envs)
    ]
    print(f"[ppo] env: max_steps={args.max_steps} shaping_coef={args.shaping_coef}")

    # ---------- training loop ----------
    n_iterations = args.total_steps // (args.rollout_steps * args.n_envs)
    print(f"[ppo] target {args.total_steps} env steps "
          f"= {n_iterations} iterations @ {args.rollout_steps}*{args.n_envs}")

    h = policy.init_hidden(args.n_envs, device)  # (B, hidden_dim)
    prev_actions = torch.full((args.n_envs,), -1, device=device, dtype=torch.long)

    ep_reward_window = deque(maxlen=200)
    ep_steps_window = deque(maxlen=200)
    total_env_steps = 0

    for it in range(n_iterations):
        # ===== ROLLOUT PHASE =====
        T, B = args.rollout_steps, args.n_envs
        # buffers
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

        with torch.no_grad():
            for t in range(T):
                # gather obs / mission / mask from workers
                obs_batch_np = np.stack([w.obs_encoded for w in workers], axis=0)  # (B, 3, 7, 7)
                obs_batch = torch.from_numpy(obs_batch_np).float().to(device)
                missions_np = np.stack([w.mission_oh for w in workers])
                missions = torch.from_numpy(missions_np).float().to(device)
                allowed_per_env = [w.allowed for w in workers]
                mask = make_action_mask(allowed_per_env, n_actions, device)

                # encode + step policy
                z = jepa.encode(obs_batch)
                z_flat = z.flatten(start_dim=1)  # (B, latent_dim)
                buf_h_init[t] = h
                buf_prev_actions[t] = prev_actions
                buf_missions[t] = missions
                buf_action_mask[t] = mask
                logits, value, h_next = policy.step_with_value(z, prev_actions, missions, h)
                masked = logits + mask
                dist = torch.distributions.Categorical(logits=masked)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                buf_z[t] = z_flat
                buf_actions[t] = action
                buf_log_probs[t] = log_prob
                buf_values[t] = value

                # step envs
                action_cpu = action.cpu().tolist()
                rewards = []
                dones = []
                for i, w in enumerate(workers):
                    _obs, r, d, info = w.step(action_cpu[i])
                    rewards.append(r)
                    dones.append(1.0 if d else 0.0)
                    if d and info:
                        ep_reward_window.append(info["ep_reward"])
                        ep_steps_window.append(info["ep_steps"])
                buf_rewards[t] = torch.tensor(rewards, device=device)
                buf_dones[t] = torch.tensor(dones, device=device)
                # Reset h on done (per-env), set prev_action to action (or -1 if done)
                done_t = buf_dones[t].bool()
                h = torch.where(done_t.unsqueeze(1), policy.init_hidden(B, device), h_next)
                prev_actions = torch.where(done_t, torch.full_like(action, -1), action)

            # bootstrap value for the last state
            obs_batch_np = np.stack([w.obs_encoded for w in workers], axis=0)
            obs_batch = torch.from_numpy(obs_batch_np).float().to(device)
            missions_np = np.stack([w.mission_oh for w in workers])
            missions = torch.from_numpy(missions_np).float().to(device)
            z = jepa.encode(obs_batch)
            _, last_value, _ = policy.step_with_value(z, prev_actions, missions, h)

        total_env_steps += T * B

        # GAE advantages and returns
        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones, last_value,
            gamma=args.gamma, lam=args.lam,
        )
        # Normalize advantages globally for this rollout.
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp(min=1e-8)
        advantages_norm = (advantages - adv_mean) / adv_std

        # ===== UPDATE PHASE =====
        # Linear schedules.
        progress = it / max(n_iterations - 1, 1)
        if args.lr_decay:
            for g in opt.param_groups:
                g["lr"] = args.lr * (1.0 - progress)
        ent_coef = args.ent_coef_start + (args.ent_coef_end - args.ent_coef_start) * progress

        # Flatten T*B → mini-batches (each mb is contiguous in env dimension
        # because we re-run the GRU per-env, but we shuffle the env-dim only).
        env_indices = np.arange(B)
        mb_size = max(B // args.n_minibatches, 1)

        last_pi_loss = last_v_loss = last_ent = last_kl = 0.0
        for epoch in range(args.ppo_epochs):
            np.random.shuffle(env_indices)
            for mb_start in range(0, B, mb_size):
                mb_envs = env_indices[mb_start:mb_start + mb_size]
                mb_envs_t = torch.from_numpy(mb_envs).to(device)

                # Re-run the policy across the rollout for this mini-batch
                # of envs. Use the recorded initial hidden state at t=0.
                mb_z = buf_z[:, mb_envs_t]                          # (T, mb, latent)
                mb_prev = buf_prev_actions[:, mb_envs_t]            # (T, mb)
                mb_missions = buf_missions[:, mb_envs_t]            # (T, mb, mdim)
                mb_mask = buf_action_mask[:, mb_envs_t]             # (T, mb, n_actions)
                mb_actions = buf_actions[:, mb_envs_t]              # (T, mb)
                mb_old_logp = buf_log_probs[:, mb_envs_t]           # (T, mb)
                mb_returns = returns[:, mb_envs_t]                  # (T, mb)
                mb_adv = advantages_norm[:, mb_envs_t]              # (T, mb)
                mb_dones = buf_dones[:, mb_envs_t]                  # (T, mb)

                # We need to handle within-rollout episode boundaries: when
                # done at step t, reset hidden for env at step t+1. Use
                # buf_h_init[0] as the very-first hidden, then re-derive.
                h_run = buf_h_init[0, mb_envs_t]
                # latent passed to policy: we kept a flat (B, latent_dim)
                # version in buf_z; reshape into the encoder's natural
                # spatial form by reading cfg.
                # The policy's latent_proj already does Flatten internally,
                # so passing flat (mb, latent_dim) works as long as the
                # input shape's last-dim matches latent_in_dim.
                logits_seq = []
                values_seq = []
                for t in range(T):
                    logits_t, value_t, h_run = policy.step_with_value(
                        mb_z[t], mb_prev[t], mb_missions[t], h_run
                    )
                    logits_t = logits_t + mb_mask[t]
                    logits_seq.append(logits_t)
                    values_seq.append(value_t)
                    # reset h on done at step t (for step t+1 onward)
                    done_t = mb_dones[t].bool()
                    h_run = torch.where(
                        done_t.unsqueeze(1),
                        policy.init_hidden(mb_z[t].shape[0], device),
                        h_run,
                    )
                logits_all = torch.stack(logits_seq, dim=0)  # (T, mb, n_actions)
                values_all = torch.stack(values_seq, dim=0)  # (T, mb)

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

        # Logging
        if (it + 1) % 5 == 0 or it == 0 or it == n_iterations - 1:
            mean_R = float(np.mean(ep_reward_window)) if ep_reward_window else float("nan")
            mean_steps = float(np.mean(ep_steps_window)) if ep_steps_window else float("nan")
            print(
                f"[iter {it+1:4d}/{n_iterations}] env_steps={total_env_steps:>7d} "
                f"window_R={mean_R:.3f} ep_steps={mean_steps:.1f} "
                f"pi={last_pi_loss:+.4f} v={last_v_loss:.4f} H={last_ent:.3f} KL={last_kl:.4f} "
                f"lr={opt.param_groups[0]['lr']:.2e} ent_coef={ent_coef:.4f}"
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
                "jepa_checkpoint": args.jepa_checkpoint,
                "bc_checkpoint": args.bc_checkpoint,
                "iteration": it + 1,
                "env_steps": total_env_steps,
                "window_mean_reward": float(np.mean(ep_reward_window)) if ep_reward_window else 0.0,
            }, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

    # final
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
        "env_steps": total_env_steps,
        "window_mean_reward": float(np.mean(ep_reward_window)) if ep_reward_window else 0.0,
    }, final_path)
    print(f"[done] saved {final_path}")
    print(f"[done] final window_mean_R = {float(np.mean(ep_reward_window)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
