"""Curriculum-driven PPO training: the ALP-bandit picks the next env
per episode based on per-env competence + learning progress.

Compares two schedulers on the SAME compute budget:
  - ALPCurriculum (Akakzia): value(env) = (1 - success_rate) × |LP|
  - RandomScheduler         : uniform random env per episode

Phase 1 emergence target: ALP curriculum gets ≥10% higher mean
final accuracy across all envs vs random scheduling.

This is a thin wrapper over scripts.ppo_train's EnvWorker — we
override only the env-selection step. The PPO loop itself (rollout +
GAE + update) is reused unchanged.

Usage:
    # ALP scheduler
    python -m scripts.cog_core.train_curriculum \
        --jepa-checkpoint $V13_JEPA \
        --bc-checkpoint runs/v2_ppo_multienv/policy_final.pt \
        --envs BabyAI-GoToObj-v0 BabyAI-GoToLocal-v0 BabyAI-GoTo-v0 \
        --scheduler alp \
        --total-steps 1000000 \
        --run-name cog_phase1_alp --device cuda

    # Random baseline (run separately for comparison)
    python -m scripts.cog_core.train_curriculum \
        ... --scheduler random --run-name cog_phase1_random ...
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prism.cog_core.curriculum import ALPCurriculum, RandomScheduler
from prism.generalize.pose_tracker_v2 import MEM_FEAT_DIM
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.models.recurrent_policy import RecurrentPolicy
from prism.perception.slots import NUM_COLORS, OBJECT_TYPES
from prism.utils.seed import set_global_seed
from scripts.ppo_train import (
    EnvWorker, compute_gae, latent_dim_for_cfg, make_action_mask,
)


def make_scheduler(kind: str, env_ids: list[str], seed: int):
    if kind == "alp":
        return ALPCurriculum(env_ids, seed=seed)
    if kind == "random":
        return RandomScheduler(env_ids, seed=seed)
    raise ValueError(f"unknown scheduler: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--bc-checkpoint", default=None,
                        help="required unless --no-bc is set")
    parser.add_argument("--no-bc", action="store_true",
                        help="initialize policy from scratch instead of "
                             "loading a BC warm-start checkpoint")
    parser.add_argument("--policy-hidden-dim", type=int, default=256)
    parser.add_argument("--policy-latent-proj-dim", type=int, default=128)
    parser.add_argument("--goal-source", choices=["rule", "lang"], default="rule")
    parser.add_argument("--lang-checkpoint", default=None)
    parser.add_argument("--vocab-checkpoint", default=None)
    parser.add_argument("--held-out-combos", nargs="*", default=[],
                        help="space-separated 'color_id,type_idx' pairs "
                             "to exclude from training (Stage 1.4)")
    parser.add_argument("--envs", nargs="+",
                        default=["BabyAI-GoToObj-v0", "BabyAI-GoToLocal-v0",
                                 "BabyAI-GoTo-v0"])
    parser.add_argument("--scheduler", choices=["alp", "random"], default="alp")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--n-minibatches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef-start", type=float, default=0.01)
    parser.add_argument("--ent-coef-end", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--shaping-coef", type=float, default=0.1)
    parser.add_argument("--mem-feat-dim", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2_200_000)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--save-every-iters", type=int, default=20)
    parser.add_argument("--reschedule-every-episodes", type=int, default=4,
                        help="How often the scheduler re-picks the env per worker. "
                             "Must be ≥ 1 — too low = slow training (env reset overhead), "
                             "too high = bandit signal is stale.")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.no_bc and args.bc_checkpoint is None:
        parser.error("--bc-checkpoint is required unless --no-bc is set")
    if args.goal_source == "lang":
        if args.lang_checkpoint is None or args.vocab_checkpoint is None:
            parser.error(
                "--goal-source lang requires --lang-checkpoint and "
                "--vocab-checkpoint"
            )
    held_out_combos: set[tuple[int, int]] = set()
    for s in args.held_out_combos:
        try:
            c_str, t_str = s.split(",")
            c_id = int(c_str)
            t_idx = int(t_str)
        except ValueError:
            parser.error(
                f"--held-out-combos entry {s!r} must be 'color_id,type_idx'"
            )
        if not (0 <= t_idx < len(OBJECT_TYPES)):
            parser.error(
                f"--held-out-combos type_idx {t_idx} out of range"
            )
        held_out_combos.add((c_id, int(OBJECT_TYPES[t_idx])))

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # ---- frozen JEPA ----
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

    # ---- policy ----
    if args.no_bc:
        mem_feat_dim = int(args.mem_feat_dim)
        policy_latent_in_dim = latent_dim
        policy_n_actions = n_actions
        policy_mission_dim = mission_dim
        policy_hidden_dim = args.policy_hidden_dim
        policy_latent_proj_dim = args.policy_latent_proj_dim
        policy = RecurrentPolicy(
            latent_in_dim=policy_latent_in_dim,
            n_actions=policy_n_actions,
            mission_dim=policy_mission_dim,
            hidden_dim=policy_hidden_dim,
            latent_proj_dim=policy_latent_proj_dim,
            mem_feat_dim=mem_feat_dim,
        ).to(device)
        print(f"[curriculum] policy initialized from scratch (no BC): "
              f"hidden={policy_hidden_dim} "
              f"latent_proj={policy_latent_proj_dim} mem={mem_feat_dim}")
    else:
        bc = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
        mem_feat_dim = max(int(args.mem_feat_dim), int(bc.get("mem_feat_dim", 0) or 0))
        policy_latent_in_dim = bc["latent_in_dim"]
        policy_n_actions = bc["n_actions"]
        policy_mission_dim = bc["mission_dim"]
        policy_hidden_dim = bc["hidden_dim"]
        policy_latent_proj_dim = bc["latent_proj_dim"]
        policy = RecurrentPolicy(
            latent_in_dim=policy_latent_in_dim,
            n_actions=policy_n_actions,
            mission_dim=policy_mission_dim,
            hidden_dim=policy_hidden_dim,
            latent_proj_dim=policy_latent_proj_dim,
            mem_feat_dim=mem_feat_dim,
        ).to(device)
        missing, unexpected = policy.load_state_dict(bc["policy_state_dict"], strict=False)
        print(f"[curriculum] policy loaded: missing={missing} unexpected={unexpected}")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[curriculum] writing to {out_dir}")
    print(f"[curriculum] scheduler={args.scheduler}  envs={args.envs}")

    # ---- scheduler + workers ----
    scheduler = make_scheduler(args.scheduler, args.envs, args.seed)
    use_pose_tracker = mem_feat_dim > 0
    if use_pose_tracker and mem_feat_dim != MEM_FEAT_DIM:
        raise SystemExit(f"--mem-feat-dim must be {MEM_FEAT_DIM} or 0")

    # Each worker is initially assigned a random env from the pool.
    # After every `reschedule_every_episodes` episodes, the worker
    # asks the scheduler for its next env.
    # Optional language goal provider (Stage 1.4).
    goal_provider = None
    if args.goal_source == "lang":
        from prism.agents.lang_goal_provider import LangGoalProvider
        goal_provider = LangGoalProvider(
            lang_checkpoint=args.lang_checkpoint,
            vocab_checkpoint=args.vocab_checkpoint,
            device=device,
        )
        print(f"[curriculum] goal source = lang  "
              f"(lang={args.lang_checkpoint})")
    else:
        print(f"[curriculum] goal source = rule")
    if held_out_combos:
        print(f"[curriculum] held-out combos ({len(held_out_combos)}): "
              f"{sorted(held_out_combos)}")

    worker_env_ids = [scheduler.propose() for _ in range(args.n_envs)]
    workers = []
    workers_episode_count = [0] * args.n_envs
    for i in range(args.n_envs):
        workers.append(EnvWorker(
            worker_env_ids[i], args.seed, i, mission_dim, n_actions,
            max_steps=args.max_steps, shaping_coef=args.shaping_coef,
            use_pose_tracker=use_pose_tracker,
            goal_provider=goal_provider,
            held_out_combos=held_out_combos,
        ))
    print(f"[curriculum] worker → env (initial): "
          + ", ".join(f"w{i}={e.replace('BabyAI-', '').replace('-v0', '')}"
                      for i, e in enumerate(worker_env_ids)))

    n_iterations = args.total_steps // (args.rollout_steps * args.n_envs)
    print(f"[curriculum] target {args.total_steps} env steps "
          f"= {n_iterations} iterations @ {args.rollout_steps}*{args.n_envs}")

    h = policy.init_hidden(args.n_envs, device)
    prev_actions = torch.full((args.n_envs,), -1, device=device, dtype=torch.long)

    per_env_R_window = {e: deque(maxlen=100) for e in args.envs}
    per_env_steps_window = {e: deque(maxlen=100) for e in args.envs}
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
        buf_mem = (torch.zeros(T, B, mem_feat_dim, device=device)
                   if use_pose_tracker else None)

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
                    z, prev_actions, missions, h, mem_feat=mem_batch,
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
                        # Update scheduler: success = ep_reward > 0.5
                        ep_r = info["ep_reward"]
                        succ = ep_r > 0.5
                        scheduler.update(worker_env_ids[i], succ)
                        per_env_R_window[worker_env_ids[i]].append(ep_r)
                        per_env_steps_window[worker_env_ids[i]].append(info["ep_steps"])
                        workers_episode_count[i] += 1
                        # Reschedule this worker's env if interval hit.
                        if workers_episode_count[i] % args.reschedule_every_episodes == 0:
                            new_env = scheduler.propose()
                            if new_env != worker_env_ids[i]:
                                # Rebuild worker on new env. EnvWorker auto-resets.
                                worker_env_ids[i] = new_env
                                workers[i] = EnvWorker(
                                    new_env, args.seed + i * 7919 + workers_episode_count[i],
                                    i, mission_dim, n_actions,
                                    max_steps=args.max_steps,
                                    shaping_coef=args.shaping_coef,
                                    use_pose_tracker=use_pose_tracker,
                                    goal_provider=goal_provider,
                                    held_out_combos=held_out_combos,
                                )
                buf_rewards[t] = torch.tensor(rewards, device=device)
                buf_dones[t] = torch.tensor(dones, device=device)
                done_t = buf_dones[t].bool()
                h = torch.where(done_t.unsqueeze(1), policy.init_hidden(B, device), h_next)
                prev_actions = torch.where(done_t,
                                           torch.full_like(action, -1), action)

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
                z, prev_actions, missions, h, mem_feat=mem_last,
            )

        total_env_steps += T * B

        # ---- GAE + PPO update (verbatim from scripts.ppo_train) ----
        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones, last_value,
            gamma=args.gamma, lam=args.lam,
        )
        adv_norm = (advantages - advantages.mean()) / advantages.std().clamp(min=1e-8)

        progress = it / max(n_iterations - 1, 1)
        for g in opt.param_groups:
            g["lr"] = args.lr * (1.0 - progress)
        ent_coef = args.ent_coef_start + (args.ent_coef_end - args.ent_coef_start) * progress

        env_indices = np.arange(B)
        mb_size = max(B // args.n_minibatches, 1)
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
                mb_adv = adv_norm[:, mb_envs_t]
                mb_dones = buf_dones[:, mb_envs_t]
                mb_mem = buf_mem[:, mb_envs_t] if buf_mem is not None else None

                h_run = buf_h_init[0, mb_envs_t]
                logits_seq = []
                values_seq = []
                for t in range(T):
                    mem_t = mb_mem[t] if mb_mem is not None else None
                    logits_t, value_t, h_run = policy.step_with_value(
                        mb_z[t], mb_prev[t], mb_missions[t], h_run, mem_feat=mem_t,
                    )
                    logits_t = logits_t + mb_mask[t]
                    logits_seq.append(logits_t)
                    values_seq.append(value_t)
                    done_t = mb_dones[t].bool()
                    h_run = torch.where(done_t.unsqueeze(1),
                                        policy.init_hidden(mb_z[t].shape[0], device),
                                        h_run)
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

        if (it + 1) % 5 == 0 or it == 0 or it == n_iterations - 1:
            per_env_R_str = " ".join(
                f"{e.replace('BabyAI-', '').replace('-v0', '')}={float(np.mean(per_env_R_window[e])):.2f}"
                if per_env_R_window[e]
                else f"{e.replace('BabyAI-', '').replace('-v0', '')}=---"
                for e in args.envs
            )
            sched_dist = scheduler.schedule_summary(last_n=200) if hasattr(scheduler, 'schedule_summary') else None
            sched_str = ""
            if sched_dist:
                sched_str = " sched[" + " ".join(
                    f"{e.replace('BabyAI-', '').replace('-v0', '')}={sched_dist.get(e, 0)*100:.0f}%"
                    for e in args.envs) + "]"
            print(f"[iter {it+1:4d}/{n_iterations}] env_steps={total_env_steps:>7d} "
                  f"per_env_R[{per_env_R_str}]{sched_str}")

        if (it + 1) % args.save_every_iters == 0 or it == n_iterations - 1:
            ckpt_path = out_dir / f"policy_iter{it+1}.pt"
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "latent_in_dim": policy_latent_in_dim,
                "n_actions": policy_n_actions,
                "mission_dim": policy_mission_dim,
                "hidden_dim": policy_hidden_dim,
                "latent_proj_dim": policy_latent_proj_dim,
                "mem_feat_dim": mem_feat_dim,
                "scheduler": args.scheduler,
                "envs": args.envs,
                "goal_source": args.goal_source,
                "lang_checkpoint": args.lang_checkpoint,
                "held_out_combos": sorted(held_out_combos),
                "iteration": it + 1,
                "env_steps": total_env_steps,
                "scheduler_history": [
                    (str(t), bool(s), float(v)) for t, s, v in scheduler.history[-1000:]
                ],
                "per_env_window_R": {e: float(np.mean(v)) if v else 0.0
                                     for e, v in per_env_R_window.items()},
            }, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

    final = out_dir / "policy_final.pt"
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "latent_in_dim": policy_latent_in_dim,
        "n_actions": policy_n_actions,
        "mission_dim": policy_mission_dim,
        "hidden_dim": policy_hidden_dim,
        "latent_proj_dim": policy_latent_proj_dim,
        "mem_feat_dim": mem_feat_dim,
        "scheduler": args.scheduler,
        "envs": args.envs,
        "goal_source": args.goal_source,
        "lang_checkpoint": args.lang_checkpoint,
        "held_out_combos": sorted(held_out_combos),
        "env_steps": total_env_steps,
        "scheduler_history": [
            (str(t), bool(s), float(v)) for t, s, v in scheduler.history
        ],
        "per_env_window_R": {e: float(np.mean(v)) if v else 0.0
                             for e, v in per_env_R_window.items()},
    }, final)
    print(f"[done] saved {final}")
    print(f"[done] per-env final mean_R:")
    for e in args.envs:
        r = float(np.mean(per_env_R_window[e])) if per_env_R_window[e] else 0.0
        print(f"  {e:30s} R={r:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
