"""Collect goal-directed trajectories from the current GroundedAgent.

The Phase 2 capstone plateaued at mean reward 0.40 because the JEPA was
trained on random rollouts only. Random walks rarely *successfully* approach
target objects, so the dynamics model never learned what happens when the
agent walks up to a ball / key / box. At inference time the agent imagines
"forward toward target → target disappears" and refuses to move forward
even when the target is visible directly in front.

This script breaks that chicken-and-egg loop. The current agent — which
solves ~58% of episodes — is run on the env, and every transition it
takes is saved to a .npz file. Re-training the JEPA with these
trajectories MIXED INTO the random rollouts gives the dynamics network
a better view of what "approach a target" looks like, which should let
the next agent generation predict it correctly and break through.

This is standard model-based RL bootstrap (cf. Dreamer, MuZero) — the
model is only ever as good as the data it sees, and with a frozen
random-rollout dataset there's a ceiling no amount of agent-side tuning
can raise.

Usage:
    python -m scripts.collect_agent_data \
        --jepa-checkpoint runs/jepa_categorical_aux1_BabyAI-GoToLocal-v0_seed0/jepa_final.pt \
        --episodes 400 \
        --output data/agent_traj_v0.6.npz \
        --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  (registers BabyAI envs)
import numpy as np
import torch

from prism.agents import GroundedAgent, goal_predicates_for_mission
from prism.agents.grounded_agent import allowed_actions_for_spec
from prism.envs.babyai import _encode_image
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.perception import (
    compute_augmented_predicates,
    compute_predicates,
    extract_slots,
)
from prism.utils.seed import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--output", required=True,
                        help="path to .npz file where transitions are saved")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--filter", default="all", choices=["all", "success", "partial"],
        help="all = save every transition. success = only episodes with reward>0.7. "
             "partial = reward>0 (any progress)."
    )
    parser.add_argument(
        "--augmented", action="store_true",
        help="emit augmented 120-d predicates (96 binary + 24 distance). "
             "Required when used to train a JEPA with --aux-distance-dim 24."
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # ---------------------------------------------------- load JEPA + agent
    ckpt = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
    cfg: JepaConfig = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    agent = GroundedAgent(
        jepa, device,
        horizon=args.horizon,
        n_samples=args.n_samples,
    )
    print(f"[collect] agent ready. horizon={args.horizon} n_samples={args.n_samples}")

    env = gym.make(args.env_id)

    # ---------------------------------------------------- run + collect
    obs_t_buf, act_buf, obs_tp1_buf = [], [], []
    pred_t_buf, pred_tp1_buf = [], []

    n_kept_episodes = 0
    n_skipped_episodes = 0
    n_total_transitions = 0
    rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep * 7919)
        mission = obs["mission"]
        parsed = goal_predicates_for_mission(mission)
        if parsed is None:
            n_skipped_episodes += 1
            continue
        goal_preds, spec = parsed
        allowed = allowed_actions_for_spec(spec, env.action_space.n)

        ep_buf = []
        ep_reward = 0.0
        for _step in range(args.max_steps):
            raw_t = obs["image"]
            encoded_t = _encode_image(raw_t)
            obs_t = torch.from_numpy(encoded_t).float()
            action, _info = agent.select_action(
                obs_t, goal_preds, allowed_actions=allowed
            )

            pred_fn = compute_augmented_predicates if args.augmented else compute_predicates
            preds_t = pred_fn(extract_slots(raw_t))

            next_obs, r, term, trunc, _ = env.step(action)
            ep_reward += float(r)
            raw_tp1 = next_obs["image"]
            encoded_tp1 = _encode_image(raw_tp1)
            preds_tp1 = pred_fn(extract_slots(raw_tp1))

            ep_buf.append((encoded_t, action, encoded_tp1, preds_t, preds_tp1))

            if term or trunc:
                break
            obs = next_obs

        rewards.append(ep_reward)

        # Filter logic
        keep = True
        if args.filter == "success" and ep_reward < 0.7:
            keep = False
        elif args.filter == "partial" and ep_reward <= 0.0:
            keep = False

        if keep:
            for (s, a, sp, pt, ptp) in ep_buf:
                obs_t_buf.append(s)
                act_buf.append(a)
                obs_tp1_buf.append(sp)
                pred_t_buf.append(pt)
                pred_tp1_buf.append(ptp)
            n_kept_episodes += 1
            n_total_transitions += len(ep_buf)

        if (ep + 1) % 25 == 0 or ep == args.episodes - 1:
            mean_r = float(np.mean(rewards)) if rewards else 0.0
            print(
                f"[ep {ep+1:3d}/{args.episodes}] mean_reward={mean_r:.3f} "
                f"kept_episodes={n_kept_episodes} kept_transitions={n_total_transitions}"
            )

    # ---------------------------------------------------- save
    if n_total_transitions == 0:
        print("[collect] WARNING: no transitions kept (filter too strict?)")
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        obs_t=np.stack(obs_t_buf).astype(np.float32),
        actions=np.array(act_buf, dtype=np.int64),
        obs_tp1=np.stack(obs_tp1_buf).astype(np.float32),
        predicates_t=np.stack(pred_t_buf).astype(np.float32),
        predicates_tp1=np.stack(pred_tp1_buf).astype(np.float32),
    )
    mean_r = float(np.mean(rewards)) if rewards else 0.0
    print(
        f"\n[saved] {out_path}\n"
        f"   episodes total       : {args.episodes}\n"
        f"   episodes kept        : {n_kept_episodes}\n"
        f"   episodes skipped     : {n_skipped_episodes}\n"
        f"   transitions saved    : {n_total_transitions}\n"
        f"   mean reward          : {mean_r:.3f}\n"
        f"   filter               : {args.filter}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
