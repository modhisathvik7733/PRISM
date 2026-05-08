"""Collect (obs, action, mission, reward) trajectories from the memory-mode
agent for behavior-cloning the recurrent policy.

The memory-mode agent (scoring_mode='memory') hits ~0.60 mean reward via
hand-coded pose tracking + frontier exploration + JEPA-grounded predicate
curriculum. We use it as a teacher: dump every (obs, action) pair from
high-reward episodes plus the mission target so a learned recurrent policy
can imitate it. Once trained, the recurrent policy replaces the hand-coded
state machine with learned memory in a single GRU.

Output: a .npz with
  obs_seqs       (N, T_max, 3, 7, 7) float32 — episodes padded to T_max
  action_seqs    (N, T_max) int64
  mission_target (N, 24) float32 — one-hot of (type, color) target pair
  ep_lengths     (N,) int64 — actual length of each episode
  ep_rewards     (N,) float32 — final episode reward (for filtering)

Usage:
    python -m scripts.collect_bc_data \
        --jepa-checkpoint runs/<...>/jepa_final.pt \
        --episodes 1000 --output data/bc_v0.9.npz --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch

from prism.agents import GroundedAgent, goal_predicates_for_mission
from prism.agents.grounded_agent import allowed_actions_for_spec
from prism.envs.babyai import _encode_image
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.perception.predicates import type_color_index
from prism.perception.slots import NUM_COLORS, OBJECT_TYPES
from prism.utils.seed import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reward-threshold", type=float, default=0.55,
                        help="only save episodes with reward >= this value")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-stride", type=int, default=7919,
                        help="per-episode seed = seed + ep * seed_stride. "
                             "Use a value coprime to the eval seeds' stride to "
                             "avoid leak when you eval at the same base seed.")
    parser.add_argument("--exclude-seeds", default="",
                        help="comma-separated list of seeds to exclude from the "
                             "training data. Set to '0,100,200' to keep eval "
                             "seeds held out from the BC distribution.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    excluded = set(int(s) for s in args.exclude_seeds.split(",") if s.strip())

    set_global_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
    cfg: JepaConfig = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()

    agent = GroundedAgent(jepa, device, scoring_mode="memory")
    print(f"[bc-collect] memory-mode teacher ready, target N≥{args.episodes} episodes")

    env = gym.make(args.env_id)

    obs_seqs = []
    action_seqs = []
    mission_targets = []
    ep_lengths = []
    ep_rewards = []

    n_kept = 0
    ep_seed = args.seed
    ep = 0
    n_attempted = 0
    while ep < args.episodes:
        ep_seed = args.seed + n_attempted * args.seed_stride
        n_attempted += 1
        if ep_seed in excluded:
            continue
        obs, _ = env.reset(seed=ep_seed)
        agent.reset()
        mission = obs["mission"]
        parsed = goal_predicates_for_mission(mission)
        if parsed is None:
            ep += 1
            continue
        goal_preds, spec = parsed
        allowed = allowed_actions_for_spec(spec, env.action_space.n)

        # Mission target = one-hot of (type, color)
        tc_idx = type_color_index(goal_preds[0].type_id, goal_preds[0].color_id)
        mission_one_hot = np.zeros(len(OBJECT_TYPES) * NUM_COLORS, dtype=np.float32)
        mission_one_hot[tc_idx] = 1.0

        ep_obs = []
        ep_actions = []
        ep_reward = 0.0

        for _ in range(args.max_steps):
            raw = obs["image"]
            encoded = _encode_image(raw)            # (3, 7, 7) float32
            obs_t = torch.from_numpy(encoded).float()
            action, _info = agent.select_action(obs_t, goal_preds, allowed_actions=allowed)
            ep_obs.append(encoded)
            ep_actions.append(action)
            obs, r, term, trunc, _ = env.step(action)
            ep_reward += float(r)
            if term or trunc:
                break

        if ep_reward >= args.reward_threshold:
            obs_seqs.append(np.stack(ep_obs).astype(np.float32))    # (T, 3, 7, 7)
            action_seqs.append(np.array(ep_actions, dtype=np.int64))
            mission_targets.append(mission_one_hot)
            ep_lengths.append(len(ep_actions))
            ep_rewards.append(ep_reward)
            n_kept += 1

        ep += 1
        if ep % 50 == 0:
            mean_r_recent = float(np.mean(ep_rewards[-50:])) if ep_rewards else 0.0
            print(f"[bc {ep:4d}/{args.episodes}] attempted={n_attempted} "
                  f"kept={n_kept}  recent_mean_R={mean_r_recent:.3f}")

    # Pad to T_max so we get a single tensor
    if n_kept == 0:
        print("[bc-collect] no episodes met threshold — try lowering --reward-threshold")
        return 1
    T_max = int(max(ep_lengths))
    N = n_kept
    obs_padded = np.zeros((N, T_max, 3, 7, 7), dtype=np.float32)
    act_padded = np.zeros((N, T_max), dtype=np.int64)
    for i, (obs_seq, act_seq) in enumerate(zip(obs_seqs, action_seqs)):
        L = ep_lengths[i]
        obs_padded[i, :L] = obs_seq
        act_padded[i, :L] = act_seq

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs_seqs=obs_padded,
        action_seqs=act_padded,
        mission_target=np.stack(mission_targets),
        ep_lengths=np.array(ep_lengths, dtype=np.int64),
        ep_rewards=np.array(ep_rewards, dtype=np.float32),
        T_max=T_max,
    )
    print(f"\n[saved] {out}")
    print(f"  episodes total      : {args.episodes}")
    print(f"  kept (R≥{args.reward_threshold:.2f})    : {n_kept}")
    print(f"  T_max               : {T_max}")
    print(f"  total transitions   : {int(np.sum(ep_lengths))}")
    print(f"  mean kept reward    : {float(np.mean(ep_rewards)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
