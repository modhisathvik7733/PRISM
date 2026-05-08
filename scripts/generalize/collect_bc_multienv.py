"""Collect BC data across multiple BabyAI levels using the InjectingTeacher.

This is the v1.3 BC data collection script (`scripts/collect_bc_data.py`)
generalized to: (a) iterate over a list of envs, and (b) wrap the memory
teacher in an InjectingTeacher so Pickup/Open episodes actually terminate.

The output .npz has the same fields as the single-env version
(obs_seqs, action_seqs, mission_target, ep_lengths, ep_rewards) plus an
`env_ids` array so per-env diagnostics are possible later. The existing
`scripts/train_recurrent_policy.py` BC trainer ignores `env_ids` entirely,
so the file is a drop-in replacement for the v1.3 BC dataset.

Usage:
    python -m scripts.generalize.collect_bc_multienv \
        --jepa-checkpoint $CKPT \
        --envs BabyAI-GoToLocal-v0 BabyAI-Pickup-v0 BabyAI-GoTo-v0 BabyAI-Open-v0 \
        --episodes-per-env 3000 --reward-threshold 0.55 \
        --output runs/multienv_bc/bc_data.npz --device cuda
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
from prism.envs.babyai import _encode_image, make_env_with_max_steps
from prism.generalize.teacher_inject import InjectingTeacher
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.perception.predicates import type_color_index
from prism.perception.slots import NUM_COLORS, OBJECT_TYPES
from prism.utils.seed import set_global_seed


def collect_for_env(
    env_id: str,
    *,
    teacher: InjectingTeacher,
    n_target: int,
    max_steps: int,
    base_seed: int,
    seed_stride: int,
    reward_threshold: float,
    log_prefix: str,
):
    """Roll teacher in `env_id` until `n_target` episodes pass the reward
    threshold (or attempts get pathological — capped at 4×). Returns lists
    of per-episode arrays."""
    env = make_env_with_max_steps(env_id, max_steps)

    obs_seqs: list[np.ndarray] = []
    action_seqs: list[np.ndarray] = []
    mission_targets: list[np.ndarray] = []
    ep_lengths: list[int] = []
    ep_rewards: list[float] = []
    action_counts = np.zeros(env.action_space.n, dtype=np.int64)

    n_kept = 0
    n_attempted = 0
    max_attempts = n_target * 4
    while n_kept < n_target and n_attempted < max_attempts:
        ep_seed = base_seed + n_attempted * seed_stride
        n_attempted += 1
        obs, _ = env.reset(seed=ep_seed)
        teacher.reset()
        mission = obs["mission"]
        parsed = goal_predicates_for_mission(mission)
        if parsed is None:
            continue
        goal_preds, spec = parsed
        allowed = allowed_actions_for_spec(spec, env.action_space.n)

        tc_idx = type_color_index(goal_preds[0].type_id, goal_preds[0].color_id)
        mission_one_hot = np.zeros(len(OBJECT_TYPES) * NUM_COLORS, dtype=np.float32)
        mission_one_hot[tc_idx] = 1.0

        ep_obs: list[np.ndarray] = []
        ep_actions: list[int] = []
        ep_reward = 0.0

        for _ in range(max_steps):
            raw = obs["image"]
            encoded = _encode_image(raw)
            obs_t = torch.from_numpy(encoded).float()
            action, _info = teacher.select_action(
                obs_t, goal_preds, allowed_actions=allowed, spec=spec
            )
            ep_obs.append(encoded)
            ep_actions.append(int(action))
            obs, r, term, trunc, _ = env.step(action)
            ep_reward += float(r)
            if term or trunc:
                break

        if ep_reward >= reward_threshold:
            obs_seqs.append(np.stack(ep_obs).astype(np.float32))
            action_seqs.append(np.array(ep_actions, dtype=np.int64))
            mission_targets.append(mission_one_hot)
            ep_lengths.append(len(ep_actions))
            ep_rewards.append(ep_reward)
            for a in ep_actions:
                action_counts[a] += 1
            n_kept += 1

        if n_attempted % 100 == 0:
            recent_R = float(np.mean(ep_rewards[-50:])) if ep_rewards else 0.0
            print(
                f"{log_prefix} attempted={n_attempted:5d} kept={n_kept:5d}/{n_target} "
                f"recent_mean_R={recent_R:.3f}"
            )

    print(
        f"{log_prefix} DONE — kept {n_kept}/{n_target} after {n_attempted} attempts. "
        f"action histogram: {action_counts.tolist()}"
    )
    return obs_seqs, action_seqs, mission_targets, ep_lengths, ep_rewards, action_counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True,
                        help="JEPA checkpoint used by the memory teacher's "
                             "predicate probe. Pass the v1.3 GoToLocal JEPA — "
                             "the predicate probe is task-agnostic spatial "
                             "predicates, so a JEPA trained on GoToLocal "
                             "still gives reasonable predicate signals on "
                             "Pickup/GoTo/Open scenes.")
    parser.add_argument("--envs", nargs="+",
                        default=["BabyAI-GoToLocal-v0", "BabyAI-Pickup-v0",
                                 "BabyAI-GoTo-v0", "BabyAI-Open-v0"])
    parser.add_argument("--episodes-per-env", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=128,
                        help="match v1.3 (128). Pickup / Open need more than "
                             "the BabyAI default 64 because the agent must "
                             "navigate then interact, not just navigate.")
    parser.add_argument("--reward-threshold", type=float, default=0.55,
                        help="keep only episodes where the teacher succeeded "
                             "(reward >= threshold). For Pickup/Open this "
                             "implicitly filters out cases where injection "
                             "failed (e.g. mis-detected adjacency).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-stride", type=int, default=7919)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
    cfg: JepaConfig = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()

    agent = GroundedAgent(jepa, device, scoring_mode="memory")
    teacher = InjectingTeacher(agent)
    print(
        f"[multienv-bc] teacher ready (memory + injection). target = "
        f"{args.episodes_per_env} kept episodes per env across {len(args.envs)} envs."
    )

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_missions: list[np.ndarray] = []
    all_lengths: list[int] = []
    all_rewards: list[float] = []
    all_env_ids: list[str] = []
    per_env_action_counts: dict[str, list[int]] = {}

    # Stagger per-env seeds so the same base_seed isn't reused across envs.
    for env_idx, env_id in enumerate(args.envs):
        base = args.seed + env_idx * 1_000_003
        prefix = f"[{env_id}]"
        obs_seqs, act_seqs, missions, lengths, rewards, action_counts = collect_for_env(
            env_id,
            teacher=teacher,
            n_target=args.episodes_per_env,
            max_steps=args.max_steps,
            base_seed=base,
            seed_stride=args.seed_stride,
            reward_threshold=args.reward_threshold,
            log_prefix=prefix,
        )
        all_obs.extend(obs_seqs)
        all_actions.extend(act_seqs)
        all_missions.extend(missions)
        all_lengths.extend(lengths)
        all_rewards.extend(rewards)
        all_env_ids.extend([env_id] * len(obs_seqs))
        per_env_action_counts[env_id] = action_counts.tolist()

    if not all_lengths:
        print("[multienv-bc] no episodes met threshold across any env")
        return 1

    T_max = int(max(all_lengths))
    N = len(all_obs)
    obs_padded = np.zeros((N, T_max, 3, 7, 7), dtype=np.float32)
    act_padded = np.zeros((N, T_max), dtype=np.int64)
    for i, (obs_seq, act_seq) in enumerate(zip(all_obs, all_actions)):
        L = all_lengths[i]
        obs_padded[i, :L] = obs_seq
        act_padded[i, :L] = act_seq

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs_seqs=obs_padded,
        action_seqs=act_padded,
        mission_target=np.stack(all_missions),
        ep_lengths=np.array(all_lengths, dtype=np.int64),
        ep_rewards=np.array(all_rewards, dtype=np.float32),
        env_ids=np.array(all_env_ids),
        T_max=T_max,
    )
    print(f"\n[saved] {out}")
    print(f"  total episodes      : {N}")
    print(f"  T_max               : {T_max}")
    print(f"  total transitions   : {int(np.sum(all_lengths))}")
    print(f"  mean kept reward    : {float(np.mean(all_rewards)):.3f}")
    print("  per-env episode counts:")
    for env_id in args.envs:
        n_env = sum(1 for e in all_env_ids if e == env_id)
        print(f"    {env_id:30s} {n_env:5d}")
    print("  per-env action histograms (a0..a6):")
    for env_id, counts in per_env_action_counts.items():
        print(f"    {env_id:30s} {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
