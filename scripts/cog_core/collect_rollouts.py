"""Collect rollouts from a trained PPO policy: save raw obs, actions,
JEPA latents, and ground-truth slot extractions per step.

Output schema (.npz):
  obs           (N, T_max, 3, 7, 7)  float32  — encoded obs
  actions       (N, T_max)           int64
  latents       (N, T_max, ...)      float32  — flat or spatial JEPA latents
  ep_lengths    (N,)                 int64
  env_ids       (N,)                 str       — which env produced each episode
  slots_jsonl   list[list[dict]]     — per (episode, step), list of slot dicts
                                        with keys (type_id, color_id, x, y)

Used by:
  - train_object_tracker.py (uses obs+latents+slots)
  - extract_operators.py (uses latents+actions)
  - eval_emergence.py (uses everything)

Usage:
    python -m scripts.cog_core.collect_rollouts \
        --jepa-checkpoint $V13_JEPA \
        --policy-checkpoint runs/ppo_v6_pathB/policy_iter400.pt \
        --envs BabyAI-GoToLocal-v0 BabyAI-GoTo-v0 BabyAI-GoToObj-v0 \
        --episodes-per-env 500 \
        --output runs/cog_core_phase1/rollouts.npz
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from prism.agents import GroundedAgent, goal_predicates_for_mission
from prism.agents.grounded_agent import allowed_actions_for_spec
from prism.cog_core.world_model_rollout import WorldModelRollout
from prism.envs.babyai import _encode_image, make_env_with_max_steps
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.models.recurrent_policy import RecurrentPolicy
from prism.perception import extract_slots
from prism.perception.predicates import type_color_index
from prism.perception.slots import NUM_COLORS, OBJECT_TYPES
from prism.utils.seed import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--envs", nargs="+",
                        default=["BabyAI-GoToLocal-v0",
                                 "BabyAI-GoTo-v0",
                                 "BabyAI-GoToObj-v0"])
    parser.add_argument("--episodes-per-env", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # --- frozen JEPA ---
    jepa_ckpt = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
    jepa_cfg: JepaConfig = upgrade_config(jepa_ckpt["cfg"])
    jepa = JepaWorldModel(jepa_cfg).to(device)
    jepa.load_state_dict(jepa_ckpt["model"])
    jepa.eval()
    world = WorldModelRollout(jepa, device)
    print(f"[collect] frozen JEPA loaded: encoder={jepa_cfg.encoder_type}")

    # --- policy ---
    pckpt = torch.load(args.policy_checkpoint, map_location=device, weights_only=False)
    ckpt_mem_dim = int(pckpt.get("mem_feat_dim", 0) or 0)
    policy = RecurrentPolicy(
        latent_in_dim=pckpt["latent_in_dim"],
        n_actions=pckpt["n_actions"],
        mission_dim=pckpt["mission_dim"],
        hidden_dim=pckpt["hidden_dim"],
        latent_proj_dim=pckpt["latent_proj_dim"],
        mem_feat_dim=ckpt_mem_dim,
    ).to(device)
    policy.load_state_dict(pckpt["policy_state_dict"])
    policy.eval()
    print(f"[collect] policy loaded: {args.policy_checkpoint} (mem_dim={ckpt_mem_dim})")

    agent = GroundedAgent(jepa, device, scoring_mode="recurrent")

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_latents: list[np.ndarray] = []
    all_lengths: list[int] = []
    all_env_ids: list[str] = []
    all_slots: list[list[list[dict]]] = []

    for env_id in args.envs:
        env = make_env_with_max_steps(env_id, args.max_steps)
        print(f"\n[collect] {env_id}: target {args.episodes_per_env} episodes")
        kept = 0
        attempted = 0
        max_attempts = args.episodes_per_env * 4
        while kept < args.episodes_per_env and attempted < max_attempts:
            ep_seed = args.seed + len(all_obs) * 7919 + attempted * 13
            attempted += 1
            obs, _ = env.reset(seed=ep_seed)
            agent.reset()
            mission = obs["mission"]
            parsed = goal_predicates_for_mission(mission)
            if parsed is None:
                continue
            goal_preds, spec = parsed
            allowed = allowed_actions_for_spec(spec, env.action_space.n)

            tc_idx = type_color_index(goal_preds[0].type_id, goal_preds[0].color_id)
            mission_oh = torch.zeros(len(OBJECT_TYPES) * NUM_COLORS)
            mission_oh[tc_idx] = 1.0
            agent.attach_recurrent_policy(
                policy, mission_oh,
                goal_type=goal_preds[0].type_id,
                goal_color=goal_preds[0].color_id,
            )

            ep_obs: list[np.ndarray] = []
            ep_actions: list[int] = []
            ep_latents: list[np.ndarray] = []
            ep_slots: list[list[dict]] = []

            for _ in range(args.max_steps):
                raw = obs["image"]
                encoded = _encode_image(raw)               # (3, 7, 7)
                slots_at_t = [
                    {"type_id": int(s.type_id), "color_id": int(s.color_id),
                     "x": int(s.x), "y": int(s.y)}
                    for s in extract_slots(raw)
                ]
                # Encode through JEPA for latent storage.
                with torch.no_grad():
                    z = world.encode(torch.from_numpy(encoded).float().unsqueeze(0))
                ep_obs.append(encoded)
                ep_slots.append(slots_at_t)
                ep_latents.append(z.squeeze(0).cpu().numpy())

                obs_t = torch.from_numpy(encoded).float()
                action, _info = agent.select_action(
                    obs_t, goal_preds, allowed_actions=allowed,
                )
                ep_actions.append(int(action))
                obs, _r, term, trunc, _ = env.step(action)
                if term or trunc:
                    break

            all_obs.append(np.stack(ep_obs).astype(np.float32))
            all_actions.append(np.array(ep_actions, dtype=np.int64))
            all_latents.append(np.stack(ep_latents).astype(np.float32))
            all_lengths.append(len(ep_actions))
            all_env_ids.append(env_id)
            all_slots.append(ep_slots)
            kept += 1

            if kept % 100 == 0:
                print(f"  [{kept}/{args.episodes_per_env}] kept, "
                      f"{attempted} attempted")

        print(f"  done: {kept}/{args.episodes_per_env} (after {attempted} attempts)")

    # Pad to T_max for the dense numpy save.
    if not all_lengths:
        print("[collect] no episodes collected — aborting")
        return 1
    T_max = max(all_lengths)
    N = len(all_obs)
    obs_padded = np.zeros((N, T_max, 3, 7, 7), dtype=np.float32)
    act_padded = np.zeros((N, T_max), dtype=np.int64)
    # Latent shape can be flat (D,) or spatial (C, H, W); pad accordingly.
    sample_latent = all_latents[0][0]
    latent_shape = sample_latent.shape
    lat_padded = np.zeros((N, T_max) + latent_shape, dtype=np.float32)
    for i in range(N):
        L = all_lengths[i]
        obs_padded[i, :L] = all_obs[i]
        act_padded[i, :L] = all_actions[i]
        lat_padded[i, :L] = all_latents[i]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=obs_padded,
        actions=act_padded,
        latents=lat_padded,
        ep_lengths=np.array(all_lengths, dtype=np.int64),
        env_ids=np.array(all_env_ids),
        T_max=T_max,
    )
    # Slots are ragged (variable count per frame) — save separately as pickle.
    with open(out.with_suffix(".slots.pkl"), "wb") as f:
        pickle.dump(all_slots, f)

    print(f"\n[saved] {out}")
    print(f"[saved] {out.with_suffix('.slots.pkl')}")
    print(f"  episodes:   {N}")
    print(f"  T_max:      {T_max}")
    print(f"  per-env count:")
    for env_id in args.envs:
        n_env = sum(1 for e in all_env_ids if e == env_id)
        print(f"    {env_id:30s} {n_env}")
    print(f"  latent shape: {latent_shape}")
    print(f"  total transitions: {sum(all_lengths):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
