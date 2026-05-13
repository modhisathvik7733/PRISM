"""Go/no-go gate for DreamerV3-style imagination training.

The plan in `~/.claude/plans/unity-demo-pretrain-concepts-py-81-shiny-abelson.md`
proposes adding an imagination-based auxiliary loss to PPO. The whole approach
hinges on JEPA's `LatentDynamics` producing trustworthy multi-step rollouts on
*our specific obs distribution* (synth Unity nav env / Unity bridge). JEPA was
trained on BabyAI procedural obs; Unity-style sparse obs is a different
distribution. We need to measure how far we can trust imagined rollouts before
they drift.

This script runs the policy in real episodes, captures (obs_t, action_t)
sequences, then for each H ∈ {5, 10, 15, 25}:
  1. Encodes obs_t to z_t.
  2. Rolls the JEPA dynamics forward H steps using the same actions.
  3. Compares the imagined ẑ_{t+i} to the real-encoded z'_{t+i}.

Reports mean cosine similarity and L2 distance per step.

**Decision threshold:** if cos@H=10 ≥ 0.9 → proceed with imagination loop at H=10.
If cos@H=10 < 0.7 → abandon imagination, fall back to pure-PPO + predicate
reward shaping (see plan, fallback section).

Usage:
    python scripts/check_rollout_fidelity.py \\
        --jepa  runs/v6_concept_phaseAB_v2/jepa.pt \\
        --policy runs/v6_concept_phaseAB_v2/policy.pt \\
        --n-episodes 50 \\
        --device mps      # or cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prism.adapters.babyai_adapter import BabyAIAdapter
from prism.cognition.policy import UniversalPolicy
from prism.models.jepa import JepaWorldModel, upgrade_config

from unity_demo.unity_nav_env import UnityNavEnv


def load_jepa(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    return jepa, cfg


def build_policy(ckpt: dict, jepa, cfg, device: torch.device, trunk: str):
    adapter = BabyAIAdapter(jepa=jepa, cfg=cfg, device=device)
    policy = UniversalPolicy.from_adapter(
        adapter,
        trunk=trunk,
        hidden_dim=ckpt["hidden_dim"],
        latent_proj_dim=ckpt["latent_proj_dim"],
        mem_feat_dim=ckpt.get("mem_feat_dim", 0),
        concept_n_slots=ckpt.get("concept_n_slots", 1024),
        operator_n_slots=ckpt.get("operator_n_slots", 64),
        concept_scaling=ckpt.get("concept_scaling", 1.0),
        operator_scaling=ckpt.get("operator_scaling", 4.0),
        use_operator_memory=ckpt.get("use_operator_memory", True),
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


@torch.no_grad()
def collect_real_trajectory(
    env: UnityNavEnv,
    policy: UniversalPolicy,
    jepa,
    device: torch.device,
    horizon: int,
) -> tuple[list[torch.Tensor], list[int]] | None:
    """Run one episode and return (encoded z's, action_ints) lists of length
    >= horizon+1. Returns None if the episode ends before reaching that length."""
    obs, _info = env.reset()
    h = policy.init_hidden(1, device)
    prev_a = torch.tensor([-1], device=device, dtype=torch.long)
    # Mission is fixed per episode; we don't need it for the fidelity check
    # because the dynamics doesn't depend on mission.
    from prism.perception.predicates import type_color_index
    from prism.perception.slots import COLOR_NAME_TO_IDX, OBJECT_NAME_TO_TYPE, NUM_COLORS, NUM_TYPES
    color = COLOR_NAME_TO_IDX.get("green", 1)
    typ = OBJECT_NAME_TO_TYPE.get("ball", 6)
    mission = torch.zeros(1, NUM_TYPES * NUM_COLORS, device=device)
    mission[0, type_color_index(typ, color)] = 1.0

    zs: list[torch.Tensor] = []
    actions: list[int] = []
    terminated = truncated = False
    while not (terminated or truncated):
        img = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(device)
        z = jepa.encode(img)
        if z.ndim > 2:
            z = z.flatten(1)
        zs.append(z.squeeze(0))  # (D,)

        # Argmax-action under the concept-pretrained policy.
        logits, h_next = policy.step(z, prev_a, mission, h)
        mask = torch.full_like(logits, float("-inf"))
        mask[..., :3] = 0.0  # only allow turn_left, turn_right, forward
        a = int((logits + mask).argmax(dim=-1).item())
        actions.append(a)

        obs, _r, terminated, truncated, _info = env.step(a)
        prev_a = torch.tensor([a], device=device, dtype=torch.long)
        h = h_next

        if len(actions) >= horizon + 1:
            break

    if len(zs) < horizon + 1:
        return None
    return zs[: horizon + 1], actions[: horizon + 1]


@torch.no_grad()
def imagine_trajectory(jepa, z0: torch.Tensor, actions: list[int], device: torch.device):
    """One-shot multi-step rollout via JEPA.predict, starting from z0."""
    zs = [z0]
    z = z0
    for a in actions:
        a_t = torch.tensor([a], device=device, dtype=torch.long)
        z_next = jepa.predict(z.unsqueeze(0), a_t).squeeze(0)
        zs.append(z_next)
        z = z_next
    return zs  # length len(actions) + 1


def compare_trajectories(real_zs: list[torch.Tensor], imag_zs: list[torch.Tensor]):
    """Per-step cosine sim and L2 distance."""
    cos_per_step = []
    l2_per_step = []
    for r, i in zip(real_zs, imag_zs):
        r_flat = r.flatten()
        i_flat = i.flatten()
        cos = float(F.cosine_similarity(r_flat.unsqueeze(0), i_flat.unsqueeze(0), dim=-1).item())
        l2 = float((r_flat - i_flat).norm().item())
        cos_per_step.append(cos)
        l2_per_step.append(l2)
    return cos_per_step, l2_per_step


def main() -> int:
    p = argparse.ArgumentParser(description="JEPA rollout-fidelity probe.")
    p.add_argument("--jepa", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--trunk", default="transformer", choices=["transformer", "gru"])
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 15, 25])
    p.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available()
                 else ("mps" if torch.backends.mps.is_available() else "cpu")),
    )
    # Env / dynamics match Unity (see unity_nav_env.py defaults).
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--reach-threshold", type=float, default=1.6)
    p.add_argument("--forward-step", type=float, default=0.07)
    p.add_argument("--obs-scale", type=float, default=2.0)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[fidelity] device={device}")
    print(f"[fidelity] loading JEPA from {args.jepa}")
    jepa, cfg = load_jepa(Path(args.jepa), device)
    print(f"[fidelity] loading policy from {args.policy}")
    base_ckpt = torch.load(args.policy, map_location=device, weights_only=False)
    policy = build_policy(base_ckpt, jepa, cfg, device, trunk=args.trunk)

    env = UnityNavEnv(
        max_steps=args.max_steps,
        reach_threshold=args.reach_threshold,
        forward_step=args.forward_step,
        obs_scale=args.obs_scale,
        randomize_target_color=True,  # exercise mission diversity
        seed=12345,
    )

    H_max = max(args.horizons)
    print(f"[fidelity] collecting trajectories of length {H_max+1} from "
          f"{args.n_episodes} episodes...")

    real_traj: list[list[torch.Tensor]] = []
    actions_per_ep: list[list[int]] = []
    attempts = 0
    while len(real_traj) < args.n_episodes:
        attempts += 1
        if attempts > args.n_episodes * 10:
            print(f"[fidelity] WARN: only got {len(real_traj)} eps after "
                  f"{attempts} attempts; episodes too short. Continuing.")
            break
        out = collect_real_trajectory(env, policy, jepa, device, H_max)
        if out is None:
            continue
        zs, acts = out
        real_traj.append(zs)
        actions_per_ep.append(acts)

    print(f"[fidelity] collected {len(real_traj)} usable trajectories\n")

    # For each H, compute cos+L2 at step H (i.e., the prediction at depth H).
    print(f"{'H':>4}  {'cos mean':>10}  {'cos std':>9}  {'L2 mean':>9}  {'L2 std':>9}")
    print("-" * 50)
    decisions = {}
    for H in args.horizons:
        cos_at_H = []
        l2_at_H = []
        for zs, acts in zip(real_traj, actions_per_ep):
            # Imagine starting from z_0 using first H actions.
            imag_zs = imagine_trajectory(jepa, zs[0], acts[:H], device)
            # zs has H+1 entries (including z_0). imag_zs also has H+1.
            cos, l2 = compare_trajectories(zs[: H + 1], imag_zs)
            # We care about the prediction quality at the END of the horizon.
            cos_at_H.append(cos[H])
            l2_at_H.append(l2[H])
        m_cos = float(np.mean(cos_at_H))
        s_cos = float(np.std(cos_at_H))
        m_l2 = float(np.mean(l2_at_H))
        s_l2 = float(np.std(l2_at_H))
        decisions[H] = m_cos
        print(f"{H:>4d}  {m_cos:>10.3f}  {s_cos:>9.3f}  {m_l2:>9.3f}  {s_l2:>9.3f}")

    print()
    # Decision logic.
    cos_at_10 = decisions.get(10)
    cos_at_15 = decisions.get(15)
    if cos_at_10 is not None and cos_at_10 >= 0.9:
        print(f"[fidelity] DECISION: PROCEED with imagination, H=10 (cos={cos_at_10:.3f})")
        if cos_at_15 is not None and cos_at_15 >= 0.9:
            print(f"[fidelity]           H=15 is also safe (cos={cos_at_15:.3f}) — "
                  "consider it for tighter credit assignment")
        return 0
    elif cos_at_10 is not None and cos_at_10 >= 0.7:
        print(f"[fidelity] DECISION: MARGINAL at H=10 (cos={cos_at_10:.3f}).")
        print("[fidelity]           Try imagination with shorter H (e.g., 5) "
              "or hybrid weight ≤0.3, or pivot to predicate-shaping fallback.")
        return 0
    else:
        cos_str = f"{cos_at_10:.3f}" if cos_at_10 is not None else "N/A"
        print(f"[fidelity] DECISION: ABORT imagination (cos@10={cos_str} < 0.7).")
        print("[fidelity]           Fall back to pure-PPO + predicate reward shaping.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
