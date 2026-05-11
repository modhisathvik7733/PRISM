"""Stage 1.3 — compositional generalization at the policy level.

Loads a trained PPO policy (typically from a run that used
`--held-out-combos`) and measures success rate stratified by whether
the episode's mission target `(color, type)` is in the held-out set
or in the in-distribution training set.

Pipeline:
  obs → JEPA.encode → z
  text/rule → mission_oh (24-d)
  (z, mission_oh, h_prev) → policy → action_logits
  argmax → step env

Reports:
  - per-combo success rate
  - ID aggregate (combos PPO trained on)
  - held-out aggregate (combos PPO never saw at training)
  - compositional gap (ID - held)

A pass = held-out success ≥ 0.7 × ID success (i.e. < 30% relative drop).

Usage:
    python -m scripts.eval_lang_policy_compositional \\
        --policy-checkpoint runs/ppo_stage1_3_lang/policy_final.pt \\
        --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \\
        --lang-checkpoint runs/grounding_floor_tt_clean/grounding_floor_final.pt \\
        --vocab-checkpoint runs/grounding_floor_tt_clean/vocab.pt \\
        --env-id BabyAI-GoToLocal-v0 \\
        --episodes-per-combo 30 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from prism.agents import goal_predicates_for_mission
from prism.agents.grounded_agent import allowed_actions_for_spec
from prism.envs.babyai import _encode_image, make_env_with_max_steps
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.models.recurrent_policy import RecurrentPolicy
from prism.perception.predicates import type_color_index
from prism.perception.slots import NUM_COLORS, NUM_TYPES, OBJECT_TYPES
from prism.utils.seed import set_global_seed


def parse_combos(args_combos) -> set[tuple[int, int]]:
    """Returns set of (color_id, type_id) from 'color,type_idx' strings."""
    out: set[tuple[int, int]] = set()
    for s in args_combos:
        c_str, t_str = s.split(",")
        c_id = int(c_str)
        t_idx = int(t_str)
        out.add((c_id, int(OBJECT_TYPES[t_idx])))
    return out


def build_mission_oh(type_id: int, color_id: int, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    tc = type_color_index(type_id, color_id)
    out[tc] = 1.0
    return out


@torch.no_grad()
def run_episode(
    env,
    jepa: JepaWorldModel,
    policy: RecurrentPolicy,
    mission_oh_np: np.ndarray,
    allowed: tuple[int, ...],
    max_steps: int,
    device: torch.device,
    seed: int,
) -> tuple[bool, int, float]:
    """Roll out one episode under the policy. Returns (success, steps, reward)."""
    obs, _ = env.reset(seed=seed)
    n_actions = env.action_space.n
    h_prev = torch.zeros(1, policy.hidden_dim, device=device)
    # RecurrentPolicy.step_with_value signature:
    #   (z, prev_action, mission, h[, mem_feat]) -> (logits, value, h_next)
    # prev_action is (B,) int64; use -1 for the first step ("no action" embed).
    prev_a = torch.tensor([-1], device=device, dtype=torch.long)
    allowed_mask = torch.full(
        (n_actions,), -1e9, device=device, dtype=torch.float32,
    )
    for a in allowed:
        allowed_mask[a] = 0.0
    mission_oh = torch.from_numpy(mission_oh_np).to(device).unsqueeze(0)
    mem_feat_dim = int(getattr(policy, "mem_feat_dim", 0) or 0)
    mem_feat = (
        torch.zeros(1, mem_feat_dim, device=device)
        if mem_feat_dim > 0 else None
    )

    for step in range(max_steps):
        encoded = _encode_image(obs["image"])
        z = jepa.encode(
            torch.from_numpy(encoded).float().unsqueeze(0).to(device),
        )
        if mem_feat is not None:
            logits, _value, h_prev = policy.step_with_value(
                z, prev_a, mission_oh, h_prev, mem_feat,
            )
        else:
            logits, _value, h_prev = policy.step_with_value(
                z, prev_a, mission_oh, h_prev,
            )
        masked = logits + allowed_mask.unsqueeze(0)
        action = int(masked.argmax(dim=-1).item())
        prev_a = torch.tensor([action], device=device, dtype=torch.long)
        obs, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            return (float(reward) > 0.0), step + 1, float(reward)
    return False, max_steps, 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-checkpoint", required=True)
    p.add_argument("--jepa-checkpoint", required=True)
    p.add_argument("--lang-checkpoint", default=None,
                   help="if set, predict goal via lang head instead of rule")
    p.add_argument("--vocab-checkpoint", default=None)
    p.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    p.add_argument("--held-out-combos", nargs="*", default=None,
                   help="combos to evaluate as held-out. If omitted, read "
                        "from the policy checkpoint's saved held_out_combos.")
    p.add_argument("--episodes-per-combo", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # ---- Load JEPA ----
    print(f"[stage1.3] loading JEPA: {args.jepa_checkpoint}")
    jck = torch.load(args.jepa_checkpoint, map_location=device,
                     weights_only=False)
    cfg = upgrade_config(jck["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(jck["model"])
    jepa.eval()

    # ---- Load policy ----
    print(f"[stage1.3] loading policy: {args.policy_checkpoint}")
    pck = torch.load(args.policy_checkpoint, map_location=device,
                     weights_only=False)
    policy = RecurrentPolicy(
        latent_in_dim=pck["latent_in_dim"],
        n_actions=pck["n_actions"],
        mission_dim=pck["mission_dim"],
        hidden_dim=pck["hidden_dim"],
        latent_proj_dim=pck["latent_proj_dim"],
        mem_feat_dim=int(pck.get("mem_feat_dim", 0) or 0),
    ).to(device)
    policy.load_state_dict(pck["policy_state_dict"])
    policy.eval()
    print(f"[stage1.3]   policy: latent_in={pck['latent_in_dim']} "
          f"mission_dim={pck['mission_dim']} hidden={pck['hidden_dim']}")

    # ---- Optional lang head (for prediction-time goal) ----
    goal_provider = None
    if args.lang_checkpoint is not None:
        from prism.agents.lang_goal_provider import LangGoalProvider
        goal_provider = LangGoalProvider(
            lang_checkpoint=args.lang_checkpoint,
            vocab_checkpoint=args.vocab_checkpoint,
            device=device,
        )
        print(f"[stage1.3]   eval goal source = lang")
    else:
        print(f"[stage1.3]   eval goal source = rule")

    # ---- Held-out combos: from CLI or checkpoint ----
    if args.held_out_combos is not None and len(args.held_out_combos) > 0:
        held = parse_combos(args.held_out_combos)
    else:
        ck_held = pck.get("held_out_combos", [])
        held = {tuple(t) for t in ck_held}
    print(f"[stage1.3]   held-out combos ({len(held)}): {sorted(held)}")

    # ---- Run episodes ----
    env = make_env_with_max_steps(args.env_id, args.max_steps)
    n_actions = env.action_space.n
    mission_dim = len(OBJECT_TYPES) * NUM_COLORS

    per_combo: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"success": [], "steps": [], "n_seen": 0}
    )
    # We need to balance episodes across the 24 possible (color, type)
    # combos. Loop episodes; if a combo already has enough samples,
    # skip and try another seed.
    target_per_combo = args.episodes_per_combo
    max_total_attempts = 24 * target_per_combo * 5

    total_done = 0
    attempts = 0
    rng = np.random.default_rng(args.seed)
    while attempts < max_total_attempts:
        attempts += 1
        ep_seed = int(rng.integers(0, 1_000_000_000))
        obs, _ = env.reset(seed=ep_seed)
        mission = obs["mission"]
        parsed = goal_predicates_for_mission(mission)
        if parsed is None:
            continue
        rule_preds, spec = parsed
        rule_t_id = int(rule_preds[0].type_id)
        rule_c_id = int(rule_preds[0].color_id)
        rule_key = (rule_c_id, rule_t_id)
        if per_combo[rule_key]["n_seen"] >= target_per_combo:
            continue

        # Determine the goal that drives the policy.
        if goal_provider is not None:
            lang_t_id, lang_c_id = goal_provider(mission)
            if lang_t_id < 0 or lang_c_id < 0:
                goal_t_id, goal_c_id = rule_t_id, rule_c_id
            else:
                goal_t_id, goal_c_id = lang_t_id, lang_c_id
        else:
            goal_t_id, goal_c_id = rule_t_id, rule_c_id

        allowed = allowed_actions_for_spec(spec, n_actions)
        mission_oh = build_mission_oh(goal_t_id, goal_c_id, mission_dim)
        success, steps, reward = run_episode(
            env, jepa, policy, mission_oh, allowed,
            args.max_steps, device, ep_seed,
        )
        per_combo[rule_key]["success"].append(int(success))
        per_combo[rule_key]["steps"].append(steps)
        per_combo[rule_key]["n_seen"] += 1
        total_done += 1

        # Stop when all combos have enough samples.
        all_full = all(
            per_combo[k]["n_seen"] >= target_per_combo
            for k in per_combo
            if per_combo[k]["n_seen"] > 0
        )
        if all_full and total_done >= 12 * target_per_combo:
            # Heuristic — if we've sampled at least 12 combos at target depth.
            pass

    env.close()

    # ---- Stratify ----
    id_records = {k: v for k, v in per_combo.items() if k not in held}
    held_records = {k: v for k, v in per_combo.items() if k in held}

    def rate(rs: dict) -> tuple[float, int]:
        all_s: list[int] = []
        for v in rs.values():
            all_s.extend(v["success"])
        return (float(np.mean(all_s)) if all_s else 0.0), len(all_s)

    id_rate, id_n = rate(id_records)
    held_rate, held_n = rate(held_records)

    print("\n=== per-combo success rates ===")
    print(f"  {'combo (c, t_id)':>16}  {'group':>8}  {'n':>4}  {'success%':>9}")
    for key in sorted(per_combo.keys()):
        v = per_combo[key]
        if not v["success"]:
            continue
        group = "HELD" if key in held else "id"
        r = float(np.mean(v["success"])) * 100
        print(f"  {str(key):>16}  {group:>8}  {len(v['success']):>4}  "
              f"{r:>8.1f}%")

    print(f"\n=== aggregates ===")
    print(f"  in-distribution combos: success={id_rate*100:.1f}%  (n={id_n})")
    print(f"  held-out combos:        success={held_rate*100:.1f}%  (n={held_n})")
    gap = id_rate - held_rate
    rel_drop = (gap / id_rate * 100) if id_rate > 1e-6 else float("inf")
    print(f"  compositional gap (ID − held):  {gap*100:+.1f} pts "
          f"(relative drop {rel_drop:.1f}%)")

    print("\n=== verdict ===")
    if held_n == 0:
        print("  N/A — no held-out episodes (checkpoint reported no "
              "held-out combos and none were passed via --held-out-combos)")
    elif id_rate < 0.05:
        print("  N/A — ID success too low to measure a meaningful gap")
    elif held_rate >= 0.7 * id_rate:
        print(f"  PASS — held-out success ≥ 70% of ID success. "
              f"Policy generalizes compositionally across held-out "
              f"(color, type) combos.")
    elif held_rate >= 0.4 * id_rate:
        print(f"  PARTIAL — held-out success {held_rate/id_rate*100:.0f}% "
              f"of ID. Some generalization but a real compositional gap.")
    else:
        print(f"  FAIL — held-out success {held_rate/id_rate*100:.0f}% of "
              f"ID. Policy memorized training combos rather than learning "
              f"a goal-conditioned strategy.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "per_combo": {
                    f"{k[0]},{k[1]}": {
                        "n": v["n_seen"],
                        "success_rate": float(np.mean(v["success"]))
                        if v["success"] else None,
                    }
                    for k, v in per_combo.items()
                },
                "id_rate": id_rate,
                "id_n": id_n,
                "held_rate": held_rate,
                "held_n": held_n,
                "gap_pts": gap * 100,
                "held_combos": sorted(list(held)),
            }, f, indent=2)
        print(f"\n[wrote] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
