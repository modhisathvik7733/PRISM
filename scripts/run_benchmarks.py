"""Run PRISM benchmarks across multiple BabyAI envs.

Loads a trained policy + JEPA + (optional) language goal head, then
evaluates success rate on each env in a list. Optionally stratifies
by held-out vs in-distribution (color, type) combos if the policy
was trained with --held-out-combos.

Output: per-env table + JSON summary + comparison to documented v2.0
multi-env PPO baseline numbers.

Usage:
    python -m scripts.run_benchmarks \\
        --policy-checkpoint runs/ppo_stage1_3_lang_heldout/policy_final.pt \\
        --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \\
        --lang-checkpoint runs/grounding_floor_tt_clean/grounding_floor_final.pt \\
        --vocab-checkpoint runs/grounding_floor_tt_clean/vocab.pt \\
        --envs BabyAI-GoToLocal-v0 BabyAI-GoTo-v0 BabyAI-GoToObj-v0 \\
        --episodes-per-env 200 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from prism.agents import goal_predicates_for_mission
from prism.agents.grounded_agent import allowed_actions_for_spec
from prism.envs.babyai import _encode_image, make_env_with_max_steps
from prism.models.jepa import JepaWorldModel, upgrade_config
from prism.models.recurrent_policy import RecurrentPolicy
from prism.perception.predicates import type_color_index
from prism.perception.slots import NUM_COLORS, OBJECT_TYPES
from prism.utils.seed import set_global_seed


# Documented v2.0 multi-env PPO baseline (from docs/EXPERIMENTS.md, v2.0
# section). v2.0 was BC-warmstarted, multi-env trained, 16-worker PPO.
V2_BASELINE: dict[str, float] = {
    "BabyAI-GoToLocal-v0": 0.946,
    "BabyAI-GoTo-v0": 0.189,
    "BabyAI-GoToObj-v0": 1.000,
}


def build_mission_oh(type_id: int, color_id: int, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    tc = type_color_index(type_id, color_id)
    out[tc] = 1.0
    return out


@torch.no_grad()
def run_episode(env, jepa, policy, mission_oh_np, allowed, max_steps, device, seed):
    obs, _ = env.reset(seed=seed)
    n_actions = env.action_space.n
    h_prev = torch.zeros(1, policy.hidden_dim, device=device)
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
            logits, _v, h_prev = policy.step_with_value(
                z, prev_a, mission_oh, h_prev, mem_feat,
            )
        else:
            logits, _v, h_prev = policy.step_with_value(
                z, prev_a, mission_oh, h_prev,
            )
        masked = logits + allowed_mask.unsqueeze(0)
        action = int(masked.argmax(dim=-1).item())
        prev_a = torch.tensor([action], device=device, dtype=torch.long)
        obs, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            return (float(reward) > 0.0), step + 1, float(reward)
    return False, max_steps, 0.0


def parse_combos(args_combos):
    out: set[tuple[int, int]] = set()
    for s in args_combos:
        c, t_idx = s.split(",")
        out.add((int(c), int(OBJECT_TYPES[int(t_idx)])))
    return out


def evaluate_env(
    env_id: str,
    jepa,
    policy,
    goal_provider,
    held_out: set,
    episodes: int,
    max_steps: int,
    seed: int,
    device: torch.device,
) -> dict:
    env = make_env_with_max_steps(env_id, max_steps)
    n_actions = env.action_space.n
    mission_dim = len(OBJECT_TYPES) * NUM_COLORS

    per_combo = defaultdict(lambda: {"success": [], "steps": []})
    successes, steps_arr, rewards = [], [], []
    n_skipped = 0
    rng = np.random.default_rng(seed)

    ep_done = 0
    attempts = 0
    while ep_done < episodes and attempts < episodes * 4:
        attempts += 1
        ep_seed = int(rng.integers(0, 1_000_000_000))
        obs, _ = env.reset(seed=ep_seed)
        parsed = goal_predicates_for_mission(obs["mission"])
        if parsed is None:
            n_skipped += 1
            continue
        rule_preds, spec = parsed
        rule_c = int(rule_preds[0].color_id)
        rule_t = int(rule_preds[0].type_id)
        rule_key = (rule_c, rule_t)

        if goal_provider is not None:
            lang_t, lang_c = goal_provider(obs["mission"])
            if lang_t < 0:
                goal_t, goal_c = rule_t, rule_c
            else:
                goal_t, goal_c = lang_t, lang_c
        else:
            goal_t, goal_c = rule_t, rule_c

        allowed = allowed_actions_for_spec(spec, n_actions)
        mission_oh = build_mission_oh(goal_t, goal_c, mission_dim)
        success, n_steps, reward = run_episode(
            env, jepa, policy, mission_oh, allowed, max_steps, device, ep_seed,
        )
        per_combo[rule_key]["success"].append(int(success))
        per_combo[rule_key]["steps"].append(n_steps)
        successes.append(int(success))
        steps_arr.append(n_steps)
        rewards.append(reward)
        ep_done += 1

    env.close()
    rate = float(np.mean(successes)) if successes else 0.0
    mean_steps = float(np.mean(steps_arr)) if steps_arr else 0.0
    mean_reward = float(np.mean(rewards)) if rewards else 0.0

    # Stratify by held-out vs ID.
    id_succ = [s for k, v in per_combo.items() if k not in held_out for s in v["success"]]
    held_succ = [s for k, v in per_combo.items() if k in held_out for s in v["success"]]
    id_rate = float(np.mean(id_succ)) if id_succ else None
    held_rate = float(np.mean(held_succ)) if held_succ else None

    return {
        "env_id": env_id,
        "n_episodes": ep_done,
        "n_skipped": n_skipped,
        "success_rate": rate,
        "mean_steps": mean_steps,
        "mean_reward": mean_reward,
        "id_rate": id_rate,
        "id_n": len(id_succ),
        "held_rate": held_rate,
        "held_n": len(held_succ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-checkpoint", required=True)
    p.add_argument("--jepa-checkpoint", required=True)
    p.add_argument("--lang-checkpoint", default=None,
                   help="if set, use lang to predict (color, type) goal")
    p.add_argument("--vocab-checkpoint", default=None)
    p.add_argument("--envs", nargs="+",
                   default=["BabyAI-GoToLocal-v0",
                            "BabyAI-GoTo-v0",
                            "BabyAI-GoToObj-v0"])
    p.add_argument("--held-out-combos", nargs="*", default=None,
                   help="if omitted, read from policy checkpoint")
    p.add_argument("--episodes-per-env", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None,
                   help="optional JSON summary path")
    args = p.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # ---- JEPA ----
    print(f"[bench] loading JEPA: {args.jepa_checkpoint}")
    jck = torch.load(args.jepa_checkpoint, map_location=device,
                     weights_only=False)
    cfg = upgrade_config(jck["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(jck["model"])
    jepa.eval()

    # ---- Policy ----
    print(f"[bench] loading policy: {args.policy_checkpoint}")
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

    # ---- Lang goal provider (optional) ----
    goal_provider = None
    if args.lang_checkpoint is not None:
        from prism.agents.lang_goal_provider import LangGoalProvider
        goal_provider = LangGoalProvider(
            lang_checkpoint=args.lang_checkpoint,
            vocab_checkpoint=args.vocab_checkpoint,
            device=device,
        )
        print(f"[bench] goal source = lang")
    else:
        print(f"[bench] goal source = rule")

    # ---- Held-out combos: CLI or checkpoint ----
    if args.held_out_combos:
        held = parse_combos(args.held_out_combos)
    else:
        ck_held = pck.get("held_out_combos", [])
        held = {tuple(t) for t in ck_held}
    if held:
        print(f"[bench] held-out combos ({len(held)}): {sorted(held)}")

    # ---- Run benchmarks ----
    results = []
    for env_id in args.envs:
        print(f"\n[bench] === {env_id} ({args.episodes_per_env} episodes) ===")
        r = evaluate_env(
            env_id, jepa, policy, goal_provider, held,
            args.episodes_per_env, args.max_steps,
            args.seed + abs(hash(env_id)) % 1000000,
            device,
        )
        results.append(r)
        print(f"  success: {r['success_rate']*100:5.1f}%  "
              f"mean_steps: {r['mean_steps']:5.1f}  "
              f"mean_reward: {r['mean_reward']:.3f}  "
              f"(n={r['n_episodes']}, skipped={r['n_skipped']})")
        if held and r["id_rate"] is not None and r["held_rate"] is not None:
            print(f"  ID combos: {r['id_rate']*100:5.1f}% (n={r['id_n']})  "
                  f"held-out: {r['held_rate']*100:5.1f}% (n={r['held_n']})  "
                  f"gap: {(r['id_rate'] - r['held_rate'])*100:+.1f} pts")

    # ---- Comparison table ----
    print("\n=== benchmark summary ===")
    header = f"  {'env':28s}  {'PRISM v4.1.4':>13s}  {'v2.0 baseline':>14s}  {'delta':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        prism = r["success_rate"] * 100
        v2 = V2_BASELINE.get(r["env_id"])
        v2_str = f"{v2*100:5.1f}%" if v2 is not None else "n/a"
        delta = (
            f"{(prism - v2*100):+.1f} pts" if v2 is not None else "—"
        )
        print(f"  {r['env_id']:28s}  {prism:12.1f}%  {v2_str:>14s}  "
              f"{delta:>8s}")

    # ---- Verdict per env ----
    print("\n=== per-env verdict ===")
    for r in results:
        v2 = V2_BASELINE.get(r["env_id"])
        prism = r["success_rate"]
        env = r["env_id"]
        if v2 is None:
            print(f"  {env}: PRISM {prism*100:.1f}% (no baseline comparison)")
            continue
        gap = v2 - prism
        if v2 < 0.3:
            print(f"  {env}: PRISM {prism*100:.1f}% vs baseline {v2*100:.1f}% "
                  f"— both weak on this env (known v2.0 ceiling); "
                  f"comparable performance")
        elif gap <= 0.10:
            print(f"  {env}: PRISM {prism*100:.1f}% within 10pp of "
                  f"BC-warmstarted v2.0 baseline {v2*100:.1f}% — "
                  f"strong, considering no BC warm-start in v4.1.4")
        elif gap <= 0.30:
            print(f"  {env}: PRISM {prism*100:.1f}% vs v2.0 {v2*100:.1f}% — "
                  f"reasonable gap ({gap*100:.0f}pp); attributable to "
                  f"no BC warm-start")
        else:
            print(f"  {env}: PRISM {prism*100:.1f}% vs v2.0 {v2*100:.1f}% — "
                  f"large gap ({gap*100:.0f}pp); the v4.1.4 policy may "
                  f"not transfer outside its training env")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "results": results,
                "v2_baseline": V2_BASELINE,
                "held_out_combos": sorted(list(held)),
                "args": {k: v for k, v in vars(args).items()
                         if not isinstance(v, torch.device)},
            }, f, indent=2)
        print(f"\n[wrote] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
