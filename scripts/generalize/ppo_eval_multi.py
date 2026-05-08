"""Evaluate a single recurrent-policy checkpoint across multiple BabyAI envs
and produce one comparison table.

Reuses `scripts/eval_agent_cohorts.run_episode` so per-cohort definitions
stay identical to the v1.3 capstone — what changes is only the outer loop:
for each env, run N episodes, collect (mean_R, success%, mean_steps), then
emit a single table at the end.

Pass the same JEPA checkpoint that was used to train the policy. For the
universal policy, that's the universal JEPA. For the v1.3 zero-shot
baseline, pass `runs/<v1.3-jepa>/jepa_final.pt`.

Usage:
    python -m scripts.generalize.ppo_eval_multi \
        --jepa-checkpoint runs/jepa_universal/jepa_final.pt \
        --policy-checkpoint runs/<run>/policy_final.pt \
        --envs BabyAI-GoToLocal-v0 BabyAI-Pickup-v0 BabyAI-GoTo-v0 BabyAI-Open-v0 \
        --episodes 1000 --max-steps 128 --device cuda
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from prism.agents import GroundedAgent
from prism.agents.grounded_agent import allowed_actions_for_spec
from prism.envs.babyai import _encode_image, make_env_with_max_steps
from prism.generalize.mission_parser_v2 import goal_predicates_for_mission_ext
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.models.recurrent_policy import RecurrentPolicy
from prism.perception import compute_predicates, extract_slots
from prism.perception.predicates import type_color_index
from prism.perception.slots import NUM_COLORS, OBJECT_TYPES
from prism.utils.seed import set_global_seed
from scripts.eval_agent_cohorts import initial_cohort


def run_episode_ext(env, agent, *, seed, max_steps, recurrent_policy=None):
    """Drop-in replacement for `eval_agent_cohorts.run_episode` that uses
    the v2 (extended) mission parser. Same return shape, so the multi-env
    eval aggregator and per-cohort breakdown work unchanged."""
    obs, _ = env.reset(seed=seed)
    agent.reset()
    mission = obs["mission"]
    parsed = goal_predicates_for_mission_ext(mission)
    if parsed is None:
        return None
    goal_preds, spec = parsed
    allowed = allowed_actions_for_spec(spec, env.action_space.n)

    if recurrent_policy is not None:
        tc_idx = type_color_index(goal_preds[0].type_id, goal_preds[0].color_id)
        mission_one_hot = torch.zeros(len(OBJECT_TYPES) * NUM_COLORS)
        mission_one_hot[tc_idx] = 1.0
        agent.attach_recurrent_policy(
            recurrent_policy, mission_one_hot,
            goal_type=goal_preds[0].type_id,
            goal_color=goal_preds[0].color_id,
        )

    raw_t0 = obs["image"]
    gt_t0 = compute_predicates(extract_slots(raw_t0))
    cohort = initial_cohort(gt_t0, goal_preds)

    actions_taken: list[int] = []
    explored_steps = 0
    max_scores: list[float] = []
    became_visible = False
    visible_idx = next((g.flat_index for g in goal_preds if g.name == "visible"), None)
    if cohort != "hidden":
        became_visible = True

    ep_reward = 0.0
    steps_to_first_visible = -1
    for step in range(max_steps):
        raw = obs["image"]
        encoded = _encode_image(raw)
        if not became_visible and visible_idx is not None:
            gt_now = compute_predicates(extract_slots(raw))
            if gt_now[visible_idx] > 0.5:
                became_visible = True
                steps_to_first_visible = step

        obs_t = torch.from_numpy(encoded).float()
        action, info = agent.select_action(obs_t, goal_preds, allowed_actions=allowed)
        actions_taken.append(int(action))
        if info.get("explored", 0.0):
            explored_steps += 1
        max_scores.append(info.get("max_score", 0.0))

        obs, r, term, trunc, _ = env.step(action)
        ep_reward += float(r)
        if term or trunc:
            break

    return {
        "mission": mission,
        "cohort": cohort,
        "reward": ep_reward,
        "steps": len(actions_taken),
        "actions": actions_taken,
        "explored_rate": explored_steps / max(len(actions_taken), 1),
        "max_score_mean": float(np.mean(max_scores)) if max_scores else 0.0,
        "became_visible": became_visible,
        "steps_to_first_visible": steps_to_first_visible,
    }


def evaluate_env(
    env_id: str,
    *,
    agent: GroundedAgent,
    recurrent_policy: RecurrentPolicy,
    n_episodes: int,
    max_steps: int,
    seed: int,
):
    env = make_env_with_max_steps(env_id, max_steps)
    by_cohort: dict[str, list[dict]] = defaultdict(list)
    n_skipped = 0
    for ep in range(n_episodes):
        result = run_episode_ext(
            env, agent,
            seed=seed + ep * 7919,
            max_steps=max_steps,
            recurrent_policy=recurrent_policy,
        )
        if result is None:
            n_skipped += 1
            continue
        by_cohort[result["cohort"]].append(result)
    env.close()

    rewards = [e["reward"] for c in by_cohort.values() for e in c]
    successes = [1.0 if e["reward"] > 0.5 else 0.0
                 for c in by_cohort.values() for e in c]
    steps = [e["steps"] for c in by_cohort.values() for e in c]
    mean_r = float(np.mean(rewards)) if rewards else 0.0
    succ = float(np.mean(successes)) if successes else 0.0
    mean_steps = float(np.mean(steps)) if steps else 0.0
    return {
        "n": len(rewards),
        "skipped": n_skipped,
        "mean_R": mean_r,
        "success": succ,
        "mean_steps": mean_steps,
        "by_cohort": by_cohort,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--envs", nargs="+",
                        default=["BabyAI-GoToLocal-v0", "BabyAI-Pickup-v0",
                                 "BabyAI-GoTo-v0", "BabyAI-Open-v0"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-cohort", action="store_true",
                        help="also print per-cohort breakdown for each env")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    jepa_ckpt = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
    cfg: JepaConfig = upgrade_config(jepa_ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(jepa_ckpt["model"])
    jepa.eval()
    print(f"[multi-eval] JEPA loaded: encoder={cfg.encoder_type}")

    agent = GroundedAgent(
        jepa, device,
        horizon=args.horizon,
        n_samples=args.n_samples,
        scoring_mode="recurrent",
    )

    pckpt = torch.load(args.policy_checkpoint, map_location=device, weights_only=False)
    ckpt_mem_dim = int(pckpt.get("mem_feat_dim", 0) or 0)
    recurrent_policy = RecurrentPolicy(
        latent_in_dim=pckpt["latent_in_dim"],
        n_actions=pckpt["n_actions"],
        mission_dim=pckpt["mission_dim"],
        hidden_dim=pckpt["hidden_dim"],
        latent_proj_dim=pckpt["latent_proj_dim"],
        mem_feat_dim=ckpt_mem_dim,
    ).to(device)
    recurrent_policy.load_state_dict(pckpt["policy_state_dict"])
    recurrent_policy.eval()
    print(
        f"[multi-eval] policy loaded: {args.policy_checkpoint} "
        f"(mem_feat_dim={ckpt_mem_dim})"
    )

    summaries: dict[str, dict] = {}
    for env_id in args.envs:
        print(f"\n[multi-eval] running {env_id} ({args.episodes} eps)…")
        summaries[env_id] = evaluate_env(
            env_id,
            agent=agent,
            recurrent_policy=recurrent_policy,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )

    # Final comparison table.
    print("\n=== multi-env summary ===")
    print(f"{'env':30s}  {'n':>4s}  {'mean_R':>7s}  {'success%':>8s}  {'mean_steps':>10s}")
    for env_id, s in summaries.items():
        print(
            f"{env_id:30s}  {s['n']:>4d}  {s['mean_R']:>7.3f}  "
            f"{s['success']*100:>7.1f}%  {s['mean_steps']:>10.1f}"
        )

    if args.per_cohort:
        print("\n=== per-env, per-cohort ===")
        cohorts = ("adjacent", "near", "facing", "visible", "hidden")
        for env_id, s in summaries.items():
            print(f"\n[{env_id}]")
            print(f"  {'cohort':10s}  {'n':>4s}  {'mean_R':>7s}  {'success%':>8s}  {'mean_steps':>10s}")
            for c in cohorts:
                eps = s["by_cohort"].get(c, [])
                if not eps:
                    continue
                r = float(np.mean([e["reward"] for e in eps]))
                ok = float(np.mean([1.0 if e["reward"] > 0.5 else 0.0 for e in eps]))
                st = float(np.mean([e["steps"] for e in eps]))
                print(f"  {c:10s}  {len(eps):>4d}  {r:>7.3f}  {ok*100:>7.1f}%  {st:>10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
