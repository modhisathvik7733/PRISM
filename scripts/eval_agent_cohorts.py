"""Layer 4 diagnostic — split agent failures by initial target visibility.

Hypothesis after the spatial refactor: Layer 3 metrics improved dramatically
(F1 0.71 → 0.876, forward F1 0.93 → 0.98) but agent reward stayed flat
(~0.36–0.43). That means the world model is NOT the bottleneck; the agent's
decision policy is leaving signal on the table.

This script splits each episode into cohorts based on whether the goal
target is visible at t=0:

  visible_t0=True   → "approach" capability test (model + scoring should drive this)
  visible_t0=False  → "exploration" capability test (random fallback handles this)

If the visible cohort succeeds (>0.7) and the hidden cohort fails (<0.2), the
bottleneck is exploration — the world model can't help when the target is
not yet in view.

Per-cohort we also log:
  - exploration rate (% of steps where max_score < threshold → random pick)
  - action histogram (was the agent spinning? walking forward? stuck?)
  - mean steps to first contact (for hidden → did it ever find the target?)
  - mean reward

Usage:
    python -m scripts.eval_agent_cohorts \
        --jepa-checkpoint runs/<run-name>/jepa_final.pt \
        --episodes 100 --device cuda
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch

from prism.agents import GroundedAgent, goal_predicates_for_mission
from prism.agents.grounded_agent import allowed_actions_for_spec
from prism.envs.babyai import _encode_image
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.perception import compute_predicates, extract_slots
from prism.utils.seed import set_global_seed


def initial_cohort(gt_preds_t0: np.ndarray, goal_preds) -> str:
    """Classify episode by tightest predicate true at t=0 for the goal target.
    Tightness order: adjacent > near > facing > visible > hidden.
    """
    by_name = {g.name: g for g in goal_preds}
    for name in ("adjacent", "near", "facing", "visible"):
        g = by_name.get(name)
        if g is not None and gt_preds_t0[g.flat_index] > 0.5:
            return name
    return "hidden"


def run_episode(env, agent, *, seed, max_steps):
    obs, _ = env.reset(seed=seed)
    agent.reset()  # zeros curriculum exploration counter
    mission = obs["mission"]
    parsed = goal_predicates_for_mission(mission)
    if parsed is None:
        return None
    goal_preds, spec = parsed
    allowed = allowed_actions_for_spec(spec, env.action_space.n)

    raw_t0 = obs["image"]
    gt_t0 = compute_predicates(extract_slots(raw_t0))
    cohort = initial_cohort(gt_t0, goal_preds)

    actions_taken = []
    explored_steps = 0
    max_scores = []
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
        actions_taken.append(action)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--scoring-mode", default="magnitude",
                        choices=["magnitude", "binary", "distance", "curriculum"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
    cfg: JepaConfig = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    encoder_type = getattr(cfg, "encoder_type", "flat")
    print(f"[cohort] loaded JEPA: encoder={encoder_type}")
    agent = GroundedAgent(
        jepa, device,
        horizon=args.horizon,
        n_samples=args.n_samples,
        scoring_mode=args.scoring_mode,
    )
    print(
        f"[cohort] horizon={args.horizon} n_samples={args.n_samples} "
        f"scoring={args.scoring_mode}"
    )

    env = gym.make(args.env_id)

    # Aggregate per cohort
    by_cohort: dict[str, list[dict]] = defaultdict(list)
    n_skipped = 0
    for ep in range(args.episodes):
        result = run_episode(
            env, agent,
            seed=args.seed + ep * 7919,
            max_steps=args.max_steps,
        )
        if result is None:
            n_skipped += 1
            continue
        by_cohort[result["cohort"]].append(result)

    print(f"\n[cohort] ran {args.episodes - n_skipped} parseable episodes ({n_skipped} skipped)")

    # ---------------------------------------------------- per-cohort report
    cohorts = ("adjacent", "near", "facing", "visible", "hidden")
    print("\n=== per-cohort summary ===")
    print(
        f"{'cohort':10s}  {'n':>4s}  {'mean_R':>7s}  {'success%':>8s}  "
        f"{'expl%':>6s}  {'max_score':>10s}  {'mean_steps':>10s}"
    )
    for c in cohorts:
        eps = by_cohort.get(c, [])
        if not eps:
            print(f"{c:10s}  {0:>4d}  (no episodes)")
            continue
        rewards = [e["reward"] for e in eps]
        n = len(eps)
        mean_r = float(np.mean(rewards))
        succ = float(np.mean([1.0 if r > 0.5 else 0.0 for r in rewards]))
        expl = float(np.mean([e["explored_rate"] for e in eps]))
        msc = float(np.mean([e["max_score_mean"] for e in eps]))
        steps = float(np.mean([e["steps"] for e in eps]))
        print(
            f"{c:10s}  {n:>4d}  {mean_r:>7.3f}  {succ*100:>7.1f}%  "
            f"{expl*100:>5.1f}%  {msc:>10.4f}  {steps:>10.1f}"
        )

    # ---------------------------------------------------- hidden-cohort detail
    hidden = by_cohort.get("hidden", [])
    if hidden:
        print(f"\n=== hidden-cohort exploration analysis (n={len(hidden)}) ===")
        n_found = sum(1 for e in hidden if e["became_visible"])
        print(f"  episodes where target became visible : {n_found}/{len(hidden)} ({n_found/len(hidden)*100:.1f}%)")
        if n_found > 0:
            steps_to_v = [e["steps_to_first_visible"] for e in hidden if e["became_visible"]]
            print(f"  mean steps to first visible          : {float(np.mean(steps_to_v)):.1f}")
        # Of those that did find the target, did they then succeed?
        found_eps = [e for e in hidden if e["became_visible"]]
        if found_eps:
            r_found = float(np.mean([e["reward"] for e in found_eps]))
            print(f"  reward | found-target                : {r_found:.3f}")
        not_found_eps = [e for e in hidden if not e["became_visible"]]
        if not_found_eps:
            r_not = float(np.mean([e["reward"] for e in not_found_eps]))
            print(f"  reward | never-found-target          : {r_not:.3f}")

    # ---------------------------------------------------- action histogram
    print("\n=== action histogram by cohort ===")
    print(f"{'cohort':10s}  " + "  ".join(f"a{i}={'?':>5s}" for i in range(7)))
    for c in cohorts:
        eps = by_cohort.get(c, [])
        if not eps:
            continue
        all_acts = [a for e in eps for a in e["actions"]]
        total = max(len(all_acts), 1)
        cnt = Counter(all_acts)
        cells = "  ".join(f"a{i}={cnt.get(i, 0)/total*100:>4.1f}%" for i in range(7))
        print(f"{c:10s}  {cells}")

    # ---------------------------------------------------- verdict
    print("\n=== verdict ===")
    visible_like = [e for c in ("visible", "facing", "near", "adjacent") for e in by_cohort.get(c, [])]
    hidden_eps = by_cohort.get("hidden", [])
    if visible_like and hidden_eps:
        r_v = float(np.mean([e["reward"] for e in visible_like]))
        r_h = float(np.mean([e["reward"] for e in hidden_eps]))
        gap = r_v - r_h
        print(f"  reward | target visible at t=0  : {r_v:.3f}  (n={len(visible_like)})")
        print(f"  reward | target hidden at t=0   : {r_h:.3f}  (n={len(hidden_eps)})")
        print(f"  cohort gap                     : {gap:.3f}")
        if r_v > 0.7 and r_h < 0.2:
            print("  → EXPLORATION FAILURE. Model + scoring work when target is in view.")
            print("    Fix: better exploration policy (frontier-based, novelty bonus, or")
            print("    biased random walk that prefers turning when target absent).")
        elif r_v < 0.5:
            print("  → APPROACH FAILURE. Even with target visible, agent does not reach it.")
            print("    Fix: investigate scoring / horizon. Try horizon=1 to test compounding noise.")
        else:
            print("  → MIXED. Both approach and exploration are partial. Address both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
