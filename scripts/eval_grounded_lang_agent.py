"""Stage 1.1 — closed-loop grounded language agent eval.

Pipeline:
    mission_text → lang_model → predicted (goal_color, goal_type)
                                              ↓
    obs → jepa.encode → z_t
                                              ↓
    for each candidate action a:
        z_pred = jepa.predict(z_t, a)
        aux    = jepa.aux_predicate_head(z_pred)   # 96-d preds + 24-d distance
        d(a)   = aux.distance_block[type_color_index(goal_type, goal_color)]
    a_best  = argmin_a d(a)
    step env

Mission grounding mode is selectable:
  --mode rule  — uses prism.agents.goal_predicates_for_mission (regex parser)
  --mode lang  — uses the trained text→(color, type) head from
                 train_grounding_floor.py / grounding_predicate_head

Compare the two modes' success rates per env. Falsifies whether the
language model's grounding signal (≥95% on text alone, 53.6% held
compositional via readout) translates to closed-loop episode success.

Usage:
    python -m scripts.eval_grounded_lang_agent \\
        --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \\
        --lang-checkpoint runs/grounding_floor_tt_clean/grounding_floor_final.pt \\
        --vocab-checkpoint runs/grounding_floor_tt_clean/vocab.pt \\
        --envs BabyAI-GoToLocal-v0 BabyAI-GoTo-v0 BabyAI-GoToObj-v0 \\
        --episodes-per-env 200 \\
        --modes rule lang \\
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

from prism.agents import goal_predicates_for_mission
from prism.agents.grounded_agent import allowed_actions_for_spec
from prism.envs.babyai import _encode_image, make_env_with_max_steps
from prism.language.grounding_head import WhitespaceVocab
from prism.language.grounding_predicate_head import make_dual_head
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.perception.predicates import PREDICATE_VECTOR_DIM, type_color_index
from prism.perception.slots import NUM_COLORS, NUM_TYPES, OBJECT_TYPES
from prism.utils.seed import set_global_seed


def lang_predict_goal(
    text: str,
    lang_model: torch.nn.Module,
    vocab: WhitespaceVocab,
    device: torch.device,
) -> tuple[int, int]:
    """Run the text→(color, type) head. Returns (color_id, type_idx).
    type_idx is the position inside OBJECT_TYPES."""
    tokens, mask = vocab.encode_batch([text])
    tokens = tokens.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        out = lang_model(tokens, mask)
        if isinstance(out, tuple):                   # dual-head returns tuple
            c_logits, t_logits = out
        else:                                        # combined-head 10-d
            c_logits = out[:, :NUM_COLORS]
            t_logits = out[:, NUM_COLORS:NUM_COLORS + NUM_TYPES]
        color = int(c_logits.argmax(-1).item())
        type_idx = int(t_logits.argmax(-1).item())
    return color, type_idx


@torch.no_grad()
def score_actions_batched(
    jepa: JepaWorldModel,
    z_t: torch.Tensor,
    goal_type_id: int,
    goal_color_id: int,
    n_actions: int,
    device: torch.device,
) -> np.ndarray:
    """For all candidate actions at once, return a score per action.
    Higher = better. Score = -distance to (goal_type, goal_color)."""
    # z_t shape: (1, ...). Expand to (n_actions, ...).
    z_batch = z_t.expand(n_actions, *z_t.shape[1:]).contiguous()
    a_batch = torch.arange(n_actions, device=device, dtype=torch.long)
    z_pred = jepa.predict(z_batch, a_batch)
    aux_logits = jepa.aux_predicate_head(z_pred)
    D = jepa.cfg.aux_distance_dim
    if D <= 0:
        # Fall back to predicate-presence score (96-d).
        pred_probs = torch.sigmoid(aux_logits[:, :PREDICATE_VECTOR_DIM])
        # "visible(goal_type, goal_color)" is at predicate_index("visible", ...)
        # but we keep it simple: sum over (visible/near/facing/adjacent) for
        # the goal (type, color).
        tc_idx = type_color_index(goal_type_id, goal_color_id)
        # Predicates: 4 types stacked → visible at 0*24+tc, near at 1*24+tc, ...
        N_TC = NUM_TYPES * NUM_COLORS
        score = sum(
            pred_probs[:, k * N_TC + tc_idx] for k in range(4)
        )
        return score.cpu().numpy()
    # Distance block at indices [PREDICATE_VECTOR_DIM : PREDICATE_VECTOR_DIM+D]
    dist_logits = aux_logits[:, PREDICATE_VECTOR_DIM:PREDICATE_VECTOR_DIM + D]
    dist = torch.sigmoid(dist_logits)
    tc_idx = type_color_index(goal_type_id, goal_color_id)
    return (-dist[:, tc_idx]).cpu().numpy()


def run_episode(
    env,
    jepa: JepaWorldModel,
    goal_type_id: int,
    goal_color_id: int,
    allowed: tuple[int, ...],
    max_steps: int,
    device: torch.device,
    seed: int,
) -> tuple[bool, int, float]:
    """Returns (success, steps_taken, final_reward)."""
    obs, _ = env.reset(seed=seed)
    n_actions = env.action_space.n
    allowed_mask = np.full(n_actions, -1e9, dtype=np.float32)
    for a in allowed:
        allowed_mask[a] = 0.0

    for step in range(max_steps):
        encoded = _encode_image(obs["image"])
        obs_t = torch.from_numpy(encoded).float().unsqueeze(0).to(device)
        z_t = jepa.encode(obs_t)
        scores = score_actions_batched(
            jepa, z_t, goal_type_id, goal_color_id, n_actions, device,
        )
        scores = scores + allowed_mask
        action = int(np.argmax(scores))
        obs, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            return (float(reward) > 0.0), step + 1, float(reward)
    return False, max_steps, 0.0


def parse_mission_rule(
    mission: str,
) -> tuple[int, int, tuple[int, ...]] | None:
    parsed = goal_predicates_for_mission(mission)
    if parsed is None:
        return None
    goal_preds, spec = parsed
    if not goal_preds:
        return None
    gp = goal_preds[0]
    # Need env's n_actions to build allowed mask; default 7 for BabyAI.
    allowed = allowed_actions_for_spec(spec, 7)
    return int(gp.color_id), int(gp.type_id), allowed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jepa-checkpoint", required=True)
    p.add_argument("--lang-checkpoint", required=True,
                   help="text→(color, type) head checkpoint from "
                        "train_grounding_floor.py or "
                        "train_grounding_predicate.py")
    p.add_argument("--vocab-checkpoint", required=True)
    p.add_argument("--lang-kind", choices=["bow", "tiny_tf"], default="tiny_tf")
    p.add_argument("--envs", nargs="+",
                   default=["BabyAI-GoToLocal-v0",
                            "BabyAI-GoTo-v0",
                            "BabyAI-GoToObj-v0"])
    p.add_argument("--episodes-per-env", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--modes", nargs="+", default=["rule", "lang"],
                   choices=["rule", "lang"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None,
                   help="optional path for a JSON summary")
    args = p.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # ---- JEPA ----
    print(f"[stage1.1] loading JEPA: {args.jepa_checkpoint}")
    ckpt = torch.load(args.jepa_checkpoint, map_location=device,
                      weights_only=False)
    cfg = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    if jepa.aux_predicate_head is None:
        raise SystemExit(
            "JEPA checkpoint has no aux_predicate_head — Stage 1.1 needs "
            "a JEPA trained with --aux-predicate-weight > 0."
        )
    print(f"[stage1.1]   encoder={cfg.encoder_type} aux_pred={cfg.aux_predicate_weight} "
          f"aux_dist_dim={getattr(cfg, 'aux_distance_dim', 0)}")

    # ---- language head + vocab ----
    print(f"[stage1.1] loading vocab: {args.vocab_checkpoint}")
    vocab = WhitespaceVocab.load(args.vocab_checkpoint)
    print(f"[stage1.1]   vocab size = {vocab.size}")
    print(f"[stage1.1] loading lang head: {args.lang_checkpoint}")
    lang_ckpt = torch.load(args.lang_checkpoint, map_location=device,
                           weights_only=False)
    lang_kind = lang_ckpt.get("kind", args.lang_kind)
    lang_model = make_dual_head(
        lang_kind, vocab.size, NUM_COLORS, NUM_TYPES,
    ).to(device)
    lang_model.load_state_dict(lang_ckpt["state_dict"])
    lang_model.eval()
    print(f"[stage1.1]   lang kind = {lang_kind}")

    # ---- run episodes ----
    results: dict = {"per_env": {}, "by_combo": {}}
    for env_id in args.envs:
        env = make_env_with_max_steps(env_id, args.max_steps)
        per_mode: dict[str, dict] = {m: defaultdict(list) for m in args.modes}
        print(f"\n[stage1.1] {env_id}: {args.episodes_per_env} episodes "
              f"per mode (modes: {args.modes})")
        for ep in range(args.episodes_per_env):
            # Same seed across modes for paired comparison.
            ep_seed = args.seed + ep * 7919 + hash(env_id) % 1_000_003
            # Sample the mission once.
            obs, _ = env.reset(seed=ep_seed)
            mission = obs["mission"]
            rule_parsed = parse_mission_rule(mission)
            if rule_parsed is None:
                continue
            rule_c, rule_t, allowed = rule_parsed

            for mode in args.modes:
                if mode == "rule":
                    goal_c, goal_t_id = rule_c, rule_t
                else:
                    color, type_idx = lang_predict_goal(
                        mission, lang_model, vocab, device,
                    )
                    if not (0 <= type_idx < len(OBJECT_TYPES)):
                        continue
                    goal_c, goal_t_id = color, int(OBJECT_TYPES[type_idx])

                success, steps, reward = run_episode(
                    env, jepa, goal_t_id, goal_c, allowed,
                    args.max_steps, device, ep_seed,
                )
                per_mode[mode]["success"].append(int(success))
                per_mode[mode]["steps"].append(steps)
                per_mode[mode]["reward"].append(reward)
                key = (rule_c, rule_t)
                results["by_combo"].setdefault(
                    str(key), {"rule": [], "lang": []}
                )
                results["by_combo"][str(key)][mode].append(int(success))

        env.close()
        env_summary = {}
        for mode in args.modes:
            arr_s = np.array(per_mode[mode]["success"])
            arr_steps = np.array(per_mode[mode]["steps"])
            arr_r = np.array(per_mode[mode]["reward"])
            env_summary[mode] = {
                "n": int(len(arr_s)),
                "success_rate": float(arr_s.mean()) if len(arr_s) else 0.0,
                "mean_steps": float(arr_steps.mean()) if len(arr_steps) else 0.0,
                "mean_reward": float(arr_r.mean()) if len(arr_r) else 0.0,
            }
        results["per_env"][env_id] = env_summary
        for mode, m in env_summary.items():
            print(f"  {mode:>4s}: success={m['success_rate']*100:5.1f}%  "
                  f"mean_steps={m['mean_steps']:5.1f}  "
                  f"mean_reward={m['mean_reward']:.3f}  (n={m['n']})")

    # ---- comparison summary ----
    print("\n=== summary ===")
    print(f"  {'env':35s}  {'rule%':>7s}  {'lang%':>7s}  {'delta':>7s}")
    for env_id, s in results["per_env"].items():
        rule = s.get("rule", {}).get("success_rate", float("nan")) * 100
        lang = s.get("lang", {}).get("success_rate", float("nan")) * 100
        print(f"  {env_id:35s}  {rule:7.1f}  {lang:7.1f}  "
              f"{lang - rule:+7.1f}")

    # Verdict.
    print("\n=== verdict ===")
    deltas = [
        s["lang"]["success_rate"] - s["rule"]["success_rate"]
        for s in results["per_env"].values()
        if "lang" in s and "rule" in s
    ]
    if not deltas:
        print("  no comparable data")
    else:
        mean_delta = float(np.mean(deltas)) * 100
        if abs(mean_delta) <= 5.0:
            print(f"  PASS — lang vs rule within 5pp (mean delta "
                  f"{mean_delta:+.1f}pp). Language grounding is faithful "
                  f"to rule-parsed goals at the action level.")
        elif mean_delta < -5.0:
            print(f"  FAIL — lang is {-mean_delta:.1f}pp worse than rule. "
                  "Language model predicts wrong goals, OR text-to-goal "
                  "translation breaks for some mission templates.")
        else:
            print(f"  ANOMALY — lang is {mean_delta:.1f}pp better than rule. "
                  "Re-check: probably a bug or a small-N artifact.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[wrote] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
