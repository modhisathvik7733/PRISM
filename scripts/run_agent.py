"""Run the first end-to-end PRISM agent on BabyAI.

This is the Phase 2 capstone. The agent has zero learned policy. It acts by:
  1. Encoding the obs with the frozen JEPA encoder.
  2. Imagining the next latent under each candidate action.
  3. Reading out predicates from each imagined latent (frozen probe).
  4. Picking the action whose imagined predicates best match the parsed
     mission's goal.

Success criterion (Phase 2 capstone falsifier):
  Mean reward on BabyAI-GoToLocal-v0 over 50 episodes should clear ~0.6.
  Vanilla mission-blind PPO clocked 0.48 — beating that with NO learned
  policy is the proof that grounded-prediction-driven action selection
  works end-to-end.

Usage:
    python -m scripts.run_agent \
        --jepa-checkpoint runs/jepa_categorical_aux1_BabyAI-GoToLocal-v0_seed0/jepa_final.pt \
        --episodes 50 --device cuda
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import minigrid  # noqa: F401  (registers BabyAI envs)
import numpy as np
import torch

from prism.agents import GroundedAgent, goal_predicates_for_mission
from prism.agents.grounded_agent import allowed_actions_for_spec
from prism.envs.babyai import _encode_image
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.perception import compute_predicates, extract_slots
from prism.utils.seed import set_global_seed


@torch.no_grad()
def _diagnose_step(jepa, agent, raw_obs, encoded_obs, goal_preds, device, n_actions):
    """Print a side-by-side of:
        - ground-truth predicates from slots
        - probe(z_t)               = baseline at current state
        - probe(predict(z_t, a))   = imagined per-action
        - improvement              = imagined − baseline   (this is what the
                                     advantage scorer actually uses)

    Useful for spotting where the train/inference distribution gap shows up
    AND for sanity-checking the advantage formulation.
    """
    gt = compute_predicates(extract_slots(raw_obs))
    z_t = jepa.encode(torch.from_numpy(encoded_obs).float().unsqueeze(0).to(device))
    probe_t = torch.sigmoid(agent.probe(z_t)).squeeze(0).cpu().numpy()

    actions = torch.arange(n_actions, device=device, dtype=torch.long)
    z_next = jepa.predict(z_t.expand(n_actions, -1), actions)
    probe_next = torch.sigmoid(agent.probe(z_next)).cpu().numpy()  # (n_actions, 96)

    print("    [diag] goal predicates: name | gt | base | next per-action | improvement per-action")
    for g in goal_preds:
        idx = g.flat_index
        next_str = " ".join(f"{probe_next[a, idx]:.2f}" for a in range(n_actions))
        imp_str = " ".join(f"{probe_next[a, idx] - probe_t[idx]:+.2f}" for a in range(n_actions))
        print(
            f"        {g.name:9s}({g.color_id},{g.type_id:>2d}) "
            f"gt={gt[idx]:.0f} base={probe_t[idx]:.2f}   "
            f"next: {next_str}   "
            f"imp: {imp_str}"
        )


def run_episode(
    env: gym.Env,
    agent: GroundedAgent,
    *,
    seed: int,
    max_steps: int = 64,
    verbose: bool = False,
    recurrent_policy=None,
) -> dict:
    obs, _ = env.reset(seed=seed)
    agent.reset()  # zeros curriculum exploration counter; no-op for other modes
    mission = obs["mission"]
    parsed = goal_predicates_for_mission(mission)
    if parsed is None:
        # Fallback: random policy. Phase 4+ will handle compositional missions.
        n_actions = env.action_space.n
        rng = np.random.default_rng(seed)
        chosen_actions = []
        ep_reward = 0.0
        for _ in range(max_steps):
            a = int(rng.integers(n_actions))
            obs, r, term, trunc, _ = env.step(a)
            ep_reward += float(r)
            chosen_actions.append(a)
            if term or trunc:
                break
        return {
            "mission": mission,
            "parsed": False,
            "reward": ep_reward,
            "steps": len(chosen_actions),
            "actions": chosen_actions,
        }
    goal_preds, spec = parsed
    allowed = allowed_actions_for_spec(spec, env.action_space.n)

    # If running the recurrent policy, attach it now (after agent.reset()) with
    # the mission one-hot derived from the parsed goal.
    if recurrent_policy is not None:
        from prism.perception.predicates import type_color_index
        from prism.perception.slots import NUM_COLORS, OBJECT_TYPES
        tc_idx = type_color_index(goal_preds[0].type_id, goal_preds[0].color_id)
        mission_one_hot = torch.zeros(len(OBJECT_TYPES) * NUM_COLORS)
        mission_one_hot[tc_idx] = 1.0
        agent.attach_recurrent_policy(recurrent_policy, mission_one_hot)

    chosen_actions = []
    ep_reward = 0.0
    for step in range(max_steps):
        raw = obs["image"]                     # (7, 7, 3) uint8
        encoded = _encode_image(raw)           # (3, 7, 7) float32 normalized
        obs_t = torch.from_numpy(encoded).float()
        action, info = agent.select_action(obs_t, goal_preds, allowed_actions=allowed)
        if verbose:
            n_actions = env.action_space.n
            scores = []
            for i in range(n_actions):
                v = info[f"score_a{i}"]
                scores.append("masked" if v == float("-inf") else round(v, 2))
            explored_tag = " (explored)" if info.get("explored", 0.0) else ""
            print(
                f"  step {step:2d} action={action}{explored_tag} "
                f"allowed={allowed} scores={scores}"
            )
            _diagnose_step(
                agent.jepa, agent, raw, encoded, goal_preds, agent.device,
                n_actions,
            )
        obs, r, term, trunc, _ = env.step(action)
        ep_reward += float(r)
        chosen_actions.append(action)
        if term or trunc:
            break

    return {
        "mission": mission,
        "parsed": True,
        "reward": ep_reward,
        "steps": len(chosen_actions),
        "actions": chosen_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--probe-checkpoint", default=None,
                        help="Optional standalone probe ckpt. If omitted, the JEPA's "
                             "internal aux_predicate_head is used (which requires the "
                             "JEPA to have been trained with aux_predicate_weight>0).")
    parser.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=4,
                        help="latent rollout horizon for action scoring (>=1). "
                             "horizon=4 averages over 4-step imagined futures.")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="random follow-up samples per first action (variance "
                             "reduction). 8 keeps single-step latency negligible.")
    parser.add_argument("--scoring-mode", default="magnitude",
                        choices=["magnitude", "binary", "distance", "curriculum",
                                 "memory", "recurrent"],
                        help="magnitude (default) = raw prob diff. binary = score "
                             "predicate FLIPS only. distance = continuous distance "
                             "head. memory = pose tracking + frontier exploration. "
                             "recurrent = learned GRU policy on frozen JEPA latents "
                             "(requires --policy-checkpoint).")
    parser.add_argument("--policy-checkpoint", default=None,
                        help="path to RecurrentPolicy .pt for scoring-mode=recurrent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true",
                        help="print per-step action scores for the first episode")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # ------------------------------------------------------ load JEPA
    ckpt = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
    cfg: JepaConfig = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    encoder_type = getattr(cfg, "encoder_type", "flat")
    aux_w = getattr(cfg, "aux_predicate_weight", 0.0)
    print(f"[agent] loaded JEPA: encoder={encoder_type} aux_predicate_weight={aux_w}")

    # ------------------------------------------------------ load probe (optional)
    external_probe = None
    if args.probe_checkpoint is not None:
        probe_ckpt = torch.load(args.probe_checkpoint, map_location=device, weights_only=False)
        from prism.models.predicate_probe import PredicateProbe
        external_probe = PredicateProbe(embed_dim=probe_ckpt["embed_dim"]).to(device)
        external_probe.load_state_dict(probe_ckpt["probe"])
        external_probe.eval()
        print(f"[agent] loaded external probe from {args.probe_checkpoint}")

    agent = GroundedAgent(
        jepa, device,
        probe=external_probe,
        horizon=args.horizon,
        n_samples=args.n_samples,
        scoring_mode=args.scoring_mode,
    )
    print(
        f"[agent] horizon={args.horizon} n_samples={args.n_samples} "
        f"scoring={args.scoring_mode} n_actions={agent.n_actions}"
    )

    # ------------------------------------------------------ optional: load
    # the BC-trained recurrent policy when scoring-mode=recurrent.
    recurrent_policy = None
    if args.scoring_mode == "recurrent":
        if args.policy_checkpoint is None:
            raise SystemExit("--policy-checkpoint is required for scoring-mode=recurrent")
        from prism.models.recurrent_policy import RecurrentPolicy
        pckpt = torch.load(args.policy_checkpoint, map_location=device, weights_only=False)
        recurrent_policy = RecurrentPolicy(
            latent_in_dim=pckpt["latent_in_dim"],
            n_actions=pckpt["n_actions"],
            mission_dim=pckpt["mission_dim"],
            hidden_dim=pckpt["hidden_dim"],
            latent_proj_dim=pckpt["latent_proj_dim"],
        ).to(device)
        recurrent_policy.load_state_dict(pckpt["policy_state_dict"])
        recurrent_policy.eval()
        print(f"[agent] loaded recurrent policy from {args.policy_checkpoint}")

    # ------------------------------------------------------ env + run
    env = gym.make(args.env_id)
    # Override env's internal step cap so agents can use the longer budget.
    # BabyAI's reward = 1 - 0.9*(steps/max_steps), so a larger max_steps also
    # raises per-episode reward for the same step count.
    from prism.envs.babyai import set_max_steps
    set_max_steps(env, args.max_steps)
    rewards = []
    parsed_count = 0
    successes = 0  # episodes with reward > 0
    for ep in range(args.episodes):
        result = run_episode(
            env, agent,
            seed=args.seed + ep * 7919,  # spread seeds
            max_steps=args.max_steps,
            verbose=args.verbose and ep == 0,
            recurrent_policy=recurrent_policy,
        )
        rewards.append(result["reward"])
        if result["parsed"]:
            parsed_count += 1
        if result["reward"] > 0:
            successes += 1
        print(
            f"[ep {ep:02d}] mission={result['mission']!r:60s} "
            f"steps={result['steps']:3d} reward={result['reward']:.3f}"
            + ("" if result["parsed"] else "  (UNPARSED — fell back to random)")
        )

    mean_reward = float(np.mean(rewards))
    print("\n=== summary ===")
    print(f"  episodes              : {args.episodes}")
    print(f"  parsed missions       : {parsed_count}/{args.episodes}")
    print(f"  reward > 0            : {successes}/{args.episodes}")
    print(f"  mean reward           : {mean_reward:.3f}")

    # Phase 2 capstone falsifier: beat the mission-blind PPO baseline (0.48).
    pass_capstone = mean_reward > 0.55
    print(
        f"\n  Phase 2 capstone (mean_reward > 0.55): "
        f"{'PASS — grounded action selection works end-to-end' if pass_capstone else 'FAIL — diagnose before Phase 3'}"
    )
    return 0 if pass_capstone else 2


if __name__ == "__main__":
    raise SystemExit(main())
