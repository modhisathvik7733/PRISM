"""E1 cross-env evaluation — measure transfer of curriculum-trained policies.

The scientific signal for E1's "developmental ordering matters" claim
is NOT the final window_R of each training arm (those are on different
final envs and so are not directly comparable). The right comparison
is: at end of training, how well does the policy do on **all** three
BabyAI levels?

Forward order trains: sensorimotor → object recognition → action composition.
Reverse trains:        action composition → object recognition → sensorimotor.
Shuffled trains:       sensorimotor → action composition → object recognition.

This script evaluates a single policy checkpoint (or multiple in a
single invocation) on the full {GoToObj, GoToLocal, PickupLoc} suite
and reports per-env success rate. The cross-curriculum comparison
table tells us whether forward order produced more transferable
competence than reverse/shuffled.

Critical details:
  - Greedy action selection by default. Stochastic sampling is
    available via --stochastic for variance estimates.
  - Action masking via the adapter's MISSION_ALLOWED_ACTIONS table —
    same as training. Critical: without the mask, the policy can
    sample disallowed actions and "succeed" by accident on simpler
    envs while failing on harder ones for the same reason.
  - Probe set is NOT involved; this is a behavioral test.
  - Each env is seeded deterministically (seed + episode index) so
    runs are reproducible.

Usage:

    # Single arm.
    python -m scripts.experiments.e1_cross_env_eval \\
        --checkpoint runs/v6_e1_forward/policy_final.pt \\
        --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \\
        --n-episodes 100 --device cuda

    # All three arms — produces the E1 comparison table.
    python -m scripts.experiments.e1_cross_env_eval \\
        --checkpoint runs/v6_e1_forward/policy_final.pt   forward \\
        --checkpoint runs/v6_e1_reverse/policy_final.pt   reverse \\
        --checkpoint runs/v6_e1_shuffled/policy_final.pt  shuffled \\
        --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \\
        --n-episodes 100 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from prism.adapters.babyai_adapter import (
    MISSION_ALLOWED_ACTIONS, BabyAIAdapter,
)
from prism.agents import goal_predicates_for_mission
from prism.cognition.policy import UniversalPolicy
from prism.envs.babyai import _encode_image, make_env_with_max_steps
from prism.perception.slots import NUM_COLORS, OBJECT_TYPES


E1_EVAL_ENVS = ["BabyAI-GoToObj-v0", "BabyAI-GoToLocal-v0", "BabyAI-PickupLoc-v0"]


def _build_policy(jepa_checkpoint: Path, device: torch.device) -> UniversalPolicy:
    """Construct the substrate with the same shape ppo_train uses."""
    adapter = BabyAIAdapter.from_jepa_checkpoint(jepa_checkpoint, device=device)
    return UniversalPolicy.from_adapter(
        adapter,
        trunk="transformer",
        D_tok=128, L=16,
        n_trunk_layers=4, n_trunk_heads=4, trunk_ffn_dim=512,
        concept_n_slots=1024, operator_n_slots=64,
    ).to(device)


def _load_policy_weights(policy: UniversalPolicy, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["policy_state_dict"]
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()


def _mission_onehot(mission_text: str) -> np.ndarray:
    """Build the same (color, type) one-hot that EnvWorker / training
    produces. Returns a length-(NUM_COLORS * len(OBJECT_TYPES)) vector."""
    mission_dim = len(OBJECT_TYPES) * NUM_COLORS
    v = np.zeros(mission_dim, dtype=np.float32)
    try:
        preds = goal_predicates_for_mission(mission_text, None)
        if preds and preds[0][1] is not None and preds[0][2] is not None:
            col, typ = preds[0][1], preds[0][2]
            if 0 <= col < NUM_COLORS and 0 <= typ < len(OBJECT_TYPES):
                v[typ * NUM_COLORS + col] = 1.0
    except Exception:
        pass
    return v


def _allowed_actions_for_mission(mission_text: str, n_actions: int) -> tuple[int, ...]:
    """Return the allowed-action indices for this mission's predicate.
    Mirrors EnvWorker's `allowed` field — same masking pipeline as
    training, so the policy isn't evaluated under different action
    constraints than it was trained under.
    """
    try:
        preds = goal_predicates_for_mission(mission_text, None)
        if preds and preds[0][0] in MISSION_ALLOWED_ACTIONS:
            return MISSION_ALLOWED_ACTIONS[preds[0][0]]
    except Exception:
        pass
    return tuple(range(n_actions))


def _make_mask(allowed: tuple[int, ...], n_actions: int,
               device: torch.device) -> torch.Tensor:
    """Additive (-inf / 0) mask of shape (1, n_actions)."""
    mask = torch.full((1, n_actions), float("-inf"), device=device)
    for a in allowed:
        mask[0, a] = 0.0
    return mask


@torch.no_grad()
def evaluate_policy_on_env(
    policy: UniversalPolicy,
    env_id: str,
    n_episodes: int,
    greedy: bool,
    seed: int,
    device: torch.device,
    max_steps: int = 64,
    n_actions: int = 7,
) -> dict:
    """Run `n_episodes` of evaluation on env_id. Return success rate +
    distributional stats."""
    env = make_env_with_max_steps(env_id, max_steps=max_steps)
    successes = 0
    ep_returns: list[float] = []
    ep_lengths: list[int] = []

    for ep in range(n_episodes):
        obs, _info = env.reset(seed=seed + ep)
        h = policy.init_hidden(1, device)
        prev_a = torch.full((1,), -1, device=device, dtype=torch.long)
        ep_return = 0.0
        ep_length = 0

        for _step in range(max_steps + 1):
            obs_img = _encode_image(obs["image"])
            obs_t = torch.from_numpy(obs_img).float().unsqueeze(0).to(device)
            mission = _mission_onehot(obs["mission"])
            mission_t = torch.from_numpy(mission).float().unsqueeze(0).to(device)
            allowed = _allowed_actions_for_mission(obs["mission"], n_actions)
            mask = _make_mask(allowed, n_actions, device)

            z = policy.adapter.encode_obs(obs_t)
            logits, _value, h = policy.step_with_value(z, prev_a, mission_t, h)
            dist = policy.action_dist(logits, mask)
            if greedy:
                # Greedy from the masked distribution.
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
            prev_a = action

            step_out = env.step(int(action.item()))
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _info = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, _info = step_out
            ep_return += float(reward)
            ep_length += 1

            if done:
                if reward > 0.0:
                    successes += 1
                break

        ep_returns.append(ep_return)
        ep_lengths.append(ep_length)

    return {
        "env_id": env_id,
        "n_episodes": n_episodes,
        "success_rate": successes / max(n_episodes, 1),
        "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "mean_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "min_return": float(np.min(ep_returns)) if ep_returns else 0.0,
        "max_return": float(np.max(ep_returns)) if ep_returns else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint", action="append", nargs="+", required=True,
        help="One or more checkpoint specs. Each spec is "
             "`<path> [label]`. Label defaults to the parent dir name. "
             "Specify multiple times for cross-curriculum comparison.",
    )
    p.add_argument("--jepa-checkpoint", type=Path, required=True)
    p.add_argument("--n-episodes", type=int, default=100,
                   help="Episodes per (checkpoint, env) cell. Default 100. "
                        "Single eval ≈ <30s per cell on CPU, faster on GPU.")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions instead of greedy. Tighter variance "
                        "estimates; harder evaluation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--envs", nargs="+", default=E1_EVAL_ENVS,
        help="Evaluation envs. Default: the E1 suite.",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    greedy = not args.stochastic

    # Parse checkpoint specs: each --checkpoint can have 1 or 2 tokens.
    parsed: list[tuple[str, Path]] = []
    for spec in args.checkpoint:
        if len(spec) == 1:
            path = Path(spec[0])
            label = path.parent.name
        elif len(spec) == 2:
            path = Path(spec[0])
            label = spec[1]
        else:
            print(f"FAIL: --checkpoint expects 1 or 2 tokens, got {spec}")
            sys.exit(2)
        if not path.exists():
            print(f"FAIL: checkpoint not found: {path}")
            sys.exit(2)
        parsed.append((label, path))

    print(f"[e1-eval] {len(parsed)} checkpoint(s) × {len(args.envs)} env(s) "
          f"× {args.n_episodes} episodes "
          f"({'greedy' if greedy else 'stochastic'})")

    print(f"[e1-eval] building substrate from {args.jepa_checkpoint.name} …")
    policy = _build_policy(args.jepa_checkpoint, device)

    results: dict[str, dict[str, dict]] = {}
    for label, ckpt_path in parsed:
        print(f"\n[e1-eval] === arm: {label} ({ckpt_path.name}) ===")
        _load_policy_weights(policy, ckpt_path)
        arm_results: dict[str, dict] = {}
        for env_id in args.envs:
            print(f"[e1-eval]   evaluating on {env_id} …")
            r = evaluate_policy_on_env(
                policy=policy, env_id=env_id,
                n_episodes=args.n_episodes, greedy=greedy,
                seed=args.seed, device=device,
            )
            arm_results[env_id] = r
            print(f"[e1-eval]   {env_id}: success={r['success_rate']:.2%} "
                  f"mean_R={r['mean_return']:.3f} "
                  f"mean_len={r['mean_length']:.1f}")
        results[label] = arm_results

    # Comparison table.
    print()
    print("=" * 86)
    print("E1 cross-env evaluation")
    print("=" * 86)
    header = f"{'arm':<14}"
    for env_id in args.envs:
        short = env_id.replace("BabyAI-", "").replace("-v0", "")
        header += f"{short:>16}"
    header += f"{'mean':>12}"
    print(header)
    print("-" * 86)
    for label in results:
        row = f"{label:<14}"
        succ = []
        for env_id in args.envs:
            s = results[label][env_id]["success_rate"]
            row += f"{s:>15.2%} "
            succ.append(s)
        row += f"{np.mean(succ):>11.2%}"
        print(row)
    print()

    # Decide the E1 verdict.
    if len(results) >= 2:
        means = {lbl: np.mean([results[lbl][e]["success_rate"]
                               for e in args.envs])
                 for lbl in results}
        sorted_arms = sorted(means.items(), key=lambda x: -x[1])
        winner = sorted_arms[0]
        worst = sorted_arms[-1]
        gap_pp = (winner[1] - worst[1]) * 100
        print(f"[e1-eval] best mean success: {winner[0]} = {winner[1]:.2%}")
        print(f"[e1-eval] worst mean success: {worst[0]} = {worst[1]:.2%}")
        print(f"[e1-eval] gap = {gap_pp:.1f}pp")
        if winner[0] == "forward":
            if gap_pp >= 5.0:
                print(f"[e1-eval] forward DOMINATES "
                      f"(≥5pp over worst arm) — developmental "
                      f"ordering hypothesis SUPPORTED.")
            else:
                print(f"[e1-eval] forward leads but by < 5pp — "
                      f"inconclusive; ordering may not be load-bearing.")
        elif "forward" in means:
            print(f"[e1-eval] forward NOT the winner — developmental "
                  f"ordering hypothesis NOT supported by mean. The "
                  f"winning arm was {winner[0]}.")
        else:
            print(f"[e1-eval] no 'forward' arm in results; cannot apply "
                  f"E1 verdict logic.")

    sys.exit(0)


if __name__ == "__main__":
    main()
