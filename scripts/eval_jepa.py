"""Phase 1 evaluation — does the trained JEPA actually predict?

Loads a checkpoint from `train_jepa.py` and runs three falsifier tests
straight from the roadmap:

  1. SINGLE-STEP PREDICTION
     For a batch of (s_t, a_t, s_{t+1}) from real (not training) rollouts,
     measure ‖predict(z_t, a_t) − encode_target(s_{t+1})‖² .
     Compare to the trivial baseline of predicting the embedding mean
     (variance of the target distribution). The model must beat that.

  2. ROLLOUT DRIFT
     From s_0, autoregressively roll the latent forward: z_{t+1} = predict(z_t, a_t).
     Compare to the actually-encoded states encode_target(s_t) at horizons
     {1, 2, 4, 8, 16}. Falsifier from the roadmap:
       "rollout drift dominates by horizon 4 → JEPA losses insufficient"

  3. ACTION SENSITIVITY
     For each state z_t, compute the spread of predict(z_t, a) across all
     actions a. If the model ignores the action input, this spread is ~0
     and the model is just predicting marginal next-state. Falsifier:
       "counterfactual error collapses to mean-prediction"
     We approximate this without env-replay by checking that different
     actions produce meaningfully different predictions.

Usage:
    python -m scripts.eval_jepa \
        --checkpoint runs/jepa_BabyAI-GoToLocal-v0_seed0/jepa_final.pt \
        --episodes 50 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from prism.envs import make_babyai_env
from prism.models.jepa import JepaConfig, JepaWorldModel
from prism.utils.seed import set_global_seed


def collect_trajectories(env, n_episodes: int, max_steps: int, rng: np.random.Generator):
    """Run n_episodes random-policy rollouts, return list of (obs_seq, act_seq).
    Each obs_seq is (T+1, C, H, W); act_seq is (T,)."""
    trajs = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        obs_seq = [obs]
        act_seq = []
        for _ in range(max_steps):
            a = int(rng.integers(env.action_space.n))
            obs, _r, term, trunc, _ = env.step(a)
            obs_seq.append(obs)
            act_seq.append(a)
            if term or trunc:
                break
        if len(act_seq) >= 2:  # need at least horizon-2 trajectories
            trajs.append((np.stack(obs_seq).astype(np.float32),
                          np.array(act_seq, dtype=np.int64)))
    return trajs


@torch.no_grad()
def eval_single_step(model: JepaWorldModel, trajs, device, batch_size: int = 256):
    """MSE of predict(z_t, a_t) vs encode_target(s_{t+1}) on real data."""
    # Pool all (s_t, a_t, s_{t+1}) triples across trajectories.
    s_t, acts, s_tp1 = [], [], []
    for obs_seq, act_seq in trajs:
        s_t.append(obs_seq[:-1])
        s_tp1.append(obs_seq[1:])
        acts.append(act_seq)
    s_t = np.concatenate(s_t)
    s_tp1 = np.concatenate(s_tp1)
    acts = np.concatenate(acts)

    n = len(acts)
    pred_mses = []
    target_norms = []  # for the "predict-the-mean" baseline
    all_targets = []

    for i in range(0, n, batch_size):
        st = torch.from_numpy(s_t[i : i + batch_size]).to(device)
        a = torch.from_numpy(acts[i : i + batch_size]).to(device)
        stp1 = torch.from_numpy(s_tp1[i : i + batch_size]).to(device)

        z_t = model.encode(st)
        z_pred = model.predict(z_t, a)
        z_target = model.encode_target(stp1)

        pred_mses.append(((z_pred - z_target) ** 2).mean(dim=-1).cpu().numpy())
        target_norms.append((z_target ** 2).mean(dim=-1).cpu().numpy())
        all_targets.append(z_target.cpu().numpy())

    pred_mse = np.concatenate(pred_mses).mean()

    # "Predict the mean" baseline: MSE of (target - mean(target)) ** 2.
    targets = np.concatenate(all_targets)  # (N, D)
    target_mean = targets.mean(0, keepdims=True)
    mean_pred_mse = ((targets - target_mean) ** 2).mean()

    return {
        "n_transitions": n,
        "pred_mse": float(pred_mse),
        "mean_baseline_mse": float(mean_pred_mse),
        "skill_ratio": float(mean_pred_mse / max(pred_mse, 1e-9)),  # >1 means we beat the baseline
    }


@torch.no_grad()
def eval_rollout_drift(
    model: JepaWorldModel, trajs, device, horizons=(1, 2, 4, 8, 16)
):
    """Autoregressive latent rollout vs actual encoded sequence."""
    max_h = max(horizons)
    results = {h: [] for h in horizons}

    for obs_seq, act_seq in trajs:
        T = len(act_seq)
        if T < max_h:
            continue
        # Encode the whole trajectory once
        obs = torch.from_numpy(obs_seq).to(device)  # (T+1, C, H, W)
        z_actual = model.encode_target(obs)  # (T+1, D)
        # Roll latent forward from z_0
        z = z_actual[0:1]  # (1, D)
        for t in range(max_h):
            a = torch.tensor([act_seq[t]], device=device)
            z = model.predict(z, a)
            h = t + 1
            if h in results:
                err = ((z - z_actual[h:h+1]) ** 2).mean().item()
                results[h].append(err)
    return {h: float(np.mean(v)) if v else float("nan") for h, v in results.items()}


@torch.no_grad()
def eval_action_sensitivity(model: JepaWorldModel, trajs, device, batch_size: int = 256):
    """For each state, compute std of predict(z, a) across actions."""
    states = np.concatenate([obs_seq[:-1] for obs_seq, _ in trajs])
    n_actions = model.cfg.n_actions
    stds = []
    target_stds = []  # spread of actual encoded states for reference

    for i in range(0, len(states), batch_size):
        s = torch.from_numpy(states[i : i + batch_size]).to(device)
        z = model.encode(s)  # (B, D)
        # For each action a, predict next latent
        preds = []
        for a_id in range(n_actions):
            a = torch.full((z.shape[0],), a_id, device=device, dtype=torch.long)
            preds.append(model.predict(z, a))
        preds = torch.stack(preds, dim=1)  # (B, n_actions, D)
        # Std across action axis, averaged over D
        std = preds.std(dim=1).mean(dim=-1)
        stds.append(std.cpu().numpy())
        target_stds.append(z.std(dim=0).mean().item())

    stds = np.concatenate(stds)
    return {
        "action_pred_std_mean": float(stds.mean()),
        "encoded_state_std_mean": float(np.mean(target_stds)),
        # ratio: how much does the prediction depend on action vs the natural
        # variation in encoded states? if << 1, the model ignores actions.
        "action_to_state_ratio": float(stds.mean() / max(np.mean(target_stds), 1e-9)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    print(f"[eval] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: JepaConfig = ckpt["cfg"]
    model = JepaWorldModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eval] model loaded ({n_params:,} params), trained for {ckpt.get('step', '?')} steps")

    print(f"[eval] collecting {args.episodes} eval episodes (held-out seeds)...")
    env = make_babyai_env(args.env_id, include_mission=False)
    rng = np.random.default_rng(args.seed)
    trajs = collect_trajectories(env, args.episodes, args.max_steps, rng)
    n_trans = sum(len(a) for _, a in trajs)
    print(f"[eval] got {len(trajs)} trajectories, {n_trans} transitions")

    # --- 1. single-step prediction ----------------------------------------
    print("\n=== 1. single-step prediction ===")
    r1 = eval_single_step(model, trajs, device)
    print(f"  pred MSE                : {r1['pred_mse']:.4f}")
    print(f"  predict-the-mean MSE    : {r1['mean_baseline_mse']:.4f}")
    print(f"  skill ratio (>1 = good) : {r1['skill_ratio']:.2f}x")
    pass1 = r1["skill_ratio"] > 2.0
    print(f"  pass (ratio > 2.0)      : {'YES' if pass1 else 'NO'}")

    # --- 2. rollout drift -------------------------------------------------
    print("\n=== 2. rollout drift over horizon ===")
    r2 = eval_rollout_drift(model, trajs, device)
    for h, mse in r2.items():
        print(f"  horizon {h:2d}: MSE = {mse:.4f}")
    pass2 = r2.get(4, float("inf")) < 5 * r2.get(1, 1e-9)  # h=4 should be < 5x h=1
    print(f"  pass (h=4 < 5x h=1)     : {'YES' if pass2 else 'NO'}")

    # --- 3. action sensitivity --------------------------------------------
    print("\n=== 3. action sensitivity ===")
    r3 = eval_action_sensitivity(model, trajs, device)
    print(f"  std of pred across acts : {r3['action_pred_std_mean']:.4f}")
    print(f"  std of encoded states   : {r3['encoded_state_std_mean']:.4f}")
    print(f"  action / state ratio    : {r3['action_to_state_ratio']:.3f}")
    # We want this to be non-negligible. If actions barely change predictions,
    # the dynamics model is ignoring its action input.
    pass3 = r3["action_to_state_ratio"] > 0.05
    print(f"  pass (ratio > 0.05)     : {'YES' if pass3 else 'NO'}")

    # --- summary ----------------------------------------------------------
    print("\n=== summary ===")
    all_pass = pass1 and pass2 and pass3
    print(f"  1. beats mean-prediction        : {'PASS' if pass1 else 'FAIL'}")
    print(f"  2. rollout drift bounded        : {'PASS' if pass2 else 'FAIL'}")
    print(f"  3. action conditioning works    : {'PASS' if pass3 else 'FAIL'}")
    print(f"\n  Phase 1 falsifiers: {'ALL PASS — proceed to Phase 2' if all_pass else 'ONE OR MORE FIRED'}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
