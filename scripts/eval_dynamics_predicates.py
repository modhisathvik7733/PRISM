"""Layer 3 diagnostic — does predicate readout survive the dynamics step?

Layers we have already verified:
  L1 (eval_jepa.py)         : encoder + dynamics in latent space — PASS
  L2 (train_predicate_probe): aux head on encoded z_t              — PASS (F1 0.96)

What the AGENT actually queries:
    sigmoid(aux_head(predict(z_t, a)))  →  predicate probs over imagined next state

If those don't match real ground-truth predicates_{t+1}, the agent is scoring
actions on noise even though every component is "fine" in isolation.

Three accuracies reported:
  acc_t             — aux_head(z_t)               vs gt preds_t          (sanity, should ~match L2)
  acc_pred_tp1      — aux_head(predict(z_t, a_t)) vs gt preds_{t+1}      ← the agent's actual query
  acc_enc_tp1       — aux_head(encode(s_{t+1}))   vs gt preds_{t+1}      (upper bound on L2 transfer)

Gap between acc_pred_tp1 and acc_enc_tp1 = "predict step degrades predicates by X%".
Gap between acc_enc_tp1  and acc_t       = "encoder generalizes to next-step distribution by Y%".

Per-action and per-predicate-name (visible/facing/near/adjacent) breakdowns
let us see whether forward/turn-left/turn-right are differentially broken.

Usage:
    python -m scripts.eval_dynamics_predicates \
        --checkpoint runs/<run-name>/jepa_final.pt \
        --episodes 200 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch

from prism.envs.babyai import _encode_image
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.perception import (
    compute_augmented_predicates,
    compute_predicates,
    extract_slots,
)
from prism.perception.predicates import (
    NUM_TYPE_COLOR_PAIRS,
    PREDICATE_NAMES,
    PREDICATE_VECTOR_DIM,
)
from prism.utils.seed import set_global_seed


def collect(env, n_episodes: int, max_steps: int, rng: np.random.Generator,
            *, augmented: bool = False):
    """Random-policy rollouts. Returns parallel arrays:
       obs_t (N,3,7,7), actions (N,), obs_tp1 (N,3,7,7),
       preds_t (N,96 or 120), preds_tp1 (N,96 or 120).
    """
    obs_t, actions, obs_tp1, preds_t, preds_tp1 = [], [], [], [], []
    pred_fn = compute_augmented_predicates if augmented else compute_predicates
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        for _ in range(max_steps):
            raw_t = obs["image"]
            a = int(rng.integers(env.action_space.n))
            next_obs, _r, term, trunc, _ = env.step(a)
            raw_tp1 = next_obs["image"]

            obs_t.append(_encode_image(raw_t))
            actions.append(a)
            obs_tp1.append(_encode_image(raw_tp1))
            preds_t.append(pred_fn(extract_slots(raw_t)))
            preds_tp1.append(pred_fn(extract_slots(raw_tp1)))

            if term or trunc:
                break
            obs = next_obs
    return (
        np.stack(obs_t).astype(np.float32),
        np.array(actions, dtype=np.int64),
        np.stack(obs_tp1).astype(np.float32),
        np.stack(preds_t).astype(np.float32),
        np.stack(preds_tp1).astype(np.float32),
    )


def f1(pred_bin: np.ndarray, gt_bin: np.ndarray) -> tuple[float, float, float, float]:
    """Macro-style scalar metrics over a flat 0/1 prediction matrix."""
    tp = float(((pred_bin == 1) & (gt_bin == 1)).sum())
    fp = float(((pred_bin == 1) & (gt_bin == 0)).sum())
    fn = float(((pred_bin == 0) & (gt_bin == 1)).sum())
    acc = float((pred_bin == gt_bin).mean())
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1_v = 2 * prec * rec / max(prec + rec, 1e-9)
    return acc, prec, rec, f1_v


@torch.no_grad()
def head_probs(model: JepaWorldModel, latent: torch.Tensor, batch: int = 512) -> np.ndarray:
    out = []
    for i in range(0, latent.shape[0], batch):
        out.append(torch.sigmoid(model.aux_predicate_head(latent[i:i+batch])).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def encode_all(model: JepaWorldModel, x: np.ndarray, device, batch: int = 512) -> torch.Tensor:
    out = []
    for i in range(0, x.shape[0], batch):
        b = torch.from_numpy(x[i:i+batch]).to(device)
        out.append(model.encode(b))
    return torch.cat(out, dim=0)


@torch.no_grad()
def encode_target_all(model: JepaWorldModel, x: np.ndarray, device, batch: int = 512) -> torch.Tensor:
    out = []
    for i in range(0, x.shape[0], batch):
        b = torch.from_numpy(x[i:i+batch]).to(device)
        out.append(model.encode_target(b))
    return torch.cat(out, dim=0)


@torch.no_grad()
def predict_all(model: JepaWorldModel, z: torch.Tensor, a: np.ndarray, device, batch: int = 512) -> torch.Tensor:
    out = []
    for i in range(0, z.shape[0], batch):
        ai = torch.from_numpy(a[i:i+batch]).to(device)
        out.append(model.predict(z[i:i+batch], ai))
    return torch.cat(out, dim=0)


def report_block(name: str, probs: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> None:
    pred_bin = (probs >= threshold).astype(np.int8)
    gt_bin = gt.astype(np.int8)
    acc, prec, rec, f1_v = f1(pred_bin, gt_bin)
    print(f"  {name:18s}  acc={acc:.4f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1_v:.3f}")


def report_per_predicate(label: str, probs: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> None:
    """Break F1 down by predicate name (visible/near/facing/adjacent).
    Slot layout per perception/predicates.py: 96 = NUM_TYPE_COLOR_PAIRS(24) * NUM_PREDICATES(4),
    indexed as predicate_idx * 24 + type_color_idx.
    """
    print(f"  per-predicate F1 [{label}]:")
    for pi, pname in enumerate(PREDICATE_NAMES):
        sl = slice(pi * NUM_TYPE_COLOR_PAIRS, (pi + 1) * NUM_TYPE_COLOR_PAIRS)
        _, _, _, f1_v = f1((probs[:, sl] >= threshold).astype(np.int8), gt[:, sl].astype(np.int8))
        pos_rate = float(gt[:, sl].mean())
        print(f"    {pname:9s}  f1={f1_v:.3f}  positive_rate={pos_rate:.4f}")


def report_per_action(probs: np.ndarray, gt: np.ndarray, actions: np.ndarray, n_actions: int) -> None:
    print("  predicted-t+1 F1 broken down by action taken:")
    for a in range(n_actions):
        mask = actions == a
        if mask.sum() < 32:
            print(f"    action {a}: n={mask.sum()} (too few)")
            continue
        _, _, _, f1_v = f1(
            (probs[mask] >= 0.5).astype(np.int8),
            gt[mask].astype(np.int8),
        )
        print(f"    action {a}: n={mask.sum():5d}  f1={f1_v:.3f}")


def report_target_preservation(probs_pred_tp1: np.ndarray, gt_t: np.ndarray, gt_tp1: np.ndarray) -> None:
    """Of cases where SOME predicate was 1 at t AND still 1 at t+1, does the
    agent's predicted-t+1 distribution preserve it? This is the agent's
    failure mode: it imagines forward and 'forgets' the visible target.
    """
    persistent = (gt_t > 0.5) & (gt_tp1 > 0.5)  # (N, 96)
    if persistent.sum() == 0:
        print("  (no persistent positives in this batch)")
        return
    pred_bin = probs_pred_tp1 >= 0.5
    preserved = pred_bin & persistent
    rate = float(preserved.sum()) / float(persistent.sum())
    print(f"  predicate-preservation rate (gt=1@t AND gt=1@t+1): {rate:.4f}  (n={int(persistent.sum())})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg: JepaConfig = upgrade_config(ckpt["cfg"])
    model = JepaWorldModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if model.aux_predicate_head is None:
        raise SystemExit("checkpoint has no aux_predicate_head — train with aux_predicate_weight > 0")
    print(f"[eval-l3] loaded checkpoint trained for {ckpt.get('step', '?')} steps")

    aux_dist_dim = getattr(cfg, "aux_distance_dim", 0)
    augmented = aux_dist_dim > 0
    print(f"[eval-l3] aux_distance_dim={aux_dist_dim} (augmented={augmented})")
    print(f"[eval-l3] collecting {args.episodes} episodes (held-out seeds) ...")
    env = gym.make(args.env_id)
    rng = np.random.default_rng(args.seed)
    obs_t, acts, obs_tp1, gt_t, gt_tp1 = collect(
        env, args.episodes, args.max_steps, rng, augmented=augmented,
    )
    n = len(acts)
    print(f"[eval-l3] N transitions = {n}")
    # Slice ground truth into binary block (first 96) and distance block (last 24).
    gt_bin_t, gt_bin_tp1 = gt_t[:, :PREDICATE_VECTOR_DIM], gt_tp1[:, :PREDICATE_VECTOR_DIM]
    gt_dist_t = gt_t[:, PREDICATE_VECTOR_DIM:] if augmented else None
    gt_dist_tp1 = gt_tp1[:, PREDICATE_VECTOR_DIM:] if augmented else None
    print(f"[eval-l3] preds_t  positive rate = {gt_bin_t.mean():.4f}")
    print(f"[eval-l3] preds_tp1 positive rate = {gt_bin_tp1.mean():.4f}")
    if augmented:
        print(f"[eval-l3] dist_t  mean = {gt_dist_t.mean():.4f}  min = {gt_dist_t.min():.4f}")
        print(f"[eval-l3] dist_tp1 mean = {gt_dist_tp1.mean():.4f}  min = {gt_dist_tp1.min():.4f}")

    # Encode and predict.
    z_t = encode_all(model, obs_t, device)                  # online encoder, what the agent uses
    z_tp1_enc = encode_target_all(model, obs_tp1, device)   # EMA target encoder, training target
    z_tp1_pred = predict_all(model, z_t, acts, device)

    # The aux head outputs PREDICATE_VECTOR_DIM (96) + aux_distance_dim (0 or 24).
    # We sigmoid the binary block for F1 and sigmoid the distance block for MAE.
    P = PREDICATE_VECTOR_DIM
    probs_t_full = head_probs(model, z_t)
    probs_enc_tp1_full = head_probs(model, z_tp1_enc)
    probs_pred_tp1_full = head_probs(model, z_tp1_pred)

    probs_t, probs_enc_tp1, probs_pred_tp1 = (
        probs_t_full[:, :P], probs_enc_tp1_full[:, :P], probs_pred_tp1_full[:, :P]
    )

    print("\n=== aggregate accuracies (threshold 0.5) ===")
    report_block("acc_t",        probs_t,         gt_bin_t)
    report_block("acc_enc_tp1",  probs_enc_tp1,   gt_bin_tp1)
    report_block("acc_pred_tp1", probs_pred_tp1,  gt_bin_tp1)

    print("\n=== per-predicate F1 ===")
    report_per_predicate("at z_t",                 probs_t,         gt_bin_t)
    report_per_predicate("at encode(s_{t+1})",     probs_enc_tp1,   gt_bin_tp1)
    report_per_predicate("at predict(z_t, a_t)",   probs_pred_tp1,  gt_bin_tp1)

    print("\n=== per-action breakdown (predicted t+1) ===")
    report_per_action(probs_pred_tp1, gt_bin_tp1, acts, model.cfg.n_actions)

    print("\n=== predicate preservation (the agent's failure mode) ===")
    report_target_preservation(probs_pred_tp1, gt_bin_t, gt_bin_tp1)

    if augmented:
        print("\n=== distance head (MAE on continuous dim, lower = better) ===")
        d_t = probs_t_full[:, P:]
        d_enc_tp1 = probs_enc_tp1_full[:, P:]
        d_pred_tp1 = probs_pred_tp1_full[:, P:]
        # Overall MAE
        print(f"  d_t           MAE = {np.abs(d_t - gt_dist_t).mean():.4f}")
        print(f"  d_enc_tp1     MAE = {np.abs(d_enc_tp1 - gt_dist_tp1).mean():.4f}")
        print(f"  d_pred_tp1    MAE = {np.abs(d_pred_tp1 - gt_dist_tp1).mean():.4f}")
        # Per-action MAE — the rotation question. If turn-action distance MAE
        # is much higher than forward, distance scoring will still under-pick
        # forward when it should win.
        print("  per-action distance MAE [predict(z_t, a)]:")
        for a in range(model.cfg.n_actions):
            mask = acts == a
            if mask.sum() < 32:
                print(f"    action {a}: n={mask.sum()} (too few)")
                continue
            mae = float(np.abs(d_pred_tp1[mask] - gt_dist_tp1[mask]).mean())
            print(f"    action {a}: n={mask.sum():5d}  MAE={mae:.4f}")
        # MAE restricted to the 1-2 distance bins where the agent actually
        # operates (target visible in view): excludes the trivial "not visible
        # → 1.0" entries which are easy.
        visible_mask = gt_dist_tp1 < 0.99
        if visible_mask.sum() > 0:
            mae_vis = float(np.abs(d_pred_tp1[visible_mask] - gt_dist_tp1[visible_mask]).mean())
            print(f"  d_pred_tp1 MAE | target visible: {mae_vis:.4f}  (n={int(visible_mask.sum())})")

    # Verdict
    _, _, _, f1_t = f1((probs_t >= 0.5).astype(np.int8), gt_bin_t.astype(np.int8))
    _, _, _, f1_pred = f1((probs_pred_tp1 >= 0.5).astype(np.int8), gt_bin_tp1.astype(np.int8))
    drop = f1_t - f1_pred
    print("\n=== verdict ===")
    print(f"  F1(z_t)              = {f1_t:.3f}")
    print(f"  F1(predict(z_t, a))  = {f1_pred:.3f}")
    print(f"  drop                 = {drop:.3f}")
    if drop > 0.15:
        print("  → DYNAMICS DEGRADES PREDICATES SIGNIFICANTLY. Agent is scoring on noisy imagined-future predicates.")
        print("    Fix candidates: (a) increase aux_predicate_weight on the predicted-t+1 head;")
        print("    (b) longer training; (c) counterfactual-prediction loss on alternative actions.")
    elif drop > 0.05:
        print("  → mild degradation; dynamics path is the bottleneck but not catastrophic.")
    else:
        print("  → predicate readout survives dynamics. Failure is elsewhere (action scoring, exploration).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
