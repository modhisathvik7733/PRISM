"""Train JEPA's dynamics module on Unity-domain transitions.

The "cause-and-effect" fix: the concept-pretrained substrate sees the
world correctly but doesn't know how its actions move things in its
view. We fix that with self-supervised dynamics training, no labels
required.

Pipeline:
  1. Run thousands of episodes in UnityNavEnv with random actions
     (random is fine — dynamics learning only needs (obs, action, next_obs)
     tuples, action quality is irrelevant).
  2. Train ONLY `jepa.dynamics` to predict next-state latents:
        L = MSE(jepa.predict(z_t, a_t), jepa.encode_target(obs_{t+1}))
     The encoder stays frozen — we don't want to undo concept learning.
  3. Save the updated JEPA checkpoint.
  4. (Optionally) re-run `scripts/check_rollout_fidelity.py` to verify
     cos@H=10 jumps from ~0.01 (broken) to ≥0.9 (good).

After this, the substrate has both:
  - sharp recognition (concept memory) ✓
  - grounded dynamics (this script) ✓
which together unblock efficient policy training.

Usage:
    python unity_demo/train_dynamics_unity.py \\
        --in-jepa  runs/v6_concept_phaseAB_v2/jepa.pt \\
        --out-jepa runs/v6_dynamics_v1/jepa.pt \\
        --n-episodes 5000 --epochs 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prism.models.jepa import JepaWorldModel, upgrade_config
from unity_demo.unity_nav_env import UnityNavEnv


# ===========================================================================
# Checkpoint loading
# ===========================================================================
def load_jepa(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    return jepa, cfg, ckpt


# ===========================================================================
# Data collection: random-action rollouts
# ===========================================================================
def collect_transitions(
    env: UnityNavEnv,
    n_episodes: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run `n_episodes` random-action episodes; return (obs_t, action_t, obs_{t+1}) arrays.

    We only emit allowed actions {0, 1, 2} = {turn_L, turn_R, forward} to
    match the inference-time mask.
    """
    obs_buf: list[np.ndarray] = []
    next_obs_buf: list[np.ndarray] = []
    action_buf: list[int] = []
    n_steps_total = 0
    t0 = time.time()
    for ep in range(n_episodes):
        obs, _info = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            a = int(rng.integers(0, 3))  # uniform over allowed actions
            cur_img = obs["image"].copy()
            obs, _r, terminated, truncated, _info = env.step(a)
            next_img = obs["image"].copy()
            obs_buf.append(cur_img)
            next_obs_buf.append(next_img)
            action_buf.append(a)
            n_steps_total += 1
        if (ep + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"[collect] {ep+1}/{n_episodes} eps, {n_steps_total} transitions, "
                  f"{n_steps_total/elapsed:.0f} steps/s")
    obs_arr = np.stack(obs_buf, axis=0)
    next_arr = np.stack(next_obs_buf, axis=0)
    act_arr = np.asarray(action_buf, dtype=np.int64)
    print(f"[collect] DONE: {len(obs_arr)} transitions from {n_episodes} eps")
    return obs_arr, act_arr, next_arr


# ===========================================================================
# Training loop
# ===========================================================================
def train_dynamics(
    jepa: JepaWorldModel,
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    N = obs.size(0)
    log = {"epoch_loss": []}
    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            o_t = obs[idx]
            a_t = actions[idx]
            o_tp1 = next_obs[idx]

            # Encoder frozen — no grad. Use target encoder (EMA) for the
            # next-state target, matching JEPA's training objective.
            with torch.no_grad():
                z_t = jepa.encode(o_t)
                z_tp1_target = jepa.encode_target(o_tp1)

            # Dynamics IS trainable. Predict next-latent given current
            # latent + action.
            z_pred = jepa.dynamics(z_t, a_t)

            # JEPA's MSE loss on the predicted latent vs target latent.
            loss = F.mse_loss(z_pred, z_tp1_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in jepa.dynamics.parameters()], max_norm=1.0,
            )
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        mean_loss = total_loss / max(1, n_batches)
        log["epoch_loss"].append(mean_loss)
        elapsed = time.time() - t0
        print(f"[train] epoch {epoch+1}/{epochs}  loss={mean_loss:.4f}  "
              f"({elapsed:.1f}s, {N/elapsed:.0f} samples/s)")
    return log


# ===========================================================================
# Verification: one-step prediction quality on held-out data
# ===========================================================================
@torch.no_grad()
def verify_one_step(
    jepa: JepaWorldModel,
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    batch_size: int,
) -> dict:
    """How well does predicted z_{t+1} match real-encoded z_{t+1}?"""
    N = obs.size(0)
    cos_total = 0.0
    l2_total = 0.0
    n = 0
    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        z_t = jepa.encode(obs[start:end])
        z_tp1 = jepa.encode(next_obs[start:end])
        z_pred = jepa.dynamics(z_t, actions[start:end])
        # Flatten any spatial dims for the comparison metrics.
        zp = z_pred.flatten(1)
        zt = z_tp1.flatten(1)
        cos = F.cosine_similarity(zp, zt, dim=-1)
        l2 = (zp - zt).norm(dim=-1)
        cos_total += float(cos.sum().item())
        l2_total += float(l2.sum().item())
        n += zp.size(0)
    return {"cos_mean": cos_total / n, "l2_mean": l2_total / n}


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    p = argparse.ArgumentParser(description="Domain-specific dynamics training.")
    p.add_argument("--in-jepa", required=True, help="JEPA checkpoint to start from")
    p.add_argument("--out-jepa", required=True, help="Where to save the dynamics-trained JEPA")
    p.add_argument("--n-episodes", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available()
                 else ("mps" if torch.backends.mps.is_available() else "cpu")),
    )
    # Env / dynamics match Unity by default.
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--reach-threshold", type=float, default=1.6)
    p.add_argument("--forward-step", type=float, default=0.07)
    p.add_argument("--obs-scale", type=float, default=2.0)
    args = p.parse_args()

    device = torch.device(args.device)
    out_path = Path(args.out_jepa)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[dyn] device={device}")
    print(f"[dyn] loading JEPA from {args.in_jepa}")
    jepa, cfg, base_ckpt = load_jepa(Path(args.in_jepa), device)

    # Freeze encoder (online and target); leave dynamics trainable.
    for p_ in jepa.parameters():
        p_.requires_grad_(False)
    for p_ in jepa.dynamics.parameters():
        p_.requires_grad_(True)
    n_train = sum(p_.numel() for p_ in jepa.dynamics.parameters())
    n_total = sum(p_.numel() for p_ in jepa.parameters())
    print(f"[dyn] training dynamics module: {n_train:,} / {n_total:,} params")

    # Encoder mode: eval (so dropout/BN behave deterministically). Dynamics
    # is part of jepa but we don't have a separate train/eval toggle on
    # it; setting jepa.eval() is fine because dynamics doesn't use dropout.
    jepa.eval()

    # ---- Collect data ----
    env = UnityNavEnv(
        max_steps=args.max_steps,
        reach_threshold=args.reach_threshold,
        forward_step=args.forward_step,
        obs_scale=args.obs_scale,
        randomize_target_color=True,
        seed=7,
    )
    eval_env = UnityNavEnv(
        max_steps=args.max_steps,
        reach_threshold=args.reach_threshold,
        forward_step=args.forward_step,
        obs_scale=args.obs_scale,
        randomize_target_color=True,
        seed=8888,
    )
    rng = np.random.default_rng(0)
    print(f"[dyn] collecting transitions from {args.n_episodes} episodes "
          f"(random actions)...")
    obs_np, act_np, next_np = collect_transitions(env, args.n_episodes, device, rng)
    # Also collect a smaller held-out set for verification.
    rng_eval = np.random.default_rng(42)
    print("[dyn] collecting held-out transitions (200 eps) for verification...")
    eobs_np, eact_np, enext_np = collect_transitions(eval_env, 200, device, rng_eval)

    # Tensors.
    obs_t = torch.from_numpy(obs_np).float().to(device)
    act_t = torch.from_numpy(act_np).to(device)
    next_t = torch.from_numpy(next_np).float().to(device)
    eobs_t = torch.from_numpy(eobs_np).float().to(device)
    eact_t = torch.from_numpy(eact_np).to(device)
    enext_t = torch.from_numpy(enext_np).float().to(device)

    # ---- Baseline (before training) ----
    print("[dyn] BASELINE one-step prediction (before training):")
    base = verify_one_step(jepa, eobs_t, eact_t, enext_t, args.batch_size)
    print(f"[dyn]   cos_mean = {base['cos_mean']:.4f}   l2_mean = {base['l2_mean']:.3f}")

    # ---- Train ----
    optimizer = torch.optim.Adam(jepa.dynamics.parameters(), lr=args.lr)
    log = train_dynamics(
        jepa, obs_t, act_t, next_t,
        optimizer=optimizer, epochs=args.epochs,
        batch_size=args.batch_size, device=device,
    )

    # ---- Eval after training ----
    print("[dyn] FINAL one-step prediction (after training):")
    final = verify_one_step(jepa, eobs_t, eact_t, enext_t, args.batch_size)
    print(f"[dyn]   cos_mean = {final['cos_mean']:.4f}   l2_mean = {final['l2_mean']:.3f}")

    # ---- Save ----
    new_ckpt = {
        **base_ckpt,
        "model": jepa.state_dict(),
        "dynamics_finetune": {
            "n_episodes": args.n_episodes,
            "epochs": args.epochs,
            "base_cos": base["cos_mean"],
            "final_cos": final["cos_mean"],
            "loss_log": log["epoch_loss"],
        },
    }
    torch.save(new_ckpt, out_path)
    print(f"[dyn] saved {out_path}")

    print()
    print(f"[dyn] BASELINE cos_mean: {base['cos_mean']:.4f}")
    print(f"[dyn] FINAL    cos_mean: {final['cos_mean']:.4f}")
    print()
    if final["cos_mean"] >= 0.9:
        print("[dyn] ✓ Dynamics training succeeded — one-step prediction is sharp.")
        print("[dyn]   Next: re-run scripts/check_rollout_fidelity.py with --jepa pointing")
        print(f"[dyn]   at {out_path} to verify multi-step (H=10) fidelity.")
    elif final["cos_mean"] >= 0.7:
        print("[dyn] ~ Marginal — one-step is decent but H=10 may still drift.")
        print("[dyn]   Consider more episodes or epochs.")
    else:
        print("[dyn] ✗ One-step prediction still poor. The dynamics module may need")
        print("[dyn]   architecture changes, or the encoder distribution may have")
        print("[dyn]   drifted too far during concept pretraining.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
