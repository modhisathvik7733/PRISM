"""Supervised pretraining for the ObjectProbe.

Loads rollouts.npz + rollouts.slots.pkl, builds (latent, type, color)
→ (presence, position) training tuples, trains the probe MLP. Saves
the probe checkpoint + a held-out accuracy number.

Usage:
    python -m scripts.cog_core.train_object_tracker \
        --rollouts runs/cog_core_phase1/rollouts.npz \
        --steps 5000 --batch-size 128 --device cuda \
        --run-name cog_phase1_objects
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from prism.cog_core.object_tracker import ObjectProbe, build_training_examples
from prism.utils.seed import set_global_seed


def collate_rollouts(npz_path: Path, slots_path: Path
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the .npz + slots pickle and emit one big training set."""
    d = np.load(npz_path)
    obs = d["obs"]                   # (N, T, 3, 7, 7) — unused here, slots have it
    latents = d["latents"]           # (N, T, ...)
    lengths = d["ep_lengths"]
    with open(slots_path, "rb") as f:
        slots_list = pickle.load(f)  # list of length N, each list of T_actual lists

    all_L: list[np.ndarray] = []
    all_T: list[np.ndarray] = []
    all_C: list[np.ndarray] = []
    all_P: list[np.ndarray] = []
    all_XY: list[np.ndarray] = []
    for i, ep_slots in enumerate(slots_list):
        L = int(lengths[i])
        rollout = {
            "latents": latents[i, :L],
            "slots": ep_slots[:L],
        }
        L_, T_, C_, P_, XY_ = build_training_examples(rollout)
        all_L.append(L_)
        all_T.append(T_)
        all_C.append(C_)
        all_P.append(P_)
        all_XY.append(XY_)
    return (np.concatenate(all_L), np.concatenate(all_T),
            np.concatenate(all_C), np.concatenate(all_P),
            np.concatenate(all_XY))


def split(idx_total: int, val_frac: float, rng: np.random.Generator
          ) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(idx_total)
    n_val = int(idx_total * val_frac)
    return perm[n_val:], perm[:n_val]


@torch.no_grad()
def eval_probe(probe: ObjectProbe, L, T, C, P, XY,
               idx: np.ndarray, device: torch.device,
               batch_size: int = 256) -> dict:
    probe.eval()
    n_correct_pres = 0
    n_total = 0
    pos_errs: list[float] = []
    for start in range(0, len(idx), batch_size):
        b = idx[start:start + batch_size]
        z = torch.from_numpy(L[b]).to(device)
        t = torch.from_numpy(T[b]).to(device)
        c = torch.from_numpy(C[b]).to(device)
        p_gt = torch.from_numpy(P[b]).to(device)
        xy_gt = torch.from_numpy(XY[b]).to(device)
        presence_logit, pos = probe(z, t, c)
        pred = (torch.sigmoid(presence_logit) > 0.5).float()
        n_correct_pres += int((pred == p_gt).sum().item())
        n_total += len(b)
        # Position error only on positive examples.
        mask = (p_gt > 0.5)
        if mask.any():
            pos_errs.extend(
                (((pos[mask] - xy_gt[mask]) ** 2).sum(dim=-1) ** 0.5).cpu().tolist()
            )
    return {
        "presence_acc": n_correct_pres / max(n_total, 1),
        "position_mae": float(np.mean(pos_errs)) if pos_errs else 0.0,
        "n_examples": n_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True,
                        help="path to rollouts.npz from collect_rollouts")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    rollouts_path = Path(args.rollouts)
    slots_path = rollouts_path.with_suffix(".slots.pkl")
    print(f"[obj-train] loading {rollouts_path} + {slots_path}")
    L, T, C, P, XY = collate_rollouts(rollouts_path, slots_path)
    print(f"[obj-train] training set: {len(L)} examples")
    print(f"  presence positives: {P.sum():.0f}  ({100 * P.mean():.1f}%)")

    # Detect latent dim from the first example.
    latent_dim = L.shape[-1] if L.ndim == 2 else int(np.prod(L.shape[1:]))
    if L.ndim > 2:
        L = L.reshape(L.shape[0], -1)
    print(f"[obj-train] latent dim: {latent_dim}")

    probe = ObjectProbe(latent_dim=latent_dim, hidden=args.hidden).to(device)
    print(f"[obj-train] probe params: "
          f"{sum(p.numel() for p in probe.parameters()):,}")

    rng = np.random.default_rng(args.seed)
    train_idx, val_idx = split(len(L), args.val_frac, rng)
    print(f"[obj-train] train={len(train_idx)} val={len(val_idx)}")

    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")

    for step in range(args.steps):
        b = rng.choice(train_idx, size=args.batch_size)
        z = torch.from_numpy(L[b]).to(device)
        t = torch.from_numpy(T[b]).to(device)
        c = torch.from_numpy(C[b]).to(device)
        p_gt = torch.from_numpy(P[b]).to(device)
        xy_gt = torch.from_numpy(XY[b]).to(device)
        out = probe.loss(z, t, c, p_gt, xy_gt)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            writer.add_scalar("train/loss", float(loss.item()), step)
            writer.add_scalar("train/bce", float(out["bce"].item()), step)
            writer.add_scalar("train/mse", float(out["mse"].item()), step)
            print(f"[step {step:5d}/{args.steps}] loss={float(loss.item()):.4f} "
                  f"bce={float(out['bce'].item()):.4f} "
                  f"mse={float(out['mse'].item()):.4f}")

        if (step + 1) % 1000 == 0 or step == args.steps - 1:
            metrics = eval_probe(probe, L, T, C, P, XY, val_idx, device)
            writer.add_scalar("val/presence_acc", metrics["presence_acc"], step)
            writer.add_scalar("val/position_mae", metrics["position_mae"], step)
            print(f"  [val @ step {step+1}] "
                  f"presence_acc={metrics['presence_acc']*100:.1f}% "
                  f"position_mae={metrics['position_mae']:.2f}")

    # Final
    metrics = eval_probe(probe, L, T, C, P, XY, val_idx, device)
    ckpt = {
        "model_state_dict": probe.state_dict(),
        "latent_dim": latent_dim,
        "hidden": args.hidden,
        "metrics": metrics,
    }
    torch.save(ckpt, out_dir / "model_final.pt")
    print(f"\n[done] saved {out_dir / 'model_final.pt'}")
    print(f"[done] presence_acc={metrics['presence_acc']*100:.2f}% "
          f"position_mae={metrics['position_mae']:.2f}")
    print(f"[done] Phase 1 emergence target: ≥85% presence accuracy "
          f"({'PASS' if metrics['presence_acc'] >= 0.85 else 'FAIL'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
