"""Offline JEPA training on Crafter rollout transitions.

Loads the .npz produced by collect_rollouts.py, trains CrafterJepaWorldModel
with the JEPA loss (L_pred + L_reg), saves the final encoder checkpoint.

Usage:
    python -m scripts.crafter.train_jepa_crafter \\
        --data data/crafter_rollouts.npz \\
        --run-name crafter_jepa \\
        --epochs 5 --batch-size 256 --device cuda

Checkpoint saved to runs/<run-name>/jepa_final.pt contains:
    online_encoder_state   — CrafterCNN state_dict  (freeze this for PPO)
    dynamics_state         — _LatentDynamics state_dict
    cfg                    — CrafterJepaConfig

Training log (every 100 batches):
    [ep E/N  batch B/T]  loss=X.XXXX  l_pred=X.XXXX  l_reg=X.XXXX  lr=X.XXe-XX
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data_utils

from prism.crafter.jepa_crafter import CrafterJepaConfig, CrafterJepaWorldModel
from prism.utils.seed import set_global_seed


class _RolloutDataset(data_utils.Dataset):
    """Loads the uint8 npz into RAM and serves (obs_t, action, obs_tp1, state_t) 4-tuples.

    Shapes served per item:
      obs_t:   (3, 64, 64) float32 [0, 1]
      action:  ()          int64
      obs_tp1: (3, 64, 64) float32 [0, 1]
      state_t: (state_dim,) float32  — zeros if not in npz
    """

    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        # uint8 stored — keep on CPU as numpy, convert to float32 on __getitem__.
        self.obs_t   = d["obs_t"]    # (N, 3, 64, 64) uint8
        self.actions = d["actions"]  # (N,)            uint8
        self.obs_tp1 = d["obs_tp1"]  # (N, 3, 64, 64) uint8
        self.game_states_t = d["game_states_t"] if "game_states_t" in d else None
        has_state = self.game_states_t is not None
        print(f"[dataset] loaded {len(self.obs_t):,} transitions from {npz_path}"
              f"  game_states={'yes' if has_state else 'no'}")

    def __len__(self) -> int:
        return len(self.obs_t)

    def __getitem__(self, idx: int):
        obs_t   = torch.from_numpy(self.obs_t[idx].astype(np.float32)) / 255.0
        action  = torch.tensor(int(self.actions[idx]), dtype=torch.long)
        obs_tp1 = torch.from_numpy(self.obs_tp1[idx].astype(np.float32)) / 255.0
        if self.game_states_t is not None:
            state_t = torch.from_numpy(self.game_states_t[idx].copy())
        else:
            state_t = torch.zeros(0, dtype=torch.float32)
        return obs_t, action, obs_tp1, state_t


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/crafter_rollouts.npz")
    p.add_argument("--run-name",   default="crafter_jepa")
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--embed-dim",  type=int,   default=256)
    p.add_argument("--ema-decay",   type=float, default=0.996)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--num-workers", type=int,  default=4)
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--state-dim",   type=int,  default=-1,
                   help="Game-state dim. -1 = auto-detect from data (12 if present, else 0).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device(args.device)

    dataset = _RolloutDataset(args.data)
    loader  = data_utils.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Auto-detect state_dim from data if not specified.
    if args.state_dim == -1:
        state_dim = 12 if dataset.game_states_t is not None else 0
    else:
        state_dim = args.state_dim

    cfg = CrafterJepaConfig(
        embed_dim=args.embed_dim,
        ema_decay=args.ema_decay,
        temperature=args.temperature,
        state_dim=state_dim,
    )
    model = CrafterJepaWorldModel(cfg).to(device)
    n_enc = sum(p.numel() for p in model.online_encoder.parameters())
    n_dyn = sum(p.numel() for p in model.dynamics.parameters())
    print(f"[jepa] encoder params: {n_enc:,}   dynamics params: {n_dyn:,}")

    opt = torch.optim.AdamW(
        list(model.online_encoder.parameters()) + list(model.dynamics.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_batches = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_batches)

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[jepa] writing to {out_dir}")
    print(f"[jepa] {args.epochs} epochs × {len(loader)} batches = {total_batches} steps")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (obs_t, action, obs_tp1, state_t) in enumerate(loader, 1):
            # obs_t:   (B, 3, 64, 64) float32
            # action:  (B,)           int64
            # obs_tp1: (B, 3, 64, 64) float32
            # state_t: (B, state_dim) float32  (or zeros if no game states in npz)
            obs_t   = obs_t.to(device)
            action  = action.to(device)
            obs_tp1 = obs_tp1.to(device)
            st = state_t.to(device) if state_dim > 0 else None

            losses = model.loss(obs_t, action, obs_tp1, state_t=st)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            model.update_target()
            scheduler.step()
            global_step += 1

            if batch_idx % 100 == 0 or batch_idx == len(loader):
                lr_now = opt.param_groups[0]["lr"]
                print(
                    f"[ep {epoch}/{args.epochs}  batch {batch_idx:>5d}/{len(loader)}]"
                    f"  loss={losses['loss'].item():.4f}"
                    f"  nce={losses['loss_pred'].item():.4f}"
                    f"  pos_cos={losses['loss_reg'].item():.4f}"
                    f"  lr={lr_now:.2e}"
                )

    final_path = out_dir / "jepa_final.pt"
    torch.save(
        {
            "online_encoder_state":  model.online_encoder.state_dict(),
            "dynamics_state":        model.dynamics.state_dict(),
            "cfg":                   cfg,
        },
        final_path,
    )
    print(f"[jepa] saved {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
