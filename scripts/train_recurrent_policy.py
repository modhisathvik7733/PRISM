"""Train the recurrent policy via behavior cloning on memory-mode trajectories.

Phase 3 step 2 training loop. The frozen JEPA encoder produces z_t for every
(B, T) observation in a trajectory. The recurrent policy then runs across
the full sequence, emitting per-step action logits that are matched against
the memory-mode agent's actions via cross-entropy.

This is supervised learning with sequence batches — fast, deterministic,
no env interaction during training. The trained policy will replace the
hand-coded curriculum + frontier with learned recurrent state.

Usage:
    python -m scripts.train_recurrent_policy \
        --jepa-checkpoint runs/<...>/jepa_final.pt \
        --bc-data data/bc_v0.9.npz \
        --steps 20000 --batch-size 64 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.models.recurrent_policy import RecurrentPolicy
from prism.utils.seed import set_global_seed


def latent_dim_for_cfg(cfg: JepaConfig) -> int:
    """Flattened latent dim for either flat or spatial JEPAs."""
    enc = getattr(cfg, "encoder_type", "flat")
    if enc == "categorical_spatial":
        C = getattr(cfg, "spatial_channels", 64)
        return C * cfg.obs_h * cfg.obs_w
    return cfg.embed_dim


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--bc-data", required=True, help="path to .npz from collect_bc_data")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-proj-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # --- load JEPA (frozen) ---
    ckpt = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
    cfg: JepaConfig = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    n_actions = cfg.n_actions
    latent_dim = latent_dim_for_cfg(cfg)
    print(f"[bc-train] frozen JEPA: encoder={cfg.encoder_type} "
          f"latent_dim={latent_dim} n_actions={n_actions}")

    # --- load BC data ---
    data = np.load(args.bc_data)
    obs_seqs = data["obs_seqs"]            # (N, T, 3, 7, 7)
    action_seqs = data["action_seqs"]      # (N, T)
    missions = data["mission_target"]       # (N, 24)
    lengths = data["ep_lengths"]            # (N,)
    N, T_max = action_seqs.shape
    mission_dim = missions.shape[1]
    print(f"[bc-train] N={N} T_max={T_max} mission_dim={mission_dim}")
    print(f"[bc-train] mean ep length = {lengths.mean():.1f}, "
          f"min = {lengths.min()}, max = {lengths.max()}")

    obs_seqs_t = torch.from_numpy(obs_seqs).to(device)
    action_seqs_t = torch.from_numpy(action_seqs).to(device)
    missions_t = torch.from_numpy(missions).to(device)
    lengths_t = torch.from_numpy(lengths).to(device)

    # --- build the recurrent policy ---
    policy = RecurrentPolicy(
        latent_in_dim=latent_dim,
        n_actions=n_actions,
        mission_dim=mission_dim,
        hidden_dim=args.hidden_dim,
        latent_proj_dim=args.latent_proj_dim,
    ).to(device)
    print(f"[bc-train] policy params: {sum(p.numel() for p in policy.parameters()):,}")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)

    # --- run-name and output dir ---
    run_name = args.run_name or f"bc_recurrent_seed{args.seed}"
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bc-train] writing to {out_dir}")

    # --- train ---
    losses = []
    accs = []
    for step in range(args.steps):
        idx = rng.integers(0, N, size=args.batch_size)
        idx_t = torch.from_numpy(idx).to(device)
        obs_b = obs_seqs_t[idx_t]              # (B, T, 3, 7, 7)
        act_b = action_seqs_t[idx_t]           # (B, T)
        mis_b = missions_t[idx_t]              # (B, 24)
        len_b = lengths_t[idx_t]               # (B,)
        B, T = act_b.shape

        # Encode all frames. Run encoder under no_grad — frozen.
        with torch.no_grad():
            obs_flat = obs_b.view(B * T, *obs_b.shape[2:])
            z_flat = jepa.encode(obs_flat)
            # Reshape back to (B, T, ...)
            z_b = z_flat.view(B, T, *z_flat.shape[1:])

        logits = policy(z_b, act_b, mis_b)     # (B, T, n_actions)
        # Mask: only count steps t < length[b].
        time_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        valid = (time_idx < len_b.unsqueeze(1)).float()              # (B, T)

        loss_per = F.cross_entropy(
            logits.reshape(B * T, n_actions),
            act_b.reshape(B * T),
            reduction="none",
        ).view(B, T)
        loss = (loss_per * valid).sum() / valid.sum().clamp(min=1.0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = ((preds == act_b).float() * valid).sum()
            acc = float((correct / valid.sum().clamp(min=1.0)).item())

        losses.append(float(loss.item()))
        accs.append(acc)

        if step % 200 == 0:
            mean_l = float(np.mean(losses[-200:]))
            mean_a = float(np.mean(accs[-200:]))
            print(f"[step {step:6d}] loss={loss.item():.4f}  "
                  f"acc={acc:.3f}  mean200(loss={mean_l:.4f} acc={mean_a:.3f})")

    final_path = out_dir / "policy_final.pt"
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "latent_in_dim": latent_dim,
        "n_actions": n_actions,
        "mission_dim": mission_dim,
        "hidden_dim": args.hidden_dim,
        "latent_proj_dim": args.latent_proj_dim,
        "jepa_checkpoint": str(args.jepa_checkpoint),
    }, final_path)
    print(f"[done] saved {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
