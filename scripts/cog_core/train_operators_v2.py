"""Train OperatorBankV2 (gradient-based MoE) on rollouts collected
across all envs. Outputs cross-env stability metrics that V1's
K-means failed.

Usage:
    python -m scripts.cog_core.train_operators_v2 \
        --rollouts runs/cog_core_phase1_devB/rollouts.npz \
        --n-ops 8 --steps 5000 --batch-size 256 \
        --device cuda \
        --run-name cog_phase1_devB_ops_v2
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from prism.cog_core.operator_bank_v2 import OperatorBankV2
from prism.utils.seed import set_global_seed


def collate_transitions(npz_path: Path
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (latents_t, latents_tp1, actions, env_ids_per_transition)."""
    d = np.load(npz_path)
    latents = d["latents"]
    actions = d["actions"]
    lengths = d["ep_lengths"]
    env_ids = d["env_ids"]

    L_t: list[np.ndarray] = []
    L_tp1: list[np.ndarray] = []
    A: list[int] = []
    E: list[str] = []
    for i in range(len(lengths)):
        L = int(lengths[i])
        if L < 2:
            continue
        for t in range(L - 1):
            L_t.append(latents[i, t])
            L_tp1.append(latents[i, t + 1])
            A.append(int(actions[i, t]))
            E.append(str(env_ids[i]))
    return (np.stack(L_t).astype(np.float32),
            np.stack(L_tp1).astype(np.float32),
            np.array(A, dtype=np.int64),
            np.array(E))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--n-ops", type=int, default=8)
    parser.add_argument("--n-actions", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--action-emb-dim", type=int, default=16)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-transitions", type=int, default=100_000,
                        help="cap to keep memory bounded")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    print(f"[ops-v2] loading {args.rollouts}")
    L_t, L_tp1, A, E = collate_transitions(Path(args.rollouts))
    print(f"[ops-v2] {len(L_t):,} transitions across "
          f"{len(set(E.tolist()))} envs")

    rng = np.random.default_rng(args.seed)
    if len(L_t) > args.max_transitions:
        idx = rng.choice(len(L_t), size=args.max_transitions, replace=False)
        L_t, L_tp1, A, E = L_t[idx], L_tp1[idx], A[idx], E[idx]
        print(f"[ops-v2] subsampled to {len(L_t):,}")

    # Determine latent dim (flat after reshape)
    sample = L_t[0]
    latent_dim = int(np.prod(sample.shape))
    if L_t.ndim > 2:
        L_t = L_t.reshape(L_t.shape[0], -1)
        L_tp1 = L_tp1.reshape(L_tp1.shape[0], -1)
    print(f"[ops-v2] latent dim: {latent_dim}")

    bank = OperatorBankV2(
        latent_dim=latent_dim,
        n_actions=args.n_actions,
        n_ops=args.n_ops,
        hidden=args.hidden,
        action_emb_dim=args.action_emb_dim,
        entropy_coef=args.entropy_coef,
    ).to(device)
    print(f"[ops-v2] bank params: "
          f"{sum(p.numel() for p in bank.parameters()):,}")

    opt = torch.optim.AdamW(bank.parameters(), lr=args.lr)

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    print(f"[ops-v2] writing to {out_dir}")

    # Train
    loss_window: deque[float] = deque(maxlen=100)
    for step in range(args.steps):
        idx = rng.integers(0, len(L_t), size=args.batch_size)
        z_t = torch.from_numpy(L_t[idx]).to(device)
        z_tp1 = torch.from_numpy(L_tp1[idx]).to(device)
        a = torch.from_numpy(A[idx]).to(device)

        out = bank.loss(z_t, a, z_tp1)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
        opt.step()

        loss_window.append(float(loss.item()))
        if step % 100 == 0:
            mean_loss = float(np.mean(loss_window)) if loss_window else float("nan")
            writer.add_scalar("train/loss", float(loss.item()), step)
            writer.add_scalar("train/mse", float(out["mse"].item()), step)
            writer.add_scalar("train/entropy", float(out["entropy"].item()), step)
            print(f"[step {step:5d}/{args.steps}] "
                  f"loss={float(loss.item()):.4f} mse={float(out['mse'].item()):.4f} "
                  f"entropy={float(out['entropy'].item()):.4f} "
                  f"mean100={mean_loss:.4f}")

    # Final analysis
    print("\n=== per-operator stats (on full eval set, hard-assignment) ===")
    z_t_full = torch.from_numpy(L_t[:5000]).to(device)
    z_tp1_full = torch.from_numpy(L_tp1[:5000]).to(device)
    a_full = torch.from_numpy(A[:5000]).to(device)
    stats = bank.analyze(z_t_full, a_full)

    print(f"{'op':>3}  {'activation':>10}  {'dominant':>8}  {'purity':>7}  action_dist")
    for s in sorted(stats, key=lambda s: -s.activation_rate):
        ad = ", ".join(f"a{a}={p*100:.0f}%"
                       for a, p in sorted(s.action_distribution.items(),
                                          key=lambda kv: -kv[1])[:4])
        print(f"{s.op_id:>3d}  {s.activation_rate:>9.4f}  "
              f"{s.dominant_action:>8d}  {s.purity*100:>6.1f}%  {ad}")

    # Cross-env stability (the test V1 failed)
    print("\n=== cross-env operator stability (the test V1 failed) ===")
    env_ids = sorted(set(E.tolist()))
    if len(env_ids) >= 2:
        per_env = {}
        for env_id in env_ids:
            mask = E == env_id
            if int(mask.sum()) < 100:
                continue
            per_env[env_id] = (
                torch.from_numpy(L_t[mask][:5000]).to(device),
                torch.from_numpy(A[mask][:5000]).to(device),
            )
        stability = bank.cross_env_stability(per_env, threshold=0.8)
        print(f"  Envs included: {list(per_env.keys())}")
        print(f"  Mean cross-env matrix cosine sim: "
              f"{stability.get('mean_cosine', 0):.4f}")
        for p in stability.get("pairwise", []):
            print(f"  {p['env1']} vs {p['env2']}: "
                  f"matrix_cos={p['matrix_cosine_sim']:.4f}  "
                  f"{'PASS' if p['pass'] else 'FAIL'} (target ≥0.8)")
        all_pass = stability.get("all_pass", False)
        print(f"\n  V1 K-means baseline (for comparison):")
        print(f"    GoTo vs GoToLocal: 0.560  FAIL")
        print(f"    GoTo vs GoToObj:   0.456  FAIL")
        print(f"  V2 mean: {stability.get('mean_cosine', 0):.4f}  "
              f"{'PASS' if all_pass else 'FAIL'}")

    bank.save(str(out_dir / "operators_v2.pt"))
    print(f"\n[saved] {out_dir / 'operators_v2.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
