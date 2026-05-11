"""Train OperatorBankV3 with anti-drift machinery.

Two training modes:

* `--mode fresh`  — train from scratch on a single rollouts npz, like V2.
                    Seeds anchors at `--anchor-seed-step`. After seeding,
                    anchor + EMA losses become active.

* `--mode continual` — load a V3 checkpoint and continue training on a
                       second rollouts npz (e.g. a different env). The
                       anchor + EMA losses prevent forgetting of behavior
                       learned in the first phase.

The script always reports anchor MSE per operator at the end so you can
directly read off drift.

Usage (Phase 1: fresh on env A):
    python -m scripts.cog_core.train_operators_v3 \
        --mode fresh \
        --rollouts runs/cog_core_phase1_devB/rollouts_envA.npz \
        --steps 32000 --anchor-seed-step 16000 \
        --run-name ops_v3_phaseA --device cuda

Usage (Phase 2: continual on env B, no forgetting expected):
    python -m scripts.cog_core.train_operators_v3 \
        --mode continual \
        --load runs/ops_v3_phaseA/operators_v3.pt \
        --rollouts runs/cog_core_phase1_devB/rollouts_envB.npz \
        --replay-rollouts runs/cog_core_phase1_devB/rollouts_envA.npz \
        --steps 32000 \
        --run-name ops_v3_phaseB --device cuda
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from prism.cog_core.operator_bank_v3 import OperatorBankV3
from prism.utils.seed import set_global_seed


def collate(npz_path: Path
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(npz_path)
    latents = d["latents"]
    actions = d["actions"]
    lengths = d["ep_lengths"]
    env_ids = d["env_ids"]
    L_t, L_tp1, A, E = [], [], [], []
    for i in range(len(lengths)):
        L = int(lengths[i])
        if L < 2:
            continue
        for t in range(L - 1):
            L_t.append(latents[i, t])
            L_tp1.append(latents[i, t + 1])
            A.append(int(actions[i, t]))
            E.append(str(env_ids[i]))
    L_t = np.stack(L_t).astype(np.float32)
    L_tp1 = np.stack(L_tp1).astype(np.float32)
    if L_t.ndim > 2:
        L_t = L_t.reshape(L_t.shape[0], -1)
        L_tp1 = L_tp1.reshape(L_tp1.shape[0], -1)
    return L_t, L_tp1, np.array(A, dtype=np.int64), np.array(E)


def sample_batch(
    L_t: np.ndarray, L_tp1: np.ndarray, A: np.ndarray,
    rng: np.random.Generator, bs: int,
    replay: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    replay_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if replay is None:
        idx = rng.integers(0, len(L_t), size=bs)
        return L_t[idx], L_tp1[idx], A[idx]
    n_old = int(bs * replay_frac)
    n_new = bs - n_old
    Lr, Lr1, Ar = replay
    i_new = rng.integers(0, len(L_t), size=n_new)
    i_old = rng.integers(0, len(Lr), size=n_old)
    return (
        np.concatenate([L_t[i_new], Lr[i_old]], axis=0),
        np.concatenate([L_tp1[i_new], Lr1[i_old]], axis=0),
        np.concatenate([A[i_new], Ar[i_old]], axis=0),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fresh", "continual"], required=True)
    parser.add_argument("--rollouts", required=True,
                        help="primary rollouts npz (current env)")
    parser.add_argument("--replay-rollouts", default=None,
                        help="optional second npz for replay (continual mode)")
    parser.add_argument("--load", default=None,
                        help="path to V3 checkpoint (required if mode=continual)")
    parser.add_argument("--n-ops", type=int, default=8)
    parser.add_argument("--n-actions", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--action-emb-dim", type=int, default=16)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--lambda-ema", type=float, default=0.1)
    parser.add_argument("--lambda-anchor", type=float, default=1.0)
    parser.add_argument("--ema-tau", type=float, default=0.995)
    parser.add_argument("--anchor-size", type=int, default=64)
    parser.add_argument("--anchor-seed-step", type=int, default=16000,
                        help="step at which to seed anchors (fresh mode)")
    parser.add_argument("--steps", type=int, default=32000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-transitions", type=int, default=100_000)
    parser.add_argument("--replay-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.mode == "continual" and args.load is None:
        raise SystemExit("--load is required when --mode continual")

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # ----- data -----
    print(f"[v3] loading {args.rollouts}")
    L_t, L_tp1, A, E = collate(Path(args.rollouts))
    rng = np.random.default_rng(args.seed)
    if len(L_t) > args.max_transitions:
        idx = rng.choice(len(L_t), size=args.max_transitions, replace=False)
        L_t, L_tp1, A, E = L_t[idx], L_tp1[idx], A[idx], E[idx]
    latent_dim = int(L_t.shape[-1])
    print(f"[v3] {len(L_t):,} transitions, latent_dim={latent_dim}, "
          f"envs={sorted(set(E.tolist()))}")

    replay = None
    if args.replay_rollouts is not None:
        Lr, Lr1, Ar, Er = collate(Path(args.replay_rollouts))
        if len(Lr) > args.max_transitions:
            idx = rng.choice(len(Lr), size=args.max_transitions, replace=False)
            Lr, Lr1, Ar = Lr[idx], Lr1[idx], Ar[idx]
        replay = (Lr, Lr1, Ar)
        print(f"[v3] replay buffer: {len(Lr):,} transitions, "
              f"frac={args.replay_frac}")

    # ----- model -----
    if args.mode == "fresh":
        bank = OperatorBankV3(
            latent_dim=latent_dim,
            n_actions=args.n_actions,
            n_ops=args.n_ops,
            hidden=args.hidden,
            action_emb_dim=args.action_emb_dim,
            entropy_coef=args.entropy_coef,
            ema_tau=args.ema_tau,
            lambda_ema=args.lambda_ema,
            lambda_anchor=args.lambda_anchor,
            anchor_size=args.anchor_size,
        ).to(device)
        print("[v3] fresh bank initialized")
    else:
        bank = OperatorBankV3.load(args.load, device,
                                    hidden=args.hidden,
                                    action_emb_dim=args.action_emb_dim)
        # Override loss weights / lr in case user wants to retune for cont.
        bank.entropy_coef = args.entropy_coef
        bank.lambda_ema = args.lambda_ema
        bank.lambda_anchor = args.lambda_anchor
        bank.ema_tau = args.ema_tau
        print(f"[v3] continued from {args.load}")
        print(f"     anchors valid for ops: "
              f"{[i for i in range(bank.n_ops) if bool(bank.anchor_valid[i])]}")

    print(f"[v3] params: {sum(p.numel() for p in bank.parameters()):,}")
    opt = torch.optim.AdamW(bank.parameters(), lr=args.lr)

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    print(f"[v3] writing to {out_dir}")

    # ----- baseline anchor MSE (continual mode only) -----
    pre_anchor_mse: dict[int, float] = {}
    if args.mode == "continual":
        pre_anchor_mse = bank.anchor_mse_per_op()
        print("\n=== pre-continual anchor MSE per op (baseline) ===")
        for k in sorted(pre_anchor_mse):
            print(f"  op {k}: {pre_anchor_mse[k]:.6f}")

    # ----- training -----
    loss_window: deque[float] = deque(maxlen=100)
    for step in range(args.steps):
        b_t, b_tp1, b_a = sample_batch(
            L_t, L_tp1, A, rng, args.batch_size,
            replay=replay, replay_frac=args.replay_frac,
        )
        z_t = torch.from_numpy(b_t).to(device)
        z_tp1 = torch.from_numpy(b_tp1).to(device)
        a = torch.from_numpy(b_a).to(device)

        use_anchor = bool(bank.anchor_valid.any().item())
        use_ema = bank._ema_init

        out = bank.loss(z_t, a, z_tp1, use_ema=use_ema, use_anchor=use_anchor)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
        opt.step()
        bank.ema_step()

        # Seed anchors at the configured step (fresh mode only).
        if args.mode == "fresh" and step == args.anchor_seed_step:
            print(f"\n[step {step}] seeding anchors...")
            stored = bank.seed_anchors(z_t, a, z_tp1)
            print(f"  stored per op: {stored}")
            print()

        loss_window.append(float(out["loss"].item()))
        if step % 200 == 0:
            mean_loss = float(np.mean(loss_window)) if loss_window else float("nan")
            writer.add_scalar("train/loss", float(out["loss"].item()), step)
            writer.add_scalar("train/mse", float(out["mse"].item()), step)
            writer.add_scalar("train/ema_loss", float(out["ema_loss"].item()), step)
            writer.add_scalar("train/anchor_loss",
                              float(out["anchor_loss"].item()), step)
            writer.add_scalar("train/entropy", float(out["entropy"].item()), step)
            print(f"[step {step:6d}/{args.steps}] "
                  f"loss={float(out['loss'].item()):.4f} "
                  f"mse={float(out['mse'].item()):.4f} "
                  f"ema={float(out['ema_loss'].item()):.4f} "
                  f"anchor={float(out['anchor_loss'].item()):.4f} "
                  f"ent={float(out['entropy'].item()):.3f} "
                  f"mean100={mean_loss:.4f}")

    # ----- final report -----
    print("\n=== final per-op stats (eval set) ===")
    z_t_eval = torch.from_numpy(L_t[:5000]).to(device)
    a_eval = torch.from_numpy(A[:5000]).to(device)
    stats = bank.analyze(z_t_eval, a_eval)
    print(f"{'op':>3} {'activation':>10} {'dom_a':>6} {'purity':>7} "
          f"{'anchor_mse':>11} {'anchor?':>7}")
    for s in sorted(stats, key=lambda s: -s.activation_rate):
        amse = (f"{s.anchor_mse:.5f}" if s.anchor_valid else "—")
        print(f"{s.op_id:>3d} {s.activation_rate:>10.4f} "
              f"{s.dominant_action:>6d} {s.purity*100:>6.1f}% "
              f"{amse:>11} {str(s.anchor_valid):>7}")

    if args.mode == "continual" and pre_anchor_mse:
        print("\n=== anchor MSE drift (forgetting measure) ===")
        print("  pre = anchor MSE BEFORE continual training (env A behavior)")
        print("  post = anchor MSE AFTER continual training (should match pre)")
        post = bank.anchor_mse_per_op()
        print(f"  {'op':>3} {'pre':>10} {'post':>10} {'delta':>10}")
        for k in sorted(pre_anchor_mse):
            pre = pre_anchor_mse[k]
            po = post.get(k, float("nan"))
            print(f"  {k:>3d} {pre:>10.6f} {po:>10.6f} {po - pre:>+10.6f}")
        delta_mean = float(np.mean(
            [post[k] - pre_anchor_mse[k] for k in pre_anchor_mse
             if k in post]
        ))
        print(f"  mean drift: {delta_mean:+.6f} "
              f"(<= ~5e-4 = pass, > 1e-3 = clear forgetting)")

    print("\n=== cross-env operator stability ===")
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
        stab = bank.cross_env_stability(per_env, threshold=0.8)
        print(f"  envs: {list(per_env.keys())}")
        print(f"  mean cosine: {stab.get('mean_cosine', 0):.4f}  "
              f"(>=0.8 = pass)")
        for p in stab.get("pairwise", []):
            print(f"  {p['env1']} vs {p['env2']}: "
                  f"cos={p['matrix_cosine_sim']:.4f}  "
                  f"{'PASS' if p['pass'] else 'FAIL'}")

    bank.save(str(out_dir / "operators_v3.pt"))
    print(f"\n[saved] {out_dir / 'operators_v3.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
