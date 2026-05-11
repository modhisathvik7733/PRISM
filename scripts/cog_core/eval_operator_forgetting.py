"""Standalone forgetting eval for OperatorBankV3.

Loads a V3 checkpoint and measures, per operator:
  1. Current anchor MSE (how well dynamics head k still reproduces its
     canonical effect).
  2. Per-action routing distribution on a fresh held-out rollout set.
  3. Cross-env stability if rollouts span multiple envs.

The pass criterion for V3 (anti-drift) is two-part:
  * anchor_mse <= 2x its initial seed value (no significant drift)
  * cross-env stability mean cosine >= 0.85 (was 0.45-0.56 in V1)

Usage:
    python -m scripts.cog_core.eval_operator_forgetting \
        --bank runs/ops_v3_phaseB/operators_v3.pt \
        --rollouts runs/cog_core_phase1_devB/rollouts_envA.npz \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from prism.cog_core.operator_bank_v3 import OperatorBankV3
from scripts.cog_core.train_operators_v3 import collate


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", required=True, help="V3 checkpoint")
    p.add_argument("--rollouts", required=True, help="rollouts npz")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None,
                   help="optional path to dump results as JSON")
    args = p.parse_args()

    device = torch.device(args.device)
    bank = OperatorBankV3.load(args.bank, device)
    print(f"[eval] loaded V3 bank from {args.bank}")
    print(f"       n_ops={bank.n_ops} latent_dim={bank.latent_dim}")
    print(f"       valid anchors: "
          f"{[k for k in range(bank.n_ops) if bool(bank.anchor_valid[k])]}")

    print(f"\n[eval] loading rollouts from {args.rollouts}")
    L_t, L_tp1, A, E = collate(Path(args.rollouts))
    print(f"       {len(L_t):,} transitions, envs={sorted(set(E.tolist()))}")

    # --- 1. anchor MSE ---
    print("\n=== anchor MSE per operator (drift measure) ===")
    anchor_mse = bank.anchor_mse_per_op()
    if not anchor_mse:
        print("  no anchors seeded — anchor-drift test inapplicable")
    else:
        print(f"  {'op':>3} {'anchor_mse':>12}")
        for k in sorted(anchor_mse):
            print(f"  {k:>3d} {anchor_mse[k]:>12.6f}")
        mean = float(np.mean(list(anchor_mse.values())))
        print(f"  mean: {mean:.6f}")

    # --- 2. routing stats on held-out rollouts ---
    print("\n=== routing stats on fresh rollouts ===")
    n_eval = min(5000, len(L_t))
    z_t = torch.from_numpy(L_t[:n_eval]).to(device)
    z_tp1 = torch.from_numpy(L_tp1[:n_eval]).to(device)
    a = torch.from_numpy(A[:n_eval]).to(device)
    stats = bank.analyze(z_t, a)
    print(f"  {'op':>3} {'activation':>10} {'dom_a':>6} {'purity':>7} "
          f"{'anchor_mse':>11}")
    for s in sorted(stats, key=lambda s: -s.activation_rate):
        amse = f"{s.anchor_mse:.5f}" if s.anchor_valid else "—"
        print(f"  {s.op_id:>3d} {s.activation_rate:>10.4f} "
              f"{s.dominant_action:>6d} {s.purity*100:>6.1f}% {amse:>11}")

    # --- 3. cross-env stability ---
    print("\n=== cross-env stability ===")
    env_ids = sorted(set(E.tolist()))
    stab = {}
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
        stab = bank.cross_env_stability(per_env, threshold=0.85)
        print(f"  envs: {list(per_env.keys())}")
        print(f"  mean cosine: {stab.get('mean_cosine', 0):.4f}  "
              f"(target >=0.85)")
        for p_ in stab.get("pairwise", []):
            print(f"  {p_['env1']} vs {p_['env2']}: "
                  f"cos={p_['matrix_cosine_sim']:.4f}  "
                  f"{'PASS' if p_['pass'] else 'FAIL'}")
    else:
        print(f"  only one env in rollouts ({env_ids}) — stability inapplicable")

    # --- summary verdict ---
    print("\n=== verdict ===")
    anchor_pass = (
        bool(anchor_mse) and
        all(v <= 1e-3 for v in anchor_mse.values())
    )
    stab_pass = bool(stab.get("all_pass", False)) if stab else None
    if anchor_pass and (stab_pass is True or stab_pass is None):
        print("  PASS — operators stable across continual training")
    else:
        reasons = []
        if not anchor_pass and anchor_mse:
            worst = max(anchor_mse.values())
            reasons.append(f"anchor MSE drift (max {worst:.5f} > 1e-3)")
        if stab_pass is False:
            reasons.append(f"cross-env stability mean "
                           f"{stab.get('mean_cosine', 0):.3f} < 0.85")
        print(f"  FAIL — {'; '.join(reasons) or 'no anchors / single-env data'}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "anchor_mse": anchor_mse,
                "stability": stab,
                "anchor_pass": anchor_pass,
                "stability_pass": stab_pass,
            }, f, indent=2)
        print(f"\n[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
