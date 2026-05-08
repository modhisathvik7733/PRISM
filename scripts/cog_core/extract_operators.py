"""Extract operators from rollouts via K-means on latent-deltas.

Reads rollouts.npz, computes (z_t, a_t, z_t+1) tuples, fits OperatorBank,
prints per-cluster purity + dominant action, and saves the centroids.

Usage:
    python -m scripts.cog_core.extract_operators \
        --rollouts runs/cog_core_phase1/rollouts.npz \
        --n-clusters 8 \
        --output runs/cog_core_phase1/operators.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from prism.cog_core.operator_bank import OperatorBank


def collate_transitions(npz_path: Path
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns flat (latents_t, latents_t+1, actions, env_ids_per_transition)."""
    d = np.load(npz_path)
    latents = d["latents"]            # (N, T, ...)
    actions = d["actions"]            # (N, T)
    lengths = d["ep_lengths"]         # (N,)
    env_ids = d["env_ids"]            # (N,)

    L_t: list[np.ndarray] = []
    L_tp1: list[np.ndarray] = []
    A: list[int] = []
    E: list[str] = []
    for i in range(len(lengths)):
        L = int(lengths[i])
        # Need at least 2 valid steps to form a transition.
        if L < 2:
            continue
        for t in range(L - 1):
            L_t.append(latents[i, t])
            L_tp1.append(latents[i, t + 1])
            A.append(int(actions[i, t]))
            E.append(str(env_ids[i]))
    return np.stack(L_t), np.stack(L_tp1), np.array(A, dtype=np.int64), np.array(E)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--max-transitions", type=int, default=20000,
                        help="cap on transitions for K-means (memory)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-env", action="store_true",
                        help="Also fit one bank per env, report cross-env stability")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"[ops] loading {args.rollouts}")
    L_t, L_tp1, A, E = collate_transitions(Path(args.rollouts))
    print(f"[ops] {len(L_t):,} transitions across "
          f"{len(set(E.tolist()))} envs")

    if len(L_t) > args.max_transitions:
        idx = rng.choice(len(L_t), size=args.max_transitions, replace=False)
        L_t, L_tp1, A, E = L_t[idx], L_tp1[idx], A[idx], E[idx]
        print(f"[ops] subsampled to {len(L_t):,} for K-means")

    print(f"[ops] fitting K-means with K={args.n_clusters}…")
    bank = OperatorBank(n_ops=args.n_clusters)
    stats = bank.fit(L_t, L_tp1, A, seed=args.seed)

    print("\n=== operator clusters (sorted by size) ===")
    print(f"{'op_id':>5}  {'n':>6}  {'within_var':>10}  "
          f"{'dominant':>8}  {'purity':>7}  action_dist")
    for s in sorted(stats, key=lambda s: -s.n_members):
        ad = ", ".join(f"a{a}={p*100:.0f}%"
                       for a, p in sorted(s.action_distribution.items(),
                                          key=lambda kv: -kv[1])[:3])
        interp = "✓" if s.is_interpretable() else "✗"
        print(f"{s.cluster_id:>5d}  {s.n_members:>6d}  "
              f"{s.within_var:>10.4f}  {s.dominant_action:>8d}  "
              f"{s.purity*100:>6.1f}% {interp}  {ad}")

    n_interp = sum(1 for s in stats if s.is_interpretable())
    print(f"\n[summary] {n_interp}/{args.n_clusters} clusters are "
          f"interpretable (dominant action ≥80%)")
    print(f"[summary] Phase 1 emergence target: ≥4 interpretable clusters "
          f"({'PASS' if n_interp >= 4 else 'FAIL'})")

    bank.save(args.output)
    print(f"\n[saved] {args.output}")

    # Per-env stability check (Phase 1 emergence criterion).
    if args.per_env:
        env_set = sorted(set(E.tolist()))
        if len(env_set) < 2:
            print("[stability] only one env present, skipping cross-env check")
            return 0
        print("\n=== per-env operator stability ===")
        per_env_banks: dict[str, OperatorBank] = {}
        for env_id in env_set:
            mask = E == env_id
            if int(mask.sum()) < args.n_clusters * 10:
                print(f"  {env_id}: too few transitions, skipping")
                continue
            sub_bank = OperatorBank(n_ops=args.n_clusters)
            sub_bank.fit(L_t[mask], L_tp1[mask], A[mask], seed=args.seed)
            per_env_banks[env_id] = sub_bank
            print(f"  {env_id}: fit on {int(mask.sum())} transitions")

        if len(per_env_banks) >= 2:
            envs = list(per_env_banks.keys())
            base = per_env_banks[envs[0]]
            for env_id in envs[1:]:
                stab = base.cross_env_stability(per_env_banks[env_id])
                print(f"\n  {envs[0]} vs {env_id}:")
                print(f"    mean best cosine = {stab['mean_best_cosine']:.3f}")
                print(f"    min  best cosine = {stab['min_best_cosine']:.3f}")
                print(f"    n stable (cos≥0.8) = {stab['n_stable_above_threshold']}/{args.n_clusters}")
                ok = stab['mean_best_cosine'] >= 0.8
                print(f"    Phase 1 emergence target ≥0.8 mean cosine: "
                      f"{'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
