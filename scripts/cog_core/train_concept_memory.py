"""Train ConceptMemory to replace fixed predicate_readout.

Uses the same data pipeline and (color, type) supervision signal as
train_predicate_readout.py so we can directly compare against v4.1.2's
53.6% held-out joint agreement baseline.

Loads pre-encoded latents from collect_rollouts.py output (no need to
re-load the JEPA — latents are already in the npz). Slots from the
companion .slots.pkl give per-frame (color, type) labels via the
primary-object heuristic.

Two prediction heads sit on top of ConceptMemory's slot_dim output:
- color head: slot_dim → NUM_COLORS=6 logits
- type  head: slot_dim → NUM_TYPES=4 logits

Both supervised with cross-entropy. The Hopfield slots learn to be
attentional prototypes that disentangle color and type via the
metastable retrieval regime.

Usage:
    python -m scripts.cog_core.train_concept_memory \\
        --rollouts runs/cog_core_phase1_factored/rollouts.npz \\
        --slots runs/cog_core_phase1_factored/rollouts.slots.pkl \\
        --run-name concept_memory_v1 \\
        --n-slots 1024 --slot-dim 64 \\
        --epochs 20 --batch-size 512 \\
        --use-sparse-opt \\
        --device cuda
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from prism.cog_core.concept_memory import ConceptMemory
from prism.perception.slots import (
    AGENT_POS,
    NUM_COLORS,
    NUM_TYPES,
    OBJECT_TYPES,
)
from prism.training.sparse_hopfield_update import SparseHopfieldOptimizer
from prism.utils.seed import set_global_seed


def _primary_object(slots_at_t: list[dict]) -> tuple[int, int] | None:
    """Pick the most-prominent visible object: prefer in-front, else nearest.

    Returns (color_id, type_idx) where type_idx is the index into OBJECT_TYPES,
    or None if no slot has a recognized type.

    Same heuristic as train_predicate_readout.py for fair comparison.
    """
    if not slots_at_t:
        return None
    type_to_idx = {t: i for i, t in enumerate(OBJECT_TYPES)}
    ax, ay = AGENT_POS

    in_col = [s for s in slots_at_t if int(s["x"]) == ax]
    if in_col:
        s = min(in_col, key=lambda s: int(s["y"]))
    else:
        s = min(
            slots_at_t,
            key=lambda s: abs(int(s["x"]) - ax) + abs(int(s["y"]) - ay),
        )

    t_id = int(s["type_id"])
    if t_id not in type_to_idx:
        return None
    return int(s["color_id"]), type_to_idx[t_id]


def build_dataset(
    npz_path: Path,
    slots_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (latents, color_labels, type_labels).

    Latents are pre-encoded (the JEPA's output), shape (N, latent_dim_flat).
    """
    d = np.load(npz_path)
    latents = d["latents"]                          # (N_eps, T_max, C, H, W) or (N, T, D)
    lengths = d["ep_lengths"]
    with open(slots_path, "rb") as f:
        slots = pickle.load(f)
    if len(slots) != latents.shape[0]:
        raise SystemExit(
            f"slot count ({len(slots)}) != episode count ({latents.shape[0]})"
        )

    Z: list[np.ndarray] = []
    Cs: list[int] = []
    Ts: list[int] = []
    for i in range(len(lengths)):
        L = int(lengths[i])
        if L < 1:
            continue
        ep_slots = slots[i]
        for t in range(L):
            if t >= len(ep_slots):
                continue
            picked = _primary_object(ep_slots[t])
            if picked is None:
                continue
            c, ti = picked
            Z.append(latents[i, t])
            Cs.append(c)
            Ts.append(ti)

    Z_arr = np.stack(Z).astype(np.float32)
    if Z_arr.ndim > 2:
        Z_arr = Z_arr.reshape(Z_arr.shape[0], -1)
    return (
        Z_arr,
        np.array(Cs, dtype=np.int64),
        np.array(Ts, dtype=np.int64),
    )


def parse_holdout_combos(s: str) -> set[tuple[int, int]]:
    """Parse "c,t c,t c,t" into a set of (color_id, type_idx) tuples."""
    out: set[tuple[int, int]] = set()
    if not s:
        return out
    for tok in s.split():
        c, t = tok.split(",")
        out.add((int(c), int(t)))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True)
    p.add_argument("--slots", required=True,
                   help="path to rollouts.slots.pkl produced by collect_rollouts")
    p.add_argument("--n-slots", type=int, default=1024)
    p.add_argument("--slot-dim", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--scaling", type=float, default=1.0)
    p.add_argument("--update-steps", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--run-name", required=True)
    p.add_argument(
        "--holdout-combos",
        default="",
        help="space-separated 'c,t' pairs to hold out for compositional eval",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--use-sparse-opt",
        action="store_true",
        help="use SparseHopfieldOptimizer for slot-localized updates",
    )
    args = p.parse_args()

    set_global_seed(42)
    device = torch.device(args.device)
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_concept_memory] loading rollouts: {args.rollouts}")
    print(f"[train_concept_memory] loading slots:    {args.slots}")
    Z_np, C_np, T_np = build_dataset(Path(args.rollouts), Path(args.slots))
    latent_dim = Z_np.shape[1]
    print(
        f"[train_concept_memory] N={Z_np.shape[0]} latent_dim={latent_dim} "
        f"color_classes={NUM_COLORS} type_classes={NUM_TYPES}"
    )

    holdout = parse_holdout_combos(args.holdout_combos)
    if holdout:
        is_holdout = np.array(
            [(int(c), int(t)) in holdout for c, t in zip(C_np, T_np)],
            dtype=bool,
        )
        Z_train, C_train, T_train = Z_np[~is_holdout], C_np[~is_holdout], T_np[~is_holdout]
        Z_test, C_test, T_test = Z_np[is_holdout], C_np[is_holdout], T_np[is_holdout]
        print(
            f"[train_concept_memory] held-out: {len(Z_test)} samples across "
            f"{len(holdout)} combos; in-dist: {len(Z_train)}"
        )
    else:
        # Random 90/10 split for in-distribution sanity check.
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(Z_np))
        cut = int(0.9 * len(idx))
        train_idx, test_idx = idx[:cut], idx[cut:]
        Z_train, C_train, T_train = Z_np[train_idx], C_np[train_idx], T_np[train_idx]
        Z_test, C_test, T_test = Z_np[test_idx], C_np[test_idx], T_np[test_idx]
        print(
            f"[train_concept_memory] no holdout; 90/10 random split "
            f"train={len(Z_train)} test={len(Z_test)}"
        )

    train_ds = TensorDataset(
        torch.from_numpy(Z_train),
        torch.from_numpy(C_train),
        torch.from_numpy(T_train),
    )
    test_ds = TensorDataset(
        torch.from_numpy(Z_test),
        torch.from_numpy(C_test),
        torch.from_numpy(T_test),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    memory = ConceptMemory(
        latent_dim=latent_dim,
        n_slots=args.n_slots,
        slot_dim=args.slot_dim,
        n_heads=args.n_heads,
        scaling=args.scaling,
        update_steps=args.update_steps,
    ).to(device)
    color_head = nn.Linear(args.slot_dim, NUM_COLORS).to(device)
    type_head = nn.Linear(args.slot_dim, NUM_TYPES).to(device)

    params = list(memory.parameters()) + list(color_head.parameters()) + list(type_head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    sparse_opt = SparseHopfieldOptimizer(memory, opt) if args.use_sparse_opt else None

    @torch.no_grad()
    def evaluate(loader) -> dict:
        memory.eval()
        color_head.eval()
        type_head.eval()
        c_correct = t_correct = joint_correct = total = 0
        for z, c, t in loader:
            z, c, t = z.to(device), c.to(device), t.to(device)
            concept = memory(z)
            cl = color_head(concept).argmax(-1)
            tl = type_head(concept).argmax(-1)
            c_correct += int((cl == c).sum())
            t_correct += int((tl == t).sum())
            joint_correct += int(((cl == c) & (tl == t)).sum())
            total += c.size(0)
        return {
            "color_acc": c_correct / max(1, total),
            "type_acc": t_correct / max(1, total),
            "joint_acc": joint_correct / max(1, total),
            "n": total,
        }

    best_joint = 0.0
    for epoch in range(args.epochs):
        memory.train(); color_head.train(); type_head.train()
        running = 0.0
        n_batches = 0
        for z, c, t in train_loader:
            z, c, t = z.to(device), c.to(device), t.to(device)
            concept, attn = memory(z, return_attention=True)
            c_logits = color_head(concept)
            t_logits = type_head(concept)
            loss_c = F.cross_entropy(c_logits, c)
            loss_t = F.cross_entropy(t_logits, t)
            loss = loss_c + loss_t

            if sparse_opt is not None:
                sparse_opt.zero_grad()
                loss.backward()
                sparse_opt.record_attention(attn)
                sparse_opt.step()
            else:
                opt.zero_grad()
                loss.backward()
                opt.step()

            running += float(loss)
            n_batches += 1

        avg = running / max(1, n_batches)
        train_eval = evaluate(train_loader)
        test_eval = evaluate(test_loader)
        print(
            f"[epoch {epoch+1:2d}/{args.epochs}] "
            f"loss={avg:.4f} "
            f"train: color={train_eval['color_acc']*100:5.1f}% "
            f"type={train_eval['type_acc']*100:5.1f}% "
            f"joint={train_eval['joint_acc']*100:5.1f}%  | "
            f"test: color={test_eval['color_acc']*100:5.1f}% "
            f"type={test_eval['type_acc']*100:5.1f}% "
            f"joint={test_eval['joint_acc']*100:5.1f}% (n={test_eval['n']})"
        )

        if test_eval["joint_acc"] > best_joint:
            best_joint = test_eval["joint_acc"]
            memory.save(str(run_dir / "concept_memory_best.pt"))
            torch.save(
                {
                    "color_head": color_head.state_dict(),
                    "type_head": type_head.state_dict(),
                    "epoch": epoch + 1,
                    "test_joint": best_joint,
                },
                str(run_dir / "heads_best.pt"),
            )

    memory.save(str(run_dir / "concept_memory_final.pt"))
    torch.save(
        {
            "color_head": color_head.state_dict(),
            "type_head": type_head.state_dict(),
            "best_test_joint": best_joint,
        },
        str(run_dir / "heads_final.pt"),
    )
    print(
        f"\n[done] best test joint accuracy: {best_joint*100:.1f}% "
        f"(v4.1.2 PRISM baseline: 53.6%)"
    )
    print(f"[done] saved to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
