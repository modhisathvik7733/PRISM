"""Train ConceptMemory to replace fixed predicate_readout.

Loads existing JEPA + collected rollouts, then trains the Hopfield-based
ConceptMemory to predict the same predicate targets — but now via attention
over 1024 growable slots instead of a fixed 96-dim linear readout.

Validation: at the end, the held-out compositional joint agreement should
match or beat v4.1.2's 53.6% baseline. If it does, Phase 1 is validated.

Usage:
    python -m scripts.cog_core.train_concept_memory \\
        --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \\
        --rollouts runs/cog_core_phase1_factored/rollouts.npz \\
        --run-name concept_memory_v1 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from prism.cog_core.concept_memory import ConceptMemory
from prism.models.jepa import JepaWorldModel, upgrade_config
from prism.training.sparse_hopfield_update import SparseHopfieldOptimizer
from prism.utils.seed import set_global_seed


def load_jepa(path: str, device: torch.device) -> JepaWorldModel:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = upgrade_config(ck["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ck["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    return jepa


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jepa-checkpoint", required=True)
    p.add_argument("--rollouts", required=True,
                   help="npz with 'images', 'predicates' fields")
    p.add_argument("--n-slots", type=int, default=1024)
    p.add_argument("--slot-dim", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--scaling", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--run-name", required=True)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use-sparse-opt", action="store_true",
                   help="use SparseHopfieldOptimizer for slot-localized updates")
    args = p.parse_args()

    set_global_seed(42)
    device = torch.device(args.device)
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load JEPA (frozen).
    print(f"[train_concept_memory] loading JEPA: {args.jepa_checkpoint}")
    jepa = load_jepa(args.jepa_checkpoint, device)

    # Load rollouts.
    print(f"[train_concept_memory] loading rollouts: {args.rollouts}")
    data = np.load(args.rollouts)
    images = torch.from_numpy(data["images"]).float()  # (N, H, W, C)
    predicates = torch.from_numpy(data["predicates"]).float()  # (N, P)
    n_predicates = predicates.shape[1]
    print(f"[train_concept_memory] N={images.size(0)} predicates_dim={n_predicates}")

    # Encode all images to latents (frozen JEPA).
    print("[train_concept_memory] encoding to latents...")
    latents = []
    with torch.no_grad():
        for i in range(0, images.size(0), args.batch_size):
            batch = images[i:i + args.batch_size].to(device)
            z = jepa.encode(batch).flatten(1)
            latents.append(z.cpu())
    latents = torch.cat(latents, dim=0)
    print(f"[train_concept_memory] latents shape: {latents.shape}")

    # Build dataset of (latent, predicate_target).
    dataset = TensorDataset(latents, predicates)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # Init ConceptMemory + a projection from slot_dim → n_predicates for supervision.
    memory = ConceptMemory(
        latent_dim=latents.shape[1],
        n_slots=args.n_slots,
        slot_dim=args.slot_dim,
        n_heads=args.n_heads,
        scaling=args.scaling,
    ).to(device)
    pred_head = nn.Linear(args.slot_dim, n_predicates).to(device)

    params = list(memory.parameters()) + list(pred_head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    sparse_opt = SparseHopfieldOptimizer(memory, opt) if args.use_sparse_opt else None

    # Training loop.
    best_loss = float("inf")
    for epoch in range(args.epochs):
        memory.train(); pred_head.train()
        total = 0.0
        n_batches = 0
        for z, target in loader:
            z = z.to(device)
            target = target.to(device)
            concept, attn = memory(z, return_attention=True)
            logits = pred_head(concept)
            loss = F.binary_cross_entropy_with_logits(logits, target)

            if sparse_opt is not None:
                sparse_opt.zero_grad()
                loss.backward()
                sparse_opt.record_attention(attn)
                sparse_opt.step()
            else:
                opt.zero_grad()
                loss.backward()
                opt.step()

            total += loss.item()
            n_batches += 1
        avg = total / max(1, n_batches)
        print(f"[epoch {epoch+1}/{args.epochs}] loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            memory.save(str(run_dir / "concept_memory_best.pt"))
            torch.save(pred_head.state_dict(), str(run_dir / "pred_head_best.pt"))

    memory.save(str(run_dir / "concept_memory_final.pt"))
    torch.save(pred_head.state_dict(), str(run_dir / "pred_head_final.pt"))
    print(f"[done] saved to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
