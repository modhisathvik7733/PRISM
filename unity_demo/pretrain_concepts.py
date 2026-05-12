"""Phase 1: Concept pretraining.

Teach the substrate's Concept memory bank to sharply discriminate
objects by (type, color) BEFORE any policy training. The thesis: once
the concept retrieval returns distinguishable tokens for "red ball" vs
"green ball", downstream policy learning (BC or PPO) doesn't need to
relearn color sensitivity from rewards — it inherits it from the
substrate's representation.

No hand-coded expert here. The supervision is "this observation
contains object X of color Y". That label is a property of the data,
not a policy decision. Anything in the rest of the pipeline that wants
color awareness gets it for free from the trained concept bank.

Approach:
  1. Generate synthetic observations: one BabyAI-shape object per obs,
     across the full (type x color) vocabulary, at random in-view
     positions and headings.
  2. Pass obs → JEPA → latent_proj → concept query → concept_bank →
     concept_token.
  3. Add a thin classification head: concept_token -> (type, color) class.
  4. Train: ONLY concept_bank.keys/values + classification head.
     Everything else (JEPA, trunk, action head, etc.) stays frozen.
  5. Verify held-out classification accuracy (target >95%).
  6. Save the checkpoint with the updated concept bank.
     Discard the classification head — it was scaffolding.

After this:
  * concept_bank.keys/values now encode sharp (type, color) prototypes.
  * The rest of the policy is identical to the base checkpoint.
  * Phase 2 (policy fine-tune) works on top of a substrate that
    *already* discriminates concepts, so it needs much less data.

Usage:
    python unity_demo/pretrain_concepts.py \\
        --jepa runs/jepa_dev_v1_factored/jepa_final.pt \\
        --base-policy runs/v6_phaseB_GoToLocal_500k/policy_iter244.pt \\
        --out-path runs/v6_concept_pretrain_v1/policy.pt \\
        --n-per-class 1024 \\
        --epochs 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.adapters.babyai_adapter import BabyAIAdapter
from prism.adapters.unity_2d import Unity2DAdapter
from prism.cognition.policy import UniversalPolicy
from prism.models.jepa import JepaWorldModel, upgrade_config
from prism.perception.predicates import type_color_index
from prism.perception.slots import (
    COLOR_NAME_TO_IDX,
    COLOR_NAMES,
    NUM_COLORS,
    NUM_TYPES,
    OBJECT_NAME_TO_TYPE,
    OBJECT_TYPE_NAMES,
    OBJECT_TYPES,
)


# All (type, color) classes in the BabyAI vocabulary: 4 types × 6 colors = 24.
TARGET_TYPE_NAMES = [OBJECT_TYPE_NAMES[t] for t in OBJECT_TYPES]
TARGET_COLOR_NAMES = list(COLOR_NAMES.values())  # ordered by COLOR_NAMES dict
NUM_CLASSES = NUM_TYPES * NUM_COLORS  # 24


# ===========================================================================
# Checkpoint loading (same as continual_finetune.py — copy-pasted to keep
# this script standalone)
# ===========================================================================
def load_jepa(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    return jepa, cfg


def build_policy(ckpt: dict, jepa, cfg, device: torch.device, trunk: str):
    adapter = BabyAIAdapter(jepa=jepa, cfg=cfg, device=device)
    policy = UniversalPolicy.from_adapter(
        adapter,
        trunk=trunk,
        hidden_dim=ckpt["hidden_dim"],
        latent_proj_dim=ckpt["latent_proj_dim"],
        mem_feat_dim=ckpt.get("mem_feat_dim", 0),
        concept_n_slots=ckpt.get("concept_n_slots", 1024),
        operator_n_slots=ckpt.get("operator_n_slots", 64),
        concept_scaling=ckpt.get("concept_scaling", 1.0),
        operator_scaling=ckpt.get("operator_scaling", 4.0),
        use_operator_memory=ckpt.get("use_operator_memory", True),
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    return policy


# ===========================================================================
# Synthetic dataset: single-object observations
# ===========================================================================
def generate_dataset(
    n_per_class: int,
    obs_scale: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """For each of the 24 (type, color) classes, render n_per_class
    observations with that one object placed at a random in-view
    position and random virtual heading.

    Returns:
      images: (N, 3, 7, 7) float32, JEPA-normalized
      labels: (N,) int64, class index = type_color_index(type, color)
    """
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []

    # Build an adapter once; we only use its render_obs_multi() method.
    adapter = Unity2DAdapter(obs_scale=obs_scale)

    for type_name in TARGET_TYPE_NAMES:
        type_id = OBJECT_NAME_TO_TYPE[type_name]
        for color_name in TARGET_COLOR_NAMES:
            color_id = COLOR_NAME_TO_IDX[color_name]
            label = type_color_index(type_id, color_id)
            for _ in range(n_per_class):
                # Random heading so obs distribution covers all 4 orientations.
                adapter.heading = int(rng.integers(0, 4))
                # Random in-view position. View covers ~±3 grid cells, each
                # cell = obs_scale Unity units. We sample in Unity space
                # so the in-view radius is obs_scale * 3.
                radius = obs_scale * 3.0 - 0.5
                pos = rng.uniform(-radius, radius, size=2).astype(np.float32)
                scene = [(type_id, color_id, (float(pos[0]), float(pos[1])))]
                obs = adapter.render_obs_multi((0.0, 0.0), scene)
                images.append(obs.astype(np.float32))
                labels.append(label)

    images_np = np.stack(images, axis=0)
    labels_np = np.asarray(labels, dtype=np.int64)
    return images_np, labels_np


# ===========================================================================
# Concept-only forward pass
# ===========================================================================
def concept_token_forward(
    policy: UniversalPolicy,
    jepa: JepaWorldModel,
    images: torch.Tensor,
) -> torch.Tensor:
    """obs -> JEPA latent -> obs_token (via latent_proj) -> concept query
    -> concept_bank retrieval -> concept_token. Skips action_emb and
    mission_proj because concepts should be perception-anchored, not
    mission-anchored — we want "this IS a red ball" regardless of task.
    """
    inner = policy._inner
    retrieval = inner.retrieval
    concept_bank = retrieval.concept_bank

    with torch.no_grad():
        z = jepa.encode(images)
    if z.ndim > 2:
        z = z.flatten(1)
    obs_token = inner.latent_proj(z)  # (B, D_tok)
    B = obs_token.size(0)
    cq = retrieval.concept_base.expand(B, -1) + retrieval.concept_cond(obs_token)
    concept_token = concept_bank.retrieve(cq)  # (B, D_tok)
    return concept_token


# ===========================================================================
# Train / eval loops
# ===========================================================================
def train_one_epoch(
    policy: UniversalPolicy,
    jepa: JepaWorldModel,
    classifier: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    opt: torch.optim.Optimizer,
    batch_size: int,
) -> tuple[float, float]:
    N = images.size(0)
    perm = torch.randperm(N, device=images.device)
    total_loss = 0.0
    n_correct = 0
    n_batches = 0
    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        imgs = images[idx]
        labs = labels[idx]
        c_tok = concept_token_forward(policy, jepa, imgs)
        logits = classifier(c_tok)
        loss = F.cross_entropy(logits, labs)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in opt.param_groups[0]["params"]],
            max_norm=1.0,
        )
        opt.step()
        total_loss += float(loss.item())
        n_correct += int((logits.argmax(-1) == labs).sum().item())
        n_batches += 1
    return total_loss / max(1, n_batches), n_correct / N


@torch.no_grad()
def evaluate(
    policy: UniversalPolicy,
    jepa: JepaWorldModel,
    classifier: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> dict:
    N = images.size(0)
    n_correct = 0
    # Per-class accuracy.
    per_class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_class_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i in range(0, N, batch_size):
        imgs = images[i:i + batch_size]
        labs = labels[i:i + batch_size]
        c_tok = concept_token_forward(policy, jepa, imgs)
        preds = classifier(c_tok).argmax(-1)
        correct = (preds == labs)
        n_correct += int(correct.sum().item())
        for lab, ok in zip(labs.cpu().numpy(), correct.cpu().numpy()):
            per_class_total[lab] += 1
            per_class_correct[lab] += int(ok)
    overall = n_correct / N
    per_class_acc = np.where(per_class_total > 0, per_class_correct / np.maximum(1, per_class_total), 0.0)
    return {
        "accuracy": overall,
        "per_class_acc": per_class_acc,
        "worst_class_acc": float(per_class_acc.min()),
        "n": N,
    }


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    p = argparse.ArgumentParser(description="Phase 1 concept pretraining.")
    p.add_argument("--jepa", required=True)
    p.add_argument("--base-policy", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--trunk", default="transformer", choices=["transformer", "gru"])
    p.add_argument("--n-per-class", type=int, default=1024,
                   help="Train samples per (type, color) class. Total = 24 * this.")
    p.add_argument("--n-eval-per-class", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--obs-scale", type=float, default=2.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[concept] device={device}")
    print(f"[concept] loading JEPA from {args.jepa}")
    jepa, cfg = load_jepa(Path(args.jepa), device)

    print(f"[concept] loading base policy from {args.base_policy}")
    base_ckpt = torch.load(args.base_policy, map_location=device, weights_only=False)
    policy = build_policy(base_ckpt, jepa, cfg, device, trunk=args.trunk)

    # Reach into the substrate to grab the concept bank.
    retrieval = policy._inner.retrieval
    if isinstance(retrieval, nn.Identity):
        print("[concept] ERROR: policy has no retrieval block; cannot pretrain concepts.")
        return 1
    concept_bank = retrieval.concept_bank
    D_tok = int(concept_bank.values.size(-1))
    print(f"[concept] concept_bank: {tuple(concept_bank.keys.shape)} keys, "
          f"{tuple(concept_bank.values.shape)} values, D_tok={D_tok}")

    # Freeze everything except concept_bank.keys/values.
    for p_ in policy.parameters():
        p_.requires_grad_(False)
    concept_bank.keys.requires_grad_(True)
    concept_bank.values.requires_grad_(True)
    # latent_proj feeds into the concept query and is essential for
    # concept_cond to receive useful gradients — we also unfreeze it so
    # the obs->query pathway can adapt.
    policy._inner.latent_proj.weight.requires_grad_(True)
    policy._inner.latent_proj.bias.requires_grad_(True)
    # Same for concept_cond (the query-conditioning MLP).
    retrieval.concept_cond.weight.requires_grad_(True)
    retrieval.concept_cond.bias.requires_grad_(True)
    retrieval.concept_base.requires_grad_(True)

    # Classification head — scaffolding, discarded at save time.
    classifier = nn.Linear(D_tok, NUM_CLASSES).to(device)

    trainable_params = [p_ for p_ in policy.parameters() if p_.requires_grad] + list(classifier.parameters())
    n_trainable = sum(p_.numel() for p_ in trainable_params)
    print(f"[concept] trainable params: {n_trainable:,}")

    print(f"[concept] generating dataset: {args.n_per_class} samples/class * "
          f"{NUM_CLASSES} classes = {args.n_per_class * NUM_CLASSES} train samples")
    t0 = time.time()
    train_images_np, train_labels_np = generate_dataset(
        n_per_class=args.n_per_class, obs_scale=args.obs_scale, seed=0,
    )
    eval_images_np, eval_labels_np = generate_dataset(
        n_per_class=args.n_eval_per_class, obs_scale=args.obs_scale, seed=42,
    )
    print(f"[concept] generated in {time.time()-t0:.1f}s. "
          f"train={train_images_np.shape}, eval={eval_images_np.shape}")

    train_images = torch.from_numpy(train_images_np).to(device)
    train_labels = torch.from_numpy(train_labels_np).to(device)
    eval_images = torch.from_numpy(eval_images_np).to(device)
    eval_labels = torch.from_numpy(eval_labels_np).to(device)

    # Baseline eval: how good is the BASE policy's concept discrimination?
    print("[concept] evaluating BASE concept discrimination...")
    base_eval = evaluate(policy, jepa, classifier, eval_images, eval_labels, args.batch_size)
    print(f"[concept] BASE  accuracy={base_eval['accuracy']:.3f} "
          f"worst_class={base_eval['worst_class_acc']:.3f}")

    opt = torch.optim.Adam(trainable_params, lr=args.lr)
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            policy, jepa, classifier, train_images, train_labels, opt, args.batch_size,
        )
        eval_result = evaluate(
            policy, jepa, classifier, eval_images, eval_labels, args.batch_size,
        )
        print(f"[concept] epoch {epoch+1}/{args.epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"eval_acc={eval_result['accuracy']:.3f} "
              f"worst_class={eval_result['worst_class_acc']:.3f}")

    print("[concept] evaluating FINAL concept discrimination...")
    final_eval = evaluate(policy, jepa, classifier, eval_images, eval_labels, args.batch_size)
    print(f"[concept] FINAL accuracy={final_eval['accuracy']:.3f} "
          f"worst_class={final_eval['worst_class_acc']:.3f}")

    # Per-class breakdown (shows which (type, color) the substrate still confuses).
    print("[concept] per-class accuracy:")
    for type_name in TARGET_TYPE_NAMES:
        type_id = OBJECT_NAME_TO_TYPE[type_name]
        row = []
        for color_name in TARGET_COLOR_NAMES:
            color_id = COLOR_NAME_TO_IDX[color_name]
            idx = type_color_index(type_id, color_id)
            row.append(f"{color_name[:3]}={final_eval['per_class_acc'][idx]:.2f}")
        print(f"  {type_name:5s}: {' '.join(row)}")

    # Save the updated policy checkpoint. Classification head is dropped
    # — it was scaffolding. The concept bank's K/V are now concept-trained.
    new_ckpt = {**base_ckpt, "policy_state_dict": policy.state_dict()}
    new_ckpt["concept_pretrain"] = {
        "n_per_class": args.n_per_class,
        "epochs": args.epochs,
        "base_accuracy": base_eval["accuracy"],
        "final_accuracy": final_eval["accuracy"],
        "worst_class_acc": final_eval["worst_class_acc"],
    }
    torch.save(new_ckpt, out_path)
    print(f"[concept] saved {out_path}")

    print(f"\n[concept] BASE concept accuracy:  {base_eval['accuracy']:.3f}")
    print(f"[concept] FINAL concept accuracy: {final_eval['accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
