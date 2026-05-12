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
def load_jepa(path: Path, device: torch.device, trainable: bool = False):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    if trainable:
        jepa.train()  # enable dropout/BN training mode
        for p in jepa.parameters():
            p.requires_grad_(True)
    else:
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
    n_extra_objects_range: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """For each of the 24 (type, color) classes, render n_per_class
    observations. Each obs has:
      - 1 PRIMARY object of the labeled (type, color)
      - 0..2 random EXTRA objects of different (type, color)
      - All objects at random in-view non-overlapping positions
      - Random virtual heading

    Multi-object scenes force the model to develop position-aware,
    object-specific features (vs. the "single object, lazy global
    classification" failure mode of single-object datasets).

    Returns:
      images: (N, 3, 7, 7) float32, JEPA-normalized
      labels: (N,) int64, class index = type_color_index(primary)
    """
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []
    adapter = Unity2DAdapter(obs_scale=obs_scale)
    radius = obs_scale * 3.0 - 0.5

    # All (type, color) combos for sampling extras.
    all_classes = [
        (OBJECT_NAME_TO_TYPE[t], COLOR_NAME_TO_IDX[c])
        for t in TARGET_TYPE_NAMES
        for c in TARGET_COLOR_NAMES
    ]

    def _sample_pos(taken: list[np.ndarray]) -> np.ndarray:
        for _ in range(30):
            p = rng.uniform(-radius, radius, size=2).astype(np.float32)
            # Quantize to a grid cell-resolution (obs_scale) to check
            # collision in obs space rather than continuous space.
            if all(
                np.abs(p - t).max() > obs_scale * 0.9 for t in taken
            ):
                return p
        return p  # fallback after rejections

    for type_name in TARGET_TYPE_NAMES:
        type_id = OBJECT_NAME_TO_TYPE[type_name]
        for color_name in TARGET_COLOR_NAMES:
            color_id = COLOR_NAME_TO_IDX[color_name]
            label = type_color_index(type_id, color_id)
            for _ in range(n_per_class):
                adapter.heading = int(rng.integers(0, 4))
                taken: list[np.ndarray] = []

                # Primary object: this is what the label refers to.
                primary_pos = _sample_pos(taken)
                taken.append(primary_pos)
                scene: list[tuple[int, int, tuple[float, float]]] = [
                    (type_id, color_id, (float(primary_pos[0]), float(primary_pos[1])))
                ]

                # Extra objects: random (type, color) != primary, at
                # non-overlapping positions. Forces the model to be
                # discriminative rather than just detecting "is there an object".
                n_extras = int(rng.integers(
                    n_extra_objects_range[0], n_extra_objects_range[1] + 1
                ))
                for _ in range(n_extras):
                    # Sample an extra class != primary.
                    while True:
                        ext_type_id, ext_color_id = all_classes[
                            int(rng.integers(0, len(all_classes)))
                        ]
                        if (ext_type_id, ext_color_id) != (type_id, color_id):
                            break
                    ext_pos = _sample_pos(taken)
                    taken.append(ext_pos)
                    scene.append((
                        ext_type_id, ext_color_id,
                        (float(ext_pos[0]), float(ext_pos[1])),
                    ))

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
    jepa_trainable: bool = False,
) -> torch.Tensor:
    """obs -> JEPA latent -> obs_token (via latent_proj) -> concept query
    -> concept_bank retrieval -> concept_token. Skips action_emb and
    mission_proj because concepts should be perception-anchored, not
    mission-anchored — we want "this IS a red ball" regardless of task.

    If jepa_trainable is True, JEPA's encode runs WITH gradient tracking
    so the encoder can adapt to the target-domain (Unity) obs
    distribution. Otherwise we wrap it in no_grad for speed.
    """
    inner = policy._inner
    retrieval = inner.retrieval
    concept_bank = retrieval.concept_bank

    if jepa_trainable:
        z = jepa.encode(images)
    else:
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
# Factored classifier — shared trunk + per-axis (color, type) heads.
# Routes gradient through TWO independent heads, avoiding the joint-CE
# degenerate basin where ~half the (type, color) classes collapse to 0%.
# ===========================================================================
def build_factored_classifier(D_tok: int) -> nn.Module:
    return nn.ModuleDict({
        "trunk": nn.Sequential(
            nn.Linear(D_tok, D_tok * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        ),
        "color": nn.Linear(D_tok * 2, NUM_COLORS),
        "type": nn.Linear(D_tok * 2, NUM_TYPES),
    })


def _factored_logits(classifier: nn.Module, c_tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = classifier["trunk"](c_tok)
    return classifier["color"](h), classifier["type"](h)


def _split_labels(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint label -> (color_label, type_axis_label).
    Mirrors `type_color_index` semantics: idx = type_axis * NUM_COLORS + color.
    """
    return labels % NUM_COLORS, labels // NUM_COLORS


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
    jepa_trainable: bool = False,
) -> tuple[float, float, float]:
    """Returns (mean_loss, color_acc, type_acc) over the epoch."""
    N = images.size(0)
    perm = torch.randperm(N, device=images.device)
    total_loss = 0.0
    n_color_correct = 0
    n_type_correct = 0
    n_batches = 0
    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        imgs = images[idx]
        labs = labels[idx]
        color_labs, type_labs = _split_labels(labs)
        c_tok = concept_token_forward(policy, jepa, imgs, jepa_trainable=jepa_trainable)
        color_logits, type_logits = _factored_logits(classifier, c_tok)
        loss = F.cross_entropy(color_logits, color_labs) + F.cross_entropy(type_logits, type_labs)
        opt.zero_grad()
        loss.backward()
        all_params = []
        for group in opt.param_groups:
            all_params.extend(group["params"])
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        opt.step()
        total_loss += float(loss.item())
        n_color_correct += int((color_logits.argmax(-1) == color_labs).sum().item())
        n_type_correct += int((type_logits.argmax(-1) == type_labs).sum().item())
        n_batches += 1
    return (
        total_loss / max(1, n_batches),
        n_color_correct / N,
        n_type_correct / N,
    )


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
    n_color_correct = 0
    n_type_correct = 0
    n_joint_correct = 0
    # Per-(type, color) joint accuracy table, 24 entries.
    per_class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_class_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i in range(0, N, batch_size):
        imgs = images[i:i + batch_size]
        labs = labels[i:i + batch_size]
        color_labs, type_labs = _split_labels(labs)
        c_tok = concept_token_forward(policy, jepa, imgs)
        color_logits, type_logits = _factored_logits(classifier, c_tok)
        color_pred = color_logits.argmax(-1)
        type_pred = type_logits.argmax(-1)
        color_ok = (color_pred == color_labs)
        type_ok = (type_pred == type_labs)
        joint_ok = color_ok & type_ok
        n_color_correct += int(color_ok.sum().item())
        n_type_correct += int(type_ok.sum().item())
        n_joint_correct += int(joint_ok.sum().item())
        for lab, ok in zip(labs.cpu().numpy(), joint_ok.cpu().numpy()):
            per_class_total[lab] += 1
            per_class_correct[lab] += int(ok)
    per_class_acc = np.where(per_class_total > 0, per_class_correct / np.maximum(1, per_class_total), 0.0)
    return {
        "color_acc": n_color_correct / N,
        "type_acc": n_type_correct / N,
        "joint_acc": n_joint_correct / N,
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
    p.add_argument(
        "--jepa-lr", type=float, default=1e-4,
        help="Lower LR for JEPA params when --train-jepa is set "
             "(encoder needs gentle updates to preserve general features).",
    )
    p.add_argument("--obs-scale", type=float, default=2.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--train-jepa", action="store_true",
        help="Unfreeze JEPA and fine-tune it on the synthetic Unity dataset. "
             "Closes the domain gap at the encoder level.",
    )
    p.add_argument(
        "--out-jepa-path", default=None,
        help="Where to save the fine-tuned JEPA checkpoint. Required if "
             "--train-jepa is set.",
    )
    p.add_argument(
        "--max-extra-objects", type=int, default=0,
        help="Max number of random distractor objects per training obs "
             "(0..N). Currently uses primary-label semantics which is "
             "ambiguous with extras >0 — leave 0 unless experimenting.",
    )
    args = p.parse_args()
    if args.train_jepa and args.out_jepa_path is None:
        p.error("--out-jepa-path is required when --train-jepa is set")

    device = torch.device(args.device)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[concept] device={device}")
    print(f"[concept] loading JEPA from {args.jepa} (trainable={args.train_jepa})")
    jepa, cfg = load_jepa(Path(args.jepa), device, trainable=args.train_jepa)

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

    # Factored classifier head — scaffolding, discarded at save time.
    # Shared trunk + per-axis (color, type) heads. Routes gradient through
    # TWO independent CE losses, avoiding the joint 24-way degenerate
    # basin where ~half the classes collapsed to 0% accuracy.
    classifier = build_factored_classifier(D_tok).to(device)

    policy_trainable_params = [p_ for p_ in policy.parameters() if p_.requires_grad] + list(classifier.parameters())
    jepa_trainable_params = [p_ for p_ in jepa.parameters() if p_.requires_grad]
    n_policy_trainable = sum(p_.numel() for p_ in policy_trainable_params)
    n_jepa_trainable = sum(p_.numel() for p_ in jepa_trainable_params)
    print(f"[concept] trainable policy/classifier params: {n_policy_trainable:,}")
    print(f"[concept] trainable JEPA params: {n_jepa_trainable:,}")

    print(f"[concept] generating dataset: {args.n_per_class} samples/class * "
          f"{NUM_CLASSES} classes = {args.n_per_class * NUM_CLASSES} train samples")
    t0 = time.time()
    extras_range = (0, args.max_extra_objects)
    train_images_np, train_labels_np = generate_dataset(
        n_per_class=args.n_per_class, obs_scale=args.obs_scale, seed=0,
        n_extra_objects_range=extras_range,
    )
    eval_images_np, eval_labels_np = generate_dataset(
        n_per_class=args.n_eval_per_class, obs_scale=args.obs_scale, seed=42,
        n_extra_objects_range=extras_range,
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
    print(f"[concept] BASE  joint={base_eval['joint_acc']:.3f} "
          f"color={base_eval['color_acc']:.3f} type={base_eval['type_acc']:.3f} "
          f"worst_class={base_eval['worst_class_acc']:.3f}")

    # Separate param groups so JEPA gets a lower LR than the concept-side params.
    param_groups = [{"params": policy_trainable_params, "lr": args.lr}]
    if args.train_jepa and jepa_trainable_params:
        param_groups.append({"params": jepa_trainable_params, "lr": args.jepa_lr})
    opt = torch.optim.Adam(param_groups)
    for epoch in range(args.epochs):
        train_loss, train_color_acc, train_type_acc = train_one_epoch(
            policy, jepa, classifier, train_images, train_labels, opt,
            args.batch_size, jepa_trainable=args.train_jepa,
        )
        eval_result = evaluate(
            policy, jepa, classifier, eval_images, eval_labels, args.batch_size,
        )
        print(f"[concept] epoch {epoch+1}/{args.epochs} "
              f"loss={train_loss:.4f} "
              f"train_color={train_color_acc:.3f} train_type={train_type_acc:.3f} "
              f"eval_joint={eval_result['joint_acc']:.3f} "
              f"color={eval_result['color_acc']:.3f} type={eval_result['type_acc']:.3f} "
              f"worst={eval_result['worst_class_acc']:.3f}")

    print("[concept] evaluating FINAL concept discrimination...")
    final_eval = evaluate(policy, jepa, classifier, eval_images, eval_labels, args.batch_size)
    print(f"[concept] FINAL joint={final_eval['joint_acc']:.3f} "
          f"color={final_eval['color_acc']:.3f} type={final_eval['type_acc']:.3f} "
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
        "base_joint_acc": base_eval["joint_acc"],
        "base_color_acc": base_eval["color_acc"],
        "base_type_acc": base_eval["type_acc"],
        "final_joint_acc": final_eval["joint_acc"],
        "final_color_acc": final_eval["color_acc"],
        "final_type_acc": final_eval["type_acc"],
        "worst_class_acc": final_eval["worst_class_acc"],
        "trained_jepa": args.train_jepa,
        "factored_heads": True,
    }
    torch.save(new_ckpt, out_path)
    print(f"[concept] saved policy to {out_path}")

    # Also save the fine-tuned JEPA, if applicable. Format matches the
    # original JEPA checkpoint so the inference server's loader works
    # unchanged (just point --jepa at the new file).
    if args.train_jepa:
        out_jepa_path = Path(args.out_jepa_path)
        out_jepa_path.parent.mkdir(parents=True, exist_ok=True)
        base_jepa_ckpt = torch.load(args.jepa, map_location=device, weights_only=False)
        jepa_out = {
            **base_jepa_ckpt,
            "model": jepa.state_dict(),
            "concept_pretrain_finetune": {
                "n_per_class": args.n_per_class,
                "epochs": args.epochs,
                "jepa_lr": args.jepa_lr,
            },
        }
        torch.save(jepa_out, out_jepa_path)
        print(f"[concept] saved fine-tuned JEPA to {out_jepa_path}")

    print(f"\n[concept] BASE  joint={base_eval['joint_acc']:.3f} "
          f"color={base_eval['color_acc']:.3f} type={base_eval['type_acc']:.3f}")
    print(f"[concept] FINAL joint={final_eval['joint_acc']:.3f} "
          f"color={final_eval['color_acc']:.3f} type={final_eval['type_acc']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
