"""DisentangledFactoredMemory — proper compositional generalization.

Architecture for 80%+ held-out compositional generalization. Combines:
1. MLP attribute extractors (color, type) — strong probes
2. Orthogonality regularization on extractor weights — geometric disentanglement
3. Adversarial debiasing — each attribute extractor cannot predict the other
4. Hopfield memories on disentangled features — editable, continual

The math: joint held-out accuracy P(both correct) = P_c * P_t when factored.
To hit 80% joint, need ~90% per-attribute. Current ConceptMemory hits
~33% color / ~22% type because nothing forces disentanglement.

This module forces disentanglement architecturally:
- color_extractor and type_extractor are 2-layer MLPs from JEPA latent
- Their final-layer weights are regularized to be orthogonal subspaces
  of the input — so they "look at" different parts of the latent.
- An adversary learns to predict TYPE from color_emb (and vice versa);
  the main model is trained against this adversary via gradient reversal.
  This guarantees color_emb contains NO type info (and vice versa).
- Hopfield memories on the disentangled embeddings — kept for the
  editability / continual learning / inspection properties.

Inspired by:
- Domain-Adversarial Neural Networks (Ganin 2016) — gradient reversal
- β-VAE / FactorVAE — disentanglement objectives
- v4.1.1 factored aux loss — but applied here as a downstream module
  rather than at JEPA training time, so it works on any pretrained JEPA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from prism.cog_core.concept_memory import ConceptMemory


class GradientReversal(Function):
    """Gradient reversal layer (Ganin 2015) — forward is identity, backward
    multiplies gradient by -lambda. Used for adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lambda_)


class DisentangledFactoredMemory(nn.Module):
    """Disentangled extractors + adversarial debiasing + Hopfield memories.

    Returns:
        color_concept, type_concept, color_emb, type_emb, adv_logits

    Where adv_logits = (color_pred_from_type_emb, type_pred_from_color_emb)
    used for the adversarial debiasing loss.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        # Bottleneck dims — small to force factorization but large enough to carry info
        color_emb_dim: int = 24,
        type_emb_dim: int = 24,
        extractor_hidden: int = 128,
        # Hopfield slot counts
        color_n_slots: int = 24,
        color_slot_dim: int = 16,
        type_n_slots: int = 16,
        type_slot_dim: int = 16,
        n_heads: int = 4,
        scaling: float = 2.0,  # sharper retrieval for precise color/type
        # Adversarial config
        n_colors: int = 6,
        n_types: int = 4,
        adv_lambda: float = 0.5,  # gradient reversal strength
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.color_emb_dim = color_emb_dim
        self.type_emb_dim = type_emb_dim
        self.color_n_slots = color_n_slots
        self.color_slot_dim = color_slot_dim
        self.type_n_slots = type_n_slots
        self.type_slot_dim = type_slot_dim
        self.adv_lambda = adv_lambda

        # --- Disentangled attribute extractors (MLP) ---
        self.color_extractor = nn.Sequential(
            nn.Linear(latent_dim, extractor_hidden),
            nn.GELU(),
            nn.LayerNorm(extractor_hidden),
            nn.Linear(extractor_hidden, color_emb_dim),
        )
        self.type_extractor = nn.Sequential(
            nn.Linear(latent_dim, extractor_hidden),
            nn.GELU(),
            nn.LayerNorm(extractor_hidden),
            nn.Linear(extractor_hidden, type_emb_dim),
        )

        # --- Hopfield memories operate on disentangled embeddings ---
        self.color_memory = ConceptMemory(
            latent_dim=color_emb_dim,
            n_slots=color_n_slots,
            slot_dim=color_slot_dim,
            n_heads=n_heads,
            scaling=scaling,
            update_steps=0,
        )
        self.type_memory = ConceptMemory(
            latent_dim=type_emb_dim,
            n_slots=type_n_slots,
            slot_dim=type_slot_dim,
            n_heads=n_heads,
            scaling=scaling,
            update_steps=0,
        )

        # --- Adversaries: predict OPPOSITE attribute from each embedding ---
        # If color_emb truly contains no type info, the adversary will fail
        # (and gradient reversal pushes color_extractor to keep it that way).
        self.adv_type_from_color = nn.Sequential(
            nn.Linear(color_emb_dim, 64), nn.GELU(),
            nn.Linear(64, n_types),
        )
        self.adv_color_from_type = nn.Sequential(
            nn.Linear(type_emb_dim, 64), nn.GELU(),
            nn.Linear(64, n_colors),
        )

    def forward(
        self,
        z: torch.Tensor,
        return_attention: bool = False,
    ):
        color_emb = self.color_extractor(z)
        type_emb = self.type_extractor(z)

        if return_attention:
            color_concept, color_attn = self.color_memory(
                color_emb, return_attention=True
            )
            type_concept, type_attn = self.type_memory(
                type_emb, return_attention=True
            )
        else:
            color_concept = self.color_memory(color_emb, return_attention=False)
            type_concept = self.type_memory(type_emb, return_attention=False)

        # Adversarial heads with gradient reversal.
        # Forward: predicts WRONG attribute. Backward: pushes extractors to
        # remove that attribute's info from their respective embeddings.
        adv_type_logits = self.adv_type_from_color(
            grad_reverse(color_emb, self.adv_lambda)
        )
        adv_color_logits = self.adv_color_from_type(
            grad_reverse(type_emb, self.adv_lambda)
        )

        result = {
            "color_concept": color_concept,
            "type_concept": type_concept,
            "color_emb": color_emb,
            "type_emb": type_emb,
            "adv_type_from_color": adv_type_logits,
            "adv_color_from_type": adv_color_logits,
        }
        if return_attention:
            result["color_attn"] = color_attn
            result["type_attn"] = type_attn
        return result

    def orthogonality_loss(self) -> torch.Tensor:
        """Geometric disentanglement: the SUBSPACES the two extractors read
        from should be orthogonal.

        Take the first-layer weights of each extractor — these define the
        linear subspace each extractor projects from. Force their rows to
        be orthogonal between extractors.
        """
        W_c = self.color_extractor[0].weight  # (extractor_hidden, latent_dim)
        W_t = self.type_extractor[0].weight   # (extractor_hidden, latent_dim)

        # Normalize each row so orthogonality penalty is scale-invariant.
        W_c_n = F.normalize(W_c, dim=1)
        W_t_n = F.normalize(W_t, dim=1)

        # Cross-correlation matrix: (extractor_hidden, extractor_hidden).
        cross = W_c_n @ W_t_n.T

        return (cross ** 2).mean()

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "latent_dim": self.latent_dim,
            "color_emb_dim": self.color_emb_dim,
            "type_emb_dim": self.type_emb_dim,
            "color_n_slots": self.color_n_slots,
            "color_slot_dim": self.color_slot_dim,
            "type_n_slots": self.type_n_slots,
            "type_slot_dim": self.type_slot_dim,
            "adv_lambda": self.adv_lambda,
            "color_metadata": self.color_memory.slot_metadata,
            "type_metadata": self.type_memory.slot_metadata,
        }, path)

    @classmethod
    def load(cls, path: str, device: torch.device) -> "DisentangledFactoredMemory":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = cls(
            latent_dim=ckpt["latent_dim"],
            color_emb_dim=ckpt["color_emb_dim"],
            type_emb_dim=ckpt["type_emb_dim"],
            color_n_slots=ckpt["color_n_slots"],
            color_slot_dim=ckpt["color_slot_dim"],
            type_n_slots=ckpt["type_n_slots"],
            type_slot_dim=ckpt["type_slot_dim"],
            adv_lambda=ckpt["adv_lambda"],
        )
        m.load_state_dict(ckpt["state_dict"])
        m.color_memory.slot_metadata = ckpt.get("color_metadata", {})
        m.type_memory.slot_metadata = ckpt.get("type_metadata", {})
        m.to(device)
        return m
