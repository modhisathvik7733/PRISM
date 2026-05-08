"""Predicate probe on a (frozen) JEPA latent.

The default `PredicateProbe` is a single linear layer — this is the *honest*
test of "is the structure in the embedding?" A linear probe that fails tells
us either the structure isn't there, or it's been mixed in a way the
downstream operator/planner layer can't easily undo.

For diagnostics we also ship `MLPProbe` with one hidden layer. It answers
the secondary question: is the info present but tangled, or genuinely
absent?
"""

from __future__ import annotations

import torch
import torch.nn as nn

from prism.perception.predicates import PREDICATE_VECTOR_DIM


class PredicateProbe(nn.Module):
    """Single-linear-layer predicate readout from a frozen JEPA latent."""

    def __init__(self, embed_dim: int, out_dim: int = PREDICATE_VECTOR_DIM):
        super().__init__()
        self.linear = nn.Linear(embed_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, embed_dim) → logits (B, out_dim)."""
        return self.linear(z)


class MLPProbe(nn.Module):
    """Two-layer MLP probe — diagnostic only.

    A linear-fail / MLP-pass result means the info is in the embedding but
    tangled: useful for confirming the JEPA isn't a total loss, but the right
    fix is still to add object-centric inductive bias to the encoder, not to
    let downstream code carry the burden of de-tangling every time.
    """

    def __init__(
        self,
        embed_dim: int,
        out_dim: int = PREDICATE_VECTOR_DIM,
        hidden: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def make_probe(
    embed_dim: int,
    *,
    hidden: int = 0,
    out_dim: int = PREDICATE_VECTOR_DIM,
) -> nn.Module:
    """Factory: hidden=0 → linear probe, hidden>0 → MLP probe with that width."""
    if hidden <= 0:
        return PredicateProbe(embed_dim, out_dim)
    return MLPProbe(embed_dim, out_dim=out_dim, hidden=hidden)
