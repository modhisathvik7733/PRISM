"""Linear predicate probe on the frozen JEPA latent.

A single linear layer from JEPA `embed_dim` to `PREDICATE_VECTOR_DIM` (96).
Output is per-predicate logits; supervised with BCE against the ground-truth
predicate vector computed from slots.

Why linear, not a deeper head:
  We're testing whether the *JEPA latent itself* contains the structured
  information we need. If a deep MLP probe is required, that's evidence that
  the JEPA encoder hasn't actually learned object-structured representations
  — it's just reaching whatever predicates it can with extra computation.
  Linear probe accuracy is the honest test of "is this in the embedding".
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
