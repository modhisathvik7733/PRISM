"""FactoredConceptMemory — two separate Hopfield memories for compositional generalization.

The architectural fix for the v5.0 Phase 1 held-out failure: a single
ConceptMemory with 1024 slots had enough capacity to memorize per-combo
prototypes, so held-out (color, type) combos collapsed to 1.5% (vs
v4.1.1's 22.5% factored-aux baseline).

This module enforces factorization at the architecture level:
- ColorMemory: small Hopfield (~24 slots × small dim) — sees only color supervision
- TypeMemory:  small Hopfield (~16 slots × small dim) — sees only type supervision

Each sub-memory is forced to learn its own attribute by gradient routing.
Even though both see the same input latent, the V-bank values for each
memory are only ever updated by their own head's loss. There's no slot
that can represent a (color, type) pair — only color slots and type slots.

This is the same factorization principle that made v4.1.1's factored aux
loss work: structurally separate the prediction paths.

Tradeoff: less expressivity (can't represent richer multi-attribute concepts
in one slot), but stronger generalization on the predicate prediction task.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torch.nn as nn

_VENDOR = os.path.join(os.path.dirname(__file__), "..", "_vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from prism.cog_core.concept_memory import ConceptMemory  # noqa: E402


class FactoredConceptMemory(nn.Module):
    """Two separate Hopfield memories, one per predicate attribute.

    Returns a tuple (color_concept, type_concept). Downstream heads read
    only from their respective concept (color head from color_concept,
    type head from type_concept). This routes gradient signal correctly
    and prevents one slot from learning a joint combo prototype.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        color_n_slots: int = 24,
        color_slot_dim: int = 16,
        type_n_slots: int = 16,
        type_slot_dim: int = 16,
        n_heads: int = 4,
        scaling: float = 1.0,
        update_steps: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.color_n_slots = color_n_slots
        self.color_slot_dim = color_slot_dim
        self.type_n_slots = type_n_slots
        self.type_slot_dim = type_slot_dim

        self.color_memory = ConceptMemory(
            latent_dim=latent_dim,
            n_slots=color_n_slots,
            slot_dim=color_slot_dim,
            n_heads=n_heads,
            scaling=scaling,
            update_steps=update_steps,
        )
        self.type_memory = ConceptMemory(
            latent_dim=latent_dim,
            n_slots=type_n_slots,
            slot_dim=type_slot_dim,
            n_heads=n_heads,
            scaling=scaling,
            update_steps=update_steps,
        )

    def forward(
        self,
        z: torch.Tensor,
        return_attention: bool = False,
    ):
        if return_attention:
            color_c, color_attn = self.color_memory(z, return_attention=True)
            type_c, type_attn = self.type_memory(z, return_attention=True)
            return (color_c, type_c), (color_attn, type_attn)
        color_c = self.color_memory(z, return_attention=False)
        type_c = self.type_memory(z, return_attention=False)
        return color_c, type_c

    @torch.no_grad()
    def get_top_k_slots(self, z: torch.Tensor, k: int = 5):
        """Returns (color_top_k, type_top_k) — each is (indices, weights)."""
        return (
            self.color_memory.get_top_k_slots(z, k=k),
            self.type_memory.get_top_k_slots(z, k=k),
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "latent_dim": self.latent_dim,
                "color_n_slots": self.color_n_slots,
                "color_slot_dim": self.color_slot_dim,
                "type_n_slots": self.type_n_slots,
                "type_slot_dim": self.type_slot_dim,
                "color_metadata": self.color_memory.slot_metadata,
                "type_metadata": self.type_memory.slot_metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: torch.device) -> "FactoredConceptMemory":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = cls(
            latent_dim=ckpt["latent_dim"],
            color_n_slots=ckpt["color_n_slots"],
            color_slot_dim=ckpt["color_slot_dim"],
            type_n_slots=ckpt["type_n_slots"],
            type_slot_dim=ckpt["type_slot_dim"],
        )
        m.load_state_dict(ckpt["state_dict"])
        m.color_memory.slot_metadata = ckpt.get("color_metadata", {})
        m.type_memory.slot_metadata = ckpt.get("type_metadata", {})
        m.to(device)
        return m
