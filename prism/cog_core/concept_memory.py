"""ConceptMemory — Hopfield-based trainable concept slot memory.

Replaces PRISM's hardcoded predicate_readout with a growable, queryable,
inspectable concept store built on modern Hopfield networks (Ramsauer 2021,
arxiv:2008.02217).

Architectural rationale:
- Hardcoded predicates (96 fixed slots) → cannot grow, cannot adapt to new domains
- HopfieldLayer with N trainable slots → growable, attention-based retrieval,
  same generalization properties as transformer attention (proven equivalent
  to Hopfield update rule by Ramsauer 2021)

Each slot is a trainable concept prototype. Inputs (JEPA latents) are queried
against all slots via attention; output is a weighted combination of slot
values. Operates in three regimes depending on scaling (β):
  - β low: global averaging (good for OOD)
  - β medium: metastable composition (where learning happens)
  - β high: single-pattern retrieval (precise recall)

Slot metadata (names, properties) is tracked Python-side for inspection and
can be populated by ConceptManager via LLM bootstrap.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torch.nn as nn

# Vendored hflayers (BSD-3-Clause, ml-jku/hopfield-layers)
_VENDOR = os.path.join(os.path.dirname(__file__), "..", "_vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
from hflayers import HopfieldLayer  # noqa: E402


class ConceptMemory(nn.Module):
    """Hopfield-based concept memory replacing fixed predicate readout.

    Parameters
    ----------
    latent_dim : int
        Dimension of the input query (JEPA latent dim).
    n_slots : int
        Number of trainable concept slots (the "database rows"). Default 1024.
    slot_dim : int
        Dimension of each retrieved concept embedding. Default 64.
    n_heads : int
        Number of attention heads. Default 4.
    scaling : float
        Hopfield β (softmax inverse temperature). Higher = sharper retrieval.
        Default 1.0 = metastable regime where learning happens.
    update_steps : int
        Iterative Hopfield update steps. 0 = single-shot (attention-like).
        For concept memory, 0 is usually best. Default 0.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        n_slots: int = 1024,
        slot_dim: int = 64,
        n_heads: int = 4,
        scaling: float = 1.0,
        update_steps: int = 0,
    ):
        super().__init__()
        # head_dim must divide slot_dim evenly across heads
        assert slot_dim % n_heads == 0, (
            f"slot_dim={slot_dim} must be divisible by n_heads={n_heads}"
        )
        self.latent_dim = latent_dim
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.n_heads = n_heads
        self.scaling = scaling
        self.update_steps = update_steps

        # HopfieldLayer = trainable K and V banks (the "database").
        # input is the query, the layer stores patterns internally.
        # lookup_weights_as_separated=True → K and V are independent trainable matrices
        # lookup_targets_as_trainable=True → V (concept embeddings) is learned
        self.hopfield = HopfieldLayer(
            input_size=latent_dim,
            output_size=slot_dim,
            hidden_size=slot_dim // n_heads,
            quantity=n_slots,
            num_heads=n_heads,
            scaling=scaling,
            update_steps_max=update_steps,
            lookup_weights_as_separated=True,
            lookup_targets_as_trainable=True,
            normalize_stored_pattern=True,
            normalize_state_pattern=True,
            normalize_pattern_projection=True,
            dropout=0.0,
        )

        # Slot metadata for human inspection (NOT part of state_dict).
        # Populated by ConceptManager when novel slots are flagged.
        # Format: {slot_idx: {"name": str, "properties": dict, "count": int}}
        self.slot_metadata: dict[int, dict[str, Any]] = {}

    def forward(
        self,
        z: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Query the memory with latent z.

        Parameters
        ----------
        z : Tensor
            Shape (B, latent_dim) or (B, T, latent_dim).
        return_attention : bool
            If True, also return attention weights over slots.

        Returns
        -------
        concept : Tensor
            Shape (B, slot_dim) or (B, T, slot_dim) — retrieved concept embedding.
        attention : Tensor (optional)
            Shape (B, n_slots) or (B, T, n_slots) — softmax weights over slots.
        """
        # Normalize input shape to (B, N, latent_dim) for HopfieldLayer.
        squeezed = False
        if z.dim() == 2:
            z = z.unsqueeze(1)  # (B, 1, latent_dim)
            squeezed = True

        out = self.hopfield(z)  # (B, N, slot_dim)

        if return_attention:
            # Get association matrix: (B, n_heads, N, n_slots)
            attn = self.hopfield.hopfield.get_association_matrix(z)
            # Average over heads → (B, N, n_slots)
            attn = attn.mean(dim=1)
            if squeezed:
                out = out.squeeze(1)
                attn = attn.squeeze(1)
            return out, attn

        if squeezed:
            out = out.squeeze(1)
        return out

    @torch.no_grad()
    def get_top_k_slots(
        self,
        z: torch.Tensor,
        k: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return indices and weights of top-k activated slots.

        Returns
        -------
        indices : LongTensor (B, k)
        weights : Tensor (B, k)
        """
        _, attn = self.forward(z, return_attention=True)
        if attn.dim() == 3:  # (B, T, n_slots) — take last timestep
            attn = attn[:, -1, :]
        weights, indices = attn.topk(k, dim=-1)
        return indices, weights

    @torch.no_grad()
    def get_active_slots(
        self,
        z: torch.Tensor,
        threshold: float = 0.05,
    ) -> list[dict[str, Any]]:
        """Return all slots that activate above threshold for this input.

        Useful for ConceptManager to detect novel patterns.
        """
        _, attn = self.forward(z, return_attention=True)
        if attn.dim() == 3:
            attn = attn[:, -1, :]
        active: list[dict[str, Any]] = []
        for b in range(z.size(0) if z.dim() > 1 else 1):
            row = attn[b] if attn.dim() == 2 else attn
            for i in torch.where(row > threshold)[0].cpu().tolist():
                meta = self.slot_metadata.get(i, {"name": f"unnamed_{i}"})
                active.append({
                    "batch_idx": b,
                    "slot_idx": i,
                    "weight": float(row[i]),
                    "metadata": meta,
                })
        return active

    def name_slot(
        self,
        slot_idx: int,
        name: str,
        properties: dict | None = None,
    ) -> None:
        """Tag a slot with a human-readable name (called by ConceptManager)."""
        if slot_idx not in self.slot_metadata:
            self.slot_metadata[slot_idx] = {
                "name": name,
                "properties": properties or {},
                "count": 0,
            }
        else:
            self.slot_metadata[slot_idx]["name"] = name
            if properties:
                self.slot_metadata[slot_idx]["properties"].update(properties)

    def increment_slot_count(self, slot_idx: int) -> None:
        """Track how often each slot has been activated (for pruning)."""
        if slot_idx not in self.slot_metadata:
            self.slot_metadata[slot_idx] = {
                "name": f"unnamed_{slot_idx}",
                "properties": {},
                "count": 0,
            }
        self.slot_metadata[slot_idx]["count"] = (
            self.slot_metadata[slot_idx].get("count", 0) + 1
        )

    def get_named_slot_count(self) -> int:
        """Number of slots that have been named (excluding unnamed_X defaults)."""
        return sum(
            1 for meta in self.slot_metadata.values()
            if not meta["name"].startswith("unnamed_")
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "latent_dim": self.latent_dim,
                "n_slots": self.n_slots,
                "slot_dim": self.slot_dim,
                "n_heads": self.n_heads,
                "scaling": self.scaling,
                "update_steps": self.update_steps,
                "slot_metadata": self.slot_metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: torch.device) -> "ConceptMemory":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = cls(
            latent_dim=ckpt["latent_dim"],
            n_slots=ckpt["n_slots"],
            slot_dim=ckpt["slot_dim"],
            n_heads=ckpt["n_heads"],
            scaling=ckpt["scaling"],
            update_steps=ckpt.get("update_steps", 0),
        )
        m.load_state_dict(ckpt["state_dict"])
        m.slot_metadata = ckpt.get("slot_metadata", {})
        m.to(device)
        return m
