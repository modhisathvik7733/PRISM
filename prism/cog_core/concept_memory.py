"""ConceptMemory — Hopfield-based trainable concept slot memory.

Replaces PRISM's hardcoded predicate_readout with a growable, queryable,
inspectable concept store built on modern Hopfield networks (Ramsauer 2021,
arxiv:2008.02217).

Architecture: bare `Hopfield` with explicit nn.Parameter K (keys) and V
(values) banks. This gives clean attention extraction via
`hopfield.get_association_matrix((K, Q, V))` and direct slot inspection
via `self.values[slot_idx]`.

Operates in three retrieval regimes depending on scaling (β):
  - β low: global averaging (good for OOD)
  - β medium: metastable composition (where learning happens)
  - β high: single-pattern retrieval (precise recall)
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
from hflayers import Hopfield  # noqa: E402


class ConceptMemory(nn.Module):
    """Hopfield-based concept memory with explicit trainable K/V banks.

    Parameters
    ----------
    latent_dim : int
        Dimension of the input query (JEPA latent dim).
    n_slots : int
        Number of trainable concept slots.
    slot_dim : int
        Dimension of each retrieved concept embedding (output dim).
    n_heads : int
        Number of attention heads. Must divide max(latent_dim, slot_dim).
    scaling : float
        Hopfield β (softmax inverse temperature). 1.0 = metastable regime.
    update_steps : int
        Iterative Hopfield update steps. 0 = single-shot (attention-like).
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
        self.latent_dim = latent_dim
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.n_heads = n_heads
        self.scaling = scaling
        self.update_steps = update_steps

        # Head dim must divide both the K and V projections cleanly.
        # We pick head_dim from the smaller of latent_dim and slot_dim
        # to avoid asymmetry; n_heads must divide it.
        head_dim = max(8, min(latent_dim, slot_dim) // n_heads)
        # Round head_dim down so n_heads * head_dim is reasonable.

        # Bare Hopfield: takes (K, Q, V) tuple, returns retrieved values.
        # We manage K and V as nn.Parameter banks below.
        self.hopfield = Hopfield(
            input_size=latent_dim,
            stored_pattern_size=latent_dim,
            pattern_projection_size=slot_dim,
            output_size=slot_dim,
            hidden_size=head_dim,
            num_heads=n_heads,
            scaling=scaling,
            update_steps_max=update_steps,
            normalize_stored_pattern=True,
            normalize_state_pattern=True,
            normalize_pattern_projection=True,
            dropout=0.0,
        )

        # K bank: keys that queries match against (in latent_dim space).
        self.keys = nn.Parameter(torch.randn(1, n_slots, latent_dim) * 0.02)
        # V bank: values retrieved by attention (in slot_dim space).
        self.values = nn.Parameter(torch.randn(1, n_slots, slot_dim) * 0.02)

        # Slot metadata for human inspection (NOT in state_dict).
        self.slot_metadata: dict[int, dict[str, Any]] = {}

    def _build_triple(self, z: torch.Tensor) -> tuple:
        """Build (K, Q, V) tuple expected by Hopfield.forward.

        z is the query Q with shape (B, N, latent_dim). K and V are
        expanded from the trainable banks to match the batch.
        """
        B = z.size(0)
        K = self.keys.expand(B, -1, -1)    # (B, n_slots, latent_dim)
        V = self.values.expand(B, -1, -1)  # (B, n_slots, slot_dim)
        Q = z                               # (B, N, latent_dim)
        return (K, Q, V)

    def forward(
        self,
        z: torch.Tensor,
        return_attention: bool = False,
    ):
        """Query the memory with latent z.

        Parameters
        ----------
        z : Tensor
            Shape (B, latent_dim) or (B, T, latent_dim).
        return_attention : bool
            If True, also return attention weights over slots.

        Returns
        -------
        concept : (B, slot_dim) or (B, T, slot_dim)
        attention : (B, n_slots) or (B, T, n_slots), if return_attention
        """
        squeezed = False
        if z.dim() == 2:
            z = z.unsqueeze(1)  # (B, 1, latent_dim)
            squeezed = True

        triple = self._build_triple(z)
        out = self.hopfield(triple)  # (B, N, slot_dim)

        if return_attention:
            # Hopfield.get_association_matrix returns (B, num_heads, N_q, N_kv).
            attn = self.hopfield.get_association_matrix(triple)
            # Average over heads → (B, N_q, n_slots).
            attn = attn.mean(dim=1)
            if squeezed:
                out = out.squeeze(1)
                attn = attn.squeeze(1)  # (B, n_slots)
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
        """Return indices and weights of top-k activated slots."""
        _, attn = self.forward(z, return_attention=True)
        if attn.dim() == 3:
            attn = attn[:, -1, :]
        weights, indices = attn.topk(k, dim=-1)
        return indices, weights

    @torch.no_grad()
    def get_slot_values(self, slot_indices: torch.Tensor) -> torch.Tensor:
        """Look up V-bank entries by slot index. (B, K) → (B, K, slot_dim)."""
        v_flat = self.values.squeeze(0)  # (n_slots, slot_dim)
        return v_flat[slot_indices]

    @torch.no_grad()
    def get_active_slots(
        self,
        z: torch.Tensor,
        threshold: float = 0.05,
    ) -> list[dict[str, Any]]:
        """Return slots that activate above threshold for this input."""
        _, attn = self.forward(z, return_attention=True)
        if attn.dim() == 3:
            attn = attn[:, -1, :]
        if attn.dim() == 1:
            attn = attn.unsqueeze(0)
        active: list[dict[str, Any]] = []
        for b in range(attn.size(0)):
            for i in torch.where(attn[b] > threshold)[0].cpu().tolist():
                meta = self.slot_metadata.get(i, {"name": f"unnamed_{i}"})
                active.append({
                    "batch_idx": b,
                    "slot_idx": i,
                    "weight": float(attn[b, i]),
                    "metadata": meta,
                })
        return active

    def name_slot(
        self,
        slot_idx: int,
        name: str,
        properties: dict | None = None,
    ) -> None:
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
