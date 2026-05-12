"""OperatorMemory — Hopfield-based operator slot memory.

Replaces the 12 hardcoded operators in OperatorBankV3 with a growable
Hopfield store. Operators are behavioral primitives (move_forward, pickup,
toggle, etc.) — fewer than concepts but need sharper retrieval (a single
correct operator should fire, not a blend).

Key differences from ConceptMemory:
- Smaller slot count (64 vs 1024 concepts)
- Higher scaling β (sharper retrieval, single-pattern regime)
- Iterative update steps for high-precision lookup
- Stores per-operator MLP indices (for compatibility with OperatorBankV3)

The retrieved operator embedding can either:
1. Be used as a soft routing signal over a frozen OperatorBankV3 (transitional)
2. Be used directly as the operator effect prediction (terminal)
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
from hflayers import HopfieldLayer  # noqa: E402


class OperatorMemory(nn.Module):
    """Hopfield-based operator memory for behavioral primitives.

    Higher scaling than ConceptMemory because operators need sharp retrieval:
    "move_forward" and "turn_left" are distinct actions, not a blend.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        n_slots: int = 64,
        slot_dim: int = 64,
        n_heads: int = 4,
        scaling: float = 4.0,       # sharper than ConceptMemory (1.0)
        update_steps: int = 3,       # iterative for precision
    ):
        super().__init__()
        assert slot_dim % n_heads == 0
        self.latent_dim = latent_dim
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.n_heads = n_heads
        self.scaling = scaling
        self.update_steps = update_steps

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
        )

        # Operator metadata Python-side: {slot_idx: {"name": str, "n_uses": int,
        # "success_rate": float, "preconditions": list, "effects": list}}
        self.operator_metadata: dict[int, dict[str, Any]] = {}

    def forward(
        self,
        z: torch.Tensor,
        return_attention: bool = False,
    ):
        """Query operator memory with context vector z."""
        squeezed = False
        if z.dim() == 2:
            z = z.unsqueeze(1)
            squeezed = True

        out = self.hopfield(z)

        if return_attention:
            attn = self.hopfield.hopfield.get_association_matrix(z)
            attn = attn.mean(dim=1)
            if squeezed:
                out = out.squeeze(1)
                attn = attn.squeeze(1)
            return out, attn

        if squeezed:
            out = out.squeeze(1)
        return out

    @torch.no_grad()
    def select_operator(self, z: torch.Tensor) -> tuple[int, float]:
        """Return (best_operator_slot, confidence) for a single state.

        For policy use: pick the highest-weighted operator and report confidence.
        """
        _, attn = self.forward(z, return_attention=True)
        if attn.dim() == 3:
            attn = attn[:, -1, :]
        if attn.dim() == 1:
            attn = attn.unsqueeze(0)
        weights, indices = attn.topk(1, dim=-1)
        return int(indices[0, 0]), float(weights[0, 0])

    def name_operator(
        self,
        slot_idx: int,
        name: str,
        preconditions: list | None = None,
        effects: list | None = None,
    ) -> None:
        self.operator_metadata[slot_idx] = {
            "name": name,
            "n_uses": 0,
            "success_rate": 0.0,
            "preconditions": preconditions or [],
            "effects": effects or [],
        }

    def record_use(self, slot_idx: int, success: bool) -> None:
        """Track operator usage statistics."""
        if slot_idx not in self.operator_metadata:
            self.operator_metadata[slot_idx] = {
                "name": f"op_{slot_idx}",
                "n_uses": 0,
                "success_rate": 0.0,
                "preconditions": [],
                "effects": [],
            }
        meta = self.operator_metadata[slot_idx]
        n = meta["n_uses"]
        prev_rate = meta["success_rate"]
        meta["success_rate"] = (prev_rate * n + (1.0 if success else 0.0)) / (n + 1)
        meta["n_uses"] = n + 1

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "latent_dim": self.latent_dim,
            "n_slots": self.n_slots,
            "slot_dim": self.slot_dim,
            "n_heads": self.n_heads,
            "scaling": self.scaling,
            "update_steps": self.update_steps,
            "operator_metadata": self.operator_metadata,
        }, path)

    @classmethod
    def load(cls, path: str, device: torch.device) -> "OperatorMemory":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = cls(
            latent_dim=ckpt["latent_dim"],
            n_slots=ckpt["n_slots"],
            slot_dim=ckpt["slot_dim"],
            n_heads=ckpt["n_heads"],
            scaling=ckpt["scaling"],
            update_steps=ckpt.get("update_steps", 3),
        )
        m.load_state_dict(ckpt["state_dict"])
        m.operator_metadata = ckpt.get("operator_metadata", {})
        m.to(device)
        return m
