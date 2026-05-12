"""SparseHopfieldOptimizer — implements Sparse Memory Finetuning (Lin 2025).

Paper: "Continual Learning via Sparse Memory Finetuning" (arxiv:2510.15103)
Strongest LLM continual-learning result of 2025: reduces NaturalQuestions F1
forgetting from -89% (full FT) and -71% (LoRA) to -11%.

Mechanism: when training a Hopfield-based memory on new data, only update
the slot parameters that ACTIVATED on that data. Other slots have their
gradients zeroed before the optimizer step. This is the inverse of EWC
(which dampens important weights) — it FREEZES inactive weights entirely,
so writing a new concept to slot 47 cannot disturb the concept in slot 12.

Combined with replay (small buffer of old activations to refresh),
this gives near-zero catastrophic forgetting in continual learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseHopfieldOptimizer:
    """Wraps a standard PyTorch optimizer to zero gradients on Hopfield slots
    that did NOT activate above threshold for the current batch.

    Usage:
        optimizer = torch.optim.Adam(concept_memory.parameters(), lr=1e-3)
        sparse_opt = SparseHopfieldOptimizer(concept_memory, optimizer, threshold=0.05)

        # Forward pass
        out, attn = concept_memory(z, return_attention=True)
        loss = ...

        sparse_opt.zero_grad()
        loss.backward()
        sparse_opt.record_attention(attn)
        sparse_opt.step()
    """

    def __init__(
        self,
        hopfield_module: nn.Module,
        optimizer: torch.optim.Optimizer,
        threshold: float = 0.05,
        slot_param_substrings: tuple[str, ...] = (
            "lookup_weights",
            "target_weights",
            "stored_pattern",
        ),
    ):
        self.module = hopfield_module
        self.opt = optimizer
        self.threshold = threshold
        self.slot_param_substrings = slot_param_substrings
        self._last_attention: torch.Tensor | None = None

        # Identify slot parameters once at construction.
        self._slot_params: list[tuple[str, nn.Parameter]] = []
        for name, param in self.module.named_parameters():
            if any(s in name for s in slot_param_substrings):
                self._slot_params.append((name, param))

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.opt.zero_grad(set_to_none=set_to_none)

    def record_attention(self, attention: torch.Tensor) -> None:
        """Record which slots activated. attention shape: (B, n_slots) or (B, T, n_slots)."""
        if attention.dim() == 3:
            attention = attention.mean(dim=1)  # (B, n_slots)
        # Average over batch — slots active for ANY example in batch get updates.
        self._last_attention = attention.max(dim=0).values.detach()

    def step(self) -> None:
        """Zero inactive slot gradients, then step the wrapped optimizer."""
        if self._last_attention is not None and self._slot_params:
            active_mask = self._last_attention > self.threshold  # (n_slots,)

            for name, param in self._slot_params:
                if param.grad is None:
                    continue
                # The slot dimension is typically dim 1 of the parameter
                # (hflayers shapes: (1, n_slots, hidden_dim)).
                # Find the dim matching n_slots.
                slot_dim_idx = None
                for d, size in enumerate(param.shape):
                    if size == active_mask.size(0):
                        slot_dim_idx = d
                        break
                if slot_dim_idx is None:
                    continue

                # Build broadcast-compatible mask: 1.0 for active, 0.0 for inactive.
                mask = active_mask.to(param.grad.dtype)
                # Reshape mask to broadcast across param.grad shape.
                shape = [1] * param.grad.dim()
                shape[slot_dim_idx] = active_mask.size(0)
                mask = mask.view(shape)
                param.grad.mul_(mask)

        self.opt.step()

    def state_dict(self) -> dict:
        return self.opt.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.opt.load_state_dict(sd)

    @property
    def param_groups(self):
        return self.opt.param_groups
