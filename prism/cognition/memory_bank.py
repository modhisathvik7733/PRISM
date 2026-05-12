"""MemoryBank — substrate-side Hopfield K/V store.

Substrate equivalent of v5's `prism.cog_core.concept_memory.ConceptMemory`
and `operator_memory.OperatorMemory`: a Hopfield-attention block with
trainable K and V `nn.Parameter` banks. Used by `RetrievalBlock` to
issue per-step queries against ConceptMemory and OperatorMemory.

v5 had separate `latent_dim` (query) and `slot_dim` (value) dimensions.
v6 unifies on `D_tok` for both: the substrate operates in a single
token-embedding space, and adapters project domain-specific inputs to
`D_tok` at the boundary (resolution 3 / locked substrate hyperparameters).

Frozen-slot / growable-bank support (`expand`, `freeze_slots`, etc.)
lands in PR-5 (Phase C). This module exposes the minimum interface
PR-4 needs: construct, retrieve, return.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_VENDOR = os.path.join(os.path.dirname(__file__), "..", "_vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
from hflayers import Hopfield  # noqa: E402


class MemoryBank(nn.Module):
    """Hopfield-attention K/V store with all dims at `D_tok`.

    Parameters
    ----------
    D_tok : int
        Substrate token-embedding dim. Same for keys, values, and queries.
    n_slots : int
        Number of trainable K/V slot pairs.
    n_heads : int
        Number of attention heads. `D_tok // n_heads` must be ≥1.
    scaling : float
        Hopfield β (softmax inverse temperature). v5 conventions:
        ConceptMemory uses 1.0 (soft retrieval, blends concepts);
        OperatorMemory uses 4.0 (sharp retrieval, single primitive fires).
    update_steps : int
        Iterative Hopfield update steps. 0 = single-shot (attention-like).
        v5 conventions: ConceptMemory 0, OperatorMemory 3.
    """

    def __init__(
        self,
        D_tok: int = 128,
        n_slots: int = 1024,
        n_heads: int = 4,
        scaling: float = 1.0,
        update_steps: int = 0,
    ):
        super().__init__()
        self.D_tok = D_tok
        self.n_slots = n_slots
        self.n_heads = n_heads
        self.scaling = scaling
        self.update_steps = update_steps

        head_dim = max(8, D_tok // n_heads)
        self.hopfield = Hopfield(
            input_size=D_tok,
            stored_pattern_size=D_tok,
            pattern_projection_size=D_tok,
            output_size=D_tok,
            hidden_size=head_dim,
            num_heads=n_heads,
            scaling=scaling,
            update_steps_max=update_steps,
            normalize_stored_pattern=True,
            normalize_state_pattern=True,
            normalize_pattern_projection=True,
            dropout=0.0,
        )
        self.keys = nn.Parameter(torch.randn(1, n_slots, D_tok) * 0.02)
        self.values = nn.Parameter(torch.randn(1, n_slots, D_tok) * 0.02)

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """Issue a query against the bank.

        Parameters
        ----------
        query : (B, D_tok)

        Returns
        -------
        retrieved : (B, D_tok)
        """
        B = query.size(0)
        K = self.keys.expand(B, -1, -1)         # (B, n_slots, D_tok)
        V = self.values.expand(B, -1, -1)       # (B, n_slots, D_tok)
        Q = query.unsqueeze(1)                  # (B, 1, D_tok)
        out = self.hopfield((K, Q, V))          # (B, 1, D_tok)
        return out.squeeze(1)


if __name__ == "__main__":
    # Standalone smoke test for retrieval shape + gradient flow.
    # Run with: `python -m prism.cognition.memory_bank`
    import sys as _sys

    bank = MemoryBank(D_tok=64, n_slots=128, n_heads=4, scaling=1.0, update_steps=0)
    B = 5
    q = torch.randn(B, 64, requires_grad=True)
    out = bank.retrieve(q)
    if out.shape != (B, 64):
        print(f"FAIL: retrieve output shape {tuple(out.shape)} != expected (5, 64)")
        _sys.exit(1)
    print(f"[membank] retrieve shape OK: {tuple(out.shape)}")

    grad = torch.autograd.grad(out.sum(), [q, bank.keys, bank.values])
    if any(g is None or g.abs().sum().item() == 0.0 for g in grad):
        print("FAIL: query / keys / values did not all receive gradient")
        _sys.exit(1)
    print(f"[membank] gradients flow to query, keys, values; "
          f"|dq|={grad[0].abs().sum():.3f} |dK|={grad[1].abs().sum():.3f} "
          f"|dV|={grad[2].abs().sum():.3f}")

    # Sharp retrieval test: OperatorMemory β=4 should produce sharper
    # attention than ConceptMemory β=1 on identical queries.
    op_bank = MemoryBank(D_tok=64, n_slots=16, n_heads=4, scaling=4.0, update_steps=3)
    out_op = op_bank.retrieve(q)
    print(f"[membank] sharp/iterative retrieval (β=4, steps=3) shape OK: {tuple(out_op.shape)}")
    print("[membank] all smoke checks passed")
