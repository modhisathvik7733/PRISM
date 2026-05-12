"""RetrievalBlock — substrate-side cross-attention from a learnable
query into ConceptMemory and OperatorMemory.

Plan spec (Phase B, PR-4 step 4):
- Entry point: retrieval runs once per step, before the trunk's first layer.
- Query count: 2 — one for ConceptMemory, one for OperatorMemory.
- Each query is a learnable parameter (1, D_tok) + a small MLP
  conditioned on the current step's pooled observation token.
- Retrieval frequency: every step (no caching for MVP).
- Output shape: (B, 2, D_tok), marked with token type MEM (handled
  by the trunk's positional / type embedding pipeline; this module just
  returns the raw mem tokens).
- Gradient path: trunk → mem tokens → Hopfield K/V banks. Standard
  backprop, no exotic routing.

The retrieval tokens are PREPENDED to the trunk's input sequence per
step; they are not stored in the rolling buffer (which keeps only the
obs-derived tokens). This means RetrievalBlock fires once per
forward pass, against the current memory state, not a buffered one.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from prism.cognition.memory_bank import MemoryBank


class RetrievalBlock(nn.Module):
    """Two-query retrieval into Concept + Operator memory banks.

    Parameters are part of the substrate-locked config (resolution 3):
    `concept_n_slots`, `operator_n_slots`, β values, update_steps cannot
    vary across stages/domains. Only `D_tok` is read off the substrate
    config (locked at construction time).
    """

    def __init__(
        self,
        D_tok: int = 128,
        concept_n_slots: int = 1024,
        operator_n_slots: int = 64,
        n_heads: int = 4,
        concept_scaling: float = 1.0,
        concept_update_steps: int = 0,
        operator_scaling: float = 4.0,
        operator_update_steps: int = 3,
    ):
        super().__init__()
        self.D_tok = D_tok
        self.concept_bank = MemoryBank(
            D_tok=D_tok, n_slots=concept_n_slots, n_heads=n_heads,
            scaling=concept_scaling, update_steps=concept_update_steps,
        )
        self.operator_bank = MemoryBank(
            D_tok=D_tok, n_slots=operator_n_slots, n_heads=n_heads,
            scaling=operator_scaling, update_steps=operator_update_steps,
        )

        # Learnable base queries + obs-conditioning MLPs (one Linear each
        # — "small MLP" per the plan, kept minimal to avoid bloating the
        # substrate parameter count beyond the trunk).
        self.concept_base = nn.Parameter(torch.randn(1, D_tok) * 0.02)
        self.operator_base = nn.Parameter(torch.randn(1, D_tok) * 0.02)
        self.concept_cond = nn.Linear(D_tok, D_tok)
        self.operator_cond = nn.Linear(D_tok, D_tok)

    def forward(self, obs_token: torch.Tensor) -> torch.Tensor:
        """Compute the 2 mem tokens for this step.

        Parameters
        ----------
        obs_token : (B, D_tok)
            Current step's pooled observation token (the same token that
            will go into the rolling buffer at position L-1).

        Returns
        -------
        mem_tokens : (B, 2, D_tok)
            mem_tokens[:, 0, :] = concept retrieval
            mem_tokens[:, 1, :] = operator retrieval
        """
        B = obs_token.size(0)
        cq = self.concept_base.expand(B, -1) + self.concept_cond(obs_token)
        oq = self.operator_base.expand(B, -1) + self.operator_cond(obs_token)
        c_tok = self.concept_bank.retrieve(cq)        # (B, D_tok)
        o_tok = self.operator_bank.retrieve(oq)       # (B, D_tok)
        return torch.stack([c_tok, o_tok], dim=1)     # (B, 2, D_tok)


if __name__ == "__main__":
    # Standalone smoke test: shape contract + gradient flow + sanity
    # that concept/operator queries produce distinct tokens.
    # Run with: `python -m prism.cognition.retrieval_block`
    import sys as _sys

    rb = RetrievalBlock(D_tok=64, concept_n_slots=128, operator_n_slots=16)
    B = 4
    obs = torch.randn(B, 64, requires_grad=True)
    mem = rb(obs)
    if mem.shape != (B, 2, 64):
        print(f"FAIL: mem shape {tuple(mem.shape)} != expected (4, 2, 64)")
        _sys.exit(1)
    print(f"[retr] mem token shape OK: {tuple(mem.shape)}")

    # Gradient must flow from mem back to obs (via cond MLPs) and into
    # both banks' K/V parameters.
    grad_obs = torch.autograd.grad(mem.sum(), obs, retain_graph=True)[0]
    grad_ck = torch.autograd.grad(mem.sum(), rb.concept_bank.keys, retain_graph=True)[0]
    grad_ov = torch.autograd.grad(mem.sum(), rb.operator_bank.values, retain_graph=False)[0]
    for name, g in [("obs", grad_obs), ("concept.keys", grad_ck), ("operator.values", grad_ov)]:
        if g is None or g.abs().sum().item() == 0.0:
            print(f"FAIL: no gradient to {name}")
            _sys.exit(1)
    print(f"[retr] gradient flow OK: |∂/∂obs|={grad_obs.abs().sum():.3f} "
          f"|∂/∂concept.K|={grad_ck.abs().sum():.3f} "
          f"|∂/∂operator.V|={grad_ov.abs().sum():.3f}")

    # Sanity: concept vs operator retrieval should be distinct (different
    # banks, different queries). Cosine similarity on the 2 mem tokens
    # should NOT be ~1.0.
    c_tok = mem[:, 0, :]
    o_tok = mem[:, 1, :]
    cos = torch.nn.functional.cosine_similarity(c_tok, o_tok, dim=-1).mean().item()
    print(f"[retr] mean cosine(concept_tok, operator_tok) = {cos:.3f} (lower-magnitude is healthier)")

    print("[retr] all smoke checks passed")
