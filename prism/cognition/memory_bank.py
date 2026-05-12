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

        # Frozen mask: True at slot index i means slot i's K and V rows are
        # frozen (no gradient, no Adam step). The mask is a buffer (not a
        # parameter), persisted in state_dict, and authoritative across all
        # writers — CurriculumEngine, ContinualBackprop, manual freeze.
        # Audit pass-2 issue 3a/3c/7e: this is the substrate's only answer
        # to "is this slot mutable?".
        self.register_buffer("frozen_mask", torch.zeros(n_slots, dtype=torch.bool))
        # Backward hooks zero the gradient on frozen rows. This catches
        # ALL paths that produce a gradient on K/V — direct backprop,
        # SparseHopfieldOptimizer, any future loss term. Adam-state
        # zeroing is a separate concern handled by `freeze_slots`.
        self.keys.register_hook(self._zero_frozen_grad)
        self.values.register_hook(self._zero_frozen_grad)

    def _zero_frozen_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Hook applied to keys.grad / values.grad on every backward.
        Zeros the rows corresponding to frozen slots. Idempotent.
        """
        if not self.frozen_mask.any():
            return grad
        # grad shape: (1, n_slots, D_tok); frozen_mask shape: (n_slots,).
        keep = (~self.frozen_mask).to(grad.dtype).view(1, -1, 1)
        return grad * keep

    def freeze_slots(self, slot_idx: torch.Tensor | list[int]) -> None:
        """Mark the given slot indices as frozen. Idempotent.

        After this call:
          - `self.frozen_mask[idx]` is True.
          - Subsequent backward passes zero gradient on K[idx] and V[idx]
            via the hook registered in __init__.
          - Callers must additionally zero the optimizer state for these
            rows when using momentum-based optimizers (Adam/RMSprop) —
            see `freeze_slots_with_optimizer` for the integrated call.
        """
        if isinstance(slot_idx, list):
            slot_idx = torch.tensor(slot_idx, dtype=torch.long, device=self.frozen_mask.device)
        self.frozen_mask[slot_idx] = True

    def freeze_slots_with_optimizer(
        self,
        slot_idx: torch.Tensor | list[int],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Freeze slots AND zero the matching optimizer-state rows.

        Audit pass-2 issue 3a: zeroing K.grad / V.grad is necessary but
        not sufficient when the optimizer carries momentum/variance state
        (Adam, AdamW, RMSprop). Stale `exp_avg` / `exp_avg_sq` would
        still produce non-zero step deltas on frozen rows. This method
        zeros those state tensors row-wise.

        Required at every stage transition. The CurriculumEngine is the
        single writer; ContinualBackpropManager / trainer are read-only
        relative to the freeze decision.
        """
        if isinstance(slot_idx, list):
            slot_idx = torch.tensor(slot_idx, dtype=torch.long, device=self.frozen_mask.device)
        self.frozen_mask[slot_idx] = True

        for p in (self.keys, self.values):
            state = optimizer.state.get(p)
            if not state:
                continue
            for key in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
                buf = state.get(key)
                if buf is not None and buf.shape == p.shape:
                    buf[:, slot_idx, :] = 0.0

    def is_writable(self, slot_idx: int) -> bool:
        """Authoritative answer to 'is this slot mutable?'.
        Audit pass-2 issue 3c: every consumer that writes to K/V (the
        optimizer, ContinualBackprop reanimation, slot reinitialization)
        MUST call this before writing. The frozen mask is the substrate's
        single source of truth.
        """
        return not bool(self.frozen_mask[slot_idx].item())

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

    # Frozen-mask contract: PR-5 step 1 (Phase C foundation).
    # 1. Pre-freeze: gradient flows to every slot.
    # 2. Post-freeze: gradient on frozen rows is exactly zero.
    # 3. After Adam.step: frozen slot weights are bit-identical to pre-step.
    # 4. Adam state (exp_avg, exp_avg_sq) is zero on frozen rows.
    frz_bank = MemoryBank(D_tok=32, n_slots=8, n_heads=2)
    q_frz = torch.randn(3, 32)
    # Warm up Adam state with one optimizer step (pre-freeze) so exp_avg ≠ 0.
    opt = torch.optim.Adam(frz_bank.parameters(), lr=1e-3)
    opt.zero_grad()
    frz_bank.retrieve(q_frz).sum().backward()
    # Pre-freeze: every slot row has nonzero grad.
    g_k_pre = frz_bank.keys.grad.detach().clone()
    g_v_pre = frz_bank.values.grad.detach().clone()
    if g_k_pre.abs().sum(dim=-1).min() == 0.0:
        print("FAIL: pre-freeze keys.grad has a zero row already (some slot got no gradient)")
        _sys.exit(1)
    opt.step()
    # After one step, exp_avg should be nonzero for all rows.
    state = opt.state[frz_bank.keys]
    if state.get("exp_avg") is None or state["exp_avg"].abs().sum(dim=-1).min() == 0.0:
        print("FAIL: Adam exp_avg should be nonzero on all rows pre-freeze")
        _sys.exit(1)
    print(f"[membank] pre-freeze: all 8 slots have grad + Adam state populated")

    # Freeze slots {0, 2, 5} AND zero optimizer state.
    frz_idx = [0, 2, 5]
    frozen_keys_before = frz_bank.keys[:, frz_idx, :].detach().clone()
    frozen_values_before = frz_bank.values[:, frz_idx, :].detach().clone()
    frz_bank.freeze_slots_with_optimizer(frz_idx, opt)

    if not all(frz_bank.is_writable(i) == (i not in frz_idx) for i in range(8)):
        print("FAIL: is_writable() doesn't agree with frozen_mask")
        _sys.exit(1)

    # Adam state for frozen rows must be zero after freeze_slots_with_optimizer.
    state = opt.state[frz_bank.keys]
    if state["exp_avg"][:, frz_idx, :].abs().sum() != 0.0:
        print("FAIL: Adam exp_avg on frozen rows was not zeroed")
        _sys.exit(1)
    if state["exp_avg_sq"][:, frz_idx, :].abs().sum() != 0.0:
        print("FAIL: Adam exp_avg_sq on frozen rows was not zeroed")
        _sys.exit(1)
    print(f"[membank] freeze_slots_with_optimizer: Adam exp_avg/exp_avg_sq zeroed on frozen rows")

    # Backward pass with the frozen mask in place: gradient on frozen rows must be 0.
    opt.zero_grad()
    frz_bank.retrieve(q_frz).sum().backward()
    g_k_post = frz_bank.keys.grad
    if g_k_post[:, frz_idx, :].abs().sum() != 0.0:
        print(f"FAIL: keys.grad on frozen rows = "
              f"{g_k_post[:, frz_idx, :].abs().sum().item()} (expected 0)")
        _sys.exit(1)
    # Unfrozen rows must still have gradient.
    unfrozen = [i for i in range(8) if i not in frz_idx]
    if g_k_post[:, unfrozen, :].abs().sum() == 0.0:
        print("FAIL: keys.grad on unfrozen rows is zero (the hook is too aggressive)")
        _sys.exit(1)
    print(f"[membank] backward with frozen mask: grad = 0 on frozen rows, nonzero on unfrozen")

    # Optimizer step: frozen slot weights must be bit-identical to pre-step.
    opt.step()
    if not torch.equal(frz_bank.keys[:, frz_idx, :], frozen_keys_before):
        print(f"FAIL: keys[frozen_idx] changed after opt.step() despite zero grad + zero state")
        _sys.exit(1)
    if not torch.equal(frz_bank.values[:, frz_idx, :], frozen_values_before):
        print("FAIL: values[frozen_idx] changed after opt.step()")
        _sys.exit(1)
    print(f"[membank] opt.step(): frozen slot weights are bit-identical to pre-step (checksum match)")

    print("[membank] all smoke checks passed")
