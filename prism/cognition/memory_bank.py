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
        n_active_init: int | None = None,
    ):
        super().__init__()
        self.D_tok = D_tok
        self.n_slots = n_slots  # pre-allocated CAPACITY (parameter shape)
        self.n_heads = n_heads
        self.scaling = scaling
        self.update_steps = update_steps
        # n_active = currently participating in retrieval. Allows growth
        # via `expand(n_new)` without nn.Parameter reshape / optimizer
        # state surgery. Defaults to full capacity (no growth planned).
        if n_active_init is None:
            n_active_init = n_slots
        if not (0 < n_active_init <= n_slots):
            raise ValueError(
                f"n_active_init={n_active_init} must be in (0, n_slots={n_slots}]"
            )
        self.n_active: int = n_active_init

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
        # Active slots get the v5 small-random init; pre-allocated inactive
        # slots are initialized to "near-null": small K, exactly zero V.
        # Plan spec for capacity growth: "Keys initialized close to a null
        # direction (small norm), values initialized at zero." On expand(),
        # the slot is flipped to active and starts learning from this baseline.
        k_init = torch.randn(1, n_slots, D_tok) * 0.02
        v_init = torch.randn(1, n_slots, D_tok) * 0.02
        # Zero out inactive slot V rows (K is already small-random).
        if n_active_init < n_slots:
            v_init[:, n_active_init:, :] = 0.0
            k_init[:, n_active_init:, :] *= 0.1  # extra-small for inactive
        self.keys = nn.Parameter(k_init)
        self.values = nn.Parameter(v_init)

        # Frozen mask: True at slot index i means slot i's K and V rows are
        # frozen (no gradient, no Adam step). The mask is a buffer (not a
        # parameter), persisted in state_dict, and authoritative across all
        # writers — CurriculumEngine, ContinualBackprop, manual freeze.
        # Audit pass-2 issue 3a/3c/7e: this is the substrate's only answer
        # to "is this slot mutable?".
        self.register_buffer("frozen_mask", torch.zeros(n_slots, dtype=torch.bool))
        # Active mask: True at slot index i means slot i participates in
        # retrieval. Slots beyond n_active are pre-allocated capacity; they
        # only become active via `expand(n_new)`. Slots are zero-initialized
        # so that on first activation their attention contribution starts
        # near the "no information" baseline. Audit pass-2 issue 7a is
        # mitigated here: capacity is always present, growth is just a
        # mask flip; the masked-softmax warmup mechanism (PR-5 step 2b)
        # layers on top.
        active_init = torch.zeros(n_slots, dtype=torch.bool)
        active_init[:n_active_init] = True
        self.register_buffer("active_mask", active_init)
        # Per-slot activation mass: accumulated attention weight received
        # over the lifetime of the bank. Feeds the activation-based
        # freezing decision at stage transitions (resolution 4 from the
        # plan). Reset between stages by the CurriculumEngine; written
        # only when `track_activations=True` is passed to `retrieve`.
        self.register_buffer("activation_mass", torch.zeros(n_slots))
        # Step counter — used to normalize activation_mass into a
        # "fraction of attention received" if the engine wants that.
        self.register_buffer("activation_steps", torch.zeros((), dtype=torch.long))
        # Tracking flag — when True, `retrieve()` accumulates activation
        # statistics implicitly (without the caller passing
        # `track_activations=True` explicitly). The engine toggles this
        # at stage entry/exit so the trainer's hot-path code doesn't
        # need to know about the curriculum.
        self.tracking: bool = False
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

    # ------------------------------------------------------------------
    # Growth (PR-5 step 2) — activate pre-allocated capacity
    # ------------------------------------------------------------------
    def expand(self, n_new: int) -> torch.Tensor:
        """Activate `n_new` previously-inactive slots. Returns the indices
        of the newly-activated slots (so the CurriculumEngine can tag
        their origin / warmup state).

        Raises if there is not enough pre-allocated capacity (n_active +
        n_new > n_slots). Constructor must size n_slots to the maximum
        capacity the curriculum will ever ask for.

        Audit pass-2 issue 5a (expand triggering rule): caller-controlled,
        once per stage transition, fixed n_new per stage. This method
        does NOT trigger itself; the CurriculumEngine is the single
        writer.

        Plan resolution: slot origin is registered at creation (resolution
        5b). With pre-allocated capacity, "creation" = "activation"; the
        engine should call `register_origin(idx, domain)` here.
        """
        if self.n_active + n_new > self.n_slots:
            raise ValueError(
                f"expand(n_new={n_new}) would exceed pre-allocated capacity "
                f"(n_active={self.n_active}, n_slots={self.n_slots}). "
                f"Construct the bank with a larger n_slots to support more growth."
            )
        new_idx = torch.arange(self.n_active, self.n_active + n_new,
                                device=self.active_mask.device)
        self.active_mask[new_idx] = True
        self.n_active += n_new
        return new_idx

    def retrieve_with_attention(
        self,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve plus per-query attention weights over slots.

        Used by E4 (audit issue 3d / resolution 5): top-K activating
        frames per slot, computed by feeding a frozen probe set through
        this method and reading attention. This is the operational
        definition of "stable abstraction" — weights bit-equal + top-K
        frame Jaccard ≥ 0.6 across stage transitions.

        Returns
        -------
        out  : (B, D_tok) — retrieval output, same as `retrieve(query)`.
        attn : (B, n_slots) — softmax weights, averaged over heads.
            Inactive slots are guaranteed to have weight 0 (via
            association_mask).
        """
        B = query.size(0)
        active = self.active_mask.to(self.values.dtype).view(1, -1, 1)
        K = (self.keys * active).expand(B, -1, -1)
        V = (self.values * active).expand(B, -1, -1)
        Q = query.unsqueeze(1)
        triple = (K, Q, V)

        attn_mask = None
        if not bool(self.active_mask.all().item()):
            inactive = (~self.active_mask).view(1, -1)
            attn_mask = torch.zeros(1, self.n_slots, device=query.device)
            attn_mask = attn_mask.masked_fill(inactive, float("-inf"))

        if attn_mask is not None:
            out = self.hopfield(triple, association_mask=attn_mask)
            raw_attn = self.hopfield.get_association_matrix(
                triple, association_mask=attn_mask
            )
        else:
            out = self.hopfield(triple)
            raw_attn = self.hopfield.get_association_matrix(triple)
        # raw_attn: (B, n_heads, N_q=1, n_slots). Average over heads,
        # squeeze the singleton query dim.
        attn = raw_attn.mean(dim=1).squeeze(1)             # (B, n_slots)
        return out.squeeze(1), attn

    def reset_activation_history(self) -> None:
        """Clear activation_mass and activation_steps. Called by the
        CurriculumEngine between stages so that the next stage's
        freezing decision uses only that stage's activation statistics.
        """
        self.activation_mass.zero_()
        self.activation_steps.zero_()

    def slot_activation_fraction(self) -> torch.Tensor:
        """Per-slot mean attention-weight share over the tracked window.
        Returns shape (n_slots,). Slots with zero activation_steps
        return zero. Used by CurriculumEngine to compute the freeze set:
            freeze_idx = where(activation_fraction > threshold)
        """
        steps = self.activation_steps.clamp(min=1)
        return self.activation_mass / steps

    def retrieve(
        self,
        query: torch.Tensor,
        track_activations: bool | None = None,
    ) -> torch.Tensor:
        """Issue a query against the bank.

        Parameters
        ----------
        query : (B, D_tok)
        track_activations : if True, accumulate attention-weight statistics
            into `self.activation_mass` for the curriculum-engine's
            activation-based freezing decision (resolution 4). Default
            False to avoid the double-forward cost during evaluation.

        Returns
        -------
        retrieved : (B, D_tok)
        """
        B = query.size(0)
        # Resolution 5 / audit pass-2 issue 3b — cold-slot leakage:
        # without an explicit mask, Hopfield's softmax over K gives
        # inactive slots ~uniform attention (50% with half the slots
        # inactive — measured). We exclude inactive K rows from the
        # softmax denominator via Hopfield's `association_mask`:
        # additive -inf at inactive positions makes their softmax
        # weight exactly zero. The V-mask is also kept as defense in
        # depth (if association_mask somehow leaks).
        active = self.active_mask.to(self.values.dtype).view(1, -1, 1)
        K = (self.keys * active).expand(B, -1, -1)
        V = (self.values * active).expand(B, -1, -1)
        Q = query.unsqueeze(1)                  # (B, 1, D_tok)
        triple = (K, Q, V)

        # Build association_mask only when there are inactive slots;
        # avoid mask-building overhead on the fully-active hot path.
        attn_mask = None
        if not bool(self.active_mask.all().item()):
            inactive = (~self.active_mask).view(1, -1)            # (1, n_slots)
            attn_mask = torch.zeros(1, self.n_slots, device=query.device)
            attn_mask = attn_mask.masked_fill(inactive, float("-inf"))
            # hflayers Hopfield accepts an (N_q, N_kv)-shaped float
            # association_mask; broadcasts over batch + heads.

        if attn_mask is not None:
            out = self.hopfield(triple, association_mask=attn_mask)
        else:
            out = self.hopfield(triple)
        # out: (B, 1, D_tok)

        # When `track_activations` is None, honor `self.tracking` so the
        # engine can toggle stage-level tracking without the trainer
        # threading a flag through every retrieval call.
        effective_track = (
            track_activations if track_activations is not None else self.tracking
        )
        if effective_track:
            # Hopfield.get_association_matrix returns (B, n_heads, N_q, N_kv).
            # Sum over batch / heads / query length → (n_slots,).
            if attn_mask is not None:
                attn = self.hopfield.get_association_matrix(
                    triple, association_mask=attn_mask
                )
            else:
                attn = self.hopfield.get_association_matrix(triple)
            slot_mass = attn.detach().sum(dim=(0, 1, 2))    # (n_slots,)
            self.activation_mass = self.activation_mass + slot_mass
            self.activation_steps = self.activation_steps + B

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

    # PR-5 step 2 — capacity growth via expand() + activation tracking.
    # 1. Construct with partial capacity active.
    # 2. expand(): flip more bits, returns new indices.
    # 3. retrieve(track_activations=True): activation_mass accumulates.
    # 4. Inactive slots have zero V → they steal some softmax mass but
    #    contribute nothing to the output (acknowledged cold-slot leakage).
    grow_bank = MemoryBank(D_tok=32, n_slots=16, n_active_init=8, n_heads=2)
    assert grow_bank.n_active == 8
    assert grow_bank.active_mask[:8].all() and not grow_bank.active_mask[8:].any()
    print(f"[membank] expand: constructed with n_slots={grow_bank.n_slots} n_active={grow_bank.n_active}")

    # Inactive slot V rows must be exactly zero.
    if grow_bank.values[:, 8:, :].abs().sum().item() != 0.0:
        print("FAIL: inactive slot V rows are not zero-initialized")
        _sys.exit(1)
    print(f"[membank] inactive slot V rows are zero (near-null init)")

    # Run some retrievals with activation tracking; verify activation_mass
    # is concentrated on active slots and zero on inactive slots.
    q_grow = torch.randn(5, 32)
    for _ in range(3):
        grow_bank.retrieve(q_grow, track_activations=True)
    am = grow_bank.activation_mass
    leak = am[8:].abs().sum().item()
    total = am.abs().sum().item()
    leak_frac = leak / max(total, 1e-12)
    print(f"[membank] cold-slot leakage = {leak_frac:.2%} of total attention "
          f"(via Hopfield association_mask, should be ~0)")
    if leak_frac > 0.01:  # 1% tolerance — anything above is a real bug.
        print(f"FAIL: cold-slot leakage {leak_frac:.2%} > 1% threshold. "
              f"The association_mask path is not actually excluding inactive "
              f"slots from softmax. Likely cause: hflayers Hopfield fork uses "
              f"a different kwarg name (try stored_pattern_padding_mask) or "
              f"a different mask shape.")
        _sys.exit(1)
    print(f"[membank] activation tracking: mass on active slots = "
          f"{am[:8].sum().item():.3f}, steps = {grow_bank.activation_steps.item()}")

    # Expand by 4 slots: 8 → 12 active.
    new_idx = grow_bank.expand(4)
    assert grow_bank.n_active == 12
    assert grow_bank.active_mask[:12].all() and not grow_bank.active_mask[12:].any()
    if new_idx.tolist() != [8, 9, 10, 11]:
        print(f"FAIL: expand() returned unexpected indices {new_idx.tolist()}")
        _sys.exit(1)
    print(f"[membank] expand(4): n_active 8 → 12, new indices = {new_idx.tolist()}")

    # Over-expand: should raise.
    try:
        grow_bank.expand(10)  # 12 + 10 > 16
    except ValueError as e:
        print(f"[membank] over-expand correctly raises ValueError: ok")
    else:
        print("FAIL: expand() past capacity did not raise")
        _sys.exit(1)

    # reset_activation_history clears the stats.
    grow_bank.reset_activation_history()
    if grow_bank.activation_mass.abs().sum().item() != 0.0:
        print("FAIL: reset_activation_history did not clear activation_mass")
        _sys.exit(1)
    if grow_bank.activation_steps.item() != 0:
        print("FAIL: reset_activation_history did not clear activation_steps")
        _sys.exit(1)
    print(f"[membank] reset_activation_history: clears mass + step counter")

    print("[membank] all smoke checks passed")
