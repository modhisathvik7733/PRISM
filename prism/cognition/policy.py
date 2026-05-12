"""UniversalPolicy — the substrate-side policy entry point.

PR-2 (Phase A) scope: thin wrapper around the existing `HybridPolicy`.
The substrate-side API surface is in place; behavior is identical to v5.

The substrate contract this file enforces:
  * `from_adapter(adapter)` is the ONLY public constructor. There is no
    way to bypass the adapter and feed raw env-specific tensors into
    the policy.
  * `step_with_value(obs, prev_a, mission, h, mem_feat=None)` returns
    `(logits, value, h_next)` in PR-2 (matches v5). The two-tensor
    buffer reshape happens in PR-4 (Phase B). The signature is locked
    here so PR-4 can change the internal hidden type without breaking
    callers.
  * `init_hidden(B, device)` returns a tensor for PR-2 (Phase A
    delegation to HybridPolicy). PR-4 will change to a `(buf_tokens,
    buf_valid_len)` tuple; ppo_train.py is updated in PR-4 to consume
    the tuple.

The encoder is NOT inside this module. It lives in the adapter (resolution
1 in the plan). `step_with_value` calls `adapter.encode_obs(obs)` to get
the JEPA latent before delegating to the internal HybridPolicy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from prism.cognition.trunk import UniversalTrunk

if TYPE_CHECKING:
    from prism.adapters.base import DomainAdapter


class _TransformerInner(nn.Module):
    """Transformer-trunk inner policy: replaces HybridPolicy's GRUCell
    with `UniversalTrunk`. The two-tensor `(buf_tokens, buf_valid_len)`
    rolling state is the substrate's hard invariant (audit pass-2,
    resolution 7g) and is never packed into a single tensor.

    Per-step computation:
        z, prev_a, mission, mem_feat → single D_tok input token
        → trunk.step → hidden at L-1
        → action_head logits + value_head value.

    `mem_feat` is accepted for ppo_train signature compatibility but
    not yet routed; RetrievalBlock cross-attention into Concept/Operator
    memory lands in PR-4 step 3.
    """

    def __init__(
        self,
        latent_in_dim: int,
        n_actions: int,
        mission_dim: int,
        D_tok: int = 128,
        L: int = 16,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 512,
        mem_feat_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_in_dim = latent_in_dim
        self.n_actions = n_actions
        self.mission_dim = mission_dim
        self.D_tok = D_tok
        self.L = L
        self.mem_feat_dim = mem_feat_dim
        # Backwards-compatibility attribute consumed by ppo_train.py;
        # used only for logging in this path (the rollout buffer's hidden
        # storage uses the tuple state directly).
        self.hidden_dim = D_tok
        self.latent_proj_dim = D_tok

        self.latent_proj = nn.Linear(latent_in_dim, D_tok)
        # +1 slot for "no previous action" sentinel (matches v5 convention).
        self.no_action_index = n_actions
        self.action_emb = nn.Embedding(n_actions + 1, D_tok)
        self.mission_proj = nn.Linear(mission_dim, D_tok)
        if mem_feat_dim > 0:
            self.mem_proj: nn.Module = nn.Linear(mem_feat_dim, D_tok)
        else:
            self.mem_proj = nn.Identity()

        self.trunk = UniversalTrunk(
            D_tok=D_tok, L=L, n_layers=n_layers,
            n_heads=n_heads, ffn_dim=ffn_dim, dropout=dropout,
        )

        self.action_head = nn.Linear(D_tok, n_actions)
        self.value_head = nn.Linear(D_tok, 1)

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.trunk.init_buffer(batch_size, device)

    def _build_input_token(
        self,
        z: torch.Tensor,
        prev_action: torch.Tensor,
        mission: torch.Tensor,
        mem_feat: torch.Tensor | None,
    ) -> torch.Tensor:
        # Flatten any spatial dims in z (matches v5 categorical_spatial encoder).
        if z.ndim > 2:
            z = z.flatten(1)
        z_tok = self.latent_proj(z)
        # prev_action == -1 → use the no-action sentinel.
        a_ids = prev_action.clamp(min=0)
        a_ids = torch.where(prev_action < 0, torch.full_like(a_ids, self.no_action_index), a_ids)
        a_tok = self.action_emb(a_ids)
        m_tok = self.mission_proj(mission)
        tok = z_tok + a_tok + m_tok
        if mem_feat is not None and self.mem_feat_dim > 0:
            tok = tok + self.mem_proj(mem_feat)
        return tok

    def step_with_value(
        self,
        z: torch.Tensor,
        prev_action: torch.Tensor,
        mission: torch.Tensor,
        h_prev: tuple[torch.Tensor, torch.Tensor],
        mem_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        buf_tokens, buf_valid_len = h_prev
        new_token = self._build_input_token(z, prev_action, mission, mem_feat)
        hidden_last, new_tokens, new_valid_len = self.trunk.step(
            new_token, buf_tokens, buf_valid_len,
        )
        logits = self.action_head(hidden_last)
        value = self.value_head(hidden_last).squeeze(-1)
        return logits, value, (new_tokens, new_valid_len)

    def step(
        self,
        z: torch.Tensor,
        prev_action: torch.Tensor,
        mission: torch.Tensor,
        h_prev: tuple[torch.Tensor, torch.Tensor],
        mem_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        logits, _value, h_next = self.step_with_value(
            z, prev_action, mission, h_prev, mem_feat=mem_feat
        )
        return logits, h_next

    def reset_buffer(
        self,
        done: torch.Tensor,
        h_prev: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.trunk.reset_buffer(done, *h_prev)


class UniversalPolicy(nn.Module):
    """Substrate-side policy.

    PR-2: delegates to HybridPolicy. PR-4: replaces the internal GRU
    trunk with UniversalTrunk (4× HopfieldEncoderLayer) and switches
    hidden state to the two-tensor buffer.

    Attribute `hidden_dim` is exposed for backwards compatibility with
    ppo_train.py's rollout buffer setup at line 573 (`buf_h_init =
    torch.zeros(T, B, policy.hidden_dim, ...)`). PR-4 will replace this
    with `policy.buffer_shape`.
    """

    def __init__(
        self,
        adapter: "DomainAdapter",
        hybrid_policy: nn.Module,
    ):
        super().__init__()
        self._adapter = adapter
        self._inner = hybrid_policy

        # Expose v5-style attributes for ppo_train.py compatibility.
        # PR-4 phases these out in favor of substrate_config_hash.
        self.hidden_dim: int = int(getattr(hybrid_policy, "hidden_dim", 0))
        self.n_actions: int = int(getattr(hybrid_policy, "n_actions", 0))
        self.mem_feat_dim: int = int(getattr(hybrid_policy, "mem_feat_dim", 0))
        self.latent_proj_dim: int = int(
            getattr(hybrid_policy, "latent_proj_dim", 0)
        )
        self.no_action_index: int = int(
            getattr(hybrid_policy, "no_action_index", self.n_actions)
        )

        # State-kind tag for ppo_train. 'tensor' = single-tensor h (GRU path);
        # 'tuple' = (buf_tokens, buf_valid_len) (transformer path). The trainer
        # branches on this to allocate the rollout buffer correctly and to
        # call the right reset path.
        self.state_kind: str = "tuple" if isinstance(hybrid_policy, _TransformerInner) else "tensor"

    # ------------------------------------------------------------------
    # PR-2 hidden-state contract (single tensor; matches v5)
    # ------------------------------------------------------------------
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initial recurrent state. PR-2: single tensor (matches v5).

        PR-4 changes this to a `(buf_tokens, buf_valid_len)` tuple. The
        signature change is breaking; ppo_train.py is updated in PR-4
        in lockstep.
        """
        return self._inner.init_hidden(batch_size, device)

    # ------------------------------------------------------------------
    # Substrate-side step entry points
    # ------------------------------------------------------------------
    # PR-2 (Phase A) accepts the JEPA latent `z`, matching v5's
    # HybridPolicy.step_with_value exactly. The adapter still OWNS the
    # encoder; ppo_train.py calls `adapter.encode_obs(obs)` explicitly
    # before delegating. This keeps the PR-2 diff in ppo_train.py to a
    # one-line substitution (`jepa.encode` → `adapter.encode_obs`).
    #
    # PR-4 promotes the signature to accept raw obs and internally
    # tokenize through the adapter; both ppo_train.py and this class
    # change together to avoid mid-flight signature drift.
    def step_with_value(
        self,
        z: torch.Tensor,
        prev_action: torch.Tensor,
        mission: torch.Tensor,
        h_prev: torch.Tensor,
        mem_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One PPO step: returns `(logits, value, h_next)`.

        PR-2 pass-through. `z` is the JEPA latent from the adapter's
        encoder. PR-4 will switch this to accept raw `obs` and call
        the adapter internally.
        """
        return self._inner.step_with_value(
            z, prev_action, mission, h_prev, mem_feat=mem_feat,
        )

    def step(
        self,
        z: torch.Tensor,
        prev_action: torch.Tensor,
        mission: torch.Tensor,
        h_prev: torch.Tensor,
        mem_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Step without value head. Used by GroundedAgent in v5 eval."""
        return self._inner.step(z, prev_action, mission, h_prev, mem_feat=mem_feat)

    # ------------------------------------------------------------------
    # State reset (audit pass-2 resolution 7g: single paired-reset API)
    # ------------------------------------------------------------------
    def reset_buffer(
        self,
        done: torch.Tensor,
        h_prev: Any,
    ) -> Any:
        """Reset h on `done`. Single function for both state kinds:
        tensor (GRU path) uses a torch.where between init and h_prev;
        tuple (transformer path) delegates to the trunk's paired reset.

        Substrate guarantee: the trainer NEVER constructs the reset
        directly. This is the only path so the two-tensor case cannot
        partially reset (issue 7g).
        """
        if self.state_kind == "tuple":
            return self._inner.reset_buffer(done, h_prev)
        # tensor path: matches v5 ppo_train.py reset semantics.
        B = h_prev.size(0)
        device = h_prev.device
        h_init = self._inner.init_hidden(B, device)
        return torch.where(done.view(B, 1), h_init, h_prev)

    # ------------------------------------------------------------------
    # Action distribution + masking (substrate-side uniform interface)
    # ------------------------------------------------------------------
    def action_dist(
        self,
        logits: torch.Tensor,
        env_state: Any = None,
    ) -> torch.distributions.Distribution:
        """Apply adapter-side masking, return a Distribution.

        Substrate ALWAYS routes through this method to construct the
        action distribution. Adapters with no masking return logits
        unchanged. This is the resolution-7 hard invariant against
        silent garbage convergence from missing masking.
        """
        masked = self._adapter.mask_logits(logits, env_state)
        return torch.distributions.Categorical(logits=masked)

    # ------------------------------------------------------------------
    # Public construction
    # ------------------------------------------------------------------
    @classmethod
    def from_adapter(
        cls,
        adapter: "DomainAdapter",
        *,
        trunk: str = "gru",
        hidden_dim: int = 256,
        latent_proj_dim: int = 128,
        action_emb_dim: int = 16,
        mem_feat_dim: int = 0,
        policy_type: str = "hybrid",
        concept_n_slots: int = 1024,
        concept_slot_dim: int = 64,
        concept_n_heads: int = 4,
        concept_scaling: float = 1.0,
        operator_n_slots: int = 64,
        operator_slot_dim: int = 64,
        operator_n_heads: int = 4,
        operator_scaling: float = 4.0,
        use_operator_memory: bool = True,
        D_tok: int = 128,
        L: int = 16,
        n_trunk_layers: int = 4,
        n_trunk_heads: int = 4,
        trunk_ffn_dim: int = 512,
    ) -> "UniversalPolicy":
        """Construct a UniversalPolicy. `trunk='gru'` (Phase A) wraps
        HybridPolicy / RecurrentPolicy. `trunk='transformer'` (PR-4)
        builds `_TransformerInner` with `UniversalTrunk` + two-tensor
        rolling state.

        The transformer-trunk hyperparameters (`D_tok`, `L`,
        `n_trunk_*`) are substrate-locked (resolution 3): they cannot
        vary across stages or domains; only adapter-side `latent_dim`
        is allowed to change.
        """
        # Dimensions come from the adapter — substrate never picks them.
        latent_in_dim = adapter.latent_dim
        n_actions = adapter.n_actions
        mission_dim = adapter.mission_dim

        if trunk == "transformer":
            inner: nn.Module = _TransformerInner(
                latent_in_dim=latent_in_dim,
                n_actions=n_actions,
                mission_dim=mission_dim,
                D_tok=D_tok,
                L=L,
                n_layers=n_trunk_layers,
                n_heads=n_trunk_heads,
                ffn_dim=trunk_ffn_dim,
                mem_feat_dim=mem_feat_dim,
                dropout=0.0,
            )
            return cls(adapter=adapter, hybrid_policy=inner)

        if trunk != "gru":
            raise ValueError(
                f"unknown trunk={trunk!r}; expected 'gru' or 'transformer'"
            )

        from prism.models.hybrid_policy import HybridPolicy
        from prism.models.recurrent_policy import RecurrentPolicy

        shared_kwargs: dict[str, Any] = dict(
            latent_in_dim=latent_in_dim,
            n_actions=n_actions,
            mission_dim=mission_dim,
            hidden_dim=hidden_dim,
            latent_proj_dim=latent_proj_dim,
            action_emb_dim=action_emb_dim,
            mem_feat_dim=mem_feat_dim,
        )

        if policy_type == "hybrid":
            inner = HybridPolicy(
                **shared_kwargs,
                concept_n_slots=concept_n_slots,
                concept_slot_dim=concept_slot_dim,
                concept_n_heads=concept_n_heads,
                concept_scaling=concept_scaling,
                operator_n_slots=operator_n_slots,
                operator_slot_dim=operator_slot_dim,
                operator_n_heads=operator_n_heads,
                operator_scaling=operator_scaling,
                use_operator_memory=use_operator_memory,
            )
        elif policy_type == "recurrent":
            inner = RecurrentPolicy(**shared_kwargs)
        else:
            raise ValueError(
                f"unknown policy_type={policy_type!r}; "
                f"expected 'hybrid' or 'recurrent'"
            )

        return cls(adapter=adapter, hybrid_policy=inner)

    # ------------------------------------------------------------------
    # Diagnostic accessors (test-only, not part of the substrate API)
    # ------------------------------------------------------------------
    @property
    def adapter(self) -> "DomainAdapter":
        return self._adapter

    @property
    def inner(self) -> nn.Module:
        """The internal HybridPolicy / RecurrentPolicy. Exposed for
        tests and v5-compatibility shims. PR-4 inlines this away."""
        return self._inner


if __name__ == "__main__":
    # Standalone smoke test for the transformer-trunk path: builds a
    # synthetic adapter (no JEPA load) and exercises init_hidden,
    # step_with_value, action_dist, and reset_buffer.
    # Run with: `python -m prism.cognition.policy`
    import sys

    class _FakeAdapter:
        name = "fake"
        latent_dim = 3136          # matches BabyAI categorical_spatial
        mission_dim = 24            # matches v5 (color, type) one-hot
        n_actions = 7               # MiniGrid
        n_obs_tokens = 1
        mission_dim_max = 1

        def mask_logits(self, logits, env_state=None):
            return logits if env_state is None else logits + env_state

    adapter = _FakeAdapter()
    policy = UniversalPolicy.from_adapter(
        adapter, trunk="transformer",
        D_tok=64, L=8, n_trunk_layers=2, n_trunk_heads=4, trunk_ffn_dim=128,
    )
    print(f"[policy] built UniversalPolicy(trunk=transformer); "
          f"state_kind={policy.state_kind} params={sum(p.numel() for p in policy.parameters()):,}")
    if policy.state_kind != "tuple":
        print(f"FAIL: state_kind expected 'tuple', got {policy.state_kind!r}")
        sys.exit(1)

    B = 4
    device = torch.device("cpu")
    h = policy.init_hidden(B, device)
    if not (isinstance(h, tuple) and len(h) == 2):
        print(f"FAIL: init_hidden should return tuple (buf_tokens, buf_valid_len)")
        sys.exit(1)
    buf_tokens, buf_valid_len = h
    if buf_tokens.shape != (B, 8, 64):
        print(f"FAIL: buf_tokens shape {tuple(buf_tokens.shape)} != expected (4, 8, 64)")
        sys.exit(1)
    if buf_valid_len.shape != (B,) or buf_valid_len.dtype != torch.long:
        print(f"FAIL: buf_valid_len shape/dtype wrong")
        sys.exit(1)
    print(f"[policy] init_hidden returns tuple with shapes "
          f"buf_tokens={tuple(buf_tokens.shape)} buf_valid_len={tuple(buf_valid_len.shape)}")

    # One step_with_value call.
    z = torch.randn(B, 3136)
    prev_a = torch.tensor([-1, 0, 3, 6])
    mission = torch.randn(B, 24)
    logits, value, h_next = policy.step_with_value(z, prev_a, mission, h)
    if logits.shape != (B, 7):
        print(f"FAIL: logits {tuple(logits.shape)} != expected (4, 7)")
        sys.exit(1)
    if value.shape != (B,):
        print(f"FAIL: value {tuple(value.shape)} != expected (4,)")
        sys.exit(1)
    h_next_tokens, h_next_valid_len = h_next
    if h_next_valid_len.tolist() != [1, 1, 1, 1]:
        print(f"FAIL: valid_len after 1 step = {h_next_valid_len.tolist()}, expected [1,1,1,1]")
        sys.exit(1)
    print(f"[policy] step_with_value OK; "
          f"logits={tuple(logits.shape)} value={tuple(value.shape)} valid_len={h_next_valid_len.tolist()}")

    # action_dist + adapter masking (no-op mask).
    dist = policy.action_dist(logits, env_state=None)
    if not isinstance(dist, torch.distributions.Categorical):
        print(f"FAIL: action_dist did not return Categorical")
        sys.exit(1)
    print(f"[policy] action_dist OK; entropy={dist.entropy().mean().item():.3f}")

    # Paired reset.
    done = torch.tensor([True, False, True, False])
    h_after_reset = policy.reset_buffer(done, h_next)
    rt, rvl = h_after_reset
    if rvl.tolist() != [0, 1, 0, 1]:
        print(f"FAIL: valid_len after reset = {rvl.tolist()}, expected [0, 1, 0, 1]")
        sys.exit(1)
    if not torch.allclose(rt[0], torch.zeros_like(rt[0])):
        print("FAIL: tokens for row 0 not zeroed after done=True")
        sys.exit(1)
    if torch.allclose(rt[1], torch.zeros_like(rt[1])):
        print("FAIL: tokens for row 1 should be unchanged after done=False")
        sys.exit(1)
    print(f"[policy] reset_buffer paired-reset OK; valid_len after = {rvl.tolist()}")

    # End-to-end gradient sanity: loss = logits.sum().
    z2 = torch.randn(B, 3136)
    h2 = policy.init_hidden(B, device)
    logits2, value2, _ = policy.step_with_value(z2, prev_a, mission, h2)
    loss = logits2.sum() + value2.sum()
    loss.backward()
    n_params_with_grad = sum(
        1 for p in policy.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    total_params = sum(1 for _ in policy.parameters())
    print(f"[policy] gradient flow OK; {n_params_with_grad}/{total_params} param tensors received nonzero grad")
    if n_params_with_grad == 0:
        print("FAIL: no parameters received gradient")
        sys.exit(1)

    # GRU path still constructible (Phase A regression).
    policy_gru = UniversalPolicy.from_adapter(
        adapter, trunk="gru", policy_type="hybrid",
        hidden_dim=64, latent_proj_dim=64, mem_feat_dim=0,
        concept_n_slots=16, concept_slot_dim=16, operator_n_slots=4, operator_slot_dim=16,
    )
    if policy_gru.state_kind != "tensor":
        print(f"FAIL: gru path state_kind = {policy_gru.state_kind!r}, expected 'tensor'")
        sys.exit(1)
    print(f"[policy] gru path still constructs; state_kind={policy_gru.state_kind}")

    print("[policy] all smoke checks passed")
