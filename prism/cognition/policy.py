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

if TYPE_CHECKING:
    from prism.adapters.base import DomainAdapter


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
    ) -> "UniversalPolicy":
        """Construct a UniversalPolicy wrapping HybridPolicy or
        RecurrentPolicy with parameters derived from the adapter.

        PR-2 scope: this is the ONLY supported constructor. PR-4
        introduces a transformer-trunk variant and the same factory
        switches on a `--trunk` flag.
        """
        from prism.models.hybrid_policy import HybridPolicy
        from prism.models.recurrent_policy import RecurrentPolicy

        # Dimensions come from the adapter — substrate never picks them.
        latent_in_dim = adapter.latent_dim
        n_actions = adapter.n_actions
        mission_dim = adapter.mission_dim

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
            inner: nn.Module = HybridPolicy(
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
