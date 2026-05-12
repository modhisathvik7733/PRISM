"""DomainAdapter Protocol — the contract every domain interface fulfills.

This is a structural type (Protocol), not an ABC. Adapters do NOT need
to subclass anything. They just need to provide the methods listed here
with compatible signatures.

The Protocol is the boundary between substrate and domain. The substrate
talks to adapters; adapters talk to environments. Nothing in
`prism.cognition` may import from `prism.envs` or any domain-specific
module — if it does, that's a contract violation.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import torch
import torch.nn as nn

from prism.cognition.tokens import TokenStream


@runtime_checkable
class DomainAdapter(Protocol):
    """Protocol that every domain adapter must implement.

    Methods are intentionally minimal. Each adapter is responsible for
    composing whatever internal complexity it needs (encoder + tokenizer
    + pose tracker + ...) behind these five entry points.
    """

    name: str
    """Short identifier for logging and checkpoint metadata. Stable
    across versions of the adapter."""

    latent_dim: int
    """Dimension of every emitted token. Must equal substrate D_tok.
    The substrate refuses to attach an adapter with a mismatched dim."""

    mission_dim_max: int
    """Maximum number of MISSION tokens emitted per step. Fixed at
    construction; used to size the rollout buffer. Variable-length
    missions are right-padded by the adapter to this length."""

    n_obs_tokens: int
    """Number of OBS tokens emitted per step. Constant. Combined with
    mission_dim_max, determines per-step adapter output length."""

    def encoder(self) -> nn.Module:
        """The domain-specific observation encoder. Adapter owns the
        weights; substrate does not. PPO optimizes encoder parameters
        alongside substrate parameters when the adapter is attached."""
        ...

    def tokenize(self, obs: Any, mission: Any = None) -> TokenStream:
        """Convert a batched env observation + optional mission into a
        TokenStream of shape (B, n_obs_tokens + mission_dim_max, D_tok).

        Caller passes a structure produced by the env; the adapter is
        responsible for unpacking it. Mission may be `None` for envs
        without a mission concept (substrate emits MISSION-typed
        zero-tokens in that case)."""
        ...

    def action_head(self) -> nn.Module:
        """The action decoder module. Must expose `.dist(h_ctrl)
        -> torch.distributions.Distribution` and `.decode(action_tok)
        -> domain_action`. See `prism.cognition.tokenizer_base.ActionDecoder`
        for the contract."""
        ...

    def mask_logits(
        self,
        logits: torch.Tensor,
        env_state: Any,
    ) -> torch.Tensor:
        """Apply action masking. For discrete action heads, this adds
        `-inf` to logits at indices that are disallowed in the current
        env_state. For continuous heads, this is typically the identity.

        Returns a tensor of the same shape as `logits`. The substrate
        always calls this before sampling — adapters with no masking
        rules return `logits` unchanged.

        Hard invariant (resolution 7): a missing masking call is a
        silent garbage-convergence path. The substrate ALWAYS calls
        this; adapters MUST implement it (returning input unchanged is
        valid for unmaskable spaces)."""
        ...

    def reward_shaper(self) -> Callable[[Any, Any, float], float] | None:
        """Optional per-step reward shaper. Returns None if no shaping.

        When provided, the substrate-side training loop calls it as
        `shaper(prev_obs, action, raw_reward) -> shaped_reward`. Shaping
        is adapter-side because reward semantics are domain-specific
        (BabyAI's `1 - 0.9 * (steps / max_steps)` is not transferable).
        """
        ...
