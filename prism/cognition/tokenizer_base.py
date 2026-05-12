"""Abstract base classes for domain-side tokenizers and action decoders.

The substrate does not depend on these directly — it depends on the
`DomainAdapter` Protocol in `prism.adapters.base`. These ABCs are
provided as scaffolding for adapter implementations so that:

  * Type checkers can verify adapter compliance.
  * Common cross-adapter logic (e.g. position assignment, mission padding)
    can be shared without forcing inheritance.
  * Tests can mock the interfaces cleanly.

Adapters MAY subclass these. They are NOT required to — the
`DomainAdapter` Protocol (in `prism.adapters.base`) is a structural type,
so any class providing the right methods qualifies.
"""

from __future__ import annotations

import abc
from typing import Any

import torch
import torch.nn as nn

from prism.cognition.tokens import TokenStream


class ObservationTokenizer(abc.ABC):
    """Domain-specific tokenizer turning raw env obs into a TokenStream.

    A tokenizer is responsible for:
      * Running the domain's observation encoder (a `nn.Module` it owns).
      * Producing OBS-typed tokens.
      * Optionally producing MISSION-typed tokens from the goal description.
      * Setting per-token positions consistently across calls.

    The tokenizer is the SOLE owner of the observation encoder; the
    substrate has no reference to it and cannot accidentally share it
    across adapters. This is the encoder-as-adapter hard invariant
    (resolution 1 in the plan).
    """

    @property
    @abc.abstractmethod
    def latent_dim(self) -> int:
        """Dimension of each emitted token. Must equal substrate D_tok."""

    @property
    @abc.abstractmethod
    def mission_dim_max(self) -> int:
        """Maximum number of MISSION tokens this adapter ever emits.

        Used to size the rollout buffer. Variable-length missions must
        be right-padded to this length before emission; the adapter MAY
        emit fewer real mission tokens with `types == MISSION` and pad
        the rest with `types == HISTORY` (signaling "ignore") — but the
        TOTAL token count per step must be deterministic across a run.
        """

    @abc.abstractmethod
    def encoder(self) -> nn.Module:
        """Return the domain-specific observation encoder.

        Owned by the adapter. The substrate's optimizer optimizes this
        module's parameters jointly with substrate parameters when the
        adapter is attached, but the substrate never instantiates it.
        """

    @abc.abstractmethod
    def tokenize(self, obs: Any, mission: Any = None) -> TokenStream:
        """Convert raw observation + optional mission into a TokenStream.

        Returned stream has shape (B, K, D_tok) where K = n_obs_tokens +
        mission_dim_max (constant for this adapter).
        """


class ActionDecoder(abc.ABC):
    """Domain-specific decoder turning a control vector into an action.

    Used by `UniversalPolicy` after the trunk produces `h_ctrl`. The
    decoder is a `nn.Module` so it has trainable parameters; PPO updates
    these jointly with substrate parameters. The decoder's
    `dist(h_ctrl)` method must return a `torch.distributions.Distribution`
    so PPO can call `.log_prob()` and `.entropy()` uniformly across
    discrete-small, discrete-large, continuous, and structured action
    spaces.

    Action masking (for envs that disallow certain actions in certain
    states) is the adapter's responsibility — see
    `DomainAdapter.mask_logits` — not the decoder's. This split lets
    the same decoder be shared by adapters with different masking rules.
    """

    @abc.abstractmethod
    def head(self) -> nn.Module:
        """Return the nn.Module implementing the head. Same as `self`
        for typical implementations; kept separate to allow non-Module
        decoders (e.g. for testing)."""

    @abc.abstractmethod
    def dist(self, h_ctrl: torch.Tensor) -> torch.distributions.Distribution:
        """Return the action distribution given the trunk control vector.

        `h_ctrl` has shape (B, D_tok). Returned distribution's
        `.sample()` and `.log_prob(action)` must produce shapes
        consistent with the env's action_space.
        """

    @abc.abstractmethod
    def decode(self, action_tok: torch.Tensor) -> Any:
        """Convert a sampled action tensor into a domain-side action object.

        For discrete-small (BabyAI), this is `int(action_tok.item())`.
        For continuous (robotics), it might unpack a (mu, sigma) into a
        float array. The substrate never inspects the result.
        """
