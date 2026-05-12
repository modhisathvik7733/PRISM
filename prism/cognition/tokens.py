"""Universal token interface for the cognition substrate.

Every input the substrate processes — observations, actions, missions,
memory retrievals — is a typed token with a position embedding. The trunk
attention operates on a single sequence of these tokens regardless of
domain.

Token type registry is small and fixed at the substrate level; adapters
do not extend it. If a domain needs to convey something that isn't an
OBS / MISSION / ACTION / MEM token, it is rejected at the adapter API
rather than silently leaking a new kind into the substrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class TokenType(IntEnum):
    """Substrate-fixed token type taxonomy. Do not extend per-adapter.

    The values are used directly as type-embedding indices, so order is
    load-bearing — appending only is safe; reordering breaks checkpoints.
    """

    OBS = 0       # observation token, emitted by adapter.tokenize
    MISSION = 1   # mission / goal token, emitted by adapter.tokenize
    ACTION = 2    # previous-action token, emitted by UniversalPolicy
    MEM = 3       # memory retrieval token, emitted by RetrievalBlock
    HISTORY = 4   # rolling-buffer history token (carries an older OBS/ACT)

    @classmethod
    def num_types(cls) -> int:
        return max(int(t) for t in cls) + 1


@dataclass(frozen=True)
class TokenStream:
    """A batch of token sequences flowing through the substrate.

    Fields:
        tokens : (B, K, D_tok) float — the actual content embeddings
        types  : (B, K) long       — TokenType for each token
        pos    : (B, K) long       — within-stream position index

    All three are required; the substrate does not infer types or positions.
    The adapter is responsible for setting them. The substrate is
    responsible for combining streams (history + adapter output + retrieval)
    into a single attended sequence.
    """

    tokens: torch.Tensor
    types: torch.Tensor
    pos: torch.Tensor

    def __post_init__(self) -> None:
        # Invariants — checked once at construction, then trusted.
        B1, K1, _ = self.tokens.shape
        B2, K2 = self.types.shape
        B3, K3 = self.pos.shape
        if not (B1 == B2 == B3 and K1 == K2 == K3):
            raise ValueError(
                f"TokenStream shape mismatch: tokens={tuple(self.tokens.shape)} "
                f"types={tuple(self.types.shape)} pos={tuple(self.pos.shape)}"
            )
        if self.types.dtype not in (torch.long, torch.int64):
            raise TypeError(f"types must be long, got {self.types.dtype}")
        if self.pos.dtype not in (torch.long, torch.int64):
            raise TypeError(f"pos must be long, got {self.pos.dtype}")

    @property
    def batch_size(self) -> int:
        return self.tokens.size(0)

    @property
    def length(self) -> int:
        return self.tokens.size(1)

    @property
    def d_tok(self) -> int:
        return self.tokens.size(2)

    def to(self, device: torch.device) -> "TokenStream":
        return TokenStream(
            tokens=self.tokens.to(device),
            types=self.types.to(device),
            pos=self.pos.to(device),
        )

    @staticmethod
    def concat(streams: list["TokenStream"]) -> "TokenStream":
        """Concatenate multiple TokenStreams along the sequence axis.

        Used by UniversalPolicy to combine adapter output, action token,
        memory tokens, and rolling history. All streams must share batch
        size and token dimension; types and positions are concatenated
        as-is (caller is responsible for sane position values across
        sub-streams).
        """
        if not streams:
            raise ValueError("concat requires at least one stream")
        B = streams[0].batch_size
        D = streams[0].d_tok
        for s in streams[1:]:
            if s.batch_size != B:
                raise ValueError(f"batch-size mismatch in concat: {s.batch_size} vs {B}")
            if s.d_tok != D:
                raise ValueError(f"d_tok mismatch in concat: {s.d_tok} vs {D}")
        return TokenStream(
            tokens=torch.cat([s.tokens for s in streams], dim=1),
            types=torch.cat([s.types for s in streams], dim=1),
            pos=torch.cat([s.pos for s in streams], dim=1),
        )
