"""Dual-head text → (color, type) classifier.

Goal of Stage 1.0-floor: confirm that the text encoder + tokenizer +
training pipeline are *functional* by predicting the most trivially
text-derivable label — the goal object's (color, type) from BabyAI
mission text like "go to the red ball".

If this fails, something is broken at the data / pipeline level and no
operator-binding experiment can work. If it passes, the text encoder is
healthy and the previous operator-binding failure was specifically
because operators aren't goal-shaped (which was the diagnosis from
milestone 1.0 v0).

The model stays domain-general: it's a `text → multi-class` head with
two output spaces (color, type). The notion of "color" and "type" is
BabyAI-specific and lives in the interface/data layer. The grounding
head itself is just a function `tokens → vector_of_logits_per_head`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from prism.language.grounding_head import PAD_ID


class BoWDualHead(nn.Module):
    """Token-embedding mean pool → two linear heads.

    Smallest model that can demonstrate compositional generalization
    in BabyAI-style structured language. If "red" tokens consistently
    nudge the color head toward red and "ball" tokens nudge the type
    head toward ball, then the held-out combo "red ball" should
    classify correctly even if that exact phrase was never seen.
    """

    def __init__(
        self,
        vocab_size: int,
        n_colors: int,
        n_types: int,
        embed_dim: int = 32,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.head_color = nn.Linear(embed_dim, n_colors)
        self.head_type = nn.Linear(embed_dim, n_types)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(token_ids)
        mask_f = mask.float().unsqueeze(-1)
        pooled = (emb * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        return self.head_color(pooled), self.head_type(pooled)


class TinyTransformerDualHead(nn.Module):
    """One transformer block + mean pool + two heads. Use if BoW plateaus."""

    def __init__(
        self,
        vocab_size: int,
        n_colors: int,
        n_types: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        ff_dim: int = 128,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.pos = nn.Embedding(64, embed_dim)
        self.block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="gelu",
        )
        self.head_color = nn.Linear(embed_dim, n_colors)
        self.head_type = nn.Linear(embed_dim, n_types)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, L)
        x = self.embed(token_ids) + self.pos(pos)
        x = self.block(x, src_key_padding_mask=~mask)
        mask_f = mask.float().unsqueeze(-1)
        pooled = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        return self.head_color(pooled), self.head_type(pooled)


def make_dual_head(
    kind: str, vocab_size: int, n_colors: int, n_types: int,
) -> nn.Module:
    if kind == "bow":
        return BoWDualHead(vocab_size, n_colors, n_types)
    if kind == "tiny_tf":
        return TinyTransformerDualHead(vocab_size, n_colors, n_types)
    raise ValueError(f"unknown dual head kind: {kind!r}")
