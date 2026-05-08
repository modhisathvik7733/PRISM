"""Reusable Transformer Block — pre-norm, multi-head attention, GELU MLP.

One block supports both:
  - Self-attention only (encoder, middle thought-tokens)
  - Self-attention + cross-attention (decoder, middle attending to encoder ctx)

Causal masking is opt-in via a constructor flag so the same Block class
backs both bidirectional encoder layers and AR decoder layers.

Deliberately vanilla — no FlashAttention, no GQA, no RoPE — because the
demo runs on a single GPU at d=256. Same primitives scale to 70B with no
code change (just bump the dims), and any of those production tricks
(flash, GQA, RoPE) drop in by replacing the attention call here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0):
    """(B, H, Tq, D) × (B, H, Tk, D) → (B, H, Tq, D).

    `mask` is broadcast-added to the attention scores BEFORE softmax —
    use 0.0 for "attend" and -inf for "block".
    """
    B, H, Tq, D = q.shape
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(D)
    if mask is not None:
        scores = scores + mask
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p, training=True)
    return torch.einsum("bhqk,bhkd->bhqd", attn, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=True)  # for cross-attn
        self.q_proj = nn.Linear(d_model, d_model, bias=True)       # for cross-attn
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) → (B, H, T, head_dim)
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, T, head_dim) → (B, T, D)
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def self_attn(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = (self._split_heads(t) for t in qkv)
        out = _scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(self._merge_heads(out))

    def cross_attn(self, x: torch.Tensor, ctx: torch.Tensor,
                   mask: torch.Tensor | None = None) -> torch.Tensor:
        # Q from x, K/V from ctx (the encoder output / memory).
        q = self._split_heads(self.q_proj(x))
        kv = self.kv_proj(ctx).chunk(2, dim=-1)
        k, v = (self._split_heads(t) for t in kv)
        out = _scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(self._merge_heads(out))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Pre-norm transformer block with optional cross-attention.

    Forward signature: `block(x, ctx=None, self_mask=None, cross_mask=None)`.
    When ctx is None, the cross-attention sublayer is skipped entirely
    (so encoder-only callers don't allocate the cross-attn linear layers'
    activations needlessly).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0, has_cross_attn: bool = False):
        super().__init__()
        self.has_cross_attn = has_cross_attn
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        if has_cross_attn:
            self.norm_cross = nn.LayerNorm(d_model)
            self.cross_attn_module = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None,
                self_mask: torch.Tensor | None = None,
                cross_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.attn.self_attn(self.norm1(x), mask=self_mask))
        if self.has_cross_attn and ctx is not None:
            # Use the dedicated cross-attn module (its own qkv/out projections).
            x = x + self.dropout(
                self.cross_attn_module.cross_attn(self.norm_cross(x), ctx, mask=cross_mask)
            )
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """(1, 1, T, T) additive mask with -inf above the diagonal."""
    m = torch.full((seq_len, seq_len), float("-inf"), device=device)
    m = torch.triu(m, diagonal=1)
    return m.unsqueeze(0).unsqueeze(0)


def padding_mask(token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Build a (B, 1, 1, T) additive mask that blocks attention to PAD tokens.

    `token_ids` is (B, T) int64. PAD positions get -inf so softmax skips them.
    """
    pad = (token_ids == pad_id).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
    return pad.float() * float("-inf")
