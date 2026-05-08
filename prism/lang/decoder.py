"""AR decoder — token embeddings + causal self-attention + cross-attention
to the latent middle's thought tokens + LM head.

This is the standard transformer decoder block; the only PRISM-specific
choice is that the cross-attention `ctx` is the K thought tokens
produced by `LatentMiddle` (not the raw encoder output). All "language
understanding" therefore must flow through the latent middle — the
decoder cannot peek directly at encoder context.

The LM head's weights are tied to the input token embeddings (standard
GPT trick — saves vocab*d params and slightly improves perplexity).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from prism.lang.config import LangConfig
from prism.lang.transformer import Block, causal_mask


class ARDecoder(nn.Module):
    def __init__(self, cfg: LangConfig, tied_emb: nn.Embedding | None = None):
        super().__init__()
        self.cfg = cfg
        # Either tie the input embedding to the encoder's (when this
        # whole model shares a single token vocab — which it should for
        # all standard configurations) or learn a separate one.
        self.tok_emb = tied_emb if tied_emb is not None \
            else nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.register_buffer(
            "pos_emb",
            self._sinusoidal_pos_emb(cfg.max_seq_len, cfg.d_model),
            persistent=False,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            Block(cfg.d_model, cfg.n_heads, cfg.d_ff,
                  dropout=cfg.dropout, has_cross_attn=True)
            for _ in range(cfg.n_dec_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        # LM head is weight-tied to tok_emb (no separate Linear).
        # We bias is left untied (zero-initialized).
        self.lm_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

    @staticmethod
    def _sinusoidal_pos_emb(seq_len: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, target_ids: torch.Tensor, thoughts: torch.Tensor) -> torch.Tensor:
        """target_ids: (B, T) (teacher-forced); thoughts: (B, K, D).
        Returns logits (B, T, vocab)."""
        B, T = target_ids.shape
        x = self.tok_emb(target_ids) + self.pos_emb[:, :T]
        x = self.dropout(x)
        cmask = causal_mask(T, x.device)
        for blk in self.blocks:
            x = blk(x, ctx=thoughts, self_mask=cmask)
        x = self.final_norm(x)
        # Tied LM head: logits = x @ tok_emb.weight.T + lm_bias
        return x @ self.tok_emb.weight.T + self.lm_bias
