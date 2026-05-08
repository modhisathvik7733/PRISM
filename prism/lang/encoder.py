"""Bidirectional AR encoder — token embeddings + N self-attention blocks.

"AR" here is a misnomer in the strictest sense: this side of the model
runs full bidirectional attention (no causal mask), like BERT not GPT.
We call it the "AR encoder" because in the broader system AR layers
sit at the language interface (input/output) while the latent middle
does non-AR thinking. The encoder reads the entire input at once.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from prism.lang.config import LangConfig
from prism.lang.transformer import Block, padding_mask


class ARCnEncoder(nn.Module):
    def __init__(self, cfg: LangConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Sinusoidal positional encoding kept as a buffer — works for any
        # max_seq_len without learned param overhead. Swap to RoPE for
        # length-extrapolation later (constant-time change in attention).
        self.register_buffer(
            "pos_emb",
            self._sinusoidal_pos_emb(cfg.max_seq_len, cfg.d_model),
            persistent=False,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            Block(cfg.d_model, cfg.n_heads, cfg.d_ff,
                  dropout=cfg.dropout, has_cross_attn=False)
            for _ in range(cfg.n_enc_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model)

    @staticmethod
    def _sinusoidal_pos_emb(seq_len: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, T, D)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """input_ids: (B, T) int64 → (context (B, T, D), pad_mask (B, 1, 1, T)).

        We return the pad_mask alongside context so downstream layers
        (middle, decoder) can use it for cross-attention masking without
        re-deriving it.
        """
        B, T = input_ids.shape
        x = self.tok_emb(input_ids) + self.pos_emb[:, :T]
        x = self.dropout(x)
        pad_mask = padding_mask(input_ids, self.cfg.pad_token_id)
        for blk in self.blocks:
            x = blk(x, self_mask=pad_mask)
        return self.final_norm(x), pad_mask
