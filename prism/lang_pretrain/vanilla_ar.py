"""Vanilla AR baseline — decoder-only transformer, matched in params
to PrismLangModel for the head-to-head comparison.

This is the "no JEPA-middle, no encoder-decoder bottleneck" control.
Same vocab, same total trainable params, same training compute. The
ONLY architectural difference vs PrismLangModel is the absence of the
structured middle and the lack of encoder-decoder split — instead,
one continuous AR stream reads the input and continues generating.

Used for the Path B clean comparison: if PrismLangModel beats this
baseline on downstream reasoning at the same total training cost, the
structured middle is buying real signal. If they tie, the structured
middle is decorative.

The transformer Block is reused from prism/lang/transformer.py — same
primitives so we're not benchmarking implementation differences.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.lang.config import LangConfig
from prism.lang.transformer import Block, causal_mask


class VanillaARModel(nn.Module):
    """Decoder-only AR transformer matched to PrismLangModel's params.

    Param accounting (with default config d=256, n_layers=4, mission_dim=24):
      PrismLangModel total: ~24.2M
        - tok_emb (tied)        : 12.9M
        - encoder (4 Blocks)    :  3.2M
        - middle (1 Block + EMA):  1.7M
        - decoder (4 cross-attn):  5.6M
        - misc                  :  0.8M

      VanillaARModel matches by stacking ~12 plain self-attn Blocks
      (no cross-attn modules, ~0.8M each at d=256), giving:
        - tok_emb (tied)        : 12.9M
        - 12 self-attn Blocks   : ~9.6M
        - misc                  :  0.5M
        TOTAL                   : ~23M  ✓ (within 5% of PrismLangModel)
    """

    def __init__(self, cfg: LangConfig, n_layers: int | None = None):
        super().__init__()
        self.cfg = cfg
        # Match the structured model's effective depth: the structured
        # model has 4 enc + 6 middle steps + 4 dec = 14 "passes" through
        # transformer blocks. We approximate with 12 vanilla AR layers
        # (vanilla AR can't "iterate" the same block N times the way the
        # middle does; we trade depth for the recurrence).
        n_layers = n_layers or 12

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.register_buffer(
            "pos_emb",
            self._sinusoidal_pos_emb(cfg.max_seq_len, cfg.d_model),
            persistent=False,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            Block(cfg.d_model, cfg.n_heads, cfg.d_ff,
                  dropout=cfg.dropout, has_cross_attn=False)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model)
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids (B, T) → logits (B, T, V)."""
        B, T = input_ids.shape
        x = self.tok_emb(input_ids) + self.pos_emb[:, :T]
        x = self.dropout(x)
        cmask = causal_mask(T, x.device)
        for blk in self.blocks:
            x = blk(x, self_mask=cmask)
        x = self.final_norm(x)
        return x @ self.tok_emb.weight.T + self.lm_bias

    def loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> dict:
        """Standard next-token prediction loss. input_ids and target_ids
        are SAME-LENGTH; target_ids[i] is the desired token at position i
        (i.e. caller has already shifted)."""
        logits = self.forward(input_ids)
        B, T, V = logits.shape
        ce = F.cross_entropy(
            logits.reshape(B * T, V),
            target_ids.reshape(B * T),
            reduction="none",
        ).view(B, T)
        if target_mask is not None:
            ce = ce * target_mask
            denom = target_mask.sum().clamp(min=1.0)
            ce_loss = ce.sum() / denom
        else:
            ce_loss = ce.mean()
        return {"loss": ce_loss, "ce_loss": ce_loss.detach(), "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,         # (B, T_prompt)
        max_new_tokens: int = 64,
        bos_id: int = 50256,
        eos_id: int = 50256,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Greedy / temperature AR generation. Returns (B, T_prompt + n_generated)."""
        B = prompt_ids.shape[0]
        seq = prompt_ids.clone()
        finished = torch.zeros(B, dtype=torch.bool, device=prompt_ids.device)
        for _ in range(max_new_tokens):
            if seq.shape[1] >= self.cfg.max_seq_len:
                break
            logits = self.forward(seq)
            next_logits = logits[:, -1]
            if temperature > 0:
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_logits, top_k)
                    next_logits = torch.where(
                        next_logits < v[:, -1:], torch.full_like(next_logits, float("-inf")),
                        next_logits,
                    )
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1)
            else:
                next_tok = next_logits.argmax(dim=-1, keepdim=True)
            next_tok = torch.where(
                finished.unsqueeze(1), torch.full_like(next_tok, eos_id), next_tok,
            )
            seq = torch.cat([seq, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(1) == eos_id)
            if finished.all():
                break
        return seq

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
