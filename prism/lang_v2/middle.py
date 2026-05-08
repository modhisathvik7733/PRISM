"""LatentMiddleV2 — Coconut-style continuous reasoning, built from
GPT2Blocks for architectural consistency with the pretrained edges.

Differences from prism/lang/middle.py (v1):
  - Uses GPT2Block internally (so layer norms / MLP / attention all
    match the pretrained encoder/decoder)
  - Operates at GPT-2's d_model (768 for gpt2-small) instead of 256
  - Same K thought tokens × N thinking steps recipe; same EMA-target
    JEPA aux loss option
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from prism.lang_v2.gpt2_backbone import GPT2Block, GPT2Dims


class LatentMiddleV2(nn.Module):
    def __init__(
        self,
        dims: GPT2Dims,
        *,
        n_thought_tokens: int = 16,
        n_thought_steps: int = 6,
        share_weights: bool = True,
        jepa_aux_weight: float = 0.1,
        ema_decay: float = 0.99,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dims = dims
        self.n_thought_tokens = n_thought_tokens
        self.n_thought_steps = n_thought_steps
        self.share_weights = share_weights
        self.jepa_aux_weight = jepa_aux_weight
        self.ema_decay = ema_decay

        # K learned thought-token embeddings + content-conditioned bias
        # from a masked pool of the encoder context.
        self.thought_init = nn.Parameter(
            torch.randn(1, n_thought_tokens, dims.d_model) * 0.02
        )
        self.ctx_to_thought = nn.Linear(dims.d_model, dims.d_model, bias=True)
        nn.init.zeros_(self.ctx_to_thought.weight)
        nn.init.zeros_(self.ctx_to_thought.bias)

        if share_weights:
            self.block = GPT2Block(dims, has_cross_attn=True, dropout=dropout)
            self.blocks = None
        else:
            self.block = None
            self.blocks = nn.ModuleList([
                GPT2Block(dims, has_cross_attn=True, dropout=dropout)
                for _ in range(n_thought_steps)
            ])

        # EMA-target copy for the JEPA aux loss.
        self._has_ema = jepa_aux_weight > 0
        if self._has_ema:
            base = self.block if share_weights else self.blocks[0]
            self.ema_block = copy.deepcopy(base)
            for p in self.ema_block.parameters():
                p.requires_grad_(False)

    def _step(self, t: int, thoughts: torch.Tensor, ctx: torch.Tensor,
              cross_mask: torch.Tensor | None) -> torch.Tensor:
        # No causal mask among the K thought tokens (they're a set,
        # not a sequence in the AR sense).
        if self.share_weights:
            return self.block(thoughts, ctx=ctx, cross_mask=cross_mask)
        return self.blocks[t](thoughts, ctx=ctx, cross_mask=cross_mask)

    @torch.no_grad()
    def update_ema(self) -> None:
        if not self._has_ema:
            return
        live = self.block if self.share_weights else self.blocks[0]
        d = self.ema_decay
        for p_t, p_l in zip(self.ema_block.parameters(), live.parameters()):
            p_t.data.mul_(d).add_(p_l.data, alpha=1.0 - d)

    def forward(
        self,
        ctx: torch.Tensor,                       # (B, T, D) encoder output
        cross_mask: torch.Tensor | None = None,  # (B, 1, 1, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = ctx.shape[0]
        # Mask-aware mean pool — PAD positions don't dilute the bias.
        if cross_mask is not None:
            not_pad = (cross_mask.squeeze(1).squeeze(1) == 0).float().unsqueeze(-1)
            ctx_sum = (ctx * not_pad).sum(dim=1, keepdim=True)
            ctx_count = not_pad.sum(dim=1, keepdim=True).clamp(min=1.0)
            ctx_pool = ctx_sum / ctx_count
        else:
            ctx_pool = ctx.mean(dim=1, keepdim=True)
        thoughts = self.thought_init.expand(B, -1, -1) + self.ctx_to_thought(ctx_pool)

        aux_loss = ctx.new_zeros(())
        n_aux = 0
        for t in range(self.n_thought_steps):
            prev = thoughts
            thoughts = self._step(t, thoughts, ctx, cross_mask)
            if self._has_ema and self.training:
                with torch.no_grad():
                    target = self.ema_block(
                        prev.detach(), ctx=ctx.detach(), cross_mask=cross_mask,
                    )
                aux_loss = aux_loss + ((thoughts - target) ** 2).mean()
                n_aux += 1
        if n_aux > 0:
            aux_loss = aux_loss / n_aux
        return thoughts, aux_loss
