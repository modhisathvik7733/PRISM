"""Latent middle — Coconut-style continuous reasoning + JEPA-style aux.

Architecture:
  - K learned "thought tokens" initialized as parameters (or projected
    from the encoder's pooled output).
  - One Block (or N distinct Blocks if cfg.middle_share_weights=False)
    repeatedly applied for cfg.n_thought_steps iterations. Each iteration:
        thoughts = Block(thoughts, ctx=encoder_output)
    so the K thoughts cross-attend to the full input on every step.
  - Output: the final thought-token sequence (B, K, d). The decoder
    cross-attends to this — so the decoder's "context" is K thought
    tokens, not the raw encoder output.

JEPA-style aux loss (when cfg.jepa_aux_weight > 0):
  - An EMA copy of the same Block computes "target thoughts" from a
    detached version of the previous step's input.
  - Loss = MSE(predicted_next_thought, ema_target_thought) averaged
    across all (step, thought-position) pairs.
  - This forces the middle's dynamics to be predictively consistent
    with itself, mirroring the V-JEPA / I-JEPA training trick.
  - Weight 0 disables it entirely (pure end-to-end, like Coconut).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from prism.lang.config import LangConfig
from prism.lang.transformer import Block


class LatentMiddle(nn.Module):
    def __init__(self, cfg: LangConfig):
        super().__init__()
        self.cfg = cfg
        # K learned "thought" embeddings. Init small so early training
        # doesn't have huge initial-thought magnitudes washing out signal.
        self.thought_init = nn.Parameter(
            torch.randn(1, cfg.n_thought_tokens, cfg.d_model) * 0.02
        )
        # Projection from a pooled encoder output, mixed into the thought
        # init so the middle has a content-conditioned starting point
        # (not just a fixed parameter every batch).
        self.ctx_to_thought = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

        if cfg.middle_share_weights:
            self.block = Block(cfg.d_model, cfg.n_heads, cfg.d_ff,
                               dropout=cfg.dropout, has_cross_attn=True)
            self.blocks = None
        else:
            self.block = None
            self.blocks = nn.ModuleList([
                Block(cfg.d_model, cfg.n_heads, cfg.d_ff,
                      dropout=cfg.dropout, has_cross_attn=True)
                for _ in range(cfg.n_thought_steps)
            ])

        # EMA target — only built when JEPA aux is enabled. Lazily so we
        # don't pay the memory cost when the user disables it.
        self._has_ema = cfg.jepa_aux_weight > 0
        if self._has_ema:
            base = self.block if cfg.middle_share_weights else self.blocks[0]
            self.ema_block = copy.deepcopy(base)
            for p in self.ema_block.parameters():
                p.requires_grad_(False)

    def _step(self, t: int, thoughts: torch.Tensor, ctx: torch.Tensor,
              cross_mask: torch.Tensor | None) -> torch.Tensor:
        if self.cfg.middle_share_weights:
            return self.block(thoughts, ctx=ctx, cross_mask=cross_mask)
        return self.blocks[t](thoughts, ctx=ctx, cross_mask=cross_mask)

    @torch.no_grad()
    def update_ema(self) -> None:
        """Pull the EMA-target's weights toward the live block. Call once
        per optimizer step (after backward + opt.step)."""
        if not self._has_ema:
            return
        live = self.block if self.cfg.middle_share_weights else self.blocks[0]
        d = self.cfg.ema_decay
        for p_t, p_l in zip(self.ema_block.parameters(), live.parameters()):
            p_t.data.mul_(d).add_(p_l.data, alpha=1.0 - d)

    def forward(
        self,
        ctx: torch.Tensor,                      # (B, T, D)  encoder output
        cross_mask: torch.Tensor | None = None, # (B, 1, 1, T) PAD mask on encoder
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (final thoughts (B, K, D), aux_loss (scalar)).

        aux_loss is 0.0 when JEPA aux is disabled."""
        B = ctx.shape[0]
        # Mix learned thought-init with a content-conditioned bias from
        # the encoder. Use mean-pool over non-PAD positions if available,
        # else simple mean.
        ctx_pool = ctx.mean(dim=1, keepdim=True)            # (B, 1, D)
        thoughts = self.thought_init.expand(B, -1, -1) + self.ctx_to_thought(ctx_pool)

        aux_loss = ctx.new_zeros(())
        n_aux_terms = 0
        for t in range(self.cfg.n_thought_steps):
            prev = thoughts
            thoughts = self._step(t, thoughts, ctx, cross_mask)

            if self._has_ema and self.training:
                with torch.no_grad():
                    target = self.ema_block(prev.detach(), ctx=ctx.detach(),
                                            cross_mask=cross_mask)
                aux_loss = aux_loss + ((thoughts - target) ** 2).mean()
                n_aux_terms += 1

        if n_aux_terms > 0:
            aux_loss = aux_loss / n_aux_terms
        return thoughts, aux_loss
