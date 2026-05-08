"""PrismLangModel — composes encoder + latent middle + decoder.

Forward signature returns logits for teacher-forced training and a
`generate()` helper for greedy autoregressive decoding at eval time.

Architectural diagram:
    input_ids (B, T_in)
        │
        ▼
    encoder ──► context (B, T_in, D), pad_mask
        │
        ▼
    middle  ──► thoughts (B, K, D), aux_loss (scalar)
        │
        ▼
    decoder ──► logits (B, T_out, vocab)
        ▲
    target_ids (B, T_out) — teacher-forced shift of the answer
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.lang.config import LangConfig
from prism.lang.decoder import ARDecoder
from prism.lang.encoder import ARCnEncoder
from prism.lang.middle import LatentMiddle


class PrismLangModel(nn.Module):
    def __init__(self, cfg: LangConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ARCnEncoder(cfg)
        self.middle = LatentMiddle(cfg)
        # Tie decoder's input embedding to the encoder's so we have a
        # single shared vocab table (saves vocab*d params, standard).
        self.decoder = ARDecoder(cfg, tied_emb=self.encoder.tok_emb)

    def forward(
        self,
        input_ids: torch.Tensor,         # (B, T_in)
        target_ids: torch.Tensor,        # (B, T_out)
    ) -> dict:
        """Teacher-forced forward pass. Returns dict with 'logits',
        'aux_loss', and 'thoughts' for diagnostics."""
        ctx, pad_mask = self.encoder(input_ids)
        thoughts, aux_loss = self.middle(ctx, cross_mask=pad_mask)
        logits = self.decoder(target_ids, thoughts)
        return {"logits": logits, "aux_loss": aux_loss, "thoughts": thoughts}

    def loss(
        self,
        input_ids: torch.Tensor,
        target_ids_in: torch.Tensor,     # decoder inputs (BOS + answer[:-1])
        target_ids_out: torch.Tensor,    # decoder targets (answer + EOS)
        target_mask: torch.Tensor | None = None,  # (B, T_out) — 1 for non-PAD
    ) -> dict:
        """Cross-entropy on `target_ids_out`, masked by `target_mask`,
        plus the JEPA aux loss (weighted)."""
        out = self.forward(input_ids, target_ids_in)
        logits = out["logits"]                                     # (B, T, V)
        B, T, V = logits.shape
        ce = F.cross_entropy(
            logits.reshape(B * T, V),
            target_ids_out.reshape(B * T),
            reduction="none",
        ).view(B, T)
        if target_mask is not None:
            ce = ce * target_mask
            denom = target_mask.sum().clamp(min=1.0)
            ce_loss = ce.sum() / denom
        else:
            ce_loss = ce.mean()
        total = ce_loss + self.cfg.jepa_aux_weight * out["aux_loss"]
        return {
            "loss": total,
            "ce_loss": ce_loss.detach(),
            "aux_loss": out["aux_loss"].detach(),
            "logits": logits,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,         # (B, T_in)
        max_new_tokens: int = 16,
        bos_id: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding. Returns generated ids (B, L)
        excluding the leading BOS."""
        cfg = self.cfg
        bos_id = cfg.bos_token_id if bos_id is None else bos_id
        eos_id = cfg.eos_token_id if eos_id is None else eos_id
        device = input_ids.device
        B = input_ids.shape[0]

        ctx, pad_mask = self.encoder(input_ids)
        thoughts, _ = self.middle(ctx, cross_mask=pad_mask)

        # Start with BOS for every batch item.
        seq = torch.full((B, 1), bos_id, device=device, dtype=torch.long)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            logits = self.decoder(seq, thoughts)              # (B, T, V)
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)  # (B, 1)
            # Once a row hits EOS, force its future tokens to EOS so the
            # tensor stays rectangular (callers strip post-EOS tokens).
            next_tok = torch.where(finished.unsqueeze(1),
                                   torch.full_like(next_tok, eos_id),
                                   next_tok)
            seq = torch.cat([seq, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(1) == eos_id)
            if finished.all():
                break
        return seq[:, 1:]  # drop the leading BOS

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
