"""PrismLangV2 — pretrained GPT-2 encoder + JEPA-style middle +
pretrained GPT-2 decoder + tied LM head.

Both encoder and decoder are GPT2Stack instances. They each get
loaded with the same pretrained GPT-2 weights at construction. The
encoder is used WITHOUT a causal mask (bidirectional reading); the
decoder is used WITH causal mask + cross-attention to the latent
middle's K thought tokens.

The decoder NEVER sees the encoder's raw output — it only sees the
middle's K thoughts. This is the same bottleneck v3.0 (prism/lang/)
used; only the edges changed.

Encoder/decoder weights are NOT tied between the two stacks (unlike
T5) — the decoder needs to learn cross-attention to thoughts, which
would conflict with the encoder's pure self-attn role. The token
embedding IS tied across both (and with the LM head) — single shared
vocabulary table.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.lang_v2.gpt2_backbone import (
    GPT2Dims, GPT2Stack, causal_mask, gpt2_dims, load_gpt2_weights,
    padding_mask,
)
from prism.lang_v2.middle import LatentMiddleV2


class PrismLangV2(nn.Module):
    def __init__(
        self,
        backbone_name: str = "gpt2",
        *,
        n_thought_tokens: int = 16,
        n_thought_steps: int = 6,
        middle_share_weights: bool = True,
        jepa_aux_weight: float = 0.1,
        ema_decay: float = 0.99,
        dropout: float = 0.1,
        load_pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.dims = gpt2_dims(backbone_name)

        # Encoder view: bidirectional, NO cross-attention (it's the source)
        self.encoder = GPT2Stack(self.dims, has_cross_attn=False, dropout=dropout)
        # Decoder view: causal self-attn + cross-attention to thoughts
        self.decoder = GPT2Stack(self.dims, has_cross_attn=True, dropout=dropout)

        # Latent middle — the only fully-from-scratch component.
        self.middle = LatentMiddleV2(
            self.dims,
            n_thought_tokens=n_thought_tokens,
            n_thought_steps=n_thought_steps,
            share_weights=middle_share_weights,
            jepa_aux_weight=jepa_aux_weight,
            ema_decay=ema_decay,
            dropout=dropout,
        )

        # Tied input embeddings: decoder.tok_emb references encoder.tok_emb
        # (so we have one vocab table). Pos embeddings stay separate
        # (encoder + decoder may attend over different positions).
        self.decoder.tok_emb = self.encoder.tok_emb
        # LM head bias (head weights tied to embeddings — see forward).
        self.lm_bias = nn.Parameter(torch.zeros(self.dims.vocab_size))

        # GPT-2 has no PAD; reuse <|endoftext|> (50256) for PAD/BOS/EOS.
        self.pad_token_id = 50256
        self.bos_token_id = 50256
        self.eos_token_id = 50256

        if load_pretrained:
            print(f"[lang_v2] loading pretrained {backbone_name} weights into encoder + decoder…")
            load_gpt2_weights(self.encoder, backbone_name)
            load_gpt2_weights(self.decoder, backbone_name)
            # The decoder's token embedding is now tied (shares storage
            # with encoder.tok_emb). After load_gpt2_weights wrote to
            # decoder.tok_emb directly, the tied alias still points there
            # — no issue, both are the same tensor object.
            print("[lang_v2] pretrained weights loaded; cross-attn modules at zero-init")

    # ------------------------------------------------------------------
    # Forward + loss + generate
    # ------------------------------------------------------------------

    def encode(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """input_ids (B, T) → (context (B, T, D), pad_mask (B, 1, 1, T))."""
        pad_mask = padding_mask(input_ids, self.pad_token_id)
        ctx = self.encoder(input_ids, self_mask=pad_mask)
        return ctx, pad_mask

    def think(self, ctx: torch.Tensor, pad_mask: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor]:
        """ctx (B, T, D) → (thoughts (B, K, D), aux_loss scalar)."""
        return self.middle(ctx, cross_mask=pad_mask)

    def decode(self, target_ids: torch.Tensor, thoughts: torch.Tensor) -> torch.Tensor:
        """target_ids (B, T) + thoughts (B, K, D) → logits (B, T, V)."""
        cmask = causal_mask(target_ids.shape[1], target_ids.device)
        h = self.decoder(target_ids, self_mask=cmask, ctx=thoughts)
        # Tied LM head: project via the (shared) token-embedding matrix.
        return h @ self.encoder.tok_emb.weight.T + self.lm_bias

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> dict:
        ctx, pad_mask = self.encode(input_ids)
        thoughts, aux_loss = self.think(ctx, pad_mask)
        logits = self.decode(target_ids, thoughts)
        return {"logits": logits, "aux_loss": aux_loss, "thoughts": thoughts}

    def loss(
        self,
        input_ids: torch.Tensor,
        target_ids_in: torch.Tensor,     # decoder inputs: BOS + answer[:-1]
        target_ids_out: torch.Tensor,    # decoder targets: answer + EOS
        target_mask: torch.Tensor | None = None,
    ) -> dict:
        out = self.forward(input_ids, target_ids_in)
        logits = out["logits"]
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
        total = ce_loss + self.middle.jepa_aux_weight * out["aux_loss"]
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
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Greedy / sampled autoregressive decoding. Returns (B, L)
        excluding the leading BOS."""
        device = input_ids.device
        B = input_ids.shape[0]
        ctx, pad_mask = self.encode(input_ids)
        thoughts, _ = self.think(ctx, pad_mask)

        seq = torch.full((B, 1), self.bos_token_id, device=device, dtype=torch.long)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            logits = self.decode(seq, thoughts)              # (B, T, V)
            next_logits = logits[:, -1]                      # (B, V)
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
                finished.unsqueeze(1), torch.full_like(next_tok, self.eos_token_id), next_tok,
            )
            seq = torch.cat([seq, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(1) == self.eos_token_id)
            if finished.all():
                break
        return seq[:, 1:]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_middle_params(self) -> int:
        return sum(p.numel() for p in self.middle.parameters() if p.requires_grad)
