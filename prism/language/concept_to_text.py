"""ConceptToText — small transformer decoder generating NL from retrieved concepts.

The first PRISM language head that GENERATES, not just consumes. Reads the
top-k active concept slots from ConceptMemory plus the transformer dynamics
hidden state, and produces a natural-language description token by token.

Two training signals:
1. Supervised: when templated text labels are available from rollouts.
2. Cycle consistency: generated text → re-encode → retrieve concepts →
   check it activates the same Hopfield slots that produced the text.

The decoder is small (~3M params) so the whole language head fits in PRISM's
sub-100M target. Vocabulary uses a learned BPE/word vocab seeded from PRISM's
existing TokenizedVocab — same vocab used by the grounding head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptToText(nn.Module):
    """Transformer decoder: (retrieved_concepts, hidden_state) → text tokens.

    Cross-attention attends over the concept memory contents; self-attention
    handles autoregressive generation.
    """

    def __init__(
        self,
        vocab_size: int = 2048,
        concept_dim: int = 64,
        hidden_dim: int = 192,
        n_layers: int = 3,
        n_heads: int = 6,
        ffn_dim: int = 384,
        max_len: int = 48,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.concept_dim = concept_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)

        # Project external memory (concepts + hidden state) into hidden_dim.
        self.concept_proj = nn.Linear(concept_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)

        # Standard transformer decoder stack.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

        # Tie input and output embeddings for parameter efficiency.
        self.output_head.weight = self.token_embed.weight

    def _build_memory(
        self,
        retrieved_concepts: torch.Tensor,  # (B, K, concept_dim)
        trunk_hidden: torch.Tensor | None,  # (B, hidden_dim) optional
    ) -> torch.Tensor:
        """Combine retrieved concepts and trunk hidden into a memory sequence."""
        mem = self.concept_proj(retrieved_concepts)  # (B, K, hidden_dim)
        if trunk_hidden is not None:
            h_proj = self.hidden_proj(trunk_hidden).unsqueeze(1)  # (B, 1, hidden_dim)
            mem = torch.cat([h_proj, mem], dim=1)  # prepend
        return mem

    def forward(
        self,
        retrieved_concepts: torch.Tensor,
        trunk_hidden: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced training pass.

        retrieved_concepts: (B, K, concept_dim) — top-k from ConceptMemory
        trunk_hidden: (B, hidden_dim) — TransformerDynamics last-step hidden
        text_tokens: (B, T) long — target tokens; if None, returns nothing
                     (use .generate() instead)

        Returns: (B, T, vocab_size) logits.
        """
        if text_tokens is None:
            raise ValueError("Pass text_tokens for training, or use .generate()")

        memory = self._build_memory(retrieved_concepts, trunk_hidden)

        B, T = text_tokens.shape
        device = text_tokens.device

        tok = self.token_embed(text_tokens)
        pos = self.pos_embed(torch.arange(T, device=device)).unsqueeze(0)
        tgt = tok + pos

        # Causal mask for autoregressive decoder self-attention.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        # Padding mask (True at padded positions).
        tgt_key_padding_mask = (text_tokens == self.pad_idx)

        h = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.output_head(h)
        return logits

    @torch.no_grad()
    def generate(
        self,
        retrieved_concepts: torch.Tensor,
        trunk_hidden: torch.Tensor | None = None,
        max_len: int | None = None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Greedy or sampled autoregressive generation.

        Returns: (B, generated_len) token sequences ending in EOS or padded.
        """
        max_len = max_len or self.max_len
        memory = self._build_memory(retrieved_concepts, trunk_hidden)
        B = memory.size(0)
        device = memory.device

        tokens = torch.full(
            (B, 1), self.bos_idx, dtype=torch.long, device=device
        )

        for _ in range(max_len - 1):
            T = tokens.size(1)
            tok = self.token_embed(tokens)
            pos = self.pos_embed(torch.arange(T, device=device)).unsqueeze(0)
            tgt = tok + pos
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
            h = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal_mask)
            last_logits = self.output_head(h[:, -1, :])  # (B, vocab)

            if temperature > 0.0:
                probs = F.softmax(last_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = last_logits.argmax(dim=-1, keepdim=True)

            tokens = torch.cat([tokens, next_tok], dim=1)

            # Stop if all sequences emitted EOS.
            if (next_tok == self.eos_idx).all():
                break

        return tokens

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "concept_dim": self.concept_dim,
            "hidden_dim": self.hidden_dim,
            "max_len": self.max_len,
            "pad_idx": self.pad_idx,
            "bos_idx": self.bos_idx,
            "eos_idx": self.eos_idx,
        }, path)

    @classmethod
    def load(cls, path: str, device: torch.device) -> "ConceptToText":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        # Filter init kwargs to those the constructor accepts.
        init_keys = {
            "vocab_size", "concept_dim", "hidden_dim",
            "max_len", "pad_idx", "bos_idx", "eos_idx",
        }
        kwargs = {k: v for k, v in ckpt.items() if k in init_keys}
        m = cls(**kwargs)
        m.load_state_dict(ckpt["state_dict"])
        m.to(device)
        return m
