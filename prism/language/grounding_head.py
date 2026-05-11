"""GroundingHead — tiny mission-text-to-operator classifier.

Milestone 1.0 of Stage 1 (grounded language). The question this answers:
*given a BabyAI mission text, can we predict which V3 operator the bank
tends to route to during episodes with that mission?*

Architecture is intentionally small (few-K params) so that any success
reflects real text → operator structure, not model capacity. Two modes:

* `bow`:   token-embedding average pool → linear → operator logits.
           Fastest, fewest params, hardest baseline to beat.
* `tiny_tf`: token embedding + single transformer block + mean pool →
             linear head. ~50k params. Used if `bow` plateaus.

A simple whitespace tokenizer is built from the training set. The
vocabulary for BabyAI go-to missions is tiny (~15-20 unique tokens) so
no BPE / subword machinery is needed.

The label is V3.assign(z_t, action) — the hard operator argmax for each
transition, then aggregated to *the most-frequent operator across the
episode* (one label per mission instance) OR per-transition (more
samples, noisier). Both modes supported by the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

PAD_ID = 0
UNK_ID = 1


@dataclass
class WhitespaceVocab:
    """Tiny whitespace tokenizer with PAD=0, UNK=1, then learned tokens."""
    token_to_id: dict[str, int]
    id_to_token: list[str]
    max_len: int

    @classmethod
    def build(cls, texts: list[str], max_len: int = 16) -> "WhitespaceVocab":
        token_to_id: dict[str, int] = {"<pad>": PAD_ID, "<unk>": UNK_ID}
        for t in texts:
            for w in t.strip().lower().split():
                if w not in token_to_id:
                    token_to_id[w] = len(token_to_id)
        id_to_token = [""] * len(token_to_id)
        for w, i in token_to_id.items():
            id_to_token[i] = w
        return cls(token_to_id, id_to_token, max_len)

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        """Returns (ids, mask) padded to max_len. mask=1 for real tokens."""
        words = text.strip().lower().split()[:self.max_len]
        ids = [self.token_to_id.get(w, UNK_ID) for w in words]
        mask = [1] * len(ids)
        pad = self.max_len - len(ids)
        ids = ids + [PAD_ID] * pad
        mask = mask + [0] * pad
        return ids, mask

    def encode_batch(self, texts: list[str]
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        ids, masks = [], []
        for t in texts:
            i, m = self.encode(t)
            ids.append(i)
            masks.append(m)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(masks, dtype=torch.bool),
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "token_to_id": self.token_to_id,
                "id_to_token": self.id_to_token,
                "max_len": self.max_len,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "WhitespaceVocab":
        d = torch.load(path, weights_only=False)
        return cls(d["token_to_id"], d["id_to_token"], d["max_len"])


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class BoWGroundingHead(nn.Module):
    """Token-embedding average pool → linear → operator logits.
    Smallest possible classifier; if this works, the binding is real."""

    def __init__(self, vocab_size: int, n_ops: int, embed_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.head = nn.Linear(embed_dim, n_ops)

    def forward(
        self, token_ids: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.embed(token_ids)                       # (B, L, D)
        mask_f = mask.float().unsqueeze(-1)                # (B, L, 1)
        pooled = (emb * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        return self.head(pooled)                           # (B, K)


class TinyTransformerGroundingHead(nn.Module):
    """One transformer block + mean pool. Use if BoW plateaus."""

    def __init__(
        self,
        vocab_size: int,
        n_ops: int,
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
        self.head = nn.Linear(embed_dim, n_ops)

    def forward(
        self, token_ids: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, L)
        x = self.embed(token_ids) + self.pos(pos)
        # Transformer key_padding_mask = True where masked.
        x = self.block(x, src_key_padding_mask=~mask)
        mask_f = mask.float().unsqueeze(-1)
        pooled = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        return self.head(pooled)


def make_grounding_head(
    kind: str, vocab_size: int, n_ops: int,
) -> nn.Module:
    if kind == "bow":
        return BoWGroundingHead(vocab_size, n_ops)
    if kind == "tiny_tf":
        return TinyTransformerGroundingHead(vocab_size, n_ops)
    raise ValueError(f"unknown grounding head kind: {kind!r}")
