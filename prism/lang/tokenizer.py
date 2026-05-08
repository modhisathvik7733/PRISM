"""Thin wrapper over HuggingFace's GPT-2 BPE tokenizer.

We deliberately use a real-world tokenizer (not a custom 200-word vocab)
so the demo's tokenization is identical to what a 1B-param scaling run
would use. Same vocab, same token boundaries, same special tokens.
GPT-2 has no native PAD; we reuse `<|endoftext|>` (id 50256) for PAD,
BOS, and EOS — the model learns to disambiguate from position.
"""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=4)
def get_tokenizer(name: str = "gpt2"):
    """Cached HF tokenizer factory. Network call only on first use; the
    tokenizer is then cached locally by HF (~/.cache/huggingface)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        # Reuse EOS for PAD — standard GPT-2 idiom.
        tok.pad_token = tok.eos_token
    return tok


def encode_batch(
    texts: list[str],
    *,
    tokenizer_name: str = "gpt2",
    max_len: int = 256,
    pad_to_max: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a list of texts → (input_ids, attention_mask) tensors.

    Returns:
        input_ids:      (B, T) int64
        attention_mask: (B, T) int64, 1 for real tokens, 0 for PAD
    """
    tok = get_tokenizer(tokenizer_name)
    enc = tok(
        texts,
        return_tensors="pt",
        padding="max_length" if pad_to_max else "longest",
        truncation=True,
        max_length=max_len,
    )
    return enc["input_ids"], enc["attention_mask"]


def decode_ids(ids: torch.Tensor, tokenizer_name: str = "gpt2",
               skip_special: bool = True) -> str:
    """Decode (T,) or (B, T) ids back to text. For batches, returns a
    space-joined concat — useful only for debugging."""
    tok = get_tokenizer(tokenizer_name)
    if ids.dim() == 1:
        return tok.decode(ids.tolist(), skip_special_tokens=skip_special)
    return tok.batch_decode(ids.tolist(), skip_special_tokens=skip_special)
