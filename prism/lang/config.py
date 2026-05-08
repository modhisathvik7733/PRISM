"""Single config dataclass for PRISM-Lang.

Every dimension, depth, and vocab parameter lives here. Scaling from
the 22M-param demo to a 200M-param run is a config change, not a code
change. Loading pretrained GPT-2 weights is also a matter of matching
these dims to GPT-2's (d=768, n_heads=12, n_layers=12).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LangConfig:
    # Vocab — 50257 is the GPT-2 BPE vocab size. Keeping it lets us
    # load pretrained GPT-2 weights into encoder/decoder later if we
    # match the rest of the dims. For tiny demos a smaller vocab would
    # save params but break that compatibility, so we don't.
    vocab_size: int = 50257
    pad_token_id: int = 50256       # GPT-2 has no PAD; we reuse <|endoftext|>
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    max_seq_len: int = 256          # bAbI stories fit comfortably; raise for bigger tasks

    # Width — d_model is shared across encoder, middle, decoder so
    # sequences flow without projection layers. n_heads must divide d_model.
    d_model: int = 256
    n_heads: int = 4                # head_dim = d_model / n_heads = 64
    d_ff: int = 1024                # standard 4x d_model

    # Depth — independent so we can stress-test "more middle vs more
    # encoder" later without architectural changes.
    n_enc_layers: int = 4
    n_dec_layers: int = 4

    # Latent middle — Coconut-style continuous thinking.
    n_thought_tokens: int = 8       # "K" — the working memory size of the middle
    n_thought_steps: int = 6        # "N" — how many times the (shared) middle block fires
    middle_share_weights: bool = True  # True = recurrent / Coconut-style; False = N distinct blocks

    # JEPA-style aux loss (optional). Setting weight to 0 disables it.
    jepa_aux_weight: float = 0.1
    ema_decay: float = 0.99

    # Regularization
    dropout: float = 0.1

    # Tokenizer name (HF). `gpt2` is the canonical small BPE; same
    # tokenizer is used by Llama-style models with extended vocabs.
    tokenizer_name: str = "gpt2"

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, \
            f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
        return self.d_model // self.n_heads

    def scale_to(self, preset: str) -> "LangConfig":
        """Quick scaling presets so the smoke test can verify the
        architecture survives bigger configs without code changes."""
        if preset == "tiny":
            return LangConfig(d_model=128, n_heads=4, d_ff=512,
                              n_enc_layers=2, n_dec_layers=2,
                              n_thought_tokens=4, n_thought_steps=4)
        if preset == "small":   # ~22M params, the default demo
            return LangConfig()
        if preset == "medium":  # ~50M
            return LangConfig(d_model=512, n_heads=8, d_ff=2048,
                              n_enc_layers=8, n_dec_layers=8,
                              n_thought_tokens=16, n_thought_steps=8)
        if preset == "gpt2-compat":  # GPT-2 small dims for pretrained-weight loading
            return LangConfig(d_model=768, n_heads=12, d_ff=3072,
                              n_enc_layers=12, n_dec_layers=12,
                              n_thought_tokens=16, n_thought_steps=8)
        raise ValueError(f"unknown preset: {preset}")
