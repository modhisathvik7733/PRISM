"""GPT-2 backbone — exposes encoder and decoder *views* of a single
shared HuggingFace GPT-2 model.

Why one shared model: GPT-2 is decoder-only, but we use the same blocks
for both edges of the PRISM-Lang stack. The "encoder view" runs the
blocks WITHOUT a causal mask (bidirectional); the "decoder view" runs
them WITH a causal mask + cross-attention to the latent middle's
thought tokens.

For the decoder we have to add cross-attention modules from scratch
(stock GPT-2 has only self-attention). We initialize them so they're a
near-no-op at start: the decoder behaves like vanilla GPT-2 until the
cross-attn weights learn to use the thought tokens.

We keep this thin and explicit rather than depending on transformers'
abstractions — we want to control exactly which weights load, which
freeze, and where the cross-attn injects.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Dims:
    """Dimensions of a GPT-2 variant. Constructor helpers below pick
    the right numbers from HF's `transformers.GPT2Config`."""
    vocab_size: int
    n_positions: int
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def gpt2_dims(name: str = "gpt2") -> GPT2Dims:
    """Get dims for a HuggingFace GPT-2 variant without loading weights."""
    from transformers import GPT2Config
    cfg = GPT2Config.from_pretrained(name)
    return GPT2Dims(
        vocab_size=cfg.vocab_size,
        n_positions=cfg.n_positions,
        d_model=cfg.n_embd,
        n_heads=cfg.n_head,
        d_ff=4 * cfg.n_embd,            # GPT-2's MLP is 4× d_model
        n_layers=cfg.n_layer,
    )


class GPT2Block(nn.Module):
    """One GPT-2 transformer block: pre-LayerNorm → MHA → residual →
    pre-LN → MLP → residual. Optionally adds a cross-attention sublayer
    after self-attn (used only by the decoder view)."""

    def __init__(self, dims: GPT2Dims, has_cross_attn: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        d = dims.d_model
        self.has_cross_attn = has_cross_attn
        self.ln_1 = nn.LayerNorm(d, eps=1e-5)
        self.attn_qkv = nn.Linear(d, 3 * d)            # GPT-2 uses fused QKV (Conv1D in HF)
        self.attn_out = nn.Linear(d, d)
        if has_cross_attn:
            self.ln_cross = nn.LayerNorm(d, eps=1e-5)
            self.cross_q = nn.Linear(d, d)
            self.cross_kv = nn.Linear(d, 2 * d)
            self.cross_out = nn.Linear(d, d)
            # Initialize cross-attn output to zero so the layer is a no-op
            # at step 0 — pretrained GPT-2 behavior is preserved until
            # the cross-attn weights learn something useful.
            nn.init.zeros_(self.cross_out.weight)
            nn.init.zeros_(self.cross_out.bias)
        self.ln_2 = nn.LayerNorm(d, eps=1e-5)
        self.mlp_fc = nn.Linear(d, dims.d_ff)
        self.mlp_proj = nn.Linear(dims.d_ff, d)
        self.dropout = dropout
        self.dims = dims

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.dims.n_heads, self.dims.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.dims.d_model)

    def _attn(self, q, k, v, mask):
        s = torch.einsum("bhqd,bhkd->bhqk", q, k) / (self.dims.head_dim ** 0.5)
        if mask is not None:
            s = s + mask
        a = F.softmax(s, dim=-1)
        if self.training and self.dropout > 0:
            a = F.dropout(a, p=self.dropout)
        return torch.einsum("bhqk,bhkd->bhqd", a, v)

    def forward(self, x: torch.Tensor,
                self_mask: torch.Tensor | None = None,
                ctx: torch.Tensor | None = None,
                cross_mask: torch.Tensor | None = None) -> torch.Tensor:
        # 1. Self-attention
        h = self.ln_1(x)
        qkv = self.attn_qkv(h).chunk(3, dim=-1)
        q, k, v = (self._split_heads(t) for t in qkv)
        attn_out = self._attn(q, k, v, self_mask)
        x = x + self.attn_out(self._merge_heads(attn_out))

        # 2. Cross-attention (decoder view only)
        if self.has_cross_attn and ctx is not None:
            h = self.ln_cross(x)
            qc = self._split_heads(self.cross_q(h))
            kc, vc = (self._split_heads(t) for t in self.cross_kv(ctx).chunk(2, dim=-1))
            cross_out = self._attn(qc, kc, vc, cross_mask)
            x = x + self.cross_out(self._merge_heads(cross_out))

        # 3. MLP
        h = self.ln_2(x)
        x = x + self.mlp_proj(F.gelu(self.mlp_fc(h)))
        return x


class GPT2Stack(nn.Module):
    """Stack of GPT2Blocks + token & position embeddings + final
    LayerNorm. The same module backs both the encoder (no causal mask)
    and the decoder (causal + cross-attn)."""

    def __init__(self, dims: GPT2Dims, *, has_cross_attn: bool,
                 dropout: float = 0.1):
        super().__init__()
        self.dims = dims
        self.has_cross_attn = has_cross_attn
        self.tok_emb = nn.Embedding(dims.vocab_size, dims.d_model)
        self.pos_emb = nn.Embedding(dims.n_positions, dims.d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GPT2Block(dims, has_cross_attn=has_cross_attn, dropout=dropout)
            for _ in range(dims.n_layers)
        ])
        self.ln_f = nn.LayerNorm(dims.d_model, eps=1e-5)

    def forward(self, input_ids: torch.Tensor, *,
                self_mask: torch.Tensor | None = None,
                ctx: torch.Tensor | None = None,
                cross_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x, self_mask=self_mask, ctx=ctx, cross_mask=cross_mask)
        return self.ln_f(x)


def load_gpt2_weights(stack: GPT2Stack, hf_name: str = "gpt2") -> None:
    """Load HuggingFace GPT-2 weights into a GPT2Stack.

    HF GPT-2 uses Conv1D for QKV/MLP projections (transposed weights vs
    nn.Linear). We handle that translation here. Only self-attn / MLP /
    LayerNorm / embedding weights are loaded — cross-attn weights stay
    at their (zero-initialized) defaults.
    """
    from transformers import GPT2LMHeadModel
    src = GPT2LMHeadModel.from_pretrained(hf_name).transformer

    sd = src.state_dict()
    own = stack.state_dict()

    def copy(dst_key: str, src_key: str, transpose: bool = False) -> None:
        t = sd[src_key]
        if transpose:
            t = t.t().contiguous()
        if own[dst_key].shape != t.shape:
            raise RuntimeError(
                f"shape mismatch loading {src_key} → {dst_key}: "
                f"src {tuple(t.shape)} vs dst {tuple(own[dst_key].shape)}"
            )
        own[dst_key].copy_(t)

    # Embeddings
    copy("tok_emb.weight", "wte.weight")
    copy("pos_emb.weight", "wpe.weight")
    # Final norm
    copy("ln_f.weight", "ln_f.weight")
    copy("ln_f.bias", "ln_f.bias")

    # Per-block: LN1, attn QKV, attn out, LN2, MLP. HF Conv1D is (in, out)
    # while nn.Linear stores (out, in), so we transpose.
    for i, blk_name in enumerate([f"h.{i}" for i in range(stack.dims.n_layers)]):
        p = f"blocks.{i}"
        copy(f"{p}.ln_1.weight",   f"{blk_name}.ln_1.weight")
        copy(f"{p}.ln_1.bias",     f"{blk_name}.ln_1.bias")
        copy(f"{p}.attn_qkv.weight", f"{blk_name}.attn.c_attn.weight", transpose=True)
        copy(f"{p}.attn_qkv.bias",   f"{blk_name}.attn.c_attn.bias")
        copy(f"{p}.attn_out.weight", f"{blk_name}.attn.c_proj.weight", transpose=True)
        copy(f"{p}.attn_out.bias",   f"{blk_name}.attn.c_proj.bias")
        copy(f"{p}.ln_2.weight",   f"{blk_name}.ln_2.weight")
        copy(f"{p}.ln_2.bias",     f"{blk_name}.ln_2.bias")
        copy(f"{p}.mlp_fc.weight",   f"{blk_name}.mlp.c_fc.weight",   transpose=True)
        copy(f"{p}.mlp_fc.bias",     f"{blk_name}.mlp.c_fc.bias")
        copy(f"{p}.mlp_proj.weight", f"{blk_name}.mlp.c_proj.weight", transpose=True)
        copy(f"{p}.mlp_proj.bias",   f"{blk_name}.mlp.c_proj.bias")

    stack.load_state_dict(own)


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """(1, 1, T, T) additive mask blocking pos i from seeing pos j>i."""
    m = torch.full((seq_len, seq_len), float("-inf"), device=device)
    m = torch.triu(m, diagonal=1)
    return m.unsqueeze(0).unsqueeze(0)


def padding_mask(token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(B, 1, 1, T) additive mask blocking attention to PAD positions.
    Uses masked_fill so we don't hit the 0*(-inf)=NaN pitfall."""
    pad = (token_ids == pad_id).unsqueeze(1).unsqueeze(1)
    mask = torch.zeros(pad.shape, dtype=torch.float32, device=token_ids.device)
    mask.masked_fill_(pad, float("-inf"))
    return mask
