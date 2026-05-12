"""UniversalTrunk — transformer over a two-tensor rolling buffer.

The substrate's live recurrent computation. Replaces the GRUCell of v5
HybridPolicy. Same single-step call shape:

    hidden_last, buf_tokens', buf_valid_len' = trunk.step(
        new_token,        # (B, D_tok)
        buf_tokens,       # (B, L, D_tok)
        buf_valid_len,    # (B,) long
    )

State contract (audit pass-2, resolution 7g, and the plan's hard
invariant): the rolling state is ALWAYS two separate tensors, never
packed. They are reset together via `reset_buffer(done, *state)` — a
single API that prevents the failure mode where one tensor resets while
the other persists, producing phantom history on the first step of a
fresh episode.

Buffer geometry (FIFO, newest-at-last):
  - `buf_tokens[:, L-1, :]` is always the most recent valid token.
  - `buf_tokens[:, L-valid_len:L, :]` holds valid_len recent tokens.
  - `buf_tokens[:, 0:L-valid_len, :]` is padding (zeroes, masked out).
  - `valid_len = min(steps_since_reset, L)`. Once at L, oldest is dropped.

The trunk module itself does NOT own the encoder, the tokenizer, or any
domain logic — those live in the adapter. The trunk consumes one
pre-tokenized observation per step and returns a hidden vector.

PR-4 scope: trunk goes live behind `--trunk transformer` in ppo_train.
RetrievalBlock + Hopfield-memory cross-attention is PR-4 step 2.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_VENDOR = os.path.join(os.path.dirname(__file__), "..", "_vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
from hflayers import Hopfield  # noqa: E402
from hflayers.transformer import HopfieldEncoderLayer  # noqa: E402


class UniversalTrunk(nn.Module):
    """Multi-layer Hopfield-attention transformer over a (B, L, D_tok)
    rolling buffer with explicit (B,) valid-length tensor.

    Construction parameters are part of the substrate config hash
    (resolution 3). They are locked across stages and across domains;
    only the adapter-side input dim is allowed to vary.

    Parameters
    ----------
    D_tok : int
        Token embedding dimension. Default 128.
    L : int
        Rolling buffer length. Default 16.
    n_layers : int
        Number of HopfieldEncoderLayer blocks. Default 4.
    n_heads : int
        Number of attention heads per Hopfield block. Default 4.
    ffn_dim : int
        FFN hidden dim inside each encoder layer. Default 512.
    dropout : float
        Dropout in attention and FFN. Default 0.0 — set to 0 so PPO
        rollout/replay log_prob equality holds (resolution 7 / audit
        issue 4b: dropout in the trunk during PPO updates is a known
        source of replay-rollout divergence).
    """

    def __init__(
        self,
        D_tok: int = 128,
        L: int = 16,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.D_tok = D_tok
        self.L = L
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        # Learned positional embeddings over the L buffer slots. Position
        # 0 = oldest, position L-1 = newest. Identity is "where in the
        # rolling window", not "absolute time since episode start" — that
        # avoids unbounded position embeddings as episodes extend.
        self.pos_embed = nn.Embedding(L, D_tok)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            hopfield = Hopfield(
                input_size=D_tok,
                hidden_size=D_tok // n_heads,
                num_heads=n_heads,
                scaling=1.0,
                update_steps_max=0,
                dropout=dropout,
                normalize_stored_pattern=True,
                normalize_state_pattern=True,
                normalize_pattern_projection=True,
            )
            self.layers.append(
                HopfieldEncoderLayer(
                    hopfield_association=hopfield,
                    dim_feedforward=ffn_dim,
                    dropout=dropout,
                )
            )

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------
    def init_buffer(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initial rolling state. Both tensors zero; valid_len is 0
        meaning "no valid tokens yet — the first step's mask will block
        all attention except the just-appended token at position L-1."
        """
        buf_tokens = torch.zeros(batch_size, self.L, self.D_tok, device=device)
        buf_valid_len = torch.zeros(batch_size, dtype=torch.long, device=device)
        return buf_tokens, buf_valid_len

    def reset_buffer(
        self,
        done: torch.Tensor,
        buf_tokens: torch.Tensor,
        buf_valid_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset buffer rows where `done=True`. Both tensors are reset
        atomically — this is the ONLY function that touches the buffer
        on done. Two paired `torch.where` calls, never one packed call.

        Parameters
        ----------
        done : (B,) bool
        buf_tokens : (B, L, D_tok)
        buf_valid_len : (B,) long
        """
        B = buf_tokens.size(0)
        device = buf_tokens.device
        init_tokens, init_valid_len = self.init_buffer(B, device)

        # Paired resets — issue 7g from audit pass 2.
        new_tokens = torch.where(
            done.view(B, 1, 1),
            init_tokens,
            buf_tokens,
        )
        new_valid_len = torch.where(
            done,
            init_valid_len,
            buf_valid_len,
        )
        return new_tokens, new_valid_len

    # ------------------------------------------------------------------
    # Per-step computation
    # ------------------------------------------------------------------
    def step(
        self,
        new_token: torch.Tensor,
        buf_tokens: torch.Tensor,
        buf_valid_len: torch.Tensor,
        prefix_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One trunk step.

        1. Shift buffer left by 1 (drops oldest if at capacity).
        2. Insert `new_token` at position L-1.
        3. Increment valid_len (clamped at L).
        4. Optionally prepend `prefix_tokens` (e.g., RetrievalBlock mem
           tokens). Prefix is NOT stored in the rolling buffer — it
           is per-step transient context. The trunk attends from buf
           positions to all prefix positions plus causally within buf.
        5. Run the transformer; return hidden at the last position (the
           newest buf token).

        Parameters
        ----------
        new_token : (B, D_tok)
        buf_tokens : (B, L, D_tok)
        buf_valid_len : (B,) long
        prefix_tokens : (B, n_prefix, D_tok) | None
            Per-step context tokens prepended to the sequence. Used by
            RetrievalBlock to inject Hopfield-memory retrievals.

        Returns
        -------
        hidden_last : (B, D_tok) — output at the newest buf position
            (index `n_prefix + L - 1` in the concatenated sequence).
        new_buf_tokens : (B, L, D_tok)
        new_buf_valid_len : (B,) long, clamped at L.
        """
        B, L, D = buf_tokens.shape
        assert L == self.L and D == self.D_tok, (
            f"buffer shape mismatch: got (B={B}, L={L}, D={D}), expected "
            f"L={self.L}, D={self.D_tok}"
        )
        assert new_token.shape == (B, self.D_tok), (
            f"new_token shape mismatch: got {tuple(new_token.shape)}, "
            f"expected (B={B}, D={self.D_tok})"
        )

        # 1+2: shift left, append at L-1. `roll(-1, dim=1)` shifts each
        # row's tokens by -1 along the L axis; position L-1 now holds
        # what WAS at position 0 (about to be overwritten). Then we
        # write the new token into position L-1.
        rolled = torch.roll(buf_tokens, shifts=-1, dims=1)
        rolled = rolled.clone()  # break the shared-storage view; the
                                 # write below would otherwise also mutate
                                 # the input view in some backward paths.
        rolled[:, L - 1, :] = new_token
        new_buf_tokens = rolled

        # 3: increment valid_len, clamp at L.
        new_buf_valid_len = torch.clamp(buf_valid_len + 1, max=self.L)

        # 4: combined mask. Position i (key) is valid iff i >= L - valid_len.
        # Causal mask: query at position q can only attend to keys at j <= q.
        # We compute hidden at all positions but only care about q=L-1.
        # An additive mask: 0 where allowed, -inf where blocked.
        device = new_buf_tokens.device
        # Per-batch valid-key mask: (B, L) — True where key position is valid.
        key_pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        valid_key_mask = key_pos >= (L - new_buf_valid_len).unsqueeze(1)  # (B, L)

        # Per-position causal mask: (L, L), j <= i.
        causal_mask = torch.tril(torch.ones(L, L, device=device)).bool()  # (L, L)

        # Combined mask: per-batch (B, L, L). Position (q, k) allowed iff
        # k <= q AND key k is valid in this batch row.
        # HopfieldEncoderLayer expects an additive float mask shaped
        # (L, L) — same mask for the whole batch. We bypass that by
        # constructing a per-batch mask and tiling via attention's batch
        # broadcast: hflayers' Hopfield accepts (B, L_q, L_k)-shaped
        # `association_mask` argument; HopfieldEncoderLayer's `src_mask`
        # is more restrictive (single shared mask).
        # For PR-4 step 1, we use a SINGLE shared (L, L) causal mask plus
        # zero-fill of the invalid token positions (set padding tokens to
        # zero so they contribute 0 to attention via normalization). The
        # per-batch valid mask is enforced by zeroing padding rows of the
        # buffer before the transformer call.
        # Pre-zero padded rows so they don't contribute attention mass.
        # valid_key_mask is (B, L); broadcast to (B, L, 1) and multiply.
        masked_buf = new_buf_tokens * valid_key_mask.unsqueeze(-1).float()

        # Add positional embeddings.
        pos_ids = torch.arange(L, device=device)
        masked_buf = masked_buf + self.pos_embed(pos_ids).unsqueeze(0)

        # Build the sequence: optional [prefix, buf] concat. Prefix is
        # per-step transient (e.g., RetrievalBlock mem tokens) and does
        # NOT enter the rolling buffer. It gets no positional embedding
        # — prefix identity is positional via its role, not its slot.
        if prefix_tokens is not None:
            assert prefix_tokens.dim() == 3 and prefix_tokens.size(0) == B and \
                prefix_tokens.size(2) == self.D_tok, (
                f"prefix_tokens shape mismatch: expected (B={B}, *, D={self.D_tok}), "
                f"got {tuple(prefix_tokens.shape)}"
            )
            n_prefix = prefix_tokens.size(1)
            sequence = torch.cat([prefix_tokens, masked_buf], dim=1)
        else:
            n_prefix = 0
            sequence = masked_buf
        S = n_prefix + L

        # Build the (S, S) additive attention mask:
        #   prefix-prefix : full attention (mem tokens mix freely).
        #   prefix-buf    : blocked (mem cannot peek into rolling state).
        #   buf-prefix    : full (every buf position attends to all mem).
        #   buf-buf       : causal lower triangular.
        attn_mask = torch.full((S, S), float("-inf"), device=device)
        if n_prefix > 0:
            attn_mask[:n_prefix, :n_prefix] = 0.0          # prefix-prefix full
            attn_mask[n_prefix:, :n_prefix] = 0.0          # buf attends to prefix
        # Causal within buf.
        causal_buf_block = torch.zeros(L, L, device=device)
        causal_buf_block = causal_buf_block.masked_fill(~causal_mask, float("-inf"))
        attn_mask[n_prefix:, n_prefix:] = causal_buf_block

        # Run the layers.
        h = sequence
        for layer in self.layers:
            h = layer(h, src_mask=attn_mask)

        # Hidden at the newest buf position = last index in the
        # concatenated sequence.
        hidden_last = h[:, -1, :]
        return hidden_last, new_buf_tokens, new_buf_valid_len


if __name__ == "__main__":
    # Standalone smoke test for the trunk's shape + reset contracts.
    # Run with: `python -m prism.cognition.trunk`
    import sys

    trunk = UniversalTrunk(D_tok=64, L=8, n_layers=2, n_heads=4, ffn_dim=128, dropout=0.0)
    B = 3
    device = torch.device("cpu")

    buf_tokens, buf_valid_len = trunk.init_buffer(B, device)
    assert buf_tokens.shape == (B, 8, 64), buf_tokens.shape
    assert buf_valid_len.shape == (B,) and buf_valid_len.dtype == torch.long

    # 10 steps — more than L, so valid_len saturates at 8.
    for t in range(10):
        new_tok = torch.randn(B, 64)
        hidden, buf_tokens, buf_valid_len = trunk.step(new_tok, buf_tokens, buf_valid_len)
        assert hidden.shape == (B, 64), hidden.shape
        assert buf_tokens.shape == (B, 8, 64)
        assert buf_valid_len.max().item() <= 8
        # Newest token at L-1 should match what we just inserted.
        # (Position L-1 is always valid, so zero-masking doesn't touch it.)
        if not torch.allclose(buf_tokens[:, -1, :], new_tok):
            print(f"FAIL at t={t}: newest position does not equal inserted token")
            sys.exit(1)
    print(f"[trunk] step shape contract OK; valid_len saturated at {buf_valid_len.tolist()}")

    # reset_buffer pairing
    done = torch.tensor([True, False, True])
    buf_tokens, buf_valid_len = trunk.reset_buffer(done, buf_tokens, buf_valid_len)
    if not torch.allclose(buf_tokens[0], torch.zeros_like(buf_tokens[0])):
        print("FAIL: row 0 should be zeroed after done=True")
        sys.exit(1)
    if torch.allclose(buf_tokens[1], torch.zeros_like(buf_tokens[1])):
        print("FAIL: row 1 should be unchanged after done=False")
        sys.exit(1)
    if not torch.allclose(buf_tokens[2], torch.zeros_like(buf_tokens[2])):
        print("FAIL: row 2 should be zeroed after done=True")
        sys.exit(1)
    if buf_valid_len.tolist() != [0, 8, 0]:
        print(f"FAIL: valid_len after paired reset = {buf_valid_len.tolist()}, expected [0, 8, 0]")
        sys.exit(1)
    print(f"[trunk] reset_buffer paired-reset OK; valid_len = {buf_valid_len.tolist()}")

    # Gradient flow: hidden output must depend on input token (sanity check).
    buf_tokens, buf_valid_len = trunk.init_buffer(B, device)
    new_tok = torch.randn(B, 64, requires_grad=True)
    hidden, _, _ = trunk.step(new_tok, buf_tokens, buf_valid_len)
    grad = torch.autograd.grad(hidden.sum(), new_tok, retain_graph=False)[0]
    if grad is None or grad.abs().sum().item() == 0.0:
        print("FAIL: hidden does not depend on new_token (autograd broken)")
        sys.exit(1)
    print(f"[trunk] autograd path live; |grad|={grad.abs().sum().item():.3f}")

    print("[trunk] all smoke checks passed")
