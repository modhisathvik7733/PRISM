"""T5-style span corruption for pretraining the encoder→middle→decoder.

The structured architecture is naturally encoder-decoder. The standard
encoder-decoder pretraining objective is span corruption (T5; Raffel
et al. 2020):

    original:    The cat sat on the mat
    corrupted:   The cat <S0> on <S1> mat                 ← encoder input
    target:      <S0> sat <S1> the                        ← decoder target

The model learns to reconstruct the masked spans from bidirectional
context. This forces the encoder to read everything, the middle to
condense it, and the decoder to generate from thoughts — exactly the
information flow we want to test.

Conventions:
- Sentinel tokens: we use the GPT-2 BPE vocab so we have to pick
  unused IDs as sentinels. We reuse the last 100 vocab IDs (50157..50256
  range, well below the EOS at 50256). NOTE: id 50256 is EOS so we use
  50156 down for sentinels.
- Mask ratio: 15% of tokens, in spans of avg length 3 (T5 defaults).
"""

from __future__ import annotations

import numpy as np
import torch


# Sentinel range — we carve out IDs at the end of the vocab. GPT-2's
# vocab is 50257 with id 50256 = <|endoftext|>. We use 50156..50056
# as 100 sentinel slots (S0 = 50156, S1 = 50155, ..., S99 = 50057).
SENTINEL_BASE = 50156
N_SENTINELS = 100


def sentinel_id(k: int) -> int:
    if k < 0 or k >= N_SENTINELS:
        raise ValueError(f"sentinel index {k} out of range [0, {N_SENTINELS})")
    return SENTINEL_BASE - k


def corrupt_sequence(
    token_ids: np.ndarray,
    *,
    rng: np.random.Generator,
    mask_ratio: float = 0.15,
    avg_span_len: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply span corruption to a single sequence.

    Args:
        token_ids: 1-D int array of token IDs.
        rng: numpy Random generator (seeded externally for reproducibility).
        mask_ratio: fraction of tokens to mask (0.15 = T5 default).
        avg_span_len: mean span length (3.0 = T5 default).

    Returns:
        (corrupted_input, decoder_target) as 1-D int arrays.
        corrupted_input  = original tokens with masked spans replaced
                           by single sentinels (sequence is SHORTER)
        decoder_target   = sentinels alternating with the masked spans
                           (also short)
    """
    L = len(token_ids)
    n_to_mask = max(1, int(round(L * mask_ratio)))
    n_spans = max(1, int(round(n_to_mask / avg_span_len)))

    # Sample span START positions uniformly without replacement, then
    # extend each by a length sampled from a geometric distribution
    # truncated to maintain the desired total mask budget. T5 uses a
    # more careful Poisson-distributed generator; this approximation
    # is close enough for our scale.
    if n_spans >= L:
        # Edge case: too few tokens — just mask everything as one span.
        return (
            np.array([sentinel_id(0)], dtype=np.int64),
            np.concatenate([[sentinel_id(0)], token_ids]).astype(np.int64),
        )

    # Pick n_spans random gaps to start spans, then sample lengths.
    starts = sorted(rng.choice(L - 1, size=n_spans, replace=False))
    # Average span length budget per span:
    target_span_len = max(1, n_to_mask // n_spans)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for s in starts:
        if s < cursor:
            continue            # would overlap previous span; drop it
        # Sample length: clamp to fit before next start or sequence end.
        max_len = L - s
        # Prefer geometric sampling around target_span_len.
        length = max(1, int(rng.geometric(p=1.0 / target_span_len)))
        length = min(length, max_len)
        spans.append((s, s + length))
        cursor = s + length

    if len(spans) > N_SENTINELS:
        spans = spans[:N_SENTINELS]

    # Build corrupted_input and decoder_target.
    corrupted: list[int] = []
    target: list[int] = []
    prev_end = 0
    for k, (s, e) in enumerate(spans):
        corrupted.extend(token_ids[prev_end:s].tolist())
        corrupted.append(sentinel_id(k))
        target.append(sentinel_id(k))
        target.extend(token_ids[s:e].tolist())
        prev_end = e
    corrupted.extend(token_ids[prev_end:].tolist())
    # Trailing sentinel marks "end of last span" in T5's recipe.
    target.append(sentinel_id(len(spans)) if len(spans) < N_SENTINELS
                  else sentinel_id(len(spans) - 1))

    return np.array(corrupted, dtype=np.int64), np.array(target, dtype=np.int64)


def corrupt_batch(
    batch_token_ids: list[np.ndarray],
    *,
    rng: np.random.Generator,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    max_in_len: int = 256,
    max_out_len: int = 64,
    mask_ratio: float = 0.15,
    avg_span_len: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Corrupt a batch of sequences and pack into model-ready tensors.

    Returns:
        input_ids   (B, T_in)
        target_in   (B, T_out) — decoder inputs: BOS + target[:-1]
        target_out  (B, T_out) — next-token targets: target + EOS
        target_mask (B, T_out) — 1 at non-PAD, 0 at PAD
    """
    B = len(batch_token_ids)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for ids in batch_token_ids:
        ci, t = corrupt_sequence(ids, rng=rng, mask_ratio=mask_ratio,
                                 avg_span_len=avg_span_len)
        inputs.append(ci[:max_in_len])
        targets.append(t[:max_out_len - 1])  # leave room for trailing EOS

    # Pad input
    in_max = max(len(x) for x in inputs)
    input_ids = np.full((B, in_max), pad_id, dtype=np.int64)
    for i, x in enumerate(inputs):
        input_ids[i, :len(x)] = x

    # Build target_in / target_out
    out_max = max(len(t) for t in targets) + 1   # +1 for EOS
    target_in = np.full((B, out_max), pad_id, dtype=np.int64)
    target_out = np.full((B, out_max), pad_id, dtype=np.int64)
    target_mask = np.zeros((B, out_max), dtype=np.float32)
    for i, t in enumerate(targets):
        L = len(t)
        target_in[i, 0] = bos_id
        target_in[i, 1:1 + L] = t
        target_out[i, :L] = t
        target_out[i, L] = eos_id
        target_mask[i, :L + 1] = 1.0

    return (
        torch.from_numpy(input_ids),
        torch.from_numpy(target_in),
        torch.from_numpy(target_out),
        torch.from_numpy(target_mask),
    )
