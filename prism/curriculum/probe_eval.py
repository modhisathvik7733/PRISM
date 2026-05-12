"""probe_eval — E4 stability metrics over the frozen probe set.

Audit pass-2 issue 3d (most likely long-term failure mode): a slot's
K and V parameter rows can be frozen and bit-identical across stages,
while the trainable query MLP (in RetrievalBlock) re-routes which slot
fires for a given input. Weight-stability checksums pass; slot
*function* drifts; the inspection metric "stable abstractions" is
satisfied tautologically.

The plan's resolution (committed in resolution 5 / interpretation b):
"weight-stable slots + adaptive query routing". The operational
definition of stability is then:
    1. K/V parameter rows unchanged (bit-equal checksum) — gates here.
    2. Top-K activating probe frames per slot have Jaccard ≥ 0.6 across
       snapshots — operational gate.
    3. JS divergence of per-slot attention distributions over the probe
       set is ≤ 0.1 across snapshots — diagnostic.

This module exposes the pure utilities (Jaccard, JS divergence,
top-K extraction). The driver that runs the probe set through the
substrate to collect attention is the caller's responsibility — it
depends on whether you want per-slot attention from the ConceptMemory
bank, the OperatorMemory bank, or both.
"""

from __future__ import annotations

import torch


def top_k_frames_per_slot(
    attn: torch.Tensor,
    k: int = 50,
) -> torch.Tensor:
    """For each slot, return the indices of the top-k probe frames that
    activate it most.

    Parameters
    ----------
    attn : (N_frames, n_slots) — per-frame attention weights, as returned
        by stacking many calls to `MemoryBank.retrieve_with_attention`.
    k : top-k size.

    Returns
    -------
    top_k_idx : (n_slots, k) long — for each slot, the frame indices in
        descending order of attention weight.
    """
    if attn.dim() != 2:
        raise ValueError(f"attn must be (N_frames, n_slots), got shape {tuple(attn.shape)}")
    N, n_slots = attn.shape
    k_eff = min(k, N)
    # topk across the frame dimension, per slot.
    # attn.t() is (n_slots, N_frames); topk over dim=1 gives top frames per slot.
    _, top_idx = attn.t().topk(k_eff, dim=1)            # (n_slots, k_eff)
    return top_idx


def per_slot_jaccard(
    top_k_a: torch.Tensor,
    top_k_b: torch.Tensor,
) -> torch.Tensor:
    """Per-slot Jaccard overlap between two top-k snapshots.

    Parameters
    ----------
    top_k_a, top_k_b : (n_slots, k) long — top-k frame indices per slot
        from the two snapshots. Must have the same shape.

    Returns
    -------
    jaccard : (n_slots,) float — |A ∩ B| / |A ∪ B| per slot. A slot
        firing for exactly the same probe frames in both snapshots
        scores 1.0; a slot whose top-k frame set has rotated completely
        scores 0.0.
    """
    if top_k_a.shape != top_k_b.shape:
        raise ValueError(
            f"top_k_a {tuple(top_k_a.shape)} != top_k_b {tuple(top_k_b.shape)}"
        )
    n_slots, k = top_k_a.shape
    jaccard = torch.zeros(n_slots)
    for s in range(n_slots):
        set_a = set(top_k_a[s].tolist())
        set_b = set(top_k_b[s].tolist())
        union = set_a | set_b
        if not union:
            jaccard[s] = 1.0   # both empty → vacuously identical
            continue
        jaccard[s] = len(set_a & set_b) / len(union)
    return jaccard


def js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Jensen-Shannon divergence between two distributions over slots.

    Symmetric, bounded in [0, ln(2)]. Lower = more similar. The E4
    diagnostic gate uses median JS ≤ 0.1 (Pass 4 resolution 5).

    Parameters
    ----------
    p, q : tensors representing distributions along `dim`. Each row must
        sum to 1 (we re-normalize defensively for numerical safety).
    dim : axis along which to compute the divergence.

    Returns
    -------
    js : tensor with `dim` reduced; shape matches inputs minus `dim`.
    """
    if p.shape != q.shape:
        raise ValueError(f"p {tuple(p.shape)} != q {tuple(q.shape)}")
    p = p / (p.sum(dim=dim, keepdim=True).clamp(min=eps))
    q = q / (q.sum(dim=dim, keepdim=True).clamp(min=eps))
    m = 0.5 * (p + q)
    # KL(p || m) and KL(q || m). Use eps to avoid log(0).
    p_safe = p.clamp(min=eps)
    q_safe = q.clamp(min=eps)
    m_safe = m.clamp(min=eps)
    kl_pm = (p_safe * (p_safe.log() - m_safe.log())).sum(dim=dim)
    kl_qm = (q_safe * (q_safe.log() - m_safe.log())).sum(dim=dim)
    return 0.5 * (kl_pm + kl_qm)


if __name__ == "__main__":
    # Standalone smoke test for the metric primitives.
    # Run with: `python -m prism.curriculum.probe_eval`
    import sys as _sys

    # Construct a synthetic attention map: (50 frames, 8 slots).
    # Each slot is "tuned" to fire strongly on a specific frame range.
    torch.manual_seed(0)
    N, S = 50, 8
    attn = torch.rand(N, S) * 0.1   # baseline noise
    for s in range(S):
        # Each slot peaks on frames [s*6 : s*6+6].
        attn[s * 6:(s + 1) * 6, s] += 1.0
    attn = attn / attn.sum(dim=1, keepdim=True)    # row-normalize

    top_k = top_k_frames_per_slot(attn, k=6)
    if top_k.shape != (S, 6):
        print(f"FAIL: top_k_frames_per_slot returned {tuple(top_k.shape)}, expected (8, 6)")
        _sys.exit(1)
    # Slot s should have its top-6 frames be [s*6 .. s*6+5].
    for s in range(S):
        expected = set(range(s * 6, (s + 1) * 6))
        got = set(top_k[s].tolist())
        if got != expected:
            print(f"FAIL: slot {s} top-6 = {sorted(got)}, expected {sorted(expected)}")
            _sys.exit(1)
    print(f"[probe_eval] top_k_frames_per_slot OK — each slot's tuned region recovered")

    # Identical snapshot: Jaccard should be 1.0 everywhere.
    j_identical = per_slot_jaccard(top_k, top_k.clone())
    if not torch.allclose(j_identical, torch.ones(S)):
        print(f"FAIL: identical snapshot Jaccard = {j_identical.tolist()}, expected all 1.0")
        _sys.exit(1)
    print(f"[probe_eval] Jaccard(A, A) = 1.0 for all slots")

    # Disjoint snapshot: shift everyone's top-k by 6 (one slot's worth)
    # so no slot's top-6 overlaps with its original.
    top_k_shifted = top_k.roll(shifts=6, dims=0)   # rotate slot indices
    j_disjoint = per_slot_jaccard(top_k, top_k_shifted)
    # Each slot's top-6 = {s*6 .. s*6+5}; after rolling slots, slot s
    # sees what slot (s-1)%S originally had, which has zero overlap with
    # {s*6 .. s*6+5}. So Jaccard should be exactly 0 for all slots.
    if j_disjoint.abs().sum() > 0.0:
        print(f"FAIL: disjoint snapshot Jaccard nonzero somewhere: {j_disjoint.tolist()}")
        _sys.exit(1)
    print(f"[probe_eval] Jaccard(A, rotated_A) = 0.0 for all slots")

    # Partial-overlap test: drop 3 frames per slot, replace with random.
    top_k_perturbed = top_k.clone()
    top_k_perturbed[:, :3] = torch.randint(0, N, (S, 3))
    j_partial = per_slot_jaccard(top_k, top_k_perturbed)
    # Each slot kept 3 of 6 originals. Best case: random replacements
    # introduce 0 new overlaps. Worst case: random replacements happen
    # to match some originals. We expect roughly 0.3-0.5 mean Jaccard.
    mean_j = float(j_partial.mean())
    if not (0.1 <= mean_j <= 0.7):
        print(f"FAIL: partial-overlap mean Jaccard = {mean_j:.3f}, expected in [0.1, 0.7]")
        _sys.exit(1)
    print(f"[probe_eval] partial-overlap mean Jaccard = {mean_j:.3f} (expected ~0.3-0.5)")

    # JS divergence: equal distributions → 0; orthogonal → ~ln(2).
    p = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    js_eq = js_divergence(p, p.clone())
    if float(js_eq) > 1e-6:
        print(f"FAIL: JS(p, p) = {float(js_eq):.6f}, expected ~0")
        _sys.exit(1)
    print(f"[probe_eval] JS(p, p) = {float(js_eq):.2e} (≈0)")

    p_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    p_b = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    js_orth = js_divergence(p_a, p_b)
    ln2 = float(torch.tensor(2.0).log())
    if abs(float(js_orth) - ln2) > 1e-4:
        print(f"FAIL: JS(orthogonal) = {float(js_orth):.4f}, expected ln(2) ≈ {ln2:.4f}")
        _sys.exit(1)
    print(f"[probe_eval] JS(orthogonal) = {float(js_orth):.4f} ≈ ln(2) = {ln2:.4f}")

    print("[probe_eval] all smoke checks passed")
