"""Episodic + semantic memory — Phase 3.

Stub for now. Architecture sketch (to implement once Phases 1–2 work):

  Episodic buffer:
    - sparse, pattern-separated traces of (z_t, a_t, z_{t+1}, instruction, reward)
    - capacity-bounded ring buffer with hash-based pattern separation
      (e.g. random projections to high-dim sparse codes, k-WTA)
    - retrieval: nearest-neighbor on z_t

  Semantic store:
    - dense slot-based store; one slot per stable predicate cluster
    - updated via slow predictive-coding rule (arxiv 2509.01987):
        e_l = mu_l - W_{l,l-1} f(mu_{l-1})        # prediction error
        d mu_l / dt = -e_l + W_{l+1,l}^T e_{l+1}  # bidirectional
    - allows BOTH directions of consolidation (episodic→semantic) AND
      semantic-shaping of episodic encoding (the 2025 result's key claim).

  Controller queries:
    - "what usually happens when..."  → semantic store
    - "have I been here before?"      → episodic buffer

Phase-3 falsifier: ablating the semantic store does NOT hurt long-horizon
recall tasks → consolidation rule isn't doing real work, or the JEPA latent
already memorizes too much (cap target encoder capacity).
"""

from __future__ import annotations


class EpisodicBuffer:
    """Stub. Implemented in Phase 3."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        # TODO(phase-3): pattern-separated codes + ring buffer

    def write(self, *args, **kwargs) -> None:
        raise NotImplementedError("Phase 3 — see prism/models/memory.py module docstring")

    def query(self, *args, **kwargs):
        raise NotImplementedError("Phase 3 — see prism/models/memory.py module docstring")


class SemanticStore:
    """Stub. Implemented in Phase 3."""

    def __init__(self, n_slots: int = 1024, embed_dim: int = 128) -> None:
        self.n_slots = n_slots
        self.embed_dim = embed_dim
        # TODO(phase-3): slot embeddings + bidirectional predictive-coding update

    def consolidate(self, *args, **kwargs) -> None:
        raise NotImplementedError("Phase 3 — see prism/models/memory.py module docstring")

    def query(self, *args, **kwargs):
        raise NotImplementedError("Phase 3 — see prism/models/memory.py module docstring")
