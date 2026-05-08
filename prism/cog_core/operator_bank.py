"""OperatorBank — discovers reusable operators by clustering JEPA
latent-deltas from PPO rollouts.

Hypothesis: when an agent acts in the world, the JEPA's
(latent_t+1 - latent_t) deltas separate into a small number of
clusters corresponding to "operator types" — e.g. rotate-clockwise
produces a different latent-delta signature than forward-success which
is different from forward-blocked. If this hypothesis is correct, we
can extract those operators from PPO trajectories and use them as
compositional primitives.

Phase 1 starts with K-means as the simplest interpretable approach.
If K-means clusters are too messy (high within-cluster variance,
operators not human-interpretable), the plan is to fall back to
gradient-based clustering per Mixture-of-World-Models
(arXiv 2602.01270, 2026) — that's a Phase 1.5 escalation, not part
of this commit.

Emergence criteria (must all pass for Phase 1):
  - ≥4 distinct operators with low within-cluster variance
  - Each operator interpretable via majority action (e.g. cluster 0
    is ≥80% "action=2" → "forward")
  - Stable across BabyAI envs (centroid cosine sim ≥0.8 between
    GoToLocal-fitted centroids and GoTo-fitted centroids)
  - Operators COMPOSE: applying op_rotate then op_forward via
    apply() should match the JEPA's actual two-action rollout
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class OperatorClusterStats:
    """Per-operator diagnostics emitted by fit()."""
    cluster_id: int
    n_members: int
    centroid: np.ndarray                          # (D,)
    within_var: float                              # mean L2 from members to centroid
    action_distribution: dict[int, float]          # action_id → fraction of members
    dominant_action: int
    purity: float                                  # fraction with dominant action

    def is_interpretable(self, purity_threshold: float = 0.8) -> bool:
        return self.purity >= purity_threshold


class OperatorBank:
    """K-means clustering over latent-deltas.

    Assumes flat-latent JEPAs OR auto-flattens spatial latents. The
    cluster centroids ARE the operators — applying operator k to a
    latent z means `z + centroid[k]` (linear approximation).
    """

    def __init__(self, n_ops: int = 8):
        self.n_ops = n_ops
        self.centroids: np.ndarray | None = None        # (K, D)
        self.cluster_stats: list[OperatorClusterStats] = []
        self._km = None

    def _flatten(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(x.shape[0], -1) if x.ndim > 2 else x

    def fit(
        self,
        latents_t: np.ndarray,                  # (N, ...)
        latents_tp1: np.ndarray,                # (N, ...)
        actions: np.ndarray,                    # (N,) int — used for purity diagnostics only
        seed: int = 0,
    ) -> list[OperatorClusterStats]:
        """K-means on (z_{t+1} - z_t). Returns per-cluster stats."""
        from sklearn.cluster import KMeans

        z_t = self._flatten(latents_t)
        z_tp1 = self._flatten(latents_tp1)
        deltas = z_tp1 - z_t

        km = KMeans(n_clusters=self.n_ops, random_state=seed, n_init=10)
        labels = km.fit_predict(deltas)
        self.centroids = km.cluster_centers_
        self._km = km

        stats: list[OperatorClusterStats] = []
        for k in range(self.n_ops):
            mask = labels == k
            members = deltas[mask]
            cluster_actions = actions[mask]

            within = float(np.linalg.norm(
                members - self.centroids[k], axis=1
            ).mean()) if len(members) > 0 else 0.0

            unique, counts = np.unique(cluster_actions, return_counts=True)
            action_dist = {int(a): float(c / counts.sum())
                           for a, c in zip(unique, counts)}
            dominant = int(unique[counts.argmax()]) if len(unique) > 0 else -1
            purity = float(counts.max() / counts.sum()) if len(unique) > 0 else 0.0

            stats.append(OperatorClusterStats(
                cluster_id=k,
                n_members=int(mask.sum()),
                centroid=self.centroids[k],
                within_var=within,
                action_distribution=action_dist,
                dominant_action=dominant,
                purity=purity,
            ))
        self.cluster_stats = stats
        return stats

    def assign(self, latent_t: np.ndarray, latent_tp1: np.ndarray) -> int:
        """Which operator does this transition match?"""
        if self._km is None:
            raise RuntimeError("OperatorBank not fit() yet")
        delta = (self._flatten(latent_tp1[None]) - self._flatten(latent_t[None]))
        return int(self._km.predict(delta)[0])

    def apply(self, latent_t: np.ndarray, op_id: int) -> np.ndarray:
        """Linear-approximation operator application: z + centroid[k]."""
        if self.centroids is None:
            raise RuntimeError("OperatorBank not fit() yet")
        flat = self._flatten(latent_t[None])[0]
        result = flat + self.centroids[op_id]
        return result.reshape(latent_t.shape)

    def compose(self, latent_t: np.ndarray, op_ids: list[int]) -> np.ndarray:
        """Apply multiple operators in sequence."""
        z = latent_t
        for op in op_ids:
            z = self.apply(z, op)
        return z

    def cross_env_stability(
        self,
        other: "OperatorBank",
        threshold: float = 0.8,
    ) -> dict:
        """Compare this bank's centroids to another bank fit on
        different env data. For each centroid in self, find the closest
        match in `other` by cosine similarity. Stable means high mean
        cosine sim across pairings."""
        if self.centroids is None or other.centroids is None:
            raise RuntimeError("Both banks must be fit() first")
        s_norm = self.centroids / (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-9)
        o_norm = other.centroids / (np.linalg.norm(other.centroids, axis=1, keepdims=True) + 1e-9)
        # For each self-centroid, find best other-centroid by cosine
        sim_matrix = s_norm @ o_norm.T          # (K_self, K_other)
        best_sims = sim_matrix.max(axis=1)
        return {
            "mean_best_cosine": float(best_sims.mean()),
            "min_best_cosine": float(best_sims.min()),
            "n_stable_above_threshold": int((best_sims >= threshold).sum()),
            "per_op_best_cosine": [float(s) for s in best_sims],
        }

    def save(self, path: str) -> None:
        if self.centroids is None:
            raise RuntimeError("Nothing to save — fit() first")
        np.savez(
            path,
            centroids=self.centroids,
            n_ops=self.n_ops,
            stats_dominant_action=np.array([s.dominant_action for s in self.cluster_stats]),
            stats_purity=np.array([s.purity for s in self.cluster_stats]),
            stats_n_members=np.array([s.n_members for s in self.cluster_stats]),
            stats_within_var=np.array([s.within_var for s in self.cluster_stats]),
        )

    @classmethod
    def load(cls, path: str) -> "OperatorBank":
        d = np.load(path)
        bank = cls(n_ops=int(d["n_ops"]))
        bank.centroids = d["centroids"]
        # Don't restore the underlying sklearn KMeans (it would need the
        # original training data); for assign() at load time, refit on
        # any new data. The centroids are the actual operators.
        bank.cluster_stats = [
            OperatorClusterStats(
                cluster_id=k,
                n_members=int(d["stats_n_members"][k]),
                centroid=d["centroids"][k],
                within_var=float(d["stats_within_var"][k]),
                action_distribution={},        # not persisted; informational only
                dominant_action=int(d["stats_dominant_action"][k]),
                purity=float(d["stats_purity"][k]),
            )
            for k in range(bank.n_ops)
        ]
        return bank
