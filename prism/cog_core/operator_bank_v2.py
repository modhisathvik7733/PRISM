"""OperatorBankV2 — gradient-based Mixture-of-Experts operator
discovery. Replaces V1's K-means with a learned routing network +
K shared dynamics heads.

Why V2 exists: V1's cross-env operator stability failed (mean cosine
0.45-0.56 between K-means clusters fit per-env, target was 0.8). The
problem with K-means is that it fits clusters to the actual delta
distribution per env — and that distribution differs across envs even
when the underlying dynamics are the same.

V2 fixes this BY CONSTRUCTION: operators ARE the dynamics, shared
across envs. There's only one set of K dynamics heads + one routing
network, trained jointly on all envs' transitions. Cross-env
stability is structural — there are no per-env operator banks to
disagree.

Inspired by Mixture-of-World-Models (arXiv 2602.01270, 2026), which
uses gradient-based clustering to allocate distinct critic networks
to different tasks.

Architecture:
    Input: (z_t, action_t)
    Step 1: action_emb = Embedding(action)
            x = concat(z_t, action_emb)
    Step 2: routing_logits = MLP_routing(x)            # (B, K)
            routing_probs  = softmax(routing_logits)
    Step 3: For each operator k:
                op_pred_k = MLP_k(x)                   # predicted delta
            op_preds = stack(op_preds, dim=1)          # (B, K, latent_dim)
    Step 4: delta_pred = sum(routing_probs * op_preds, dim=1)
                                                       # (B, latent_dim)
    Loss:   MSE(delta_pred, z_t+1 - z_t) - entropy_coef * H(routing)

The entropy term prevents collapse to one always-active head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OperatorV2Stats:
    """Per-operator diagnostics from V2 routing analysis."""
    op_id: int
    activation_rate: float                # mean routing prob across all transitions
    dominant_action: int                  # action that most often routes here
    purity: float                         # fraction of activations from dominant action
    action_distribution: dict[int, float]  # routing-weighted action distribution


class OperatorBankV2(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_actions: int,
        *,
        n_ops: int = 8,
        hidden: int = 256,
        action_emb_dim: int = 16,
        entropy_coef: float = 0.01,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self.n_ops = n_ops
        self.entropy_coef = entropy_coef

        self.action_emb = nn.Embedding(n_actions, action_emb_dim)
        in_dim = latent_dim + action_emb_dim

        # Routing: which operator fires for this (state, action)?
        self.routing = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_ops),
        )

        # K dynamics heads: each predicts a latent-delta
        self.ops = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, latent_dim),
            )
            for _ in range(n_ops)
        ])

    def _flatten(self, z: torch.Tensor) -> torch.Tensor:
        return z.flatten(1) if z.dim() > 2 else z

    def forward(
        self,
        z_t: torch.Tensor,                            # (B, latent_dim) or (B, ...)
        action: torch.Tensor,                          # (B,) int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (delta_pred (B, latent_dim), routing_probs (B, K),
        op_preds (B, K, latent_dim))."""
        z_flat = self._flatten(z_t)
        a_emb = self.action_emb(action)
        x = torch.cat([z_flat, a_emb], dim=-1)

        routing_logits = self.routing(x)              # (B, K)
        routing_probs = F.softmax(routing_logits, dim=-1)
        op_preds = torch.stack([op(x) for op in self.ops], dim=1)
                                                       # (B, K, latent_dim)
        delta_pred = (routing_probs.unsqueeze(-1) * op_preds).sum(dim=1)
                                                       # (B, latent_dim)
        return delta_pred, routing_probs, op_preds

    def loss(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        z_tp1: torch.Tensor,
    ) -> dict:
        z_t_flat = self._flatten(z_t)
        z_tp1_flat = self._flatten(z_tp1)
        delta_actual = z_tp1_flat - z_t_flat

        delta_pred, routing_probs, _ = self.forward(z_t, action)
        # Reconstruction
        mse = F.mse_loss(delta_pred, delta_actual)
        # Entropy: encourage routing diversity (don't collapse to one head)
        entropy = -(routing_probs * (routing_probs + 1e-9).log()).sum(dim=-1).mean()
        total = mse - self.entropy_coef * entropy
        return {
            "loss": total,
            "mse": mse.detach(),
            "entropy": entropy.detach(),
            "routing": routing_probs.detach(),
        }

    @torch.no_grad()
    def assign(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Hard operator assignment per transition (argmax of routing)."""
        _, routing, _ = self.forward(z_t, action)
        return routing.argmax(dim=-1)

    @torch.no_grad()
    def routing_for(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        _, routing, _ = self.forward(z_t, action)
        return routing

    @torch.no_grad()
    def analyze(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
    ) -> list[OperatorV2Stats]:
        """Return per-operator stats given a batch of transitions."""
        routing = self.routing_for(z_t, action)        # (B, K)
        # Hard-assign each transition to argmax operator for purity calc.
        hard = routing.argmax(dim=-1).cpu().numpy()
        actions = action.cpu().numpy()

        stats: list[OperatorV2Stats] = []
        for k in range(self.n_ops):
            mask = hard == k
            n = int(mask.sum())
            mean_prob = float(routing[:, k].mean().item())
            if n == 0:
                stats.append(OperatorV2Stats(
                    op_id=k, activation_rate=mean_prob,
                    dominant_action=-1, purity=0.0,
                    action_distribution={},
                ))
                continue
            assigned_actions = actions[mask]
            unique, counts = np.unique(assigned_actions, return_counts=True)
            action_dist = {int(a): float(c / counts.sum())
                           for a, c in zip(unique, counts)}
            dominant = int(unique[counts.argmax()])
            purity = float(counts.max() / counts.sum())
            stats.append(OperatorV2Stats(
                op_id=k, activation_rate=mean_prob,
                dominant_action=dominant, purity=purity,
                action_distribution=action_dist,
            ))
        return stats

    @torch.no_grad()
    def cross_env_stability(
        self,
        per_env_data: dict[str, tuple[torch.Tensor, torch.Tensor]],
        threshold: float = 0.8,
    ) -> dict:
        """Test whether the SAME operator (same k) fires for similar
        (action, transition-context) pairs across envs.

        Per env, compute the routing distribution conditioned on each
        action. Then compare distributions across env pairs.

        Stability metric: for each operator k and each action a, compute
        P(k | a, env=e1) and P(k | a, env=e2). Lower KL divergence =
        more stable. We summarize as the cosine similarity between the
        per-env action→operator matrices.

        per_env_data: dict of env_id → (latents, actions) tensors.
        """
        env_ids = sorted(per_env_data.keys())
        # Build P(k | a) matrix per env: shape (n_actions, n_ops)
        env_matrices: dict[str, np.ndarray] = {}
        for env_id, (latents, actions) in per_env_data.items():
            routing = self.routing_for(latents, actions).cpu().numpy()
            actions_np = actions.cpu().numpy()
            mat = np.zeros((self.n_actions, self.n_ops))
            counts = np.zeros(self.n_actions)
            for i, a in enumerate(actions_np):
                mat[a] += routing[i]
                counts[a] += 1
            counts = np.maximum(counts, 1)
            mat = mat / counts[:, None]                # average per action
            env_matrices[env_id] = mat

        results: dict = {"per_env_action_op_matrix": {}}
        for env_id, mat in env_matrices.items():
            results["per_env_action_op_matrix"][env_id] = mat.tolist()

        # Pairwise cosine similarity between flattened matrices
        if len(env_ids) >= 2:
            pair_results = []
            for i in range(len(env_ids)):
                for j in range(i + 1, len(env_ids)):
                    e1, e2 = env_ids[i], env_ids[j]
                    m1 = env_matrices[e1].flatten()
                    m2 = env_matrices[e2].flatten()
                    cos = float((m1 @ m2) / (
                        (np.linalg.norm(m1) + 1e-9) *
                        (np.linalg.norm(m2) + 1e-9)
                    ))
                    pair_results.append({
                        "env1": e1, "env2": e2,
                        "matrix_cosine_sim": cos,
                        "pass": cos >= threshold,
                    })
            results["pairwise"] = pair_results
            results["mean_cosine"] = float(np.mean(
                [p["matrix_cosine_sim"] for p in pair_results]
            ))
            results["all_pass"] = all(p["pass"] for p in pair_results)
        return results

    def save(self, path: str) -> None:
        torch.save({
            "model_state_dict": self.state_dict(),
            "latent_dim": self.latent_dim,
            "n_actions": self.n_actions,
            "n_ops": self.n_ops,
        }, path)

    @classmethod
    def load(cls, path: str, device: torch.device,
             hidden: int = 256, action_emb_dim: int = 16) -> "OperatorBankV2":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        bank = cls(
            latent_dim=ckpt["latent_dim"],
            n_actions=ckpt["n_actions"],
            n_ops=ckpt["n_ops"],
            hidden=hidden,
            action_emb_dim=action_emb_dim,
        )
        bank.load_state_dict(ckpt["model_state_dict"])
        bank.to(device)
        return bank
