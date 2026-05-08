"""CrafterJepaWorldModel -- JEPA with InfoNCE prediction loss for Crafter.

Encoder:  CrafterCNN (IMPALA-style, 4 strided convs) -> (B, 256)
Dynamics: MLP concat(z_t, a_embed) -> z_pred, all in embed_dim space
Target:   EMA copy of online encoder, no grad

Loss:
  InfoNCE(norm(z_pred), sg(norm(target(obs_{t+1}))), temperature=0.07)

  Batch-i positive: (z_pred_i, z_target_i)  -- same transition
  Batch-i negatives: all (z_pred_i, z_target_j) for j != i

Why InfoNCE instead of MSE:
  MSE-based JEPA has a trivial collapsed fixed point: if both encoders
  output the same constant, l_pred=0 and the gradient to the encoder is
  exactly zero. VICReg cannot escape this because the gradient magnitude
  from the variance term is ~1e-10 per weight at collapse.

  InfoNCE has NO collapsed fixed point: if all z_pred are identical, the
  loss equals log(B) ~= 5.5 (the maximum), not the minimum. The contrastive
  pressure always pushes representations apart. Same pattern as MoCo v3.

Shapes locked:
  obs:    (B, 3, 64, 64)  float32  [0, 1]
  action: (B,)            int64    [0, 16]
  z:      (B, embed_dim)  float32  (default embed_dim = 256)
  z_pred: (B, embed_dim)  float32
  logits: (B, B)          float32  similarity matrix / temperature
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.crafter.cnn_encoder import CrafterCNN


@dataclass
class CrafterJepaConfig:
    embed_dim: int = 256
    n_actions: int = 17
    ema_decay: float = 0.996
    temperature: float = 0.07   # InfoNCE temperature; 0.07 matches MoCo v3 / CLIP
    dynamics_hidden: int = 256


def _info_nce(
    q: torch.Tensor,   # (B, E) queries  (z_pred, normalised)
    k: torch.Tensor,   # (B, E) keys     (z_target, normalised, stop-grad)
    temperature: float,
) -> torch.Tensor:
    """InfoNCE with in-batch negatives. Positive pair on the diagonal.

    loss = mean(-log(exp(q_i . k_i / T) / sum_j exp(q_i . k_j / T)))
         = cross_entropy(logits, arange(B))

    Shapes: q, k -> (B, E); returns scalar.
    """
    logits = (q @ k.T) / temperature          # (B, B)
    labels = torch.arange(len(q), device=q.device)
    return F.cross_entropy(logits, labels)


class _LatentDynamics(nn.Module):
    """concat(z_t, a_embed) -> MLP -> z_pred.

    Shapes:
      z:       (B, E)   float32
      a:       (B,)     int64
      a_embed: (B, E)   float32  -- same dim as z, avoids asymmetric concat
      output:  (B, E)   float32
    """

    def __init__(self, embed_dim: int, n_actions: int, hidden: int):
        super().__init__()
        E = embed_dim
        self.action_embed = nn.Embedding(n_actions, E)
        self.net = nn.Sequential(
            nn.Linear(E * 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, E),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        ae = self.action_embed(a)
        return self.net(torch.cat([z, ae], dim=-1))  # (B, E)


class CrafterJepaWorldModel(nn.Module):
    """Online encoder + EMA target encoder + latent dynamics for Crafter.

    Training:
        model = CrafterJepaWorldModel()
        losses = model.loss(obs_t, action_t, obs_tp1)
        losses["loss"].backward()
        optimizer.step()
        model.update_target()   # after every optimizer step

    Frozen inference (PPO policy):
        z = model.encode(obs)   # (B, 256)
    """

    def __init__(self, cfg: CrafterJepaConfig | None = None):
        super().__init__()
        self.cfg = cfg or CrafterJepaConfig()
        E = self.cfg.embed_dim
        self.online_encoder = CrafterCNN(embed_dim=E)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.dynamics = _LatentDynamics(E, self.cfg.n_actions, self.cfg.dynamics_hidden)

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA: target <- decay * target + (1 - decay) * online."""
        m = self.cfg.ema_decay
        for tp, op in zip(
            self.target_encoder.parameters(),
            self.online_encoder.parameters(),
        ):
            tp.data.mul_(m).add_(op.data, alpha=1.0 - m)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, 3, 64, 64) -> z: (B, embed_dim)"""
        return self.online_encoder(obs)

    def loss(
        self,
        obs_t: torch.Tensor,     # (B, 3, 64, 64) float32 [0, 1]
        action_t: torch.Tensor,  # (B,) int64
        obs_tp1: torch.Tensor,   # (B, 3, 64, 64) float32 [0, 1]
    ) -> dict[str, torch.Tensor]:
        z_t   = self.online_encoder(obs_t)       # (B, E)
        z_pred = self.dynamics(z_t, action_t)    # (B, E)

        with torch.no_grad():
            z_target = self.target_encoder(obs_tp1)  # (B, E)

        # L2-normalise both before InfoNCE (cosine similarity matrix).
        q = F.normalize(z_pred,   dim=-1)        # (B, E)
        k = F.normalize(z_target, dim=-1)        # (B, E)  stop-grad above

        loss = _info_nce(q, k, self.cfg.temperature)

        # Diagnostic: cosine similarity of positive pairs (should rise toward 1).
        with torch.no_grad():
            pos_cos = (q * k).sum(dim=-1).mean()

        return {
            "loss":      loss,
            "loss_pred": loss.detach(),
            "loss_reg":  pos_cos,           # logged as "l_reg"; ideally -> 1
        }
