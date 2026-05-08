"""CrafterJepaWorldModel — JEPA latent world model for Crafter RGB observations.

Encoder:  CrafterCNN (IMPALA-style, 4 strided convs) → (B, 256)
Dynamics: MLP  concat(z_t, a_embed) → z_pred,  all in embed_dim space
Target:   EMA copy of online encoder, no grad

Losses:
  L_pred = MSE(norm(z_pred), sg(norm(target(obs_{t+1}))))  cosine-equivalent
  L_var  = mean(relu(1 - std(z_t, dim=0)))                 VICReg variance floor

Shapes locked:
  obs:     (B, 3, 64, 64)  float32  [0, 1]
  action:  (B,)            int64    [0, 16]
  z:       (B, embed_dim)  float32  (default embed_dim = 256)
  a_embed: (B, embed_dim)  float32  (same dim, avoids asymmetric concat)
  z_pred:  (B, embed_dim)  float32
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
    reg_weight: float = 1e-2   # weight on VICReg variance floor (was 1e-3 Gaussian reg -- see loss())
    dynamics_hidden: int = 256


class _LatentDynamics(nn.Module):
    """concat(z_t, a_embed) → MLP → z_pred.

    Shapes:
      z:      (B, E)    float32
      a:      (B,)      int64
      a_embed:(B, E)    float32   action embedded to same dim as z
      input:  (B, 2*E)  float32
      output: (B, E)    float32
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
        ae = self.action_embed(a)                    # (B, E)
        return self.net(torch.cat([z, ae], dim=-1))  # (B, E)


class CrafterJepaWorldModel(nn.Module):
    """Online encoder + EMA target encoder + latent dynamics for Crafter.

    Usage — training:
        model = CrafterJepaWorldModel()
        losses = model.loss(obs_t, action_t, obs_tp1)   # dict
        losses["loss"].backward()
        optimizer.step()
        model.update_target()   # call after every optimizer step

    Usage — frozen inference (PPO policy encoder):
        z = model.encode(obs)   # (B, 256)  no grad if model is frozen
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
        """EMA update: target ← decay * target + (1 - decay) * online."""
        m = self.cfg.ema_decay
        for tp, op in zip(
            self.target_encoder.parameters(),
            self.online_encoder.parameters(),
        ):
            tp.data.mul_(m).add_(op.data, alpha=1.0 - m)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, 3, 64, 64) → z: (B, embed_dim)"""
        return self.online_encoder(obs)

    def loss(
        self,
        obs_t: torch.Tensor,     # (B, 3, 64, 64) float32 [0, 1]
        action_t: torch.Tensor,  # (B,) int64
        obs_tp1: torch.Tensor,   # (B, 3, 64, 64) float32 [0, 1]
    ) -> dict[str, torch.Tensor]:
        z_t = self.online_encoder(obs_t)             # (B, E)
        z_pred = self.dynamics(z_t, action_t)        # (B, E)
        with torch.no_grad():
            z_target = self.target_encoder(obs_tp1)  # (B, E)  stop-grad

        # L2-normalize before MSE: equivalent to cosine distance, immune to
        # scale drift. Prevents the encoder from minimising l_pred by simply
        # shrinking or expanding all outputs uniformly.
        z_pred_n   = F.normalize(z_pred,   dim=-1)  # (B, E) unit-norm
        z_target_n = F.normalize(z_target, dim=-1)  # (B, E) unit-norm
        l_pred = F.mse_loss(z_pred_n, z_target_n)   # in [0, 4]

        # VICReg variance floor: push per-dim std >= 1, no penalty above 1.
        # Anti-collapse only -- unlike (var-1)^2 it cannot overshoot.
        std = z_t.std(dim=0).clamp(min=1e-4)        # (E,)
        l_var = F.relu(1.0 - std).mean()            # in [0, 1]

        loss = l_pred + self.cfg.reg_weight * l_var
        return {
            "loss": loss,
            "loss_pred": l_pred.detach(),
            "loss_reg": l_var.detach(),
        }
