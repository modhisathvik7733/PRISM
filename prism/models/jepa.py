"""JEPA latent world model — Phase 1 core.

Architectural template: LeWorldModel (le-wm.github.io). Two losses only at
training time:

    L_pred = || sg(z_{t+1}) - p_theta(z_t, a_t) ||^2
    L_reg  = KL( N(mu, sigma) || N(0, I) )       # Gaussian-regularized embeds

`sg` = stop-gradient on the target encoder (EMA of the online encoder).

This file deliberately keeps the encoder small (BabyAI partial views are 7x7x3).
For Phase 5 the same `JepaWorldModel` is reused with a video encoder swap.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class JepaConfig:
    obs_channels: int = 3
    obs_h: int = 7
    obs_w: int = 7
    n_actions: int = 7  # MiniGrid action space
    embed_dim: int = 128
    hidden_dim: int = 256
    ema_decay: float = 0.996  # target encoder EMA, V-JEPA-style
    reg_weight: float = 1e-3


class GridEncoder(nn.Module):
    """Tiny conv encoder for 7x7x3 BabyAI partial views.

    Phase 5 will replace this with a V-JEPA-style ViT video encoder; the rest
    of the world model treats the output as an opaque (B, embed_dim) embedding.
    """

    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cfg.obs_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * cfg.obs_h * cfg.obs_w, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentDynamics(nn.Module):
    """Predicts z_{t+1} from (z_t, a_t)."""

    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.action_embed = nn.Embedding(cfg.n_actions, cfg.embed_dim)
        self.net = nn.Sequential(
            nn.Linear(cfg.embed_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        ae = self.action_embed(a)
        return self.net(torch.cat([z, ae], dim=-1))


class JepaWorldModel(nn.Module):
    """Online encoder + EMA target encoder + latent dynamics.

    Counterfactual prediction (`predict_counterfactual`) shares the dynamics
    network — the inductive bias is that *the same dynamics function* should
    explain factual and counterfactual transitions.
    """

    def __init__(self, cfg: JepaConfig | None = None):
        super().__init__()
        self.cfg = cfg or JepaConfig()
        self.online_encoder = GridEncoder(self.cfg)
        self.target_encoder = GridEncoder(self.cfg)
        self.dynamics = LatentDynamics(self.cfg)
        self._init_target_from_online()

    @torch.no_grad()
    def _init_target_from_online(self) -> None:
        for tp, op in zip(
            self.target_encoder.parameters(), self.online_encoder.parameters(), strict=True
        ):
            tp.data.copy_(op.data)
            tp.requires_grad_(False)

    @torch.no_grad()
    def update_target(self) -> None:
        m = self.cfg.ema_decay
        for tp, op in zip(
            self.target_encoder.parameters(), self.online_encoder.parameters(), strict=True
        ):
            tp.data.mul_(m).add_(op.data, alpha=1 - m)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(obs)

    @torch.no_grad()
    def encode_target(self, obs: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(obs)

    def predict(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.dynamics(z, a)

    def predict_counterfactual(self, z: torch.Tensor, a_prime: torch.Tensor) -> torch.Tensor:
        """Same dynamics as `predict`, just called with an alternative action.

        Kept as a separate method so call sites read clearly and so the
        counterfactual loss can target it explicitly.
        """
        return self.dynamics(z, a_prime)

    def loss(
        self,
        obs_t: torch.Tensor,
        action_t: torch.Tensor,
        obs_tp1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z_t = self.online_encoder(obs_t)
        z_pred = self.dynamics(z_t, action_t)
        with torch.no_grad():
            z_target = self.target_encoder(obs_tp1)

        l_pred = F.mse_loss(z_pred, z_target)
        # Gaussian regularizer: keep online embeddings near unit-norm Gaussian.
        # (LeWM uses an explicit KL; we approximate with mean+var penalty for stability.)
        mu, var = z_t.mean(0), z_t.var(0)
        l_reg = (mu.pow(2).mean() + (var - 1).pow(2).mean())
        return {
            "loss": l_pred + self.cfg.reg_weight * l_reg,
            "loss_pred": l_pred.detach(),
            "loss_reg": l_reg.detach(),
        }
