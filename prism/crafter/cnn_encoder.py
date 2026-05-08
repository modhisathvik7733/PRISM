"""CNN encoder for 64×64×3 RGB Crafter observations.

This is the analog of `prism.models.jepa`'s `categorical_spatial` encoder
but for continuous RGB inputs. Architecture is the small Impala-style CNN
that DreamerV3 / IMPALA / Crafter baselines all use:

    Input:  (B, 3, 64, 64)  float32 in [0, 1]
    Conv:   3 → 32, k=4, s=2  → (B, 32, 31, 31)
    Conv:  32 → 64, k=4, s=2  → (B, 64, 14, 14)
    Conv:  64 → 128, k=4, s=2 → (B, 128, 6, 6)
    Conv: 128 → 256, k=4, s=2 → (B, 256, 2, 2)
    Flatten + Linear → (B, embed_dim)

Used by both the JEPA wrapper (frozen encoder for latent prediction) and
the from-scratch baseline policy (encoder learned end-to-end with PPO).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrafterCNN(nn.Module):
    def __init__(self, embed_dim: int = 256, in_channels: int = 3, state_dim: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        # Compute conv flatten dim once with a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            flat = self.conv(dummy).flatten(1).shape[1]
        self._flat = flat
        # Head is wider when state_dim > 0 (state vector concatenated before fc).
        self.head = nn.Linear(flat + state_dim, embed_dim)

    def forward(self, obs: torch.Tensor, state_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """obs: (B, 3, 64, 64), state_vec: (B, state_dim) optional → (B, embed_dim)"""
        h = self.conv(obs).flatten(1)   # (B, flat)
        if self.state_dim > 0 and state_vec is not None:
            h = torch.cat([h, state_vec], dim=-1)   # (B, flat + state_dim)
        return F.relu(self.head(h))     # (B, embed_dim)
