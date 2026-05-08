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

import torch
import torch.nn as nn


class CrafterCNN(nn.Module):
    def __init__(self, embed_dim: int = 256, in_channels: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
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
        # Compute flatten dim once with a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            flat = self.conv(dummy).flatten(1).shape[1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, embed_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, 3, 64, 64) → (B, embed_dim)"""
        h = self.conv(obs)
        return self.fc(h)
