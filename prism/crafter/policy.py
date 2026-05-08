"""CrafterPolicy — CNN encoder + GRU + actor-critic heads, trained
end-to-end with PPO.

This is the from-scratch baseline (commit 2 of the Crafter port). It does
NOT use a frozen JEPA — the CNN encoder is trained jointly with the
policy via PPO gradients. The point of the baseline is to confirm the
env + RL infrastructure work and produce a "PPO from scratch" number we
can compare to (a) the published ~5% Crafter paper baseline and (b) the
JEPA-based variant in commit 3.

Architecture mirrors `prism.models.recurrent_policy.RecurrentPolicy` but
with the CNN encoder fused in and the BabyAI-specific bits removed:
  - No mission projection (Crafter has no language goal)
  - No mem_feat residual (Crafter scrolls around the agent — no pose to
    track)
  - No latent_proj (encoder already produces a flat embed_dim vector)

Inputs at each step:
  obs           — (B, 3, 64, 64) float32 in [0, 1]
  prev_action   — (B,) int64; -1 for the first step
Hidden state h_t carries memory across steps; h_0 = zeros.

Output: (B, n_actions) logits + (B,) value.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from prism.crafter.cnn_encoder import CrafterCNN


class CrafterPolicy(nn.Module):
    def __init__(
        self,
        n_actions: int = 17,
        embed_dim: int = 256,
        action_emb_dim: int = 16,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.cnn = CrafterCNN(embed_dim=embed_dim)
        # Previous-action embedding. Index n_actions reserved for the
        # "no previous action" sentinel at t=0 (matches the v1.x policy
        # so checkpoints could in theory be loaded across the families).
        self.action_emb = nn.Embedding(n_actions + 1, action_emb_dim)
        self.no_action_index = n_actions
        # GRU input is concat[encoder_out, prev_action_embed].
        gru_in = embed_dim + action_emb_dim
        self.gru = nn.GRUCell(gru_in, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step_with_value(
        self,
        obs: torch.Tensor,           # (B, 3, 64, 64)
        prev_action: torch.Tensor,   # (B,) int64; -1 for first step
        h_prev: torch.Tensor,        # (B, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One recurrent step. Returns (logits, value, h_next)."""
        idx = prev_action.clone()
        idx[idx < 0] = self.no_action_index
        ae = self.action_emb(idx)
        ze = self.cnn(obs)
        x = torch.cat([ze, ae], dim=-1)
        h_next = self.gru(x, h_prev)
        logits = self.policy_head(h_next)
        value = self.value_head(h_next).squeeze(-1)
        return logits, value, h_next
