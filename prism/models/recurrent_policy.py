"""Recurrent policy that consumes the JEPA's frozen latent.

Phase 3 step 2: replace the hand-coded memory-mode state machine (pose
tracking + frontier exploration + curriculum) with a learned GRU that
ingests (z_t, previous-action embedding, mission one-hot) and emits an
action distribution. The JEPA encoder stays frozen — we're only learning
the policy head.

Inputs at each step:
  z_t           — (B, embed_dim) flat or (B, C, H, W) spatial latent
  prev_action   — (B,) int64; -1 for the first step (we use a 'no action' embed)
  mission       — (B, 24) one-hot of the goal (type, color) pair
Hidden state h_t carries memory across steps; h_0 = zeros.

Output: (B, n_actions) logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RecurrentPolicy(nn.Module):
    def __init__(
        self,
        latent_in_dim: int,        # flattened JEPA latent dim
        n_actions: int,
        mission_dim: int = 24,
        action_emb_dim: int = 16,
        hidden_dim: int = 256,
        latent_proj_dim: int = 128,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        # Latent projection — flatten any spatial structure and reduce dim.
        self.latent_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_in_dim, latent_proj_dim),
            nn.GELU(),
        )
        # Previous-action embedding. We allocate n_actions + 1 slots; index
        # n_actions is the "no previous action" sentinel for t=0.
        self.action_emb = nn.Embedding(n_actions + 1, action_emb_dim)
        self.no_action_index = n_actions
        # Mission projection — keep it cheap, the one-hot already carries the info.
        self.mission_proj = nn.Linear(mission_dim, action_emb_dim)
        # GRU input is concat[latent_proj, action_emb, mission_proj].
        gru_in = latent_proj_dim + action_emb_dim + action_emb_dim
        self.gru = nn.GRUCell(gru_in, hidden_dim)
        # Action head reads from h_t.
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        # Value head — added for PPO actor-critic. State-value V(h_t).
        # Old BC checkpoints don't have these weights; loading them with
        # strict=False leaves the value head random-initialized, which is
        # exactly what we want for PPO fine-tune from BC: critic is learned
        # fresh while policy is fine-tuned from BC.
        self.value_head = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step(
        self,
        z: torch.Tensor,             # (B, embed) or (B, C, H, W)
        prev_action: torch.Tensor,   # (B,) int64 with -1 for first step
        mission: torch.Tensor,       # (B, mission_dim)
        h_prev: torch.Tensor,        # (B, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrent step. Returns (logits, h_next)."""
        # Map -1 sentinel → n_actions (the dedicated 'no prev action' index).
        idx = prev_action.clone()
        idx[idx < 0] = self.no_action_index
        ae = self.action_emb(idx)
        me = self.mission_proj(mission)
        ze = self.latent_proj(z)
        x = torch.cat([ze, ae, me], dim=-1)
        h_next = self.gru(x, h_prev)
        logits = self.policy_head(h_next)
        return logits, h_next

    def step_with_value(
        self,
        z: torch.Tensor,
        prev_action: torch.Tensor,
        mission: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One recurrent step that ALSO returns the state value.
        Used by PPO. Returns (logits, value, h_next), where value has
        shape (B,) — squeezed scalar.
        """
        idx = prev_action.clone()
        idx[idx < 0] = self.no_action_index
        ae = self.action_emb(idx)
        me = self.mission_proj(mission)
        ze = self.latent_proj(z)
        x = torch.cat([ze, ae, me], dim=-1)
        h_next = self.gru(x, h_prev)
        logits = self.policy_head(h_next)
        value = self.value_head(h_next).squeeze(-1)  # (B,)
        return logits, value, h_next

    def forward(
        self,
        z_seq: torch.Tensor,         # (B, T, embed) or (B, T, C, H, W)
        action_seq: torch.Tensor,    # (B, T) int64; action_seq[:, t] is taken AT step t
        mission: torch.Tensor,       # (B, mission_dim)
        lengths: torch.Tensor | None = None,  # (B,) optional, for masking
    ) -> torch.Tensor:
        """Run the policy across a full sequence.

        Returns logits of shape (B, T, n_actions). The action used as
        'previous' at step t is action_seq[:, t-1] for t > 0, and the
        'no-action' sentinel for t = 0.
        """
        B, T = action_seq.shape
        h = self.init_hidden(B, z_seq.device)
        logits_seq = []
        for t in range(T):
            if t == 0:
                prev_a = torch.full((B,), -1, device=z_seq.device, dtype=torch.long)
            else:
                prev_a = action_seq[:, t - 1]
            z_t = z_seq[:, t]
            logits, h = self.step(z_t, prev_a, mission, h)
            logits_seq.append(logits)
        return torch.stack(logits_seq, dim=1)  # (B, T, n_actions)
