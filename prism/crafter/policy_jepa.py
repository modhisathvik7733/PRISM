"""CrafterPolicyJepa — GRU + actor-critic on top of a FROZEN JEPA encoder.

Drop-in replacement for CrafterPolicy: same step_with_value interface,
same hidden-state shape, same action space. The only difference is the
visual encoder: instead of a CNN trained end-to-end with PPO gradients,
we load a CrafterCNN that was pre-trained with the JEPA objective and
freeze it for the duration of RL fine-tuning.

This lets us compare fairly:
  baseline:   score from ppo_train_baseline  (CNN trained jointly with PPO)
  this:       score from ppo_train_jepa      (same GRU+heads, frozen JEPA enc)

Architecture:
  obs (B, 3, 64, 64)
    → frozen CrafterCNN → z (B, embed_dim=256)
  prev_action (B,) [−1 sentinel at t=0]
    → Embedding(n_actions+1, 16) → ae (B, 16)
  GRU input: cat(z, ae) (B, 272)
    → GRUCell(272, hidden_dim=256) → h_next (B, 256)
  policy_head: Linear(256, 17) → logits (B, 17)
  value_head:  Linear(256,  1) → value  (B,)

Trainable params (no encoder):
  action_emb  + GRU  + policy_head + value_head
  ≈ 288 + 406 272 + 4 369 + 257 ≈ 411 K   (vs 1.36 M for CrafterPolicy)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from prism.crafter.cnn_encoder import CrafterCNN
from prism.crafter.jepa_crafter import CrafterJepaConfig, CrafterJepaWorldModel


class CrafterPolicyJepa(nn.Module):
    def __init__(
        self,
        jepa_checkpoint: str,
        n_actions: int = 17,
        action_emb_dim: int = 16,
        hidden_dim: int = 256,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.n_actions  = n_actions
        self.hidden_dim = hidden_dim

        # Load and freeze the JEPA online encoder.
        ckpt = torch.load(jepa_checkpoint, map_location=device)
        cfg: CrafterJepaConfig = ckpt["cfg"]
        encoder = CrafterCNN(embed_dim=cfg.embed_dim)
        encoder.load_state_dict(ckpt["online_encoder_state"])
        for p in encoder.parameters():
            p.requires_grad_(False)
        self.encoder = encoder
        embed_dim = cfg.embed_dim

        # Trainable recurrent policy head (identical layout to CrafterPolicy).
        self.action_emb = nn.Embedding(n_actions + 1, action_emb_dim)
        self.no_action_index = n_actions
        gru_in = embed_dim + action_emb_dim          # 256 + 16 = 272
        self.gru = nn.GRUCell(gru_in, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head  = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step_with_value(
        self,
        obs: torch.Tensor,          # (B, 3, 64, 64) float32 [0, 1]
        prev_action: torch.Tensor,  # (B,) int64; −1 for first step
        h_prev: torch.Tensor,       # (B, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One recurrent step. Returns (logits (B,17), value (B,), h_next (B,256))."""
        idx = prev_action.clone()
        idx[idx < 0] = self.no_action_index
        ae = self.action_emb(idx)               # (B, 16)

        with torch.no_grad():
            z = self.encoder(obs)               # (B, 256)  — frozen, no grad

        x      = torch.cat([z, ae], dim=-1)     # (B, 272)
        h_next = self.gru(x, h_prev)            # (B, 256)
        logits = self.policy_head(h_next)       # (B, 17)
        value  = self.value_head(h_next).squeeze(-1)  # (B,)
        return logits, value, h_next
