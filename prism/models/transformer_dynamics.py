"""TransformerDynamics — Hopfield-attention transformer as world model + trunk.

Replaces RecurrentPolicy's GRU trunk with a stack of HopfieldEncoderLayer.
Reads a sequence of [concept_t, action_t] tokens and predicts:
  - next concept embedding (world model)
  - reward (value function helper)
  - state value (for PPO)
  - action logits (policy head)

Why HopfieldEncoderLayer instead of standard nn.TransformerEncoderLayer?
- Mathematically equivalent (attention IS Hopfield retrieval) but configurable
- Can use update_steps_max>0 for iterative refinement
- Inspectable: get_association_matrix shows what's being attended to
- The pre-wired transformer block from hflayers is BSD-3 and well-tested

This becomes the central reasoning module: takes working memory of recent
concepts/actions, produces next-step predictions for everything downstream.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_VENDOR = os.path.join(os.path.dirname(__file__), "..", "_vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
from hflayers import Hopfield  # noqa: E402
from hflayers.transformer import HopfieldEncoderLayer  # noqa: E402


class TransformerDynamics(nn.Module):
    """Hopfield-attention transformer over working memory of (concept, action)
    token sequences. Single trunk produces predictions for next concept,
    reward, value, and action.

    Inputs (per step):
      concept_seq : (B, T, concept_dim)   — retrieved concepts from ConceptMemory
      action_seq  : (B, T) long           — actions taken at each step
      mission_emb : (B, mission_dim) opt  — goal embedding from language head

    Outputs:
      next_concept : (B, T, concept_dim)
      reward       : (B, T)
      value        : (B, T)
      action_logits: (B, T, n_actions)
    """

    def __init__(
        self,
        concept_dim: int = 64,
        n_actions: int = 7,
        mission_dim: int = 64,
        token_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 512,
        max_seq_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.concept_dim = concept_dim
        self.n_actions = n_actions
        self.mission_dim = mission_dim
        self.token_dim = token_dim
        self.max_seq_len = max_seq_len

        # Projection layers: bring concept, action, mission to shared token_dim.
        self.concept_proj = nn.Linear(concept_dim, token_dim)
        self.action_embed = nn.Embedding(n_actions + 1, token_dim)  # +1 for no-action
        self.no_action_idx = n_actions
        self.mission_proj = nn.Linear(mission_dim, token_dim)
        self.pos_embed = nn.Embedding(max_seq_len, token_dim)

        # Stack of Hopfield-attention encoder layers.
        # Each HopfieldEncoderLayer wraps a Hopfield + FFN + LayerNorm + residual,
        # just like nn.TransformerEncoderLayer but with Hopfield attention.
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            hopfield = Hopfield(
                input_size=token_dim,
                hidden_size=token_dim // n_heads,
                num_heads=n_heads,
                scaling=1.0,
                update_steps_max=0,   # single-shot in trunk
                dropout=dropout,
                normalize_stored_pattern=True,
                normalize_state_pattern=True,
                normalize_pattern_projection=True,
            )
            self.layers.append(
                HopfieldEncoderLayer(
                    hopfield_association=hopfield,
                    dim_feedforward=ffn_dim,
                    dropout=dropout,
                )
            )

        # Output heads.
        self.next_concept_head = nn.Linear(token_dim, concept_dim)
        self.reward_head = nn.Linear(token_dim, 1)
        self.value_head = nn.Linear(token_dim, 1)
        self.action_head = nn.Linear(token_dim, n_actions)

    def forward(
        self,
        concept_seq: torch.Tensor,
        action_seq: torch.Tensor,
        mission_emb: torch.Tensor | None = None,
        causal: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        concept_seq : (B, T, concept_dim)
        action_seq  : (B, T) long. Use no_action_idx for first step.
        mission_emb : (B, mission_dim) optional. Added to every position.
        causal      : if True, apply causal mask (each step sees only past).
        """
        B, T, _ = concept_seq.shape
        device = concept_seq.device

        # Tokenize: project concept, action, position; sum with optional mission.
        c_tok = self.concept_proj(concept_seq)                       # (B, T, D)
        a_tok = self.action_embed(action_seq.clamp(min=0))           # (B, T, D)
        pos = self.pos_embed(torch.arange(T, device=device))         # (T, D)
        tokens = c_tok + a_tok + pos.unsqueeze(0)

        if mission_emb is not None:
            m_tok = self.mission_proj(mission_emb).unsqueeze(1)      # (B, 1, D)
            tokens = tokens + m_tok

        # Build causal mask if requested.
        attn_mask = None
        if causal:
            # HopfieldEncoderLayer accepts src_mask in (T, T) form.
            # An additive float mask: 0 for allowed positions, -inf otherwise.
            attn_mask = torch.zeros(T, T, device=device)
            attn_mask = attn_mask.masked_fill(
                torch.triu(torch.ones(T, T, device=device), diagonal=1).bool(),
                float("-inf"),
            )

        # Pass through Hopfield encoder layers.
        h = tokens
        for layer in self.layers:
            h = layer(h, src_mask=attn_mask)

        # Output predictions.
        return {
            "next_concept": self.next_concept_head(h),
            "reward": self.reward_head(h).squeeze(-1),
            "value": self.value_head(h).squeeze(-1),
            "action_logits": self.action_head(h),
            "hidden": h,  # for downstream consumers (language head)
        }

    @torch.no_grad()
    def imagine_rollout(
        self,
        concept_seq: torch.Tensor,
        action_seq: torch.Tensor,
        mission_emb: torch.Tensor | None,
        n_steps: int = 5,
    ) -> dict[str, torch.Tensor]:
        """Use the world model to imagine future trajectories.

        Greedy: at each imagined step, pick the highest-logit action and
        advance the latent via the next_concept prediction.
        """
        device = concept_seq.device
        B = concept_seq.size(0)
        cur_concepts = concept_seq
        cur_actions = action_seq

        imagined_concepts = []
        imagined_actions = []
        imagined_rewards = []

        for _ in range(n_steps):
            out = self.forward(cur_concepts, cur_actions, mission_emb)
            next_c = out["next_concept"][:, -1:, :]      # (B, 1, concept_dim)
            action_logits = out["action_logits"][:, -1, :]
            next_a = action_logits.argmax(dim=-1, keepdim=True)
            next_r = out["reward"][:, -1:]

            imagined_concepts.append(next_c)
            imagined_actions.append(next_a)
            imagined_rewards.append(next_r)

            # Append and slide window if at max len.
            cur_concepts = torch.cat([cur_concepts, next_c], dim=1)
            cur_actions = torch.cat([cur_actions, next_a], dim=1)
            if cur_concepts.size(1) > self.max_seq_len:
                cur_concepts = cur_concepts[:, -self.max_seq_len:]
                cur_actions = cur_actions[:, -self.max_seq_len:]

        return {
            "concepts": torch.cat(imagined_concepts, dim=1),
            "actions": torch.cat(imagined_actions, dim=1),
            "rewards": torch.cat(imagined_rewards, dim=1),
        }


class TransformerDynamicsStep(nn.Module):
    """Single-step wrapper of TransformerDynamics with internal buffer.

    Maintains a fixed-size rolling buffer of recent (concept, action) tokens
    so the policy can be called step-by-step like an RNN, but internally
    uses transformer attention over the full recent window.

    Replaces RecurrentPolicy's per-step hidden state with a sliding window.
    """

    def __init__(self, dynamics: TransformerDynamics, buffer_size: int = 16):
        super().__init__()
        self.dynamics = dynamics
        self.buffer_size = buffer_size

    def init_buffer(self, batch_size: int, device: torch.device) -> dict:
        return {
            "concepts": torch.zeros(
                batch_size, 0, self.dynamics.concept_dim, device=device
            ),
            "actions": torch.zeros(
                batch_size, 0, dtype=torch.long, device=device
            ),
        }

    def step(
        self,
        concept: torch.Tensor,         # (B, concept_dim)
        prev_action: torch.Tensor,     # (B,) long
        buffer: dict,
        mission_emb: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """One policy step. Returns predictions + updated buffer."""
        # Append current concept and action to buffer.
        c_in = concept.unsqueeze(1)              # (B, 1, concept_dim)
        a_in = prev_action.unsqueeze(1)          # (B, 1)
        new_concepts = torch.cat([buffer["concepts"], c_in], dim=1)
        new_actions = torch.cat([buffer["actions"], a_in], dim=1)

        # Trim to buffer_size.
        if new_concepts.size(1) > self.buffer_size:
            new_concepts = new_concepts[:, -self.buffer_size:]
            new_actions = new_actions[:, -self.buffer_size:]

        # Run dynamics on the full window.
        out = self.dynamics(new_concepts, new_actions, mission_emb, causal=True)

        # Take last-step predictions.
        last = {
            "next_concept": out["next_concept"][:, -1, :],
            "reward": out["reward"][:, -1],
            "value": out["value"][:, -1],
            "action_logits": out["action_logits"][:, -1, :],
            "hidden": out["hidden"][:, -1, :],
        }
        new_buffer = {"concepts": new_concepts, "actions": new_actions}
        return last, new_buffer
