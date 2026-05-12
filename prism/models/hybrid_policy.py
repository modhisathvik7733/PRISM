"""HybridPolicy — drop-in replacement for RecurrentPolicy with Hopfield perception.

Design decisions for clean PPO integration:

1. Live PPO path uses a GRU trunk (tensor hidden state) — matches
   RecurrentPolicy.step_with_value exactly so ppo_train.py works
   unchanged except for the import.
2. ConceptMemory + OperatorMemory sit BEFORE the GRU as a Hopfield
   perception front-end. The Hopfield slot retrieval is the value-add
   over raw JEPA latent: editable concepts, continual learning friendly,
   inspectable, transformer-attention-equivalent generalization.
3. TransformerDynamics and ConceptToText are NOT in the live PPO path.
   They are standalone modules in `prism/models/transformer_dynamics.py`
   and `prism/language/concept_to_text.py` that can be called offline
   for imagination rollouts and language generation. The trained
   ConceptMemory checkpoint is shared across both paths.

This is the production-realistic split: simple efficient runtime, complex
offline reasoning. Real game agents almost always do this — runtime needs
to be fast, planning/dialogue can be batched offline.

The previous version of this class used TransformerDynamicsStep with a
dict-based buffer; that broke ppo_train.py's tensor-state assumption.
This version restores tensor compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from prism.cog_core.concept_memory import ConceptMemory
from prism.cog_core.operator_memory import OperatorMemory


class HybridPolicy(nn.Module):
    """Hopfield-augmented RecurrentPolicy. Drop-in replacement for PPO.

    Pipeline per step:
        z (JEPA latent)
            ↓ latent_proj
        z_proj (working_dim)
            ↓ ConceptMemory (Hopfield retrieval)
        concept_emb (concept_slot_dim)
            ↓ concat with action_emb + mission_proj
            ↓ GRU
        h_next (hidden_dim)
            ↓ policy_head, value_head
        (logits, value, h_next)

    Same step_with_value contract as RecurrentPolicy. Same checkpoint
    serialization keys. Same init_hidden returning a tensor.
    """

    def __init__(
        self,
        latent_in_dim: int,
        n_actions: int,
        mission_dim: int = 24,
        action_emb_dim: int = 16,
        hidden_dim: int = 256,
        latent_proj_dim: int = 128,
        mem_feat_dim: int = 0,
        # Hopfield perception config
        concept_n_slots: int = 1024,
        concept_slot_dim: int = 64,
        concept_n_heads: int = 4,
        concept_scaling: float = 1.0,
        operator_n_slots: int = 64,
        operator_slot_dim: int = 64,
        operator_n_heads: int = 4,
        operator_scaling: float = 4.0,
        use_operator_memory: bool = True,
    ):
        super().__init__()
        # Public attributes matching RecurrentPolicy for ppo_train.py compatibility.
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.mem_feat_dim = mem_feat_dim
        self.latent_proj_dim = latent_proj_dim
        self.no_action_index = n_actions
        # Hopfield config — saved in checkpoints so we can reconstruct.
        self.concept_n_slots = concept_n_slots
        self.concept_slot_dim = concept_slot_dim
        self.concept_n_heads = concept_n_heads
        self.concept_scaling = concept_scaling
        self.operator_n_slots = operator_n_slots
        self.operator_slot_dim = operator_slot_dim
        self.use_operator_memory = use_operator_memory

        # Latent projection — same shape as RecurrentPolicy.
        self.latent_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_in_dim, latent_proj_dim),
            nn.GELU(),
        )

        # Hopfield concept memory — the new piece. Reads from the projected
        # latent, returns a retrieved concept embedding via attention over
        # learnable slot prototypes.
        self.concept_memory = ConceptMemory(
            latent_dim=latent_proj_dim,
            n_slots=concept_n_slots,
            slot_dim=concept_slot_dim,
            n_heads=concept_n_heads,
            scaling=concept_scaling,
            update_steps=0,
        )

        # Optional operator memory — sharper retrieval for behavioral primitives.
        # Concatenated alongside the concept embedding before the GRU.
        if use_operator_memory:
            self.operator_memory = OperatorMemory(
                latent_dim=latent_proj_dim,
                n_slots=operator_n_slots,
                slot_dim=operator_slot_dim,
                n_heads=operator_n_heads,
                scaling=operator_scaling,
                update_steps=3,
            )
            perception_dim = concept_slot_dim + operator_slot_dim
        else:
            self.operator_memory = None
            perception_dim = concept_slot_dim

        # Previous-action embedding + mission projection (same as RecurrentPolicy).
        self.action_emb = nn.Embedding(n_actions + 1, action_emb_dim)
        self.mission_proj = nn.Linear(mission_dim, action_emb_dim)

        # GRU trunk. Input = concat[perception, action_emb, mission_proj].
        gru_in = perception_dim + action_emb_dim + action_emb_dim
        self.gru = nn.GRUCell(gru_in, hidden_dim)

        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Memory-features residual head (matches RecurrentPolicy interface).
        if mem_feat_dim > 0:
            self.mem_proj = nn.Linear(mem_feat_dim, hidden_dim)
            nn.init.zeros_(self.mem_proj.weight)
            nn.init.zeros_(self.mem_proj.bias)
        else:
            self.mem_proj = None

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Returns a tensor of shape (batch_size, hidden_dim). Compatible
        with ppo_train.py's torch.where(done, init, h_next) reset pattern."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def _perceive(self, z: torch.Tensor) -> torch.Tensor:
        """Project z and retrieve from Hopfield memories.

        Returns the perception embedding fed to the GRU.
        """
        z_proj = self.latent_proj(z)  # (B, latent_proj_dim)
        concept_emb = self.concept_memory(z_proj, return_attention=False)
        if self.operator_memory is not None:
            operator_emb = self.operator_memory(z_proj, return_attention=False)
            return torch.cat([concept_emb, operator_emb], dim=-1)
        return concept_emb

    def _head_input(
        self,
        h_next: torch.Tensor,
        mem_feat: torch.Tensor | None,
    ) -> torch.Tensor:
        if mem_feat is None or self.mem_proj is None:
            return h_next
        return h_next + self.mem_proj(mem_feat)

    def step(
        self,
        z: torch.Tensor,
        prev_action: torch.Tensor,
        mission: torch.Tensor,
        h_prev: torch.Tensor,
        mem_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrent step. Returns (logits, h_next)."""
        idx = prev_action.clone()
        idx[idx < 0] = self.no_action_index
        ae = self.action_emb(idx)
        me = self.mission_proj(mission)
        pe = self._perceive(z)
        x = torch.cat([pe, ae, me], dim=-1)
        h_next = self.gru(x, h_prev)
        h_eff = self._head_input(h_next, mem_feat)
        logits = self.policy_head(h_eff)
        return logits, h_next

    def step_with_value(
        self,
        z: torch.Tensor,
        prev_action: torch.Tensor,
        mission: torch.Tensor,
        h_prev: torch.Tensor,
        mem_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One recurrent step that ALSO returns the state value.
        Drop-in compatible with RecurrentPolicy.step_with_value.
        """
        idx = prev_action.clone()
        idx[idx < 0] = self.no_action_index
        ae = self.action_emb(idx)
        me = self.mission_proj(mission)
        pe = self._perceive(z)
        x = torch.cat([pe, ae, me], dim=-1)
        h_next = self.gru(x, h_prev)
        h_eff = self._head_input(h_next, mem_feat)
        logits = self.policy_head(h_eff)
        value = self.value_head(h_eff).squeeze(-1)
        return logits, value, h_next

    def forward(
        self,
        z_seq: torch.Tensor,
        action_seq: torch.Tensor,
        mission: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the policy across a full sequence (used by BC training)."""
        B, T = action_seq.shape
        h = self.init_hidden(B, z_seq.device)
        logits_seq = []
        for t in range(T):
            if t == 0:
                prev_a = torch.full(
                    (B,), -1, device=z_seq.device, dtype=torch.long
                )
            else:
                prev_a = action_seq[:, t - 1]
            z_t = z_seq[:, t]
            logits, h = self.step(z_t, prev_a, mission, h)
            logits_seq.append(logits)
        return torch.stack(logits_seq, dim=1)

    # -- Slot inspection / continual learning hooks --

    @torch.no_grad()
    def get_active_concepts(self, z: torch.Tensor, threshold: float = 0.05):
        """Return active concept slots for inspection (e.g., by ConceptManager)."""
        z_proj = self.latent_proj(z)
        return self.concept_memory.get_active_slots(z_proj, threshold=threshold)

    @torch.no_grad()
    def get_active_operator(self, z: torch.Tensor) -> tuple[int, float]:
        """Return (best_operator_slot, confidence) for inspection."""
        if self.operator_memory is None:
            return -1, 0.0
        z_proj = self.latent_proj(z)
        return self.operator_memory.select_operator(z_proj)
