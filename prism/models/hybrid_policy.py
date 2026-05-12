"""HybridPolicy — the full PRISM-Hybrid v5.0 architecture.

Combines:
- JEPA encoder (frozen or co-trained)
- ConceptMemory (Hopfield) for concept slots
- OperatorMemory (Hopfield) for behavioral primitives
- TransformerDynamics (HopfieldEncoderLayer stack) for world model + policy trunk
- ConceptToText (transformer decoder) for language generation
- Hooks for ConceptManager (async LLM bootstrap)

This is a drop-in replacement for RecurrentPolicy. It exposes the same
PPO-compatible interface (step_with_value), plus new methods for:
- Generating natural language descriptions of state
- Inspecting active concept slots
- Imagining future rollouts via the world model
"""

from __future__ import annotations

import torch
import torch.nn as nn

from prism.cog_core.concept_memory import ConceptMemory
from prism.cog_core.operator_memory import OperatorMemory
from prism.language.concept_to_text import ConceptToText
from prism.models.transformer_dynamics import TransformerDynamics, TransformerDynamicsStep


class HybridPolicy(nn.Module):
    """PRISM-Hybrid v5.0 policy.

    Drop-in replacement for RecurrentPolicy with the same step_with_value
    interface used by ppo_train.py. Internally uses Hopfield memories
    and a transformer dynamics trunk instead of GRU + fixed predicates.
    """

    def __init__(
        self,
        latent_in_dim: int,
        n_actions: int,
        mission_dim: int = 24,
        concept_n_slots: int = 1024,
        concept_slot_dim: int = 64,
        operator_n_slots: int = 64,
        operator_slot_dim: int = 64,
        dynamics_token_dim: int = 128,
        dynamics_layers: int = 4,
        dynamics_heads: int = 4,
        dynamics_buffer: int = 16,
        vocab_size: int = 2048,
        lang_hidden_dim: int = 192,
        lang_n_heads: int = 4,
        lang_n_layers: int = 3,
        latent_proj_dim: int = 128,
        enable_language: bool = True,
    ):
        # Validate head/dim constraints up front with clear messages.
        if lang_hidden_dim % lang_n_heads != 0:
            valid = [n for n in (1, 2, 4, 6, 8, 12, 16) if lang_hidden_dim % n == 0]
            raise ValueError(
                f"lang_hidden_dim ({lang_hidden_dim}) must be divisible by "
                f"lang_n_heads ({lang_n_heads}). Valid n_heads for this dim: {valid}"
            )
        if dynamics_token_dim % dynamics_heads != 0:
            valid = [n for n in (1, 2, 4, 6, 8, 12, 16) if dynamics_token_dim % n == 0]
            raise ValueError(
                f"dynamics_token_dim ({dynamics_token_dim}) must be divisible by "
                f"dynamics_heads ({dynamics_heads}). Valid n_heads for this dim: {valid}"
            )
        super().__init__()
        self.n_actions = n_actions
        self.latent_in_dim = latent_in_dim
        self.mission_dim = mission_dim
        self.concept_n_slots = concept_n_slots
        self.concept_slot_dim = concept_slot_dim
        self.operator_n_slots = operator_n_slots
        self.dynamics_buffer = dynamics_buffer
        self.enable_language = enable_language
        # Keep these attrs for ppo_train.py checkpoint compatibility.
        self.hidden_dim = dynamics_token_dim
        self.latent_proj_dim = latent_proj_dim
        self.mem_feat_dim = 0

        # Project flat JEPA latent down to working dim.
        self.latent_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_in_dim, latent_proj_dim),
            nn.GELU(),
        )

        # Mission embedding (one-hot → continuous).
        self.mission_emb = nn.Linear(mission_dim, dynamics_token_dim)

        # Concept memory: stores object/predicate concepts.
        self.concept_memory = ConceptMemory(
            latent_dim=latent_proj_dim,
            n_slots=concept_n_slots,
            slot_dim=concept_slot_dim,
            n_heads=4,
            scaling=1.0,    # metastable regime for composition
            update_steps=0,
        )

        # Operator memory: stores behavioral primitives.
        self.operator_memory = OperatorMemory(
            latent_dim=latent_proj_dim,
            n_slots=operator_n_slots,
            slot_dim=operator_slot_dim,
            n_heads=4,
            scaling=4.0,    # sharper for precise operator selection
            update_steps=3,
        )

        # Transformer dynamics — the central trunk.
        self.dynamics = TransformerDynamics(
            concept_dim=concept_slot_dim,
            n_actions=n_actions,
            mission_dim=dynamics_token_dim,
            token_dim=dynamics_token_dim,
            n_layers=dynamics_layers,
            n_heads=dynamics_heads,
            ffn_dim=dynamics_token_dim * 4,
            max_seq_len=dynamics_buffer * 2,
        )
        self.dynamics_step = TransformerDynamicsStep(
            self.dynamics, buffer_size=dynamics_buffer
        )

        # Language head (optional — saves memory if disabled).
        if enable_language:
            self.language_head = ConceptToText(
                vocab_size=vocab_size,
                concept_dim=concept_slot_dim,
                hidden_dim=lang_hidden_dim,
                n_layers=lang_n_layers,
                n_heads=lang_n_heads,
                ffn_dim=lang_hidden_dim * 2,
                max_len=48,
            )
        else:
            self.language_head = None
        self.lang_n_heads = lang_n_heads
        self.lang_n_layers = lang_n_layers

    def init_hidden(self, batch_size: int, device: torch.device) -> dict:
        """Returns the buffer used as recurrent state."""
        return self.dynamics_step.init_buffer(batch_size, device)

    def _project_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.latent_proj(z)

    def _encode_mission(self, mission: torch.Tensor) -> torch.Tensor:
        # (B, mission_dim_in) → (B, token_dim)
        return self.mission_emb(mission)

    def forward(
        self,
        z: torch.Tensor,
        prev_action: torch.Tensor,
        mission: torch.Tensor,
        buffer: dict,
    ) -> dict[str, torch.Tensor]:
        """One full step. Returns logits, value, retrieved concepts, attention,
        operator selection, new buffer."""
        # 1. Project JEPA latent.
        z_proj = self._project_latent(z)  # (B, latent_proj_dim)

        # 2. Retrieve concept from memory.
        concept, concept_attn = self.concept_memory(
            z_proj, return_attention=True
        )  # concept: (B, concept_slot_dim), attn: (B, n_slots)

        # 3. Retrieve operator (used as residual context for policy).
        operator, operator_attn = self.operator_memory(
            z_proj, return_attention=True
        )

        # 4. Encode mission.
        mission_e = self._encode_mission(mission)

        # 5. Replace no-action sentinel (-1) with no_action_idx for embedding.
        prev_a_safe = prev_action.clamp(min=0)
        prev_a_safe = torch.where(
            prev_action < 0,
            torch.full_like(prev_a_safe, self.dynamics.no_action_idx),
            prev_a_safe,
        )

        # 6. Step the transformer dynamics over working memory.
        out, new_buffer = self.dynamics_step.step(
            concept=concept,
            prev_action=prev_a_safe,
            buffer=buffer,
            mission_emb=mission_e,
        )

        # 7. Outputs.
        return {
            "logits": out["action_logits"],
            "value": out["value"],
            "next_concept_pred": out["next_concept"],
            "reward_pred": out["reward"],
            "hidden": out["hidden"],
            "concept": concept,
            "concept_attn": concept_attn,
            "operator": operator,
            "operator_attn": operator_attn,
            "z_proj": z_proj,
            "buffer": new_buffer,
        }

    # --- ppo_train.py-compatible interface ---

    def step_with_value(
        self,
        z: torch.Tensor,
        prev_action: torch.Tensor,
        mission: torch.Tensor,
        h_prev,
        mem_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Mirrors RecurrentPolicy.step_with_value for PPO loop compatibility.

        h_prev is reinterpreted as the dynamics buffer. mem_feat is ignored
        (the transformer's working memory replaces explicit mem features).
        """
        # If h_prev is the zero tensor that ppo_train.py initializes, build
        # an empty buffer instead.
        if isinstance(h_prev, torch.Tensor):
            buffer = self.dynamics_step.init_buffer(z.size(0), z.device)
        else:
            buffer = h_prev

        out = self.forward(z, prev_action, mission, buffer)
        return out["logits"], out["value"], out["buffer"]

    # --- Language generation ---

    @torch.no_grad()
    def generate_description(
        self,
        z: torch.Tensor,
        mission: torch.Tensor,
        buffer: dict | None = None,
        max_len: int = 32,
        top_k_concepts: int = 5,
    ) -> tuple[torch.Tensor, dict]:
        """Generate a natural-language description of the current scene + intent.

        Returns:
            tokens: (B, generated_len) — token IDs from the language head
            debug_info: dict with active concepts, attention weights, etc.
        """
        if self.language_head is None:
            raise RuntimeError("Language head is disabled. Set enable_language=True.")

        z_proj = self._project_latent(z)
        if buffer is None:
            buffer = self.dynamics_step.init_buffer(z.size(0), z.device)

        # Get top-k retrieved concepts.
        concept, concept_attn = self.concept_memory(z_proj, return_attention=True)

        # Get top-k slot embeddings as cross-attention memory for the language head.
        topk_indices, topk_weights = self.concept_memory.get_top_k_slots(
            z_proj, k=top_k_concepts
        )
        # Build (B, K, concept_dim) memory: use concept embedding repeated weighted
        # by topk_weights. For richer memory, we extract the actual slot vectors.
        retrieved = self._extract_slot_vectors(topk_indices)  # (B, K, slot_dim)

        # Run the dynamics to get current hidden state.
        mission_e = self._encode_mission(mission)
        prev_a_dummy = torch.full(
            (z.size(0),), self.dynamics.no_action_idx,
            dtype=torch.long, device=z.device,
        )
        out, _ = self.dynamics_step.step(
            concept=concept,
            prev_action=prev_a_dummy,
            buffer=buffer,
            mission_emb=mission_e,
        )

        tokens = self.language_head.generate(
            retrieved_concepts=retrieved,
            trunk_hidden=out["hidden"],
            max_len=max_len,
        )
        debug_info = {
            "topk_slots": topk_indices,
            "topk_weights": topk_weights,
            "concept_attn": concept_attn,
        }
        return tokens, debug_info

    @torch.no_grad()
    def _extract_slot_vectors(self, slot_indices: torch.Tensor) -> torch.Tensor:
        """Look up V-bank entries by slot index.

        slot_indices: (B, K) long → (B, K, concept_slot_dim).
        """
        return self.concept_memory.get_slot_values(slot_indices)


def build_hybrid_from_checkpoint_args(args_dict: dict) -> HybridPolicy:
    """Construct a HybridPolicy from a saved-args dict for ppo_train compatibility."""
    return HybridPolicy(
        latent_in_dim=args_dict["latent_in_dim"],
        n_actions=args_dict["n_actions"],
        mission_dim=args_dict.get("mission_dim", 24),
        concept_n_slots=args_dict.get("concept_n_slots", 1024),
        concept_slot_dim=args_dict.get("concept_slot_dim", 64),
        operator_n_slots=args_dict.get("operator_n_slots", 64),
        operator_slot_dim=args_dict.get("operator_slot_dim", 64),
        dynamics_token_dim=args_dict.get("dynamics_token_dim", 128),
        dynamics_layers=args_dict.get("dynamics_layers", 4),
        dynamics_heads=args_dict.get("dynamics_heads", 4),
        dynamics_buffer=args_dict.get("dynamics_buffer", 16),
        vocab_size=args_dict.get("vocab_size", 2048),
        lang_hidden_dim=args_dict.get("lang_hidden_dim", 192),
        lang_n_heads=args_dict.get("lang_n_heads", 4),
        lang_n_layers=args_dict.get("lang_n_layers", 3),
        latent_proj_dim=args_dict.get("latent_proj_dim", 128),
        enable_language=args_dict.get("enable_language", True),
    )
