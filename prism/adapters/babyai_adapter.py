"""BabyAIAdapter — domain interface for BabyAI / MiniGrid environments.

This adapter owns:
  * The JEPA observation encoder (loaded from a v4.x checkpoint).
  * The (color, type) one-hot mission encoder.
  * The MiniGrid action-masking rules (which actions are allowed for
    each mission kind — from v5's `grounded_agent.MISSION_ALLOWED_ACTIONS`).
  * The BabyAI vocabulary (object types, colors).

Resolution 1 (encoder-as-adapter): the JEPA encoder lives HERE, not in
the substrate. The substrate sees only post-encoder latents.

PR-2 scope: thin Phase A adapter. The mission encoding still produces
the v5-compatible 24-d one-hot (color, type) tensor; this is sized via
`mission_dim` and is identical to what v5 EnvWorker builds.
PR-4 / Phase D will revisit mission encoding when adapting to non-BabyAI
domains where missions are variable-length text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from prism.cognition.tokens import TokenStream, TokenType
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.perception.slots import NUM_COLORS, OBJECT_TYPES

# Reproduces v5's allowed-action table (was `grounded_agent.MISSION_ALLOWED_ACTIONS`).
# Resolution: this lives in the BabyAI adapter, NOT in `prism.agents`. The
# substrate never references it.
#   0 = turn left, 1 = turn right, 2 = forward,
#   3 = pickup,    4 = drop,        5 = toggle,    6 = done
MISSION_ALLOWED_ACTIONS: dict[str, tuple[int, ...]] = {
    "at":      (0, 1, 2),         # "go to <X>"
    "holding": (0, 1, 2, 3),      # "pick up <X>"
    "open":    (0, 1, 2, 5),      # "open <door>"
}


def _latent_dim_for_cfg(cfg: JepaConfig) -> int:
    """Mirror of v5 ppo_train.py:latent_dim_for_cfg — moved into the
    adapter so the substrate never inspects JepaConfig directly."""
    enc = getattr(cfg, "encoder_type", "flat")
    if enc == "categorical_spatial":
        C = getattr(cfg, "spatial_channels", 64)
        return C * cfg.obs_h * cfg.obs_w
    return cfg.embed_dim


class BabyAIAdapter:
    """Adapter for BabyAI-family environments (BabyAI-GoToLocal, GoTo,
    GoToObj, PickupLoc, OpenDoor, ...).

    Construction:
        adapter = BabyAIAdapter.from_jepa_checkpoint(
            path="runs/jepa_dev_v1_factored/jepa_final.pt",
            device=torch.device("cuda"),
        )

    The adapter holds the JEPA encoder in eval mode with frozen
    parameters (matches v5 ppo_train.py:428-430). The substrate's
    optimizer never sees these parameters.
    """

    name: str = "babyai"

    def __init__(
        self,
        jepa: JepaWorldModel,
        cfg: JepaConfig,
        device: torch.device,
    ):
        self._jepa = jepa
        self._cfg = cfg
        self._device = device

        # Substrate-facing dimensions. These are READ by UniversalPolicy
        # at construction; they cannot change later (substrate-config-hash).
        self.latent_dim: int = _latent_dim_for_cfg(cfg)
        self.n_actions: int = int(cfg.n_actions)
        self.mission_dim: int = len(OBJECT_TYPES) * NUM_COLORS  # 24 for BabyAI

        # Token-stream advertisements (currently unused in Phase A; the
        # substrate still operates on the latent vector directly via
        # `encode_obs`. PR-4 wires `tokenize` into the trunk path).
        self.n_obs_tokens: int = 1
        self.mission_dim_max: int = 1  # 1 token holding the 24-d one-hot

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_jepa_checkpoint(
        cls,
        path: str | Path,
        device: torch.device,
    ) -> "BabyAIAdapter":
        """Load a frozen JEPA encoder from a v4.x checkpoint.

        Mirrors v5 ppo_train.py:424-430. The JEPA is set to eval mode
        with all parameters detached from the autograd graph.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = upgrade_config(ckpt["cfg"])
        jepa = JepaWorldModel(cfg).to(device)
        jepa.load_state_dict(ckpt["model"])
        jepa.eval()
        for p in jepa.parameters():
            p.requires_grad_(False)
        return cls(jepa=jepa, cfg=cfg, device=device)

    # ------------------------------------------------------------------
    # Encoder ownership (resolution 1)
    # ------------------------------------------------------------------
    def encoder(self) -> nn.Module:
        """The domain-specific observation encoder. Frozen JEPA for
        BabyAI. The substrate's optimizer does NOT iterate over its
        parameters (they have `requires_grad=False`)."""
        return self._jepa

    @torch.no_grad()
    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of (B, 3, 7, 7) BabyAI observations into the
        JEPA latent space. Output shape depends on encoder_type:
        flat → (B, latent_dim); categorical_spatial → (B, C, H, W).

        Used by `UniversalPolicy.step_with_value`. This is the substrate's
        ONLY entry point to the BabyAI encoder.
        """
        return self._jepa.encode(obs)

    # ------------------------------------------------------------------
    # Tokenization (PR-4 will use this for the trunk path; Phase A only)
    # ------------------------------------------------------------------
    def tokenize(
        self,
        obs: torch.Tensor,
        mission: torch.Tensor | None = None,
    ) -> TokenStream:
        """PR-4 entry point. Constructs a TokenStream from raw obs +
        mission one-hot. PR-2 (Phase A) does not route through this —
        the substrate calls `encode_obs` directly.

        For BabyAI:
          - 1 OBS token = flattened JEPA latent (shape (B, latent_dim))
          - 1 MISSION token = (color, type) one-hot embedded to D_tok via
            a learned linear projection. For PR-2, we surface the raw
            one-hot zero-padded to match latent_dim, which is a stand-in
            until PR-4 introduces a proper projection.
        """
        B = obs.size(0)
        device = obs.device

        z = self.encode_obs(obs)  # (B, latent_dim) or (B, C, H, W)
        if z.ndim > 2:
            z = z.flatten(1)
        # OBS token: (B, 1, latent_dim)
        obs_tok = z.unsqueeze(1)

        # MISSION token: pad to latent_dim. Phase A doesn't actually use
        # tokenize() in the live path; this is a placeholder for PR-4.
        if mission is None:
            mission_padded = torch.zeros(B, 1, self.latent_dim, device=device)
        else:
            m = mission
            if m.ndim == 1:
                m = m.unsqueeze(0)
            if m.size(-1) < self.latent_dim:
                pad = torch.zeros(
                    B, self.latent_dim - m.size(-1), device=device, dtype=m.dtype
                )
                m = torch.cat([m, pad], dim=-1)
            mission_padded = m.unsqueeze(1)

        tokens = torch.cat([obs_tok, mission_padded], dim=1)
        types = torch.tensor(
            [int(TokenType.OBS), int(TokenType.MISSION)],
            device=device, dtype=torch.long,
        ).unsqueeze(0).expand(B, 2).contiguous()
        pos = torch.arange(2, device=device, dtype=torch.long).unsqueeze(0).expand(B, 2).contiguous()
        return TokenStream(tokens=tokens, types=types, pos=pos)

    # ------------------------------------------------------------------
    # Action head + masking
    # ------------------------------------------------------------------
    def action_head(self) -> nn.Module | None:
        """The action head is constructed by HybridPolicy in Phase A and
        is therefore not separately owned by the adapter. Returns None
        in PR-2; PR-4 splits the head out as a `DiscreteSmall` decoder
        that the adapter constructs.
        """
        return None

    def mask_logits(
        self,
        logits: torch.Tensor,
        env_state: Any = None,
    ) -> torch.Tensor:
        """Apply per-env action masking.

        `env_state` for BabyAI is a `torch.Tensor` of shape `(B, n_actions)`
        with 1.0 at allowed actions and 0.0 at disallowed actions. This
        matches what EnvWorker already builds; the adapter just applies
        the mask to logits with `-inf` at disallowed positions.

        Substrate guarantee: this is called BEFORE every action sampling.
        Adapters with no masking rules can pass an all-ones tensor or
        None; we treat None as no-op.
        """
        if env_state is None:
            return logits
        if not isinstance(env_state, torch.Tensor):
            raise TypeError(
                f"BabyAIAdapter.mask_logits expects env_state as a Tensor "
                f"(allowed-action mask), got {type(env_state).__name__}"
            )
        # Apply -inf at disallowed positions (where mask == 0).
        return torch.where(
            env_state > 0.5,
            logits,
            torch.full_like(logits, float("-inf")),
        )

    def reward_shaper(self) -> Callable[[Any, Any, float], float] | None:
        """BabyAI's natural reward `1 - 0.9 * (steps/max_steps)` is
        applied by the env itself; no adapter-side shaping. v5 supports
        a distance-based shaping coefficient via `--shaping-coef`;
        that's applied at the training-loop level, not here.
        """
        return None

    # ------------------------------------------------------------------
    # Adapter-side BabyAI utilities (kept here so substrate never imports them)
    # ------------------------------------------------------------------
    @staticmethod
    def allowed_actions_for_predicate(
        predicate: str, n_actions: int
    ) -> tuple[int, ...]:
        """Mirror of v5 grounded_agent.allowed_actions_for_spec. The
        substrate doesn't need this; it's exposed so EnvWorker (or
        whoever builds the env_state masks) can ask the adapter."""
        return MISSION_ALLOWED_ACTIONS.get(predicate, tuple(range(n_actions)))
