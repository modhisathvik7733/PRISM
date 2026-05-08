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
    # "flat"        — original conv-on-normalized-floats encoder.
    # "categorical" — looks up per-cell embeddings for (type, color, state)
    #                 triples, then convs. Required for object-grounded
    #                 representations to be linearly readable downstream.
    encoder_type: str = "categorical"
    # Vocabulary sizes for categorical encoder. Defaults match minigrid.
    n_types: int = 12      # 0..10 used; +1 safety
    n_colors: int = 7      # 0..5 used; +1 safety (covers "no color" sentinel cells)
    n_states: int = 4
    type_emb_dim: int = 32
    color_emb_dim: int = 16
    state_emb_dim: int = 8
    # Per-cell normalization maxes (must match prism.envs.babyai._CHANNEL_MAX).
    # Used by the categorical encoder to recover integer codes from the
    # normalized [0, 1] obs that wrappers produce.
    channel_max: tuple[float, float, float] = (11.0, 6.0, 4.0)
    # Auxiliary predicate-supervised loss weight. >0 attaches a small predicate
    # readout head to the online encoder during training and adds BCE against
    # ground-truth predicates to the total loss. This forces the encoder to
    # preserve object-typed information that JEPA's pure predictive loss would
    # otherwise discard (in BabyAI random rollouts, objects don't move, so the
    # encoder can satisfy next-state prediction without encoding object types).
    aux_predicate_weight: float = 0.0
    aux_predicate_dim: int = 96  # PREDICATE_VECTOR_DIM — keep in sync
    # Continuous distance head dimension. >0 extends the aux head output to
    # `aux_predicate_dim + aux_distance_dim`; the binary block uses BCE, the
    # distance block uses MSE on sigmoid output. 24 = NUM_TYPE_COLOR_PAIRS.
    # Distance gives the agent a smooth gradient signal that the binary set
    # cannot — for "target visible 5 cells away," every binary predicate is
    # constant across forward steps, so the agent has no reason to prefer
    # forward over turn. Distance is monotone-decreasing under forward when
    # facing, so forward becomes scoreable. Default 0 = back-compat with old
    # 96-d-only checkpoints.
    aux_distance_dim: int = 0
    aux_distance_weight: float = 1.0  # MSE scale relative to BCE
    # LatentDynamics capacity. Defaults reproduce the original 3-linear-layer
    # MLP (Linear(in,h)-GELU-Linear(h,h)-GELU-Linear(h,out)). dynamics_layers
    # counts the (Linear+GELU) blocks before the output projection — so 2 = current.
    # Bumping these is the Fix-A test for the rotation-prediction failure
    # (turn-action F1 ~0.55 while forward F1 ~0.93 in eval_dynamics_predicates).
    dynamics_hidden_dim: int = 256
    dynamics_layers: int = 2
    # "mlp"          — concat(z, action_embed) → MLP. Original architecture.
    # "film"         — flat-latent FiLM: action produces (gamma, beta) per linear
    #                  block. Falsified for rotation: turn F1 unchanged at ~0.55.
    # "spatial_film" — convolutional FiLM dynamics over a spatial latent
    #                  (B, C, H, W). Action gates each conv block channel-wise.
    #                  Combined with encoder_type="categorical_spatial", this is
    #                  the LeWorldModel-style architecture: rotations become
    #                  approximate spatial permutations a CNN can express.
    dynamics_type: str = "mlp"
    # Channel count for the spatial encoder / spatial dynamics. Only used when
    # encoder_type="categorical_spatial" / dynamics_type="spatial_film".
    spatial_channels: int = 64
    spatial_action_dim: int = 32  # FiLM action-embedding dim for spatial dynamics


class GridEncoder(nn.Module):
    """Tiny conv encoder for 7x7x3 BabyAI partial views, treating the obs as
    a normalized continuous tensor. Kept for backward compat with the
    `flat` encoder type — diagnostic runs showed that on BabyAI's symbolic
    obs this encoder doesn't preserve object identity well enough to support
    linear predicate readout. Prefer `CategoricalGridEncoder` for new runs.
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


class CategoricalGridEncoder(nn.Module):
    """Categorical-embedding conv encoder for BabyAI symbolic obs.

    Why this exists:
      The default `GridEncoder` consumes (3, 7, 7) normalized floats. That
      treats categorical codes (object type, color, state) as continuous —
      the encoder has to reverse-engineer "type ≈ 0.55 means ball" from a
      scalar, and conjunctions like "type==ball AND color==red at the same
      cell" become hard to compute. Probe diagnostics (Phase 2 v0) showed
      this fails: linear probe F1 ≈ 0.09, MLP-256 probe F1 ≈ 0.23.

    What this does instead:
      For each cell, look up three small embeddings (type, color, state),
      concatenate them, and treat the result as a per-cell feature map.
      Conv over that, then global pool / flatten + MLP to embed_dim.

      This makes object identity a first-class feature: the embedding
      table for `type` directly distinguishes 'ball' from 'box' as
      orthogonal vectors instead of nearby scalars, and the conv can
      cleanly compute "ball-and-red co-located at this cell".

    Input contract:
      The JEPA wrappers in prism.envs.babyai produce (3, 7, 7) float32 in
      roughly [0, 1] (each channel divided by its max). We undo that here
      to recover integer codes for the embedding lookup. This keeps the
      env wrapper interface stable while letting the encoder treat the
      input properly as categorical.
    """

    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.cfg = cfg
        self.type_emb = nn.Embedding(cfg.n_types, cfg.type_emb_dim)
        self.color_emb = nn.Embedding(cfg.n_colors, cfg.color_emb_dim)
        self.state_emb = nn.Embedding(cfg.n_states, cfg.state_emb_dim)
        per_cell_dim = cfg.type_emb_dim + cfg.color_emb_dim + cfg.state_emb_dim

        self.conv = nn.Sequential(
            nn.Conv2d(per_cell_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * cfg.obs_h * cfg.obs_w, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

        # Buffer so it moves with .to(device) and saves with state_dict.
        self.register_buffer(
            "_channel_max",
            torch.tensor(cfg.channel_max, dtype=torch.float32).reshape(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) float32 in [0, 1]. Recover integer codes.
        codes = (x * self._channel_max).round().long()
        codes = codes.clamp(min=0)
        # Clamp each channel to its vocab size to be safe against off-by-one.
        codes_t = codes[:, 0].clamp(max=self.cfg.n_types - 1)     # (B, H, W)
        codes_c = codes[:, 1].clamp(max=self.cfg.n_colors - 1)
        codes_s = codes[:, 2].clamp(max=self.cfg.n_states - 1)

        e_t = self.type_emb(codes_t)    # (B, H, W, type_emb_dim)
        e_c = self.color_emb(codes_c)   # (B, H, W, color_emb_dim)
        e_s = self.state_emb(codes_s)   # (B, H, W, state_emb_dim)
        # Concat along last dim, then permute to (B, C, H, W) for conv.
        feat = torch.cat([e_t, e_c, e_s], dim=-1).permute(0, 3, 1, 2).contiguous()

        h = self.conv(feat)
        return self.head(h)


class CategoricalSpatialEncoder(nn.Module):
    """Categorical-embedding conv encoder that PRESERVES spatial structure.

    Unlike CategoricalGridEncoder (which flattens to a 1-D embed), this
    returns a (B, C, H, W) feature map. The downstream dynamics can then
    operate convolutionally so that rotations of the underlying grid show
    up as approximate spatial permutations — which a CNN can learn — rather
    than as arbitrary 128-d coordinate remappings (which a flat-latent MLP
    cannot, as eval_dynamics_predicates demonstrated empirically: turn-action
    F1 capped at ~0.55 across 4 different flat-latent architectures).
    """

    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.cfg = cfg
        self.type_emb = nn.Embedding(cfg.n_types, cfg.type_emb_dim)
        self.color_emb = nn.Embedding(cfg.n_colors, cfg.color_emb_dim)
        self.state_emb = nn.Embedding(cfg.n_states, cfg.state_emb_dim)
        per_cell_dim = cfg.type_emb_dim + cfg.color_emb_dim + cfg.state_emb_dim
        C = getattr(cfg, "spatial_channels", 64)
        self.conv = nn.Sequential(
            nn.Conv2d(per_cell_dim, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, C, 3, padding=1), nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1),
        )
        self.register_buffer(
            "_channel_max",
            torch.tensor(cfg.channel_max, dtype=torch.float32).reshape(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        codes = (x * self._channel_max).round().long().clamp(min=0)
        codes_t = codes[:, 0].clamp(max=self.cfg.n_types - 1)
        codes_c = codes[:, 1].clamp(max=self.cfg.n_colors - 1)
        codes_s = codes[:, 2].clamp(max=self.cfg.n_states - 1)
        e_t = self.type_emb(codes_t)
        e_c = self.color_emb(codes_c)
        e_s = self.state_emb(codes_s)
        feat = torch.cat([e_t, e_c, e_s], dim=-1).permute(0, 3, 1, 2).contiguous()
        return self.conv(feat)  # (B, C, H, W)


def _make_encoder(cfg: JepaConfig) -> nn.Module:
    # Backward compat: checkpoints saved before encoder_type existed unpickle
    # to a JepaConfig instance missing the new fields. Default to "flat" for
    # those — that's what they were trained with.
    encoder_type = getattr(cfg, "encoder_type", "flat")
    if encoder_type == "flat":
        return GridEncoder(cfg)
    if encoder_type == "categorical":
        return CategoricalGridEncoder(cfg)
    if encoder_type == "categorical_spatial":
        return CategoricalSpatialEncoder(cfg)
    raise ValueError(
        f"unknown encoder_type {encoder_type!r} "
        "(use 'flat', 'categorical', or 'categorical_spatial')"
    )


def _is_spatial_encoder(cfg: JepaConfig) -> bool:
    return getattr(cfg, "encoder_type", "flat") == "categorical_spatial"


def upgrade_config(cfg: JepaConfig) -> JepaConfig:
    """Patch a (possibly old) JepaConfig instance with default values for any
    fields added after it was saved. Idempotent."""
    fresh = JepaConfig()
    for f in fresh.__dataclass_fields__:
        if not hasattr(cfg, f):
            setattr(cfg, f, getattr(fresh, f))
    return cfg


class LatentDynamics(nn.Module):
    """Predicts z_{t+1} from (z_t, a_t)."""

    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.action_embed = nn.Embedding(cfg.n_actions, cfg.embed_dim)
        # getattr with defaults so older checkpoints (no dynamics_* fields)
        # unpickle with the original architecture.
        h = getattr(cfg, "dynamics_hidden_dim", cfg.hidden_dim)
        n_blocks = max(1, getattr(cfg, "dynamics_layers", 2))
        layers: list[nn.Module] = [nn.Linear(cfg.embed_dim * 2, h), nn.GELU()]
        for _ in range(n_blocks - 1):
            layers.append(nn.Linear(h, h))
            layers.append(nn.GELU())
        layers.append(nn.Linear(h, cfg.embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        ae = self.action_embed(a)
        return self.net(torch.cat([z, ae], dim=-1))


class FiLMBlock(nn.Module):
    """One layer of action-conditioned dynamics.

    h' = (1 + gamma(ae)) * gelu(W h) + beta(ae)

    The (1 + gamma) form initializes near identity-modulation so the block
    does not destroy signal at init.
    """

    def __init__(self, hidden: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.to_gamma = nn.Linear(action_dim, hidden)
        self.to_beta = nn.Linear(action_dim, hidden)
        # Initialize gamma/beta projections to ~zero so blocks start near identity.
        nn.init.zeros_(self.to_gamma.weight); nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight);  nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, ae: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(ae)
        beta = self.to_beta(ae)
        return (1.0 + gamma) * F.gelu(self.linear(h)) + beta


class FiLMDynamics(nn.Module):
    """Action-conditioned dynamics via FiLM modulation at every layer.

    Unlike LatentDynamics (concat-then-MLP), here z is projected first and
    then *every* hidden layer is modulated by the action embedding. The action
    cannot be averaged out across actions because it gates each block's
    activations multiplicatively and additively.
    """

    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.action_embed = nn.Embedding(cfg.n_actions, cfg.embed_dim)
        h = getattr(cfg, "dynamics_hidden_dim", cfg.hidden_dim)
        n_blocks = max(1, getattr(cfg, "dynamics_layers", 2))
        self.in_proj = nn.Linear(cfg.embed_dim, h)
        self.blocks = nn.ModuleList(
            [FiLMBlock(h, cfg.embed_dim) for _ in range(n_blocks)]
        )
        self.out_proj = nn.Linear(h, cfg.embed_dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        ae = self.action_embed(a)
        h = F.gelu(self.in_proj(z))
        for block in self.blocks:
            h = block(h, ae)
        return self.out_proj(h)


class SpatialFiLMBlock(nn.Module):
    """Conv layer with channel-wise action FiLM modulation.

    h' = (1 + gamma(ae))[:, :, None, None] * gelu(conv(h))
         + beta(ae)[:, :, None, None]

    The action embedding produces per-channel (gamma, beta) scalars that
    multiplicatively + additively modulate the GELU-activated conv output.
    Combined with a residual + zero-init out-projection in SpatialFiLMDynamics,
    the dynamics starts as identity (z → z) at initialization, so no-op-like
    actions get the right behavior for free and gradient signal early in
    training perturbs an already-correct baseline.
    """

    def __init__(self, channels: int, action_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.to_gamma = nn.Linear(action_dim, channels)
        self.to_beta = nn.Linear(action_dim, channels)
        nn.init.zeros_(self.to_gamma.weight); nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight);  nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, ae: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(ae).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.to_beta(ae).unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * F.gelu(self.conv(h)) + beta


class SpatialFiLMDynamics(nn.Module):
    """Convolutional dynamics for spatial latents with action-FiLM at every block.

    Operates on (B, C, H, W). Predicts a residual delta added to the input
    latent — so at init (zero out_proj), dynamics(z, a) = z, the correct
    behavior for no-op actions and a stable starting point for learning
    rotation/translation effects.

    Why this fixes what flat-latent FiLM couldn't: rotations correspond to
    coherent spatial transformations of the latent feature map. A 3x3 conv
    naturally expresses local spatial reasoning; stacked convs build up
    receptive field. A flat-latent MLP cannot — empirically, turn-action F1
    capped at ~0.55 regardless of capacity or action conditioning.
    """

    def __init__(self, cfg: JepaConfig):
        super().__init__()
        action_dim = getattr(cfg, "spatial_action_dim", 32)
        self.action_embed = nn.Embedding(cfg.n_actions, action_dim)
        C = getattr(cfg, "spatial_channels", 64)
        n_blocks = max(1, getattr(cfg, "dynamics_layers", 2))
        self.blocks = nn.ModuleList(
            [SpatialFiLMBlock(C, action_dim) for _ in range(n_blocks)]
        )
        self.out_proj = nn.Conv2d(C, C, 1)
        # Zero-init output projection so dynamics(z, a) = z + 0 = z at init.
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        ae = self.action_embed(a)
        h = z
        for block in self.blocks:
            h = block(h, ae)
        return z + self.out_proj(h)


def _make_dynamics(cfg: JepaConfig) -> nn.Module:
    dyn_type = getattr(cfg, "dynamics_type", "mlp")
    if dyn_type == "mlp":
        return LatentDynamics(cfg)
    if dyn_type == "film":
        return FiLMDynamics(cfg)
    if dyn_type == "spatial_film":
        return SpatialFiLMDynamics(cfg)
    raise ValueError(
        f"unknown dynamics_type {dyn_type!r} "
        "(use 'mlp', 'film', or 'spatial_film')"
    )


class JepaWorldModel(nn.Module):
    """Online encoder + EMA target encoder + latent dynamics.

    Counterfactual prediction (`predict_counterfactual`) shares the dynamics
    network — the inductive bias is that *the same dynamics function* should
    explain factual and counterfactual transitions.
    """

    def __init__(self, cfg: JepaConfig | None = None):
        super().__init__()
        self.cfg = cfg or JepaConfig()
        self.online_encoder = _make_encoder(self.cfg)
        self.target_encoder = _make_encoder(self.cfg)
        self.dynamics = _make_dynamics(self.cfg)
        # Auxiliary predicate head — only created if the loss weight is > 0.
        # Linear by design: if a deeper head is required, the encoder isn't
        # actually preserving the structure, just providing material for the
        # head to reconstruct it.
        # Aux head input dim depends on encoder output shape:
        #   flat encoders     → (B, embed_dim)         → Linear(embed_dim, P)
        #   spatial encoder   → (B, C, H, W) flattened → Linear(C*H*W, P)
        # We use Sequential([Flatten, Linear]) for the spatial case so calling
        # `self.aux_predicate_head(z_spatial)` works without callers special-casing.
        # The flat case stays as a bare Linear so older checkpoints' state_dict
        # keys (`aux_predicate_head.weight/bias`) still load.
        self.aux_predicate_head: nn.Module | None = None
        if getattr(self.cfg, "aux_predicate_weight", 0.0) > 0.0:
            # Output is [predicate_logits (P) ‖ distance_logits (D)]. D=0 keeps
            # the head shape identical to old 96-only checkpoints.
            P = self.cfg.aux_predicate_dim
            D = getattr(self.cfg, "aux_distance_dim", 0)
            out_dim = P + D
            if _is_spatial_encoder(self.cfg):
                C = getattr(self.cfg, "spatial_channels", 64)
                in_dim = C * self.cfg.obs_h * self.cfg.obs_w
                self.aux_predicate_head = nn.Sequential(
                    nn.Flatten(), nn.Linear(in_dim, out_dim)
                )
            else:
                self.aux_predicate_head = nn.Linear(self.cfg.embed_dim, out_dim)
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
        predicates_t: torch.Tensor | None = None,
        predicates_tp1: torch.Tensor | None = None,
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

        total = l_pred + self.cfg.reg_weight * l_reg
        out = {
            "loss_pred": l_pred.detach(),
            "loss_reg": l_reg.detach(),
        }

        # Auxiliary predicate-supervised loss. Two heads:
        #   * On `z_t`        — forces encoder to preserve object structure.
        #   * On `z_pred`     — closes the train/inference gap for the agent.
        #     The agent runs `aux_head(dynamics(z_t, a))` at inference, so we
        #     must train the head to read predicates from the dynamics output
        #     (predicates_tp1) and not just from the encoder output (predicates_t).
        # Without the second term, the head was only ever optimized on encoded
        # observations — predicate readout from imagined states was a
        # distribution-shift away, and the agent's score for `forward` actions
        # came out wrong (capstone failure mode in v0.2).
        aux_w = getattr(self.cfg, "aux_predicate_weight", 0.0)
        if aux_w > 0.0 and self.aux_predicate_head is not None:
            P = self.cfg.aux_predicate_dim
            D = getattr(self.cfg, "aux_distance_dim", 0)
            d_w = getattr(self.cfg, "aux_distance_weight", 1.0)
            if predicates_t is not None:
                logits_t = self.aux_predicate_head(z_t)
                l_aux_t = F.binary_cross_entropy_with_logits(
                    logits_t[:, :P], predicates_t[:, :P]
                )
                total = total + aux_w * l_aux_t
                out["loss_aux_t"] = l_aux_t.detach()
                out["loss_aux"] = l_aux_t.detach()  # back-compat
                if D > 0 and predicates_t.shape[1] >= P + D:
                    dist_pred_t = torch.sigmoid(logits_t[:, P:P + D])
                    l_dist_t = F.mse_loss(dist_pred_t, predicates_t[:, P:P + D])
                    total = total + aux_w * d_w * l_dist_t
                    out["loss_dist_t"] = l_dist_t.detach()
            if predicates_tp1 is not None:
                logits_tp1 = self.aux_predicate_head(z_pred)
                l_aux_tp1 = F.binary_cross_entropy_with_logits(
                    logits_tp1[:, :P], predicates_tp1[:, :P]
                )
                total = total + aux_w * l_aux_tp1
                out["loss_aux_tp1"] = l_aux_tp1.detach()
                if D > 0 and predicates_tp1.shape[1] >= P + D:
                    dist_pred_tp1 = torch.sigmoid(logits_tp1[:, P:P + D])
                    l_dist_tp1 = F.mse_loss(dist_pred_tp1, predicates_tp1[:, P:P + D])
                    total = total + aux_w * d_w * l_dist_tp1
                    out["loss_dist_tp1"] = l_dist_tp1.detach()

        out["loss"] = total
        return out
