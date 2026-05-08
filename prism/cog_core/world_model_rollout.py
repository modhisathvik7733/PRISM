"""WorldModelRollout — wraps the frozen v1.3 JEPA for multi-step
imagination.

JepaWorldModel.predict(z, a) gives one-step prediction. Cognitive-core
components (counterfactual, operators, curriculum) need to roll forward
N steps. This module exposes:

    encode(obs)       — single-step encoding (delegates to jepa.encode)
    rollout(z, acts)  — N-step rollout, returns trajectory in latent space
    predicates(z)     — read aux predicate head (if JEPA was trained with
                         aux_predicate_weight > 0)

Everything is no-grad. JEPA stays frozen.
"""

from __future__ import annotations

import torch

from prism.models.jepa import JepaWorldModel


class WorldModelRollout:
    def __init__(self, jepa: JepaWorldModel, device: torch.device):
        self.jepa = jepa.to(device).eval()
        for p in self.jepa.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, 3, H, W) → latent. Same shape contract as JEPA.encode."""
        return self.jepa.encode(obs.to(self.device))

    @torch.no_grad()
    def step(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """One-step prediction. z (B, ...) + action (B,) → next_z (B, ...)."""
        return self.jepa.predict(z, action.to(self.device))

    @torch.no_grad()
    def rollout(
        self,
        z_0: torch.Tensor,                # (B, ...) initial latent
        actions: torch.Tensor,            # (B, T) int64 action sequence
    ) -> torch.Tensor:
        """Roll forward T steps. Returns (B, T+1, ...) trajectory
        including z_0. Each step uses the JEPA's deterministic
        action-conditioned dynamics."""
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)            # (1, T)
        if z_0.dim() == len(self.jepa.cfg_obs_shape()) if hasattr(self.jepa, "cfg_obs_shape") else False:
            z_0 = z_0.unsqueeze(0)
        actions = actions.to(self.device)
        T = actions.shape[1]
        trajectory = [z_0]
        z = z_0
        for t in range(T):
            z = self.jepa.predict(z, actions[:, t])
            trajectory.append(z)
        return torch.stack(trajectory, dim=1)         # (B, T+1, ...)

    @torch.no_grad()
    def predicates(self, z: torch.Tensor) -> torch.Tensor | None:
        """Read predicate logits from JEPA's aux head. Returns None
        if the JEPA was trained without the aux predicate head."""
        head = getattr(self.jepa, "aux_predicate_head", None)
        if head is None:
            return None
        # The aux head expects flattened latent for flat encoders, full
        # spatial for spatial encoders — JepaWorldModel wires this.
        return torch.sigmoid(head(z))

    @torch.no_grad()
    def latent_diff(self, z_a: torch.Tensor, z_b: torch.Tensor) -> dict:
        """Standard divergence metrics between two latents (for
        counterfactual + operator-stability eval)."""
        za = z_a.flatten(1) if z_a.dim() > 2 else z_a
        zb = z_b.flatten(1) if z_b.dim() > 2 else z_b
        l2 = ((za - zb) ** 2).sum(dim=-1).sqrt()
        cos = torch.nn.functional.cosine_similarity(za, zb, dim=-1)
        return {"l2": l2, "cos": cos}
