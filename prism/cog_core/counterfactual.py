"""CounterfactualEngine — given (state, actual_action, cf_action),
roll both forward via the frozen JEPA and measure how their predictions
diverge.

Tests the cognitive-core question: does the world model encode CAUSAL
structure, not just statistical correlation? If yes, swapping
"forward" for "rotate" at the same state should produce a divergence
in latents whose DIRECTION is physically meaningful (e.g. predicate
flips happen in the right way: "facing(goal)" stays True after
forward but flips after rotate).

Phase 1 emergence test: ≥80% of counterfactual swaps produce
predicate divergence in the physically correct direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prism.cog_core.world_model_rollout import WorldModelRollout


@dataclass
class CounterfactualResult:
    actual_z: torch.Tensor                     # (B, ...) state after actual action
    cf_z: torch.Tensor                         # (B, ...) state after counterfactual action
    l2: torch.Tensor                           # (B,) L2 latent distance
    cosine: torch.Tensor                       # (B,) cosine similarity (high = similar)
    actual_predicates: torch.Tensor | None     # (B, n_pred) sigmoid probs (None if no aux head)
    cf_predicates: torch.Tensor | None
    predicate_flips: torch.Tensor | None       # (B, n_pred) -1/0/+1 of how each pred flipped


class CounterfactualEngine:
    def __init__(self, world: WorldModelRollout):
        self.world = world

    @torch.no_grad()
    def compare(
        self,
        z_0: torch.Tensor,                     # (B, ...) start state(s)
        actual_action: torch.Tensor,           # (B,) int64 action that was taken
        cf_action: torch.Tensor,               # (B,) int64 alternative action
        n_steps: int = 1,
    ) -> CounterfactualResult:
        """Roll both branches n_steps forward. Returns full divergence info."""
        # Build (B, n_steps) action sequences. For n_steps > 1 we repeat
        # the cf_action for the first step then apply random follow-ups
        # (callers can also pass full sequences via the rollout method).
        if n_steps == 1:
            actual_z = self.world.step(z_0, actual_action)
            cf_z = self.world.step(z_0, cf_action)
        else:
            # Just the first step diverges; following steps use the same
            # actions to isolate the divergence to the first decision.
            # (Callers wanting full divergent rollouts use rollout() directly.)
            tail_actions = torch.zeros(z_0.shape[0], n_steps - 1,
                                       dtype=torch.long, device=z_0.device)
            actual_seq = torch.cat([actual_action.unsqueeze(1), tail_actions], dim=1)
            cf_seq = torch.cat([cf_action.unsqueeze(1), tail_actions], dim=1)
            actual_z = self.world.rollout(z_0, actual_seq)[:, -1]
            cf_z = self.world.rollout(z_0, cf_seq)[:, -1]

        diff = self.world.latent_diff(actual_z, cf_z)
        actual_pred = self.world.predicates(actual_z)
        cf_pred = self.world.predicates(cf_z)

        flips = None
        if actual_pred is not None and cf_pred is not None:
            actual_b = (actual_pred > 0.5).float()
            cf_b = (cf_pred > 0.5).float()
            # +1 = flipped from False → True under cf; -1 = True → False; 0 = same.
            flips = cf_b - actual_b

        return CounterfactualResult(
            actual_z=actual_z,
            cf_z=cf_z,
            l2=diff["l2"],
            cosine=diff["cos"],
            actual_predicates=actual_pred,
            cf_predicates=cf_pred,
            predicate_flips=flips,
        )

    @torch.no_grad()
    def coherence_score(
        self,
        z_0: torch.Tensor,
        actual_action: torch.Tensor,
        cf_action: torch.Tensor,
        expected_flip_indices: list[int] | None = None,
    ) -> float:
        """For 'physically correct direction' check: pass the predicate
        indices we EXPECT to flip given the action swap. Returns the
        fraction of (B,) examples where at least one expected predicate
        actually flipped.

        If expected_flip_indices is None, returns 0.0 — caller must
        supply expectations from env-specific knowledge.
        """
        result = self.compare(z_0, actual_action, cf_action, n_steps=1)
        if result.predicate_flips is None or expected_flip_indices is None:
            return 0.0
        rel = result.predicate_flips[:, expected_flip_indices]
        # "At least one expected predicate flipped" — direction-agnostic
        # for now (flipped True→False or False→True both count). Phase 2
        # can require the SIGN to match expectation too.
        any_flipped = (rel.abs().sum(dim=-1) > 0).float()
        return float(any_flipped.mean().item())
