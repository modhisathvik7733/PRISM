"""LatentPlanner — reward-free planning in JEPA latent space.

Two planning modes:
  beam_search(z_start, z_goal)    — goal-directed: find the action sequence
      whose predicted dynamics trajectory ends closest to z_goal.
  novelty_plan(z_start, z_memory) — curiosity-driven: find the sequence that
      maximizes distance from all previously visited psi-states.

Both use batched beam expansion: all K beams × 17 actions expanded in a
single dynamics call per horizon step (no Python loop over actions).

Shapes:
  obs:      (3, 64, 64)   float32 or uint8  — single frame, no batch dim
  z:        (E,)          float32            — single latent (E=embed_dim)
  z_batch:  (B, E)       float32            — beam states
  z_memory: (M, E)       float32            — visited-state buffer (M may be 0)

Beam expansion per step:
  K beams × n_actions = K*n candidates
  z_tiled: (K*n, E)  via repeat_interleave(n_actions, dim=0)
  a_all:   (K*n,)    via arange(n_actions).repeat(K)
  z_next:  (K*n, E) — one dynamics forward pass, no loop
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPlanner:
    def __init__(
        self,
        encoder: nn.Module,
        dynamics: nn.Module,
        n_actions: int = 17,
        device: torch.device | str = "cpu",
    ):
        self.encoder   = encoder
        self.dynamics  = dynamics
        self.n_actions = n_actions
        self.device    = torch.device(device)
        self.encoder.eval()
        self.dynamics.eval()

    @torch.no_grad()
    def encode_obs(self, obs: np.ndarray) -> torch.Tensor:
        """obs: (3, 64, 64) uint8 or float32 → z: (E,) no batch dim."""
        if obs.dtype == np.uint8:
            obs = obs.astype(np.float32) / 255.0
        t = torch.from_numpy(obs).to(self.device).unsqueeze(0)  # (1, 3, 64, 64)
        return self.encoder(t).squeeze(0)                        # (E,)

    @torch.no_grad()
    def rollout(self, z_start: torch.Tensor, actions: List[int]) -> List[torch.Tensor]:
        """Unroll dynamics for an action sequence. Returns [z_1, ..., z_H] each (E,)."""
        z = z_start.unsqueeze(0)  # (1, E)
        zs: List[torch.Tensor] = []
        for a in actions:
            a_t = torch.tensor([a], dtype=torch.long, device=self.device)
            z = self.dynamics(z, a_t)
            zs.append(z.squeeze(0))
        return zs

    @torch.no_grad()
    def _expand(self, z_batch: torch.Tensor) -> torch.Tensor:
        """Expand K beams × n_actions → (K*n, E) next latent states."""
        K = len(z_batch)
        z_tiled = z_batch.repeat_interleave(self.n_actions, dim=0)            # (K*n, E)
        a_all   = torch.arange(self.n_actions, device=self.device).repeat(K)  # (K*n,)
        return self.dynamics(z_tiled, a_all)                                   # (K*n, E)

    @torch.no_grad()
    def beam_search(
        self,
        z_start: torch.Tensor,   # (E,)
        z_goal: torch.Tensor,    # (E,)
        horizon: int = 15,
        beam_k: int = 10,
    ) -> List[int]:
        """Beam search toward z_goal scored on the TERMINAL latent only.

        Pruning at each intermediate step uses current cosine similarity to
        keep the beam diverse, but the final ranking uses only the cos_sim of
        the horizon-th predicted state to z_goal.  This avoids selecting plans
        that pass near the goal early and then drift away.

        Returns the action sequence of the highest-scoring beam (len = horizon).
        """
        z_goal_n = F.normalize(z_goal.unsqueeze(0), dim=-1)  # (1, E)

        beam_z     = z_start.unsqueeze(0)               # (1, E)
        beam_acts: List[List[int]] = [[]]
        # Pruning score: intermediate cosine similarity (keeps beams on track).
        # We track this separately from the terminal score used for final ranking.
        beam_prune = torch.zeros(1, device=self.device)  # (K,)

        for step in range(horizon):
            K      = len(beam_z)
            z_next = self._expand(beam_z)               # (K*n, E)

            z_next_n   = F.normalize(z_next, dim=-1)    # (K*n, E)
            step_cos   = (z_next_n * z_goal_n).sum(dim=-1)  # (K*n,)

            # Prune by cumulative intermediate similarity to keep beam alive.
            prune_scores = beam_prune.repeat_interleave(self.n_actions) + step_cos

            topk = min(beam_k, len(prune_scores))
            _, top_idx = prune_scores.topk(topk)

            parent = top_idx // self.n_actions
            action = top_idx  % self.n_actions

            beam_z     = z_next[top_idx]
            beam_acts  = [beam_acts[int(parent[i])] + [int(action[i])] for i in range(topk)]
            beam_prune = prune_scores[top_idx]

        # Final ranking: cosine similarity of terminal state to goal only.
        terminal_n    = F.normalize(beam_z, dim=-1)                 # (K, E)
        terminal_cos  = (terminal_n * z_goal_n).sum(dim=-1)         # (K,)
        best          = int(terminal_cos.argmax())
        return beam_acts[best]

    @torch.no_grad()
    def novelty_plan(
        self,
        z_start: torch.Tensor,   # (E,)
        z_memory: torch.Tensor,  # (M, E) — M may be 0
        horizon: int = 15,
        beam_k: int = 10,
    ) -> List[int]:
        """Beam search maximizing novelty = 1 - max_cosine_sim to z_memory.

        If z_memory is empty, every state scores 1.0 (all equally novel).
        Returns the action sequence of the highest-scoring beam (len = horizon).
        """
        has_memory = z_memory.numel() > 0
        if has_memory:
            z_mem_n = F.normalize(z_memory, dim=-1)  # (M, E)

        beam_z      = z_start.unsqueeze(0)            # (1, E)
        beam_acts:  List[List[int]] = [[]]
        beam_scores = torch.zeros(1, device=self.device)

        for _ in range(horizon):
            K      = len(beam_z)
            z_next = self._expand(beam_z)              # (K*n, E)

            if has_memory:
                z_next_n   = F.normalize(z_next, dim=-1)         # (K*n, E)
                sim_mat    = z_next_n @ z_mem_n.T                 # (K*n, M)
                max_sim    = sim_mat.max(dim=-1).values           # (K*n,)
                step_score = 1.0 - max_sim
            else:
                step_score = torch.ones(K * self.n_actions, device=self.device)

            cum_scores = beam_scores.repeat_interleave(self.n_actions) + step_score

            topk = min(beam_k, len(cum_scores))
            top_vals, top_idx = cum_scores.topk(topk)

            parent = top_idx // self.n_actions
            action = top_idx  % self.n_actions

            beam_z      = z_next[top_idx]
            beam_acts   = [beam_acts[int(parent[i])] + [int(action[i])] for i in range(topk)]
            beam_scores = top_vals

        return beam_acts[0]
