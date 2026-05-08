"""ObjectTracker — probes JEPA latents for entity presence + position,
then assigns persistent IDs across consecutive frames.

Two-part design:

  1. ObjectProbe (learned MLP)
       Input : (latent_z, query_type_id, query_color_id)
       Output: (is_present_logit, world_x, world_y)
     Trained supervised: ground-truth labels come from
     `prism.perception.slots.extract_slots_from_normalized` applied to
     the raw observation each latent was produced from. The probe is a
     small MLP — the JEPA's representations should already separate
     entities, so we just need to read them out.

  2. PersistentTracker (greedy matching)
       Per consecutive frame pair, match (type, color) detections from
     the probe by nearest-neighbor world-position. Assign / re-use
     entity IDs. New detections get fresh IDs. Lost detections are
     remembered for `lifespan` frames so flickering doesn't break
     persistence.

Phase 1 emergence test: ≥85% per-step entity-ID accuracy on held-out
BabyAI rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- probe
class ObjectProbe(nn.Module):
    """MLP probe: (latent, query) → (presence, world_pos).

    The latent is encoded once per frame; the probe is queried per
    candidate (type, color). For BabyAI's 4 types × 6 colors = 24
    queries per frame, this is cheap.
    """

    def __init__(self, latent_dim: int, n_types: int = 11, n_colors: int = 6,
                 hidden: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.type_emb = nn.Embedding(n_types, 32)
        self.color_emb = nn.Embedding(n_colors, 32)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 32 + 32, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),               # (presence_logit, x, y)
        )

    def forward(self, z: torch.Tensor, type_id: torch.Tensor,
                color_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z (B, D), type/color ids (B,) → (presence_logit (B,), pos (B, 2))."""
        if z.dim() > 2:
            z = z.flatten(1)
        te = self.type_emb(type_id)
        ce = self.color_emb(color_id)
        h = self.net(torch.cat([z, te, ce], dim=-1))
        presence = h[:, 0]
        pos = h[:, 1:]                          # (B, 2)
        return presence, pos

    def loss(self, z, type_id, color_id, gt_present, gt_pos) -> dict:
        """Binary cross-entropy on presence + MSE on pos (only for present
        objects). gt_present (B,) float 0/1. gt_pos (B, 2)."""
        presence_logit, pos_pred = self.forward(z, type_id, color_id)
        bce = F.binary_cross_entropy_with_logits(presence_logit, gt_present)
        # Only count position loss where gt_present == 1
        mse = ((pos_pred - gt_pos) ** 2).sum(dim=-1)
        mse_masked = (mse * gt_present).sum() / gt_present.sum().clamp(min=1.0)
        return {
            "loss": bce + 0.1 * mse_masked,
            "bce": bce.detach(),
            "mse": mse_masked.detach(),
            "presence_logit": presence_logit,
            "pos_pred": pos_pred,
        }


# ------------------------------------------------------------------- tracker
@dataclass
class TrackedEntity:
    entity_id: int
    type_id: int
    color_id: int
    last_pos: tuple[float, float]
    last_seen_t: int


@dataclass
class PersistentTracker:
    """Greedy nearest-neighbor matcher across frames.

    Call .step(t, detections) per frame with a list of detections —
    each detection is (type_id, color_id, x, y, presence_score). Returns
    list of (entity_id, type_id, color_id, x, y) for that frame.
    """
    match_radius: float = 2.0          # max world-frame distance to consider a match
    lifespan: int = 5                  # frames an entity can be unseen before pruning
    presence_threshold: float = 0.5
    next_id: int = 0
    entities: dict[int, TrackedEntity] = field(default_factory=dict)

    def reset(self) -> None:
        self.next_id = 0
        self.entities.clear()

    def step(self, t: int, detections: list[tuple[int, int, float, float, float]]
             ) -> list[tuple[int, int, int, float, float]]:
        # Filter by presence threshold first.
        live = [(ti, ci, x, y) for (ti, ci, x, y, p) in detections
                if p >= self.presence_threshold]

        # For each live detection, find best matching existing entity of
        # the same (type, color) within match_radius. Greedy by distance.
        used_existing: set[int] = set()
        out: list[tuple[int, int, int, float, float]] = []

        # Compute candidate (det_idx, ent_id, dist) tuples, sort by dist.
        candidates: list[tuple[float, int, int]] = []
        for di, (ti, ci, x, y) in enumerate(live):
            for eid, ent in self.entities.items():
                if ent.type_id != ti or ent.color_id != ci:
                    continue
                ex, ey = ent.last_pos
                dist = ((ex - x) ** 2 + (ey - y) ** 2) ** 0.5
                if dist <= self.match_radius:
                    candidates.append((dist, di, eid))
        candidates.sort()

        matched_dets: set[int] = set()
        for dist, di, eid in candidates:
            if di in matched_dets or eid in used_existing:
                continue
            # Match this detection to this entity.
            ti, ci, x, y = live[di]
            self.entities[eid].last_pos = (float(x), float(y))
            self.entities[eid].last_seen_t = t
            out.append((eid, ti, ci, float(x), float(y)))
            matched_dets.add(di)
            used_existing.add(eid)

        # Unmatched detections → new entities.
        for di, (ti, ci, x, y) in enumerate(live):
            if di in matched_dets:
                continue
            eid = self.next_id
            self.next_id += 1
            self.entities[eid] = TrackedEntity(
                entity_id=eid, type_id=ti, color_id=ci,
                last_pos=(float(x), float(y)), last_seen_t=t,
            )
            out.append((eid, ti, ci, float(x), float(y)))

        # Prune entities not seen recently.
        stale = [eid for eid, e in self.entities.items()
                 if t - e.last_seen_t > self.lifespan]
        for eid in stale:
            del self.entities[eid]

        return out


# ------------------------------------------------------- supervision builder
def build_training_examples(
    rollout: dict[str, Any],
    *,
    n_types: int = 11,
    n_colors: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert one rollout into supervised training tuples for ObjectProbe.

    For each (latent_t, slot_t) pair in the rollout, generate one
    training example per (type, color) query — most are negative
    (presence=0), positives come from the slot extraction.

    Returns (latents, type_ids, color_ids, gt_present, gt_pos):
      latents     (N, D)     float32
      type_ids    (N,)       int64
      color_ids   (N,)       int64
      gt_present  (N,)       float32 (0 or 1)
      gt_pos      (N, 2)     float32
    """
    latents = rollout["latents"]                # (T, D) or (T, C, H, W)
    slots = rollout["slots"]                    # list of length T, each a list of slot dicts
    if latents.ndim > 2:
        latents = latents.reshape(latents.shape[0], -1)
    T, D = latents.shape

    items: list[tuple[np.ndarray, int, int, float, np.ndarray]] = []
    for t in range(T):
        # Build a (type, color) → (x, y) map for this frame.
        present_map: dict[tuple[int, int], tuple[float, float]] = {}
        for s in slots[t]:
            present_map[(int(s["type_id"]), int(s["color_id"]))] = (
                float(s["x"]), float(s["y"])
            )
        # Emit one example per (type, color) — sample negatives at low rate
        # to keep the dataset balanced (most queries are absent).
        for ti in range(n_types):
            for ci in range(n_colors):
                key = (ti, ci)
                if key in present_map:
                    items.append((latents[t], ti, ci, 1.0, np.array(present_map[key], dtype=np.float32)))
                # Else: sample a negative with prob 0.1 to avoid 90:1 imbalance
                elif np.random.random() < 0.1:
                    items.append((latents[t], ti, ci, 0.0, np.zeros(2, dtype=np.float32)))

    L = np.stack([i[0] for i in items])
    T_ = np.array([i[1] for i in items], dtype=np.int64)
    C_ = np.array([i[2] for i in items], dtype=np.int64)
    P = np.array([i[3] for i in items], dtype=np.float32)
    XY = np.stack([i[4] for i in items])
    return L, T_, C_, P, XY
