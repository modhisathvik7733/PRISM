"""Open up the concept-pretrained PRISM substrate and show what it
'sees' for canonical scenes.

This is interpretability, not retraining. We feed canonical observations
into the trained substrate and report:

  1. The 7x7 obs grid (printable view of what the model sees).
  2. Top-K concept-bank slot activations — which "memories" fire.
  3. Top-K action logits — what the policy would choose right now.
  4. Side-by-side red vs green comparison — does the model truly
     discriminate, and where in the pipeline does the difference appear?

Run:
    python scripts/inspect_substrate_awareness.py \\
        --jepa  runs/v6_concept_phaseAB_v2/jepa.pt \\
        --policy runs/v6_concept_phaseAB_v2/policy.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prism.adapters.babyai_adapter import BabyAIAdapter
from prism.adapters.unity_2d import Unity2DAdapter
from prism.cognition.policy import UniversalPolicy
from prism.models.jepa import JepaWorldModel, upgrade_config
from prism.perception.predicates import type_color_index
from prism.perception.slots import (
    AGENT_POS,
    COLOR_NAMES,
    NUM_COLORS,
    NUM_TYPES,
    OBJECT_NAME_TO_TYPE,
    OBJECT_TYPE_NAMES,
    OBJECT_TYPES,
)


_ACTION_NAMES = {0: "turn_L", 1: "turn_R", 2: "forward",
                 3: "pickup", 4: "drop", 5: "toggle", 6: "done"}


# --------------------------------------------------------------------------
# Checkpoint loading
# --------------------------------------------------------------------------
def load_jepa(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    return jepa, cfg


def build_policy(ckpt: dict, jepa, cfg, device: torch.device, trunk: str):
    adapter = BabyAIAdapter(jepa=jepa, cfg=cfg, device=device)
    policy = UniversalPolicy.from_adapter(
        adapter,
        trunk=trunk,
        hidden_dim=ckpt["hidden_dim"],
        latent_proj_dim=ckpt["latent_proj_dim"],
        mem_feat_dim=ckpt.get("mem_feat_dim", 0),
        concept_n_slots=ckpt.get("concept_n_slots", 1024),
        operator_n_slots=ckpt.get("operator_n_slots", 64),
        concept_scaling=ckpt.get("concept_scaling", 1.0),
        operator_scaling=ckpt.get("operator_scaling", 4.0),
        use_operator_memory=ckpt.get("use_operator_memory", True),
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


# --------------------------------------------------------------------------
# Scene printing (text rendering of the 7x7 obs)
# --------------------------------------------------------------------------
_TYPE_GLYPH = {
    0: ".",      # unseen
    1: " ",      # empty (visible floor)
    2: "#",      # wall
    4: "D",      # door
    5: "k",      # key
    6: "o",      # ball
    7: "B",      # box
    10: "A",     # agent
}
_COLOR_CHAR = {0: "r", 1: "g", 2: "b", 3: "p", 4: "y", 5: "w"}


def print_grid(image_normalized: np.ndarray) -> None:
    """image_normalized: (3, 7, 7) JEPA-normalized obs. Print a 7x7 view."""
    # Un-normalize the type and color channels.
    img = image_normalized * np.array([11.0, 6.0, 4.0]).reshape(3, 1, 1)
    img = img.round().astype(int)
    type_grid = img[0]
    color_grid = img[1]
    print("  +" + "-" * 14 + "+")
    for y in range(7):
        row = "  |"
        for x in range(7):
            t = int(type_grid[y, x])
            c = int(color_grid[y, x])
            glyph = _TYPE_GLYPH.get(t, "?")
            # For colorable objects, append color letter; for agent show A.
            if t in (4, 5, 6, 7):
                ch = _COLOR_CHAR.get(c, "?")
                row += f"{glyph}{ch}"
            elif t == 10:
                row += "A "
            elif t == 0:
                row += ". "
            else:
                row += "  "
        row += "|"
        print(row)
    print("  +" + "-" * 14 + "+")
    print("  legend: A=agent  o*=ball  k*=key  D*=door  B*=box  .=unseen")


# --------------------------------------------------------------------------
# Probe the substrate's concept retrieval + action head
# --------------------------------------------------------------------------
@torch.no_grad()
def probe_one(policy, jepa, obs_np: np.ndarray, mission_str: str,
              device: torch.device, top_k_slots: int = 5) -> dict:
    """Return: concept-slot top-K, action-logit top-3 for the given obs+mission."""
    img = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
    z = jepa.encode(img)
    z_flat = z.flatten(1) if z.ndim > 2 else z

    # Build mission one-hot.
    from prism.language.mission_parser import parse_mission
    spec = parse_mission(mission_str)
    mission = torch.zeros(1, NUM_TYPES * NUM_COLORS, device=device)
    if spec and spec.color_id is not None:
        mission[0, type_color_index(spec.type_id, spec.color_id)] = 1.0

    inner = policy._inner
    retrieval = inner.retrieval
    concept_bank = retrieval.concept_bank

    # Concept query (perception-anchored; no action/mission in the query path).
    obs_token = inner.latent_proj(z_flat)               # (1, D_tok)
    cq = retrieval.concept_base.expand(1, -1) + retrieval.concept_cond(obs_token)
    # Compute the attention weights over concept slots.
    keys = concept_bank.keys                            # (1, n_slots, D_tok)
    K = keys.squeeze(0)                                 # (n_slots, D_tok)
    scaling = float(getattr(concept_bank, "scaling", 1.0))
    attn_logits = (cq @ K.t()) * scaling                # (1, n_slots)
    attn = F.softmax(attn_logits, dim=-1).squeeze(0)    # (n_slots,)
    top_attn, top_idx = attn.topk(top_k_slots)

    # Full policy step (with action_head) — sentinel prev_action.
    h0 = policy.init_hidden(1, device)
    prev_a = torch.tensor([-1], device=device, dtype=torch.long)
    logits, _h_next = policy.step(z_flat, prev_a, mission, h0)
    # Apply the same mask the inference server uses.
    mask = torch.full_like(logits, float("-inf"))
    mask[..., :3] = 0.0
    masked_logits = (logits + mask).squeeze(0)
    top_log, top_act = masked_logits.topk(3)

    return {
        "top_slot_idx": top_idx.cpu().numpy(),
        "top_slot_attn": top_attn.cpu().numpy(),
        "top_actions": top_act.cpu().numpy(),
        "top_action_logits": top_log.cpu().numpy(),
        "full_attn": attn.cpu().numpy(),
        "concept_query": cq.cpu().numpy().squeeze(0),
    }


# --------------------------------------------------------------------------
# Canonical scenes
# --------------------------------------------------------------------------
def make_scene(adapter: Unity2DAdapter, agent_pos, target_xz, target_type,
               target_color, distractor_xz=None, distractor_color=None) -> np.ndarray:
    """Render one scene to (3, 7, 7) obs."""
    objs = [
        (OBJECT_NAME_TO_TYPE[target_type], target_color, target_xz),
    ]
    if distractor_xz is not None:
        objs.append((OBJECT_NAME_TO_TYPE[target_type], distractor_color, distractor_xz))
    return adapter.render_obs_multi(agent_pos, objs)


def main() -> int:
    p = argparse.ArgumentParser(description="Substrate-awareness diagnostic.")
    p.add_argument("--jepa", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--trunk", default="transformer", choices=["transformer", "gru"])
    p.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available()
                 else ("mps" if torch.backends.mps.is_available() else "cpu")),
    )
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[probe] device={device}")
    jepa, cfg = load_jepa(Path(args.jepa), device)
    base_ckpt = torch.load(args.policy, map_location=device, weights_only=False)
    policy = build_policy(base_ckpt, jepa, cfg, device, trunk=args.trunk)
    adapter = Unity2DAdapter(obs_scale=2.0)

    # Canonical scenes designed to isolate one variable each.
    scenes = [
        ("Green ball directly ahead", (0, 0), (0.0, 4.0), "ball", 1, None, None, "go to the green ball"),
        ("Red ball directly ahead",   (0, 0), (0.0, 4.0), "ball", 0, None, None, "go to the green ball"),
        ("Green ball to the right",   (0, 0), (4.0, 0.0), "ball", 1, None, None, "go to the green ball"),
        ("Green ball to the left",    (0, 0), (-4.0, 0.0), "ball", 1, None, None, "go to the green ball"),
        ("Green AND red ball",        (0, 0), (3.0, 3.0), "ball", 1, (-3.0, 3.0), 0, "go to the green ball"),
        ("Red AND green ball, mission says RED", (0, 0), (3.0, 3.0), "ball", 0, (-3.0, 3.0), 1, "go to the red ball"),
    ]

    results = []
    for name, agent_pos, target_xz, ttype, tcolor, dxz, dcolor, mission in scenes:
        adapter.heading = 0  # always face north for these tests
        obs = make_scene(adapter, agent_pos, target_xz, ttype, tcolor, dxz, dcolor)
        out = probe_one(policy, jepa, obs, mission, device, top_k_slots=args.top_k)

        print()
        print("=" * 70)
        print(f"SCENE: {name}")
        print(f"  mission: '{mission}'")
        print()
        print_grid(obs)
        print()
        print(f"  TOP-{args.top_k} concept slots fired:")
        for i in range(args.top_k):
            print(f"    slot #{out['top_slot_idx'][i]:4d}  attn={out['top_slot_attn'][i]:.4f}")
        print()
        print(f"  TOP-3 action choices:")
        for i in range(3):
            a = int(out['top_actions'][i])
            print(f"    {_ACTION_NAMES[a]:10s}  logit={out['top_action_logits'][i]:+.3f}")
        chosen = int(out['top_actions'][0])
        print(f"  → would choose: {_ACTION_NAMES[chosen]}")
        results.append(out)

    # --------- Cross-scene comparisons ---------
    print()
    print("=" * 70)
    print("DISCRIMINATION CHECK")
    print("=" * 70)

    def slot_overlap(a, b, k=5):
        return len(set(a[:k]) & set(b[:k]))

    print()
    print("Green-ball-ahead vs Red-ball-ahead (identical position, "
          "different color):")
    print(f"  shared concept slots in top-{args.top_k}: "
          f"{slot_overlap(results[0]['top_slot_idx'], results[1]['top_slot_idx'], args.top_k)} "
          f"of {args.top_k}")
    diff = np.linalg.norm(results[0]['concept_query'] - results[1]['concept_query'])
    print(f"  concept-query L2 distance: {diff:.3f}")
    print("  Interpretation:")
    print("    - high slot overlap + low L2 ⇒ model treats red/green similarly (BAD)")
    print("    - low slot overlap OR high L2 ⇒ model discriminates them (GOOD)")

    print()
    print("Green-ball-ahead vs Green-ball-right "
          "(same color, different position):")
    print(f"  shared concept slots in top-{args.top_k}: "
          f"{slot_overlap(results[0]['top_slot_idx'], results[2]['top_slot_idx'], args.top_k)} "
          f"of {args.top_k}")
    print(f"  picked action (ahead/right): "
          f"{_ACTION_NAMES[int(results[0]['top_actions'][0])]} / "
          f"{_ACTION_NAMES[int(results[2]['top_actions'][0])]}")
    print("  Interpretation:")
    print("    - if positions yield different actions ⇒ spatial awareness present")
    print("    - if actions identical ⇒ model is position-blind")

    print()
    print("Mission-flip test (same scene, different mission):")
    print(f"  scene #5 (mission=green, target=green at right, distractor=red at left):")
    print(f"    chosen action: {_ACTION_NAMES[int(results[4]['top_actions'][0])]}")
    print(f"  scene #6 (mission=red,   target=red   at right, distractor=green at left):")
    print(f"    chosen action: {_ACTION_NAMES[int(results[5]['top_actions'][0])]}")
    print("  Interpretation:")
    print("    - if actions match the mission's color ⇒ mission grounding works")
    print("    - if same action in both ⇒ mission ignored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
