"""Train a PredicateReadout from JEPA latents to (color, type) of the
most-prominent visible object — a perceptual grounding signal that's
independent of whether the agent reached its goal.

This is the Stage 1.0-proper Phase 1 experiment, v2 (slot-derived).

Why slot-derived and not mission-derived: with random-policy rollouts,
the agent rarely reaches the goal, so "mission text" is a poor label
for "what's actually in the latent at frame t." The agent's view at
some random step encodes whatever objects happen to be visible *now*.
That's what the readout should read.

Pipeline per frame:
  * Look up the slot list (from rollouts.slots.pkl) for that (episode, t).
  * Find the most-prominent visible object — the one closest to the agent
    along its facing direction (smallest y at x=3, with fallback to the
    closest object by Manhattan distance).
  * Label = (color_idx, type_idx_in_OBJECT_TYPES) of that object.
  * Skip frames with no visible object.

The readout outputs `NUM_COLORS + NUM_TYPES` logits = two factored heads
(color softmax of size 6, type softmax of size 4). Two CE losses.
Compositional generalization is now structurally possible: even if
(c=3, t=3) is never seen as a *combination* in training, the color head
sees c=3 in other combos and the type head sees t=3 in other combos.

Pass criteria:
  * ID joint acc (both correct on seen combos) ≥ 70%
  * Held-out joint acc (both correct on unseen combos) ≥ 50%

Usage:
    python -m scripts.cog_core.train_predicate_readout \
        --rollouts runs/cog_core_phase1_devB/rollouts.npz \
        --slots runs/cog_core_phase1_devB/rollouts.slots.pkl \
        --steps 5000 --batch-size 1024 \
        --holdout-combos 4 \
        --run-name predicate_readout_v1 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from prism.cog_core.predicate_readout import PredicateReadout
from prism.perception.slots import (
    AGENT_POS,
    NUM_COLORS,
    NUM_TYPES,
    OBJECT_TYPES,
)
from prism.utils.seed import set_global_seed


N_PRED = NUM_COLORS + NUM_TYPES                            # 6 + 4 = 10 logits


def _primary_object(slots_at_t: list[dict]) -> tuple[int, int] | None:
    """Pick the most-prominent visible object: prefer one in agent's
    facing column (x=AGENT_POS[0]) with smallest y (closest in front);
    fall back to the slot with smallest Manhattan distance to AGENT_POS.

    Returns (color_id, type_idx) where type_idx is the position of
    type_id inside OBJECT_TYPES, or None if no slot has a recognized
    type."""
    if not slots_at_t:
        return None
    type_to_idx = {t: i for i, t in enumerate(OBJECT_TYPES)}
    ax, ay = AGENT_POS

    in_col = [s for s in slots_at_t if int(s["x"]) == ax]
    if in_col:
        s = min(in_col, key=lambda s: int(s["y"]))
    else:
        s = min(
            slots_at_t,
            key=lambda s: abs(int(s["x"]) - ax) + abs(int(s["y"]) - ay),
        )

    t_id = int(s["type_id"])
    if t_id not in type_to_idx:
        return None
    return int(s["color_id"]), type_to_idx[t_id]


def build_dataset(
    npz_path: Path,
    slots_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (latents, color_labels, type_labels).

    Labels come from per-frame slot inspection — see _primary_object.
    Frames with no recognized object in view are skipped.
    """
    d = np.load(npz_path)
    latents = d["latents"]
    lengths = d["ep_lengths"]
    with open(slots_path, "rb") as f:
        slots = pickle.load(f)
    if len(slots) != latents.shape[0]:
        raise SystemExit(
            f"slot count ({len(slots)}) != episode count ({latents.shape[0]})"
        )

    Z: list[np.ndarray] = []
    Cs: list[int] = []
    Ts: list[int] = []
    for i in range(len(lengths)):
        L = int(lengths[i])
        if L < 1:
            continue
        ep_slots = slots[i]
        for t in range(L):
            if t >= len(ep_slots):
                continue
            picked = _primary_object(ep_slots[t])
            if picked is None:
                continue
            c, ti = picked
            Z.append(latents[i, t])
            Cs.append(c)
            Ts.append(ti)

    Z = np.stack(Z).astype(np.float32)
    if Z.ndim > 2:
        Z = Z.reshape(Z.shape[0], -1)
    return (
        Z,
        np.array(Cs, dtype=np.int64),
        np.array(Ts, dtype=np.int64),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True)
    p.add_argument("--slots", required=True,
                   help="path to rollouts.slots.pkl produced by collect_rollouts")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--holdout-combos", type=int, default=4)
    p.add_argument("--id-val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", required=True)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    print(f"[readout] loading rollouts: {args.rollouts}")
    print(f"[readout] loading slots:    {args.slots}")
    Z, C, T = build_dataset(Path(args.rollouts), Path(args.slots))
    print(f"[readout]   {len(Z):,} frames with a visible recognized object")
    print(f"[readout]   latent_dim={Z.shape[-1]}")
    print(f"[readout]   color histogram: "
          f"{dict(sorted(Counter(C.tolist()).items()))}")
    print(f"[readout]   type  histogram: "
          f"{dict(sorted(Counter(T.tolist()).items()))}")

    # Compositional split over (color, type) combos.
    CT = np.stack([C, T], axis=1)
    combos = sorted({(int(c), int(t)) for c, t in CT})
    rng = random.Random(args.seed)
    rng.shuffle(combos)
    n_hold = min(args.holdout_combos, max(1, len(combos) // 4))
    held = set(combos[:n_hold])
    print(f"[readout]   {len(combos)} distinct combos, "
          f"{n_hold} held out: {sorted(held)}")

    held_mask = np.array(
        [(int(c), int(t)) in held for c, t in CT]
    )
    seen_idx = np.flatnonzero(~held_mask)
    held_idx = np.flatnonzero(held_mask)
    np_rng = np.random.default_rng(args.seed)
    np_rng.shuffle(seen_idx)
    n_val = int(len(seen_idx) * args.id_val_frac)
    val_idx = seen_idx[:n_val]
    train_idx = seen_idx[n_val:]
    print(f"[readout]   frames: train={len(train_idx):,}  "
          f"id_val={len(val_idx):,}  held={len(held_idx):,}")

    Z_gpu = torch.from_numpy(Z).to(device)
    C_gpu = torch.from_numpy(C).to(device)
    T_gpu = torch.from_numpy(T).to(device)
    train_t = torch.from_numpy(train_idx).to(device)
    val_t = torch.from_numpy(val_idx).to(device)
    held_t = torch.from_numpy(held_idx).to(device)

    latent_dim = int(Z.shape[-1])
    model = PredicateReadout(
        latent_dim=latent_dim,
        n_predicates=N_PRED,                           # 10 = 6 color + 4 type
        hidden=args.hidden,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[readout]   readout: {n_params:,} params  "
          f"(latent {latent_dim} -> 6 color + 4 type)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")

    def split_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return logits[:, :NUM_COLORS], logits[:, NUM_COLORS:]

    def eval_split(idx: torch.Tensor) -> dict:
        model.eval()
        with torch.no_grad():
            logits = model(Z_gpu[idx])
            lc, lt = split_logits(logits)
            ce = float(
                (F.cross_entropy(lc, C_gpu[idx]) +
                 F.cross_entropy(lt, T_gpu[idx])).item()
            )
            pc = lc.argmax(-1)
            pt = lt.argmax(-1)
            acc_c = float((pc == C_gpu[idx]).float().mean().item())
            acc_t = float((pt == T_gpu[idx]).float().mean().item())
            joint = float(
                ((pc == C_gpu[idx]) & (pt == T_gpu[idx]))
                .float().mean().item()
            )
        model.train()
        return {"ce": ce, "acc_c": acc_c, "acc_t": acc_t, "joint": joint}

    print(f"[readout] training {args.steps} steps, batch={args.batch_size}")
    for step in range(args.steps):
        batch = train_t[
            torch.randint(0, train_t.shape[0],
                          (args.batch_size,), device=device)
        ]
        z = Z_gpu[batch]
        c = C_gpu[batch]
        t = T_gpu[batch]
        logits = model(z)
        lc, lt = split_logits(logits)
        loss = F.cross_entropy(lc, c) + F.cross_entropy(lt, t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0:
            with torch.no_grad():
                tr_joint = float(
                    ((lc.argmax(-1) == c) & (lt.argmax(-1) == t))
                    .float().mean().item()
                )
            v = eval_split(val_t)
            h = eval_split(held_t)
            writer.add_scalar("train/loss", float(loss.item()), step)
            writer.add_scalar("val_id/joint", v["joint"], step)
            writer.add_scalar("val_held/joint", h["joint"], step)
            print(f"[step {step:5d}/{args.steps}]  "
                  f"loss={float(loss.item()):.3f}  "
                  f"tr={tr_joint*100:5.1f}%  "
                  f"id={v['joint']*100:5.1f}% "
                  f"(c={v['acc_c']*100:.0f}/t={v['acc_t']*100:.0f})  "
                  f"held={h['joint']*100:5.1f}% "
                  f"(c={h['acc_c']*100:.0f}/t={h['acc_t']*100:.0f})")

    # ---- final ----
    v = eval_split(val_t)
    h = eval_split(held_t)
    print("\n=== final results ===")
    print(f"  in-distribution (seen combos):")
    print(f"    color  acc: {v['acc_c']*100:5.1f}%")
    print(f"    type   acc: {v['acc_t']*100:5.1f}%")
    print(f"    joint  acc: {v['joint']*100:5.1f}%   (target ≥ 70%)")
    print(f"  held-out (compositional, unseen combos):")
    print(f"    color  acc: {h['acc_c']*100:5.1f}%")
    print(f"    type   acc: {h['acc_t']*100:5.1f}%")
    print(f"    joint  acc: {h['joint']*100:5.1f}%   (target ≥ 50%)")
    print(f"  random color: {100.0/NUM_COLORS:5.1f}%   "
          f"random type: {100.0/NUM_TYPES:5.1f}%   "
          f"random joint: {100.0/(NUM_COLORS*NUM_TYPES):5.1f}%")

    # Per-held-combo breakdown.
    print("\n=== held-out combo breakdown ===")
    model.eval()
    with torch.no_grad():
        held_logits = model(Z_gpu[held_t])
        lc, lt = split_logits(held_logits)
        held_pc = lc.argmax(-1).cpu().numpy()
        held_pt = lt.argmax(-1).cpu().numpy()
    held_c = C[held_idx]
    held_t_np = T[held_idx]
    print(f"  {'(color, type)':>16}  {'n':>6}  "
          f"{'color%':>7}  {'type%':>6}  {'joint%':>7}")
    for combo in sorted(held):
        c0, t0 = combo
        mask = (held_c == c0) & (held_t_np == t0)
        n = int(mask.sum())
        if n == 0:
            continue
        c_acc = float((held_pc[mask] == c0).mean())
        t_acc = float((held_pt[mask] == t0).mean())
        j_acc = float(((held_pc[mask] == c0) & (held_pt[mask] == t0)).mean())
        print(f"  {str(combo):>16}  {n:>6d}  "
              f"{c_acc*100:>6.1f}%  {t_acc*100:>5.1f}%  {j_acc*100:>6.1f}%")

    pass_id = v["joint"] >= 0.70
    pass_held = h["joint"] >= 0.50
    print("\n=== verdict ===")
    if pass_id and pass_held:
        print("  PASS — JEPA latents encode (color, type) of the visible "
              "object compositionally.")
        print("  Next: train text -> predicate (Stage 1.0-proper Phase 2).")
    elif pass_id and not pass_held:
        print("  PARTIAL — readout learns seen combos but doesn't compose.")
        print("  Means: color and type are encoded but *entangled* in the "
              "latent.")
    elif not pass_id:
        print("  FAIL — readout can't even predict seen (color, type).")
        print("  Means: JEPA latents don't encode object identity in the "
              "agent's view well enough to read out.")

    model.save(str(out_dir / "predicate_readout_final.pt"))
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "id_color": v["acc_c"], "id_type": v["acc_t"],
            "id_joint": v["joint"],
            "held_color": h["acc_c"], "held_type": h["acc_t"],
            "held_joint": h["joint"],
            "pass_id": pass_id, "pass_held": pass_held,
            "held_combos": sorted(list(held)),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_held": int(len(held_idx)),
            "label_scheme": "slot-derived primary-object (color, type), "
                            "factored heads",
        }, f, indent=2)
    print(f"\n[saved] {out_dir / 'predicate_readout_final.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
