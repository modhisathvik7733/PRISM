"""Stage 1.0-floor — predict goal (color, type) from BabyAI mission text.

This is the *floor* test before any operator-binding can work: confirm
the text encoder + tokenizer + training pipeline are functional by
asking the model to predict the most trivially text-derivable label
(the goal object's color and type, from missions like "go to the red
ball").

If this fails → there is a data / pipeline problem and no semantic
grounding experiment can succeed. If it passes → the text encoder is
healthy and the previous milestone-1.0 failure was specifically because
V3 operators aren't goal-shaped (already diagnosed; we now reframe to
predicate-based binding for Stage 1.0-proper).

The held-out split is a true compositional split: specific
(color, type) combinations never seen during training. Pass requires
the model to assemble color + type votes from individual tokens —
demonstrating it has learned the structure rather than memorized
phrases.

Usage:
    python -m scripts.lang.train_grounding_floor \
        --rollouts runs/cog_core_phase1_devB/rollouts.npz \
        --kind bow \
        --steps 3000 --batch-size 1024 \
        --run-name grounding_floor_v0 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from prism.agents import goal_predicates_for_mission
from prism.language.grounding_head import WhitespaceVocab
from prism.language.grounding_predicate_head import make_dual_head
from prism.perception.slots import NUM_COLORS, NUM_TYPES, OBJECT_TYPES
from prism.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# data: missions -> (color, type) labels via the existing parser
# ---------------------------------------------------------------------------

def parse_missions(missions: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (color_ids, type_indices, keep_mask).

    For each mission string, attempts to parse the goal predicate. Some
    missions may be unparseable (returns None) — those are masked out.
    type_index is the position of the goal's type_id inside OBJECT_TYPES
    (0..NUM_TYPES-1), NOT the raw type_id used by the env.
    """
    type_to_idx = {t: i for i, t in enumerate(OBJECT_TYPES)}
    colors = np.full(len(missions), -1, dtype=np.int64)
    types = np.full(len(missions), -1, dtype=np.int64)
    keep = np.zeros(len(missions), dtype=bool)
    for i, m in enumerate(missions):
        parsed = goal_predicates_for_mission(str(m))
        if parsed is None:
            continue
        goal_preds, _spec = parsed
        if not goal_preds:
            continue
        gp = goal_preds[0]
        if gp.type_id not in type_to_idx:
            continue
        colors[i] = int(gp.color_id)
        types[i] = type_to_idx[gp.type_id]
        keep[i] = True
    return colors, types, keep


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True)
    p.add_argument("--kind", choices=["bow", "tiny_tf"], default="bow")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-seq-len", type=int, default=16)
    p.add_argument("--holdout-combos", type=int, default=4,
                   help="number of (color, type) combinations to reserve "
                        "as held-out compositional test set")
    p.add_argument("--id-val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", required=True)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # ---- load rollouts (only need missions per episode) ----
    print(f"[floor] loading rollouts: {args.rollouts}")
    d = np.load(args.rollouts)
    if "missions" not in d.files:
        raise SystemExit("rollouts.npz missing `missions` field")
    missions = d["missions"]
    print(f"[floor]   {len(missions)} episodes, "
          f"{len(set(missions.tolist()))} unique missions")

    # ---- parse to (color, type) labels ----
    colors, types, keep = parse_missions(missions)
    n_parsed = int(keep.sum())
    print(f"[floor]   parsed {n_parsed}/{len(missions)} missions "
          f"into (color, type)")
    if n_parsed == 0:
        raise SystemExit("zero missions parsed — check goal_predicates_for_mission")

    missions = missions[keep]
    colors = colors[keep]
    types = types[keep]

    # Distribution check.
    combo_counts = Counter(
        (int(c), int(t)) for c, t in zip(colors, types)
    )
    print(f"[floor]   distinct (color, type) combos: {len(combo_counts)}")
    print(f"[floor]   color histogram: "
          f"{dict(sorted(Counter(colors.tolist()).items()))}")
    print(f"[floor]   type  histogram: "
          f"{dict(sorted(Counter(types.tolist()).items()))}")

    # ---- compositional split: hold out specific (color, type) combos ----
    all_combos = sorted(combo_counts.keys())
    rng = random.Random(args.seed)
    rng.shuffle(all_combos)
    n_hold = min(args.holdout_combos, max(1, len(all_combos) // 4))
    held_combos = set(all_combos[:n_hold])
    seen_combos = set(all_combos[n_hold:])
    print(f"[floor]   held-out combos ({n_hold}): "
          f"{sorted(held_combos)}")

    held_mask = np.array([
        (int(c), int(t)) in held_combos for c, t in zip(colors, types)
    ])
    seen_mask = ~held_mask

    seen_idx = np.flatnonzero(seen_mask)
    held_idx = np.flatnonzero(held_mask)
    np_rng = np.random.default_rng(args.seed)
    np_rng.shuffle(seen_idx)
    n_val = int(len(seen_idx) * args.id_val_frac)
    val_idx = seen_idx[:n_val]
    train_idx = seen_idx[n_val:]
    print(f"[floor]   episodes: train={len(train_idx):,}  "
          f"id_val={len(val_idx):,}  held={len(held_idx):,}")

    # ---- vocab (built from train only) ----
    vocab = WhitespaceVocab.build(
        [str(missions[i]) for i in train_idx], max_len=args.max_seq_len,
    )
    print(f"[floor]   vocab size = {vocab.size}")

    def enc(idx: np.ndarray):
        tok, msk = vocab.encode_batch([str(missions[i]) for i in idx])
        c = torch.from_numpy(colors[idx])
        t = torch.from_numpy(types[idx])
        return tok.to(device), msk.to(device), c.to(device), t.to(device)

    tr_tok, tr_msk, tr_c, tr_t = enc(train_idx)
    va_tok, va_msk, va_c, va_t = enc(val_idx)
    he_tok, he_msk, he_c, he_t = enc(held_idx)

    # ---- model ----
    model = make_dual_head(args.kind, vocab.size, NUM_COLORS, NUM_TYPES).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[floor]   {args.kind} dual-head: {n_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    vocab.save(str(out_dir / "vocab.pt"))

    def eval_split(tok, msk, c, t) -> dict:
        model.eval()
        with torch.no_grad():
            lc, lt = model(tok, msk)
            ce_c = float(F.cross_entropy(lc, c).item())
            ce_t = float(F.cross_entropy(lt, t).item())
            pc = lc.argmax(-1)
            pt = lt.argmax(-1)
            acc_c = float((pc == c).float().mean().item())
            acc_t = float((pt == t).float().mean().item())
            joint = float(((pc == c) & (pt == t)).float().mean().item())
        model.train()
        return {
            "ce_c": ce_c, "ce_t": ce_t,
            "acc_c": acc_c, "acc_t": acc_t,
            "joint_acc": joint,
        }

    # ---- train ----
    print(f"[floor] training {args.steps} steps, batch={args.batch_size}")
    for step in range(args.steps):
        idx = torch.randint(0, tr_tok.shape[0], (args.batch_size,), device=device)
        tok = tr_tok[idx]
        msk = tr_msk[idx]
        c = tr_c[idx]
        t = tr_t[idx]

        lc, lt = model(tok, msk)
        loss_c = F.cross_entropy(lc, c)
        loss_t = F.cross_entropy(lt, t)
        loss = loss_c + loss_t

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
            v = eval_split(va_tok, va_msk, va_c, va_t)
            h = eval_split(he_tok, he_msk, he_c, he_t)
            writer.add_scalar("train/loss", float(loss.item()), step)
            writer.add_scalar("val_id/joint", v["joint_acc"], step)
            writer.add_scalar("val_held/joint", h["joint_acc"], step)
            print(f"[step {step:5d}/{args.steps}]  "
                  f"loss={float(loss.item()):.3f}  "
                  f"tr_joint={tr_joint*100:5.1f}%  "
                  f"id={v['joint_acc']*100:5.1f}% "
                  f"(c={v['acc_c']*100:.0f}/t={v['acc_t']*100:.0f})  "
                  f"held={h['joint_acc']*100:5.1f}% "
                  f"(c={h['acc_c']*100:.0f}/t={h['acc_t']*100:.0f})")

    # ---- final ----
    v = eval_split(va_tok, va_msk, va_c, va_t)
    h = eval_split(he_tok, he_msk, he_c, he_t)
    print("\n=== final results ===")
    print(f"  in-distribution (seen combos):")
    print(f"    color  acc: {v['acc_c']*100:5.1f}%")
    print(f"    type   acc: {v['acc_t']*100:5.1f}%")
    print(f"    joint  acc: {v['joint_acc']*100:5.1f}%   (target ≥ 95%)")
    print(f"  held-out (compositional, unseen combos):")
    print(f"    color  acc: {h['acc_c']*100:5.1f}%")
    print(f"    type   acc: {h['acc_t']*100:5.1f}%")
    print(f"    joint  acc: {h['joint_acc']*100:5.1f}%   (target ≥ 80%)")

    # Confusion-style: which held-out combos pass / fail.
    print("\n=== held-out combo breakdown ===")
    model.eval()
    with torch.no_grad():
        lc, lt = model(he_tok, he_msk)
        pc = lc.argmax(-1).cpu().numpy()
        pt = lt.argmax(-1).cpu().numpy()
    true_c = he_c.cpu().numpy()
    true_t = he_t.cpu().numpy()
    print(f"  {'(color, type)':>16}  {'n':>5}  {'color%':>7}  "
          f"{'type%':>6}  {'joint%':>7}")
    for combo in sorted(held_combos):
        c0, t0 = combo
        mask = (true_c == c0) & (true_t == t0)
        n = int(mask.sum())
        if n == 0:
            continue
        c_acc = float((pc[mask] == c0).mean())
        t_acc = float((pt[mask] == t0).mean())
        j_acc = float(((pc[mask] == c0) & (pt[mask] == t0)).mean())
        print(f"  {str(combo):>16}  {n:>5d}  "
              f"{c_acc*100:>6.1f}%  {t_acc*100:>5.1f}%  {j_acc*100:>6.1f}%")

    # Verdict.
    pass_id = v["joint_acc"] >= 0.95
    pass_held = h["joint_acc"] >= 0.80
    print("\n=== verdict ===")
    if pass_id and pass_held:
        print("  PASS — text encoder is healthy AND BoW composes color+type "
              "votes correctly across unseen combos.")
        print("  Next: build proper Stage 1.0 (text → goal-predicate "
              "vector via predicate readout from JEPA latents).")
    elif pass_id and not pass_held:
        print("  PARTIAL — text encoder learns seen combos perfectly but "
              "doesn't compose across unseen combos.")
        print("  This means BoW's avg-pool isn't enough for compositional "
              "generalization. Re-run with --kind tiny_tf.")
    elif not pass_id:
        print("  FAIL — text encoder can't even learn seen combos.")
        print("  Something is broken at the data / pipeline level.")
        print("  Check the vocab and label histograms printed at start.")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "kind": args.kind,
            "n_colors": NUM_COLORS,
            "n_types": NUM_TYPES,
            "args": vars(args),
        },
        out_dir / "grounding_floor_final.pt",
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "id_color": v["acc_c"], "id_type": v["acc_t"],
            "id_joint": v["joint_acc"],
            "held_color": h["acc_c"], "held_type": h["acc_t"],
            "held_joint": h["joint_acc"],
            "pass_id": pass_id, "pass_held": pass_held,
            "held_combos": sorted(list(held_combos)),
            "vocab_size": vocab.size,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_held": int(len(held_idx)),
        }, f, indent=2)
    print(f"\n[saved] {out_dir / 'grounding_floor_final.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
