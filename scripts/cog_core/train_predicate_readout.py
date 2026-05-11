"""Train a PredicateReadout from JEPA latents to (color, type) goal
predicates extracted from BabyAI rollouts.

This is the Stage 1.0-proper Phase 1 experiment: does the JEPA latent
encode the goal predicate well enough to read out compositionally?

For each episode:
  * Parse the mission to (color_idx, type_idx) via goal_predicates_for_mission.
  * Take the last K frames of the episode (the agent is most likely
    near the target there — even with random policy, terminal frames
    concentrate where the agent stopped, often facing/adjacent to an
    object).
  * Label each of those frames with the mission's (color, type) as a
    flat 24-class index: `label = color_idx * NUM_TYPES + type_idx`.

Split missions compositionally — specific (color, type) combos held out,
never seen during training. Pass if the readout generalizes to held-out
combos: that proves the JEPA latent encodes color and type as
linearly-separable features (not entangled phrases).

Usage:
    python -m scripts.cog_core.train_predicate_readout \
        --rollouts runs/cog_core_phase1_devB/rollouts.npz \
        --steps 5000 --batch-size 1024 \
        --last-k 8 --holdout-combos 4 \
        --run-name predicate_readout_v0 \
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
from prism.cog_core.predicate_readout import PredicateReadout
from prism.perception.slots import NUM_COLORS, NUM_TYPES, OBJECT_TYPES
from prism.utils.seed import set_global_seed


N_PRED = NUM_COLORS * NUM_TYPES                            # 24


def build_dataset(
    npz_path: Path,
    last_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (latents, labels, episode_ids, combo_labels).

    `labels` is a flat 24-class index = color_idx * NUM_TYPES + type_idx.
    `combo_labels` is the (color, type) tuple per row (for split filtering).
    """
    d = np.load(npz_path)
    if "missions" not in d.files:
        raise SystemExit("rollouts.npz has no `missions` — re-collect with the "
                         "updated collect_rollouts.py")
    latents = d["latents"]
    lengths = d["ep_lengths"]
    missions = d["missions"]
    type_to_idx = {t: i for i, t in enumerate(OBJECT_TYPES)}

    Z: list[np.ndarray] = []
    Y: list[int] = []
    EP: list[int] = []
    CT: list[tuple[int, int]] = []
    for i in range(len(lengths)):
        L = int(lengths[i])
        if L < 1:
            continue
        parsed = goal_predicates_for_mission(str(missions[i]))
        if parsed is None:
            continue
        goal_preds, _ = parsed
        if not goal_preds:
            continue
        gp = goal_preds[0]
        if gp.type_id not in type_to_idx:
            continue
        c_idx = int(gp.color_id)
        t_idx = type_to_idx[gp.type_id]
        label = c_idx * NUM_TYPES + t_idx
        start = max(0, L - last_k)
        for t in range(start, L):
            Z.append(latents[i, t])
            Y.append(label)
            EP.append(i)
            CT.append((c_idx, t_idx))

    Z = np.stack(Z).astype(np.float32)
    if Z.ndim > 2:
        Z = Z.reshape(Z.shape[0], -1)
    return (
        Z,
        np.array(Y, dtype=np.int64),
        np.array(EP, dtype=np.int64),
        np.array(CT, dtype=np.int64),                  # (N, 2)
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True)
    p.add_argument("--last-k", type=int, default=8,
                   help="number of terminal frames per episode used as "
                        "training data (the agent is most likely near the "
                        "target in these frames)")
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

    print(f"[readout] loading rollouts: {args.rollouts} (last_k={args.last_k})")
    Z, Y, EP, CT = build_dataset(Path(args.rollouts), args.last_k)
    print(f"[readout]   {len(Z):,} frames, latent_dim={Z.shape[-1]}")
    counts = Counter(Y.tolist())
    print(f"[readout]   label histogram: {dict(sorted(counts.items()))}")

    # Compositional split.
    combos = sorted({(int(c), int(t)) for c, t in CT})
    rng = random.Random(args.seed)
    rng.shuffle(combos)
    n_hold = min(args.holdout_combos, max(1, len(combos) // 4))
    held = set(combos[:n_hold])
    print(f"[readout]   {len(combos)} distinct combos, {n_hold} held out: "
          f"{sorted(held)}")

    held_mask = np.array([(int(c), int(t)) in held for c, t in CT])
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
    Y_gpu = torch.from_numpy(Y).to(device)
    train_idx_t = torch.from_numpy(train_idx).to(device)
    val_idx_t = torch.from_numpy(val_idx).to(device)
    held_idx_t = torch.from_numpy(held_idx).to(device)

    latent_dim = int(Z.shape[-1])
    model = PredicateReadout(
        latent_dim=latent_dim,
        n_predicates=N_PRED,
        hidden=args.hidden,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[readout]   readout: {n_params:,} params "
          f"(latent_dim={latent_dim} -> {N_PRED})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")

    def eval_split(idx_t: torch.Tensor) -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            logits = model(Z_gpu[idx_t])
            ce = float(F.cross_entropy(logits, Y_gpu[idx_t]).item())
            acc = float(
                (logits.argmax(-1) == Y_gpu[idx_t]).float().mean().item()
            )
        model.train()
        return ce, acc

    print(f"[readout] training {args.steps} steps, batch={args.batch_size}")
    for step in range(args.steps):
        batch = train_idx_t[
            torch.randint(0, train_idx_t.shape[0],
                          (args.batch_size,), device=device)
        ]
        z = Z_gpu[batch]
        y = Y_gpu[batch]
        logits = model(z)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0:
            tr_acc = float((logits.argmax(-1) == y).float().mean().item())
            id_ce, id_acc = eval_split(val_idx_t)
            held_ce, held_acc = eval_split(held_idx_t)
            writer.add_scalar("train/loss", float(loss.item()), step)
            writer.add_scalar("val_id/acc", id_acc, step)
            writer.add_scalar("val_held/acc", held_acc, step)
            print(f"[step {step:5d}/{args.steps}]  "
                  f"loss={float(loss.item()):.3f}  "
                  f"tr_acc={tr_acc*100:5.1f}%  "
                  f"id_acc={id_acc*100:5.1f}%  "
                  f"held_acc={held_acc*100:5.1f}%")

    # ---- final ----
    id_ce, id_acc = eval_split(val_idx_t)
    held_ce, held_acc = eval_split(held_idx_t)
    print("\n=== final results ===")
    print(f"  in-distribution (seen combos)   acc: {id_acc*100:5.1f}%  "
          f"(target ≥ 70%)")
    print(f"  held-out (compositional combos) acc: {held_acc*100:5.1f}%  "
          f"(target ≥ 50%)")
    print(f"  random-baseline:                     {100.0/N_PRED:5.1f}%  "
          f"(1 / {N_PRED})")

    # Per-combo held-out breakdown.
    print("\n=== held-out combo breakdown ===")
    model.eval()
    with torch.no_grad():
        held_logits = model(Z_gpu[held_idx_t])
        held_pred = held_logits.argmax(-1).cpu().numpy()
    held_y = Y[held_idx]
    held_ct = CT[held_idx]
    print(f"  {'(color, type)':>16}  {'n':>6}  {'top1%':>6}  "
          f"{'top3%':>6}")
    for combo in sorted(held):
        c, t = combo
        true_label = c * NUM_TYPES + t
        mask = (held_y == true_label)
        n = int(mask.sum())
        if n == 0:
            continue
        top1 = float((held_pred[mask] == true_label).mean())
        # top-3 acc
        top3 = float(
            (held_logits[mask].topk(3, dim=-1).indices ==
             torch.tensor(true_label, device=device)).any(-1).float().mean().item()
        )
        print(f"  {str(combo):>16}  {n:>6d}  "
              f"{top1*100:>5.1f}%  {top3*100:>5.1f}%")

    pass_id = id_acc >= 0.70
    pass_held = held_acc >= 0.50
    print("\n=== verdict ===")
    if pass_id and pass_held:
        print("  PASS — JEPA latents encode (color, type) goal predicate in "
              "a compositionally-readable way.")
        print("  Next: train text -> predicate (Stage 1.0-proper Phase 2).")
    elif pass_id and not pass_held:
        print("  PARTIAL — readout memorizes seen combos but doesn't "
              "compose to unseen ones.")
        print("  Means: JEPA latent has color and type information but "
              "*entangled* (color and type don't factorize linearly).")
        print("  Mitigations:")
        print("    - Add object-centric inductive bias to JEPA encoder")
        print("    - Train readout with auxiliary disentanglement loss")
        print("    - Increase --last-k (more frames -> stronger goal signal)")
    elif not pass_id:
        print("  FAIL — readout can't even predict seen combos.")
        print("  Means: JEPA latents don't encode which object the agent "
              "is near.")
        print("  Likely causes:")
        print("    - Random-policy rollouts: agent rarely actually reaches "
              "the goal, so last-K frames don't show goal proximity")
        print("    - Increase --last-k or filter to successful episodes only")

    model.save(str(out_dir / "predicate_readout_final.pt"))
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "id_acc": id_acc, "held_acc": held_acc,
            "pass_id": pass_id, "pass_held": pass_held,
            "held_combos": sorted(list(held)),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_held": int(len(held_idx)),
            "n_predicates": N_PRED,
            "random_baseline": 1.0 / N_PRED,
        }, f, indent=2)
    print(f"\n[saved] {out_dir / 'predicate_readout_final.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
