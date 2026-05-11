"""Stage 1 Milestone 1.0 — train a small grounding head that predicts
V3's operator routing from BabyAI mission text.

Pipeline:
  1. Load V3 operator bank + rollouts (with `missions` field).
  2. For every transition, compute V3.assign(z_t, a) → operator label.
  3. Split missions into seen / held-out (compositional generalization split).
  4. Train BoW or tiny-transformer text → operator classifier.
  5. Report:
     - in-distribution test accuracy (held-out transitions from seen missions)
     - held-out-composition accuracy (transitions from missions never seen
       during training)
     - top-1 confusion matrix over operators

Pass criteria for v4.1 → Stage 1.0:
  ID acc ≥ 80%, held-out composition acc ≥ 60%.

Usage:
    python -m scripts.lang.train_grounding \
        --bank runs/ops_v3_phaseB/operators_v3.pt \
        --rollouts runs/cog_core_phase1_devB/rollouts.npz \
        --kind bow \
        --steps 5000 \
        --batch-size 1024 \
        --run-name grounding_v0 \
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

from prism.cog_core.operator_bank_v3 import OperatorBankV3
from prism.language.grounding_head import (
    WhitespaceVocab,
    make_grounding_head,
)
from prism.utils.seed import set_global_seed


def build_transitions(
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns flat arrays of (latents, actions, env_ids, missions,
    episode_ids) — one row per transition (with t < L-1)."""
    d = np.load(npz_path)
    if "missions" not in d.files:
        raise SystemExit(
            "rollouts.npz has no `missions` field — re-collect with the "
            "updated collect_rollouts.py that saves mission text per episode."
        )
    latents = d["latents"]
    actions = d["actions"]
    lengths = d["ep_lengths"]
    env_ids = d["env_ids"]
    missions = d["missions"]

    L_t, A, E, M, EP = [], [], [], [], []
    for i in range(len(lengths)):
        L = int(lengths[i])
        if L < 2:
            continue
        for t in range(L - 1):
            L_t.append(latents[i, t])
            A.append(int(actions[i, t]))
            E.append(str(env_ids[i]))
            M.append(str(missions[i]))
            EP.append(i)

    L_t = np.stack(L_t).astype(np.float32)
    if L_t.ndim > 2:
        L_t = L_t.reshape(L_t.shape[0], -1)
    return (
        L_t,
        np.array(A, dtype=np.int64),
        np.array(E),
        np.array(M),
        np.array(EP, dtype=np.int64),
    )


def split_missions(
    missions: np.ndarray, holdout_frac: float, seed: int,
) -> tuple[set, set]:
    """Split unique missions into seen / held-out. All transitions from a
    held-out mission go to the compositional test set."""
    unique = sorted(set(missions.tolist()))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_holdout = max(1, int(len(unique) * holdout_frac))
    holdout = set(unique[:n_holdout])
    seen = set(unique[n_holdout:])
    return seen, holdout


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", required=True, help="V3 operator bank checkpoint")
    p.add_argument("--rollouts", required=True)
    p.add_argument("--kind", choices=["bow", "tiny_tf"], default="bow")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--holdout-frac", type=float, default=0.2,
                   help="fraction of unique missions reserved as held-out "
                        "compositions (never seen during training)")
    p.add_argument("--id-val-frac", type=float, default=0.1,
                   help="fraction of seen-mission transitions reserved for "
                        "in-distribution validation")
    p.add_argument("--max-seq-len", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", required=True)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # ---- data ----
    print(f"[ground] loading rollouts: {args.rollouts}")
    L_t, A, E, M, EP = build_transitions(Path(args.rollouts))
    print(f"[ground]   {len(L_t):,} transitions across "
          f"{len(set(E.tolist()))} envs, "
          f"{len(set(M.tolist()))} unique missions")

    # Sample of missions for sanity-check.
    sample = sorted(set(M.tolist()))[:8]
    print(f"[ground]   sample missions: {sample}")

    # ---- V3 bank → operator labels ----
    print(f"[ground] loading V3 bank: {args.bank}")
    bank = OperatorBankV3.load(args.bank, device)
    bank.eval()
    print(f"[ground]   n_ops={bank.n_ops} latent_dim={bank.latent_dim}")

    print("[ground] computing operator labels per transition...")
    labels: list[int] = []
    bs = 8192
    with torch.no_grad():
        for i in range(0, len(L_t), bs):
            z = torch.from_numpy(L_t[i:i + bs]).to(device)
            a = torch.from_numpy(A[i:i + bs]).to(device)
            assign = bank.assign(z, a).cpu().numpy().tolist()
            labels.extend(assign)
    labels = np.array(labels, dtype=np.int64)
    label_hist = Counter(labels.tolist())
    print(f"[ground]   label histogram: "
          f"{dict(sorted(label_hist.items()))}")

    # ---- splits ----
    seen, holdout = split_missions(M, args.holdout_frac, args.seed)
    print(f"[ground] mission split: "
          f"{len(seen)} seen, {len(holdout)} held-out")
    seen_mask = np.array([m in seen for m in M])
    held_mask = ~seen_mask

    rng = np.random.default_rng(args.seed)
    seen_idx = np.flatnonzero(seen_mask)
    rng.shuffle(seen_idx)
    n_val = int(len(seen_idx) * args.id_val_frac)
    val_idx = seen_idx[:n_val]
    train_idx = seen_idx[n_val:]
    held_idx = np.flatnonzero(held_mask)
    print(f"[ground] transitions: "
          f"train={len(train_idx):,}  id_val={len(val_idx):,}  "
          f"held_out_comp={len(held_idx):,}")

    # ---- vocab ----
    vocab = WhitespaceVocab.build(
        [M[i] for i in train_idx], max_len=args.max_seq_len,
    )
    print(f"[ground] vocab size = {vocab.size}  "
          f"(max_seq_len={vocab.max_len})")

    # Pre-encode all missions for fast indexing.
    def encode_all(idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        texts = [M[i] for i in idx]
        return vocab.encode_batch(texts)

    train_tokens, train_mask = encode_all(train_idx)
    train_labels = torch.from_numpy(labels[train_idx])
    val_tokens, val_mask = encode_all(val_idx)
    val_labels = torch.from_numpy(labels[val_idx])
    held_tokens, held_mask_t = encode_all(held_idx)
    held_labels = torch.from_numpy(labels[held_idx])

    train_tokens = train_tokens.to(device)
    train_mask = train_mask.to(device)
    train_labels = train_labels.to(device)
    val_tokens = val_tokens.to(device)
    val_mask = val_mask.to(device)
    val_labels = val_labels.to(device)
    held_tokens = held_tokens.to(device)
    held_mask_t = held_mask_t.to(device)
    held_labels = held_labels.to(device)

    # ---- model ----
    model = make_grounding_head(args.kind, vocab.size, bank.n_ops).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ground] {args.kind} head: {n_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    vocab.save(str(out_dir / "vocab.pt"))

    # ---- train ----
    def eval_split(tokens, mask, lbls) -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            logits = model(tokens, mask)
            ce = float(F.cross_entropy(logits, lbls).item())
            pred = logits.argmax(dim=-1)
            acc = float((pred == lbls).float().mean().item())
        model.train()
        return ce, acc

    print(f"[ground] training {args.steps} steps, batch={args.batch_size}")
    for step in range(args.steps):
        idx = torch.randint(
            0, train_tokens.shape[0], (args.batch_size,), device=device,
        )
        tb = train_tokens[idx]
        mb = train_mask[idx]
        lb = train_labels[idx]

        logits = model(tb, mb)
        loss = F.cross_entropy(logits, lb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0:
            with torch.no_grad():
                tr_acc = float(
                    (logits.argmax(-1) == lb).float().mean().item()
                )
            val_ce, val_acc = eval_split(val_tokens, val_mask, val_labels)
            held_ce, held_acc = eval_split(
                held_tokens, held_mask_t, held_labels,
            )
            writer.add_scalar("train/loss", float(loss.item()), step)
            writer.add_scalar("train/acc", tr_acc, step)
            writer.add_scalar("val_id/loss", val_ce, step)
            writer.add_scalar("val_id/acc", val_acc, step)
            writer.add_scalar("val_held/loss", held_ce, step)
            writer.add_scalar("val_held/acc", held_acc, step)
            print(f"[step {step:5d}/{args.steps}] "
                  f"loss={float(loss.item()):.4f}  "
                  f"train_acc={tr_acc*100:.1f}%  "
                  f"id_acc={val_acc*100:.1f}%  "
                  f"held_acc={held_acc*100:.1f}%")

    # ---- final report ----
    val_ce, val_acc = eval_split(val_tokens, val_mask, val_labels)
    held_ce, held_acc = eval_split(held_tokens, held_mask_t, held_labels)

    print("\n=== final results ===")
    print(f"  in-distribution val acc:  {val_acc*100:5.1f}%  "
          f"(target ≥ 80%)")
    print(f"  held-out composition acc: {held_acc*100:5.1f}%  "
          f"(target ≥ 60%)")

    # Confusion matrix on held-out compositions.
    model.eval()
    with torch.no_grad():
        held_pred = model(held_tokens, held_mask_t).argmax(dim=-1).cpu().numpy()
    held_true = held_labels.cpu().numpy()
    K = bank.n_ops
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p_ in zip(held_true, held_pred):
        cm[t, p_] += 1
    print("\n=== confusion matrix on held-out compositions ===")
    print(f"  rows = V3 ground truth, cols = predicted")
    print("       " + " ".join(f"p{k:>4d}" for k in range(K)))
    for k in range(K):
        row = " ".join(f"{cm[k, j]:5d}" for j in range(K))
        diag = cm[k, k] / max(cm[k].sum(), 1)
        print(f"  t{k}: {row}   recall={diag*100:5.1f}%")

    # Save model + summary.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "kind": args.kind,
            "vocab_size": vocab.size,
            "n_ops": bank.n_ops,
            "args": vars(args),
        },
        out_dir / "grounding_final.pt",
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "id_acc": val_acc,
                "held_acc": held_acc,
                "id_pass": val_acc >= 0.80,
                "held_pass": held_acc >= 0.60,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_held": int(len(held_idx)),
                "n_seen_missions": len(seen),
                "n_held_missions": len(holdout),
                "vocab_size": vocab.size,
                "label_histogram": dict(label_hist),
            },
            f,
            indent=2,
        )

    print(f"\n=== verdict ===")
    if val_acc >= 0.80 and held_acc >= 0.60:
        print("  PASS — milestone 1.0 cleared.")
        print("  Next: Stage 1.1 (language-driven action selection).")
    else:
        print("  FAIL — milestone 1.0 not cleared.")
        if val_acc < 0.80:
            print(f"  - id_acc {val_acc*100:.1f}% < 80%")
        if held_acc < 0.60:
            print(f"  - held_acc {held_acc*100:.1f}% < 60%")
        print("  Diagnose:")
        print("    - if id_acc is fine but held fails -> need larger vocab "
              "OR transformer encoder")
        print("    - if both fail -> operator labels too noisy "
              "OR mission text doesn't carry enough operator signal")

    print(f"\n[saved] {out_dir / 'grounding_final.pt'}")
    print(f"[saved] {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
