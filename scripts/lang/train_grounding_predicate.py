"""Stage 1.0-proper Phase 2 — train text → predicate vector and measure
agreement with the PredicateReadout's view from latents.

Pipeline:
  1. Load PredicateReadout (z -> predicate logits) trained in Phase 1.
  2. For each episode, label = goal (color, type) parsed from mission.
  3. Train a text encoder: mission_text -> predicate logits, supervised
     against the same label.
  4. Evaluate on:
       - in-distribution missions  (seen combos)
       - held-out compositional   (unseen (color, type) combos)
  5. **Agreement metric** — does the text encoder agree with the
     readout's view from latents on held-out compositions?
     This is the grounding closure test.

The agreement metric is the crucial Stage 1 signal:
  * argmax(T(text)) == argmax(R(z_last)) on held-out episodes
  * If high → text and latent agree on what the goal IS, *without* either
    of them having been trained on this specific (color, type) combo.
    That is grounded compositional language.

Usage:
    python -m scripts.lang.train_grounding_predicate \
        --rollouts runs/cog_core_phase1_devB/rollouts.npz \
        --readout runs/predicate_readout_v0/predicate_readout_final.pt \
        --kind tiny_tf --steps 500 --batch-size 1024 \
        --holdout-combos 4 --run-name grounding_predicate_v0 \
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from prism.agents import goal_predicates_for_mission
from prism.cog_core.predicate_readout import PredicateReadout
from prism.language.grounding_head import (
    PAD_ID,
    WhitespaceVocab,
)
from prism.perception.slots import NUM_COLORS, NUM_TYPES, OBJECT_TYPES
from prism.utils.seed import set_global_seed


N_PRED = NUM_COLORS * NUM_TYPES                            # 24


class TinyTransformerPredicateHead(nn.Module):
    """text -> predicate logits. Single transformer block + mean pool."""

    def __init__(
        self,
        vocab_size: int,
        n_predicates: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        ff_dim: int = 128,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.pos = nn.Embedding(64, embed_dim)
        self.block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="gelu",
        )
        self.head = nn.Linear(embed_dim, n_predicates)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.embed(tokens) + self.pos(pos)
        x = self.block(x, src_key_padding_mask=~mask)
        m = mask.float().unsqueeze(-1)
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled)


def build_episode_data(npz_path: Path
                       ) -> tuple[
                           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One row per episode. Returns (z_last, label, mission, combo)."""
    d = np.load(npz_path)
    if "missions" not in d.files:
        raise SystemExit("rollouts.npz missing `missions` field")
    latents = d["latents"]
    lengths = d["ep_lengths"]
    missions = d["missions"]
    type_to_idx = {t: i for i, t in enumerate(OBJECT_TYPES)}

    Z, Y, M, CT = [], [], [], []
    for i in range(len(lengths)):
        L = int(lengths[i])
        if L < 1:
            continue
        parsed = goal_predicates_for_mission(str(missions[i]))
        if parsed is None:
            continue
        gp = parsed[0][0]
        if gp.type_id not in type_to_idx:
            continue
        c = int(gp.color_id)
        t = type_to_idx[gp.type_id]
        Z.append(latents[i, L - 1])                    # last frame
        Y.append(c * NUM_TYPES + t)
        M.append(str(missions[i]))
        CT.append((c, t))

    Z = np.stack(Z).astype(np.float32)
    if Z.ndim > 2:
        Z = Z.reshape(Z.shape[0], -1)
    return (Z, np.array(Y, dtype=np.int64),
            np.array(M), np.array(CT, dtype=np.int64))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True)
    p.add_argument("--readout", required=True,
                   help="trained PredicateReadout checkpoint from Phase 1")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--holdout-combos", type=int, default=4)
    p.add_argument("--id-val-frac", type=float, default=0.1)
    p.add_argument("--max-seq-len", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", required=True)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # ---- data ----
    print(f"[ground-p] loading rollouts: {args.rollouts}")
    Z, Y, M, CT = build_episode_data(Path(args.rollouts))
    print(f"[ground-p]   {len(Z):,} episodes, latent_dim={Z.shape[-1]}")
    print(f"[ground-p]   label histogram: "
          f"{dict(sorted(Counter(Y.tolist()).items()))}")

    # ---- compositional split ----
    combos = sorted({(int(c), int(t)) for c, t in CT})
    rng = random.Random(args.seed)
    rng.shuffle(combos)
    n_hold = min(args.holdout_combos, max(1, len(combos) // 4))
    held = set(combos[:n_hold])
    held_mask = np.array([(int(c), int(t)) in held for c, t in CT])
    seen_idx = np.flatnonzero(~held_mask)
    held_idx = np.flatnonzero(held_mask)
    np_rng = np.random.default_rng(args.seed)
    np_rng.shuffle(seen_idx)
    n_val = int(len(seen_idx) * args.id_val_frac)
    val_idx = seen_idx[:n_val]
    train_idx = seen_idx[n_val:]
    print(f"[ground-p]   held-out combos: {sorted(held)}")
    print(f"[ground-p]   episodes: train={len(train_idx):,}  "
          f"id_val={len(val_idx):,}  held={len(held_idx):,}")

    # ---- vocab ----
    vocab = WhitespaceVocab.build(
        [str(M[i]) for i in train_idx], max_len=args.max_seq_len,
    )
    print(f"[ground-p]   vocab size = {vocab.size}")

    def enc(idx: np.ndarray):
        tok, msk = vocab.encode_batch([str(M[i]) for i in idx])
        return (
            tok.to(device), msk.to(device),
            torch.from_numpy(Y[idx]).to(device),
            torch.from_numpy(Z[idx]).to(device),
        )

    tr_tok, tr_msk, tr_y, tr_z = enc(train_idx)
    va_tok, va_msk, va_y, va_z = enc(val_idx)
    he_tok, he_msk, he_y, he_z = enc(held_idx)

    # ---- text model ----
    text_model = TinyTransformerPredicateHead(
        vocab_size=vocab.size,
        n_predicates=N_PRED,
    ).to(device)
    n_params = sum(p_.numel() for p_ in text_model.parameters())
    print(f"[ground-p]   tiny_tf predicate head: {n_params:,} params")

    # ---- load PredicateReadout (frozen) ----
    readout = PredicateReadout.load(args.readout, device)
    readout.eval()
    for p_ in readout.parameters():
        p_.requires_grad_(False)
    print(f"[ground-p]   loaded frozen readout from {args.readout}")

    opt = torch.optim.AdamW(text_model.parameters(), lr=args.lr)
    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    vocab.save(str(out_dir / "vocab.pt"))

    def eval_split(tok, msk, y, z) -> dict:
        text_model.eval()
        with torch.no_grad():
            t_logits = text_model(tok, msk)
            t_pred = t_logits.argmax(-1)
            t_acc = float((t_pred == y).float().mean().item())
            # readout view from latents
            r_logits = readout(z)
            r_pred = r_logits.argmax(-1)
            r_acc = float((r_pred == y).float().mean().item())
            agreement = float((t_pred == r_pred).float().mean().item())
        text_model.train()
        return {"text_acc": t_acc, "readout_acc": r_acc, "agreement": agreement}

    # ---- train ----
    print(f"[ground-p] training {args.steps} steps, batch={args.batch_size}")
    for step in range(args.steps):
        idx = torch.randint(0, tr_tok.shape[0], (args.batch_size,), device=device)
        logits = text_model(tr_tok[idx], tr_msk[idx])
        loss = F.cross_entropy(logits, tr_y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(text_model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            tr_acc = float((logits.argmax(-1) == tr_y[idx]).float().mean().item())
            v = eval_split(va_tok, va_msk, va_y, va_z)
            h = eval_split(he_tok, he_msk, he_y, he_z)
            writer.add_scalar("train/loss", float(loss.item()), step)
            writer.add_scalar("val_id/text_acc", v["text_acc"], step)
            writer.add_scalar("val_id/agreement", v["agreement"], step)
            writer.add_scalar("val_held/text_acc", h["text_acc"], step)
            writer.add_scalar("val_held/agreement", h["agreement"], step)
            print(f"[step {step:5d}/{args.steps}]  "
                  f"loss={float(loss.item()):.3f}  "
                  f"tr_acc={tr_acc*100:5.1f}%  "
                  f"id: text={v['text_acc']*100:.0f}/"
                  f"readout={v['readout_acc']*100:.0f}/"
                  f"agree={v['agreement']*100:.0f}  "
                  f"held: text={h['text_acc']*100:.0f}/"
                  f"readout={h['readout_acc']*100:.0f}/"
                  f"agree={h['agreement']*100:.0f}")

    # ---- final ----
    v = eval_split(va_tok, va_msk, va_y, va_z)
    h = eval_split(he_tok, he_msk, he_y, he_z)
    print("\n=== final results ===")
    print(f"  in-distribution (seen combos):")
    print(f"    text accuracy:               {v['text_acc']*100:5.1f}%")
    print(f"    readout-from-latent accuracy:{v['readout_acc']*100:5.1f}%")
    print(f"    agreement(text, readout):    {v['agreement']*100:5.1f}%")
    print(f"  held-out (compositional):")
    print(f"    text accuracy:               {h['text_acc']*100:5.1f}%")
    print(f"    readout-from-latent accuracy:{h['readout_acc']*100:5.1f}%")
    print(f"    agreement(text, readout):    {h['agreement']*100:5.1f}%  "
          f"<<<  Stage 1 grounding signal")

    pass_text = h["text_acc"] >= 0.80
    pass_agreement = h["agreement"] >= 0.70
    print("\n=== verdict ===")
    if pass_text and pass_agreement:
        print("  PASS — text encoder predicts the same goal predicate "
              "as readout-from-latent on unseen (color, type) combinations.")
        print("  Stage 1.0-proper cleared. Language is grounded.")
    else:
        reasons = []
        if not pass_text:
            reasons.append(f"text held-out acc {h['text_acc']*100:.1f}% < 80%")
        if not pass_agreement:
            reasons.append(
                f"text/readout agreement {h['agreement']*100:.1f}% < 70%"
            )
        print(f"  FAIL — {'; '.join(reasons)}")
        if h["readout_acc"] < 0.50:
            print("  Note: readout itself is weak on held-out — bottleneck "
                  "is at the JEPA-latent level, not the text encoder.")
        elif h["text_acc"] >= 0.80 and h["agreement"] < 0.70:
            print("  Note: text and readout each work individually but "
                  "disagree on held-out. Could mean they're each correct "
                  "on different subsets, OR one of them is being clever "
                  "without grounding.")

    torch.save(
        {
            "state_dict": text_model.state_dict(),
            "vocab_size": vocab.size,
            "n_predicates": N_PRED,
            "args": vars(args),
        },
        out_dir / "grounding_predicate_final.pt",
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "id_text_acc": v["text_acc"],
            "id_readout_acc": v["readout_acc"],
            "id_agreement": v["agreement"],
            "held_text_acc": h["text_acc"],
            "held_readout_acc": h["readout_acc"],
            "held_agreement": h["agreement"],
            "pass_text": pass_text,
            "pass_agreement": pass_agreement,
            "held_combos": sorted(list(held)),
        }, f, indent=2)
    print(f"\n[saved] {out_dir / 'grounding_predicate_final.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
