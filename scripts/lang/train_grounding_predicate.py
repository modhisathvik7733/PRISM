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


N_PRED = NUM_COLORS + NUM_TYPES                            # 6 + 4 = 10 (factored)


class TinyTransformerPredicateHead(nn.Module):
    """text -> factored predicate logits (NUM_COLORS + NUM_TYPES).
    First NUM_COLORS are color logits, remaining are type logits."""

    def __init__(
        self,
        vocab_size: int,
        n_predicates: int = N_PRED,
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


def build_episode_data(
    npz_path: Path,
    *,
    slots_path: Path | None = None,
    require_goal_visible: bool = False,
):
    """One row per episode. Returns (z, color_label, type_label, mission).

    By default, `z` is the latent at the *final* frame of the episode and
    the labels come from parsing the mission ("go to the red ball" →
    (red, ball)). With random rollouts the agent often did NOT reach the
    goal, so z_last and the mission may not agree on what's in view.

    If `require_goal_visible=True` and `slots_path` is given, `z` is
    instead the latent at the *latest* frame where the mission's target
    `(color, type)` is actually visible in slots. Episodes where the
    target never appears in view are skipped. This isolates the
    perception question (does the JEPA encode the goal predicate when
    visible?) from the policy question (does the agent reach the goal?).
    """
    d = np.load(npz_path)
    if "missions" not in d.files:
        raise SystemExit("rollouts.npz missing `missions` field")
    latents = d["latents"]
    lengths = d["ep_lengths"]
    missions = d["missions"]
    type_to_idx = {t: i for i, t in enumerate(OBJECT_TYPES)}

    slots_all = None
    if require_goal_visible:
        if slots_path is None:
            raise SystemExit("--require-goal-visible needs --slots path")
        import pickle as _pickle
        with open(slots_path, "rb") as f:
            slots_all = _pickle.load(f)
        if len(slots_all) != latents.shape[0]:
            raise SystemExit(
                f"slot count {len(slots_all)} != episode count "
                f"{latents.shape[0]}"
            )

    Z, C, T, M = [], [], [], []
    skipped = 0
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
        goal_color = int(gp.color_id)
        goal_type_id = int(gp.type_id)
        goal_type_idx = type_to_idx[gp.type_id]

        if require_goal_visible:
            # Find the latest frame index where the goal target is visible.
            ep_slots = slots_all[i]
            visible_t = -1
            for t in range(min(L, len(ep_slots))):
                for s in ep_slots[t]:
                    if (int(s["type_id"]) == goal_type_id
                            and int(s["color_id"]) == goal_color):
                        visible_t = t
                        break
            if visible_t < 0:
                skipped += 1
                continue
            Z.append(latents[i, visible_t])
        else:
            Z.append(latents[i, L - 1])

        C.append(goal_color)
        T.append(goal_type_idx)
        M.append(str(missions[i]))

    if require_goal_visible:
        print(f"[ground-p]   goal-visible filter: kept {len(Z)}, "
              f"skipped {skipped} episodes where goal never appeared")

    Z = np.stack(Z).astype(np.float32)
    if Z.ndim > 2:
        Z = Z.reshape(Z.shape[0], -1)
    return (
        Z,
        np.array(C, dtype=np.int64),
        np.array(T, dtype=np.int64),
        np.array(M),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True)
    p.add_argument("--readout", required=True,
                   help="trained PredicateReadout checkpoint from Phase 1")
    p.add_argument("--slots", default=None,
                   help="path to rollouts.slots.pkl; required when "
                        "--require-goal-visible is set")
    p.add_argument("--require-goal-visible", action="store_true",
                   help="use the latest frame where the mission target is "
                        "actually visible in slots, instead of z_last. "
                        "Isolates the perception question from the policy "
                        "question — proper test for v4.1.1 JEPA grounding.")
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
    Z, C, T, M = build_episode_data(
        Path(args.rollouts),
        slots_path=Path(args.slots) if args.slots else None,
        require_goal_visible=args.require_goal_visible,
    )
    print(f"[ground-p]   {len(Z):,} episodes, latent_dim={Z.shape[-1]}")
    print(f"[ground-p]   color hist: "
          f"{dict(sorted(Counter(C.tolist()).items()))}")
    print(f"[ground-p]   type  hist: "
          f"{dict(sorted(Counter(T.tolist()).items()))}")

    # ---- compositional split over (color, type) combos ----
    CT_pairs = list(zip(C.tolist(), T.tolist()))
    combos = sorted({(int(c), int(t)) for c, t in CT_pairs})
    rng = random.Random(args.seed)
    rng.shuffle(combos)
    n_hold = min(args.holdout_combos, max(1, len(combos) // 4))
    held = set(combos[:n_hold])
    held_mask = np.array([(int(c), int(t)) in held for c, t in CT_pairs])
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
            torch.from_numpy(C[idx]).to(device),
            torch.from_numpy(T[idx]).to(device),
            torch.from_numpy(Z[idx]).to(device),
        )

    tr_tok, tr_msk, tr_c, tr_t, tr_z = enc(train_idx)
    va_tok, va_msk, va_c, va_t, va_z = enc(val_idx)
    he_tok, he_msk, he_c, he_t, he_z = enc(held_idx)

    # ---- text model (factored heads: NUM_COLORS + NUM_TYPES) ----
    text_model = TinyTransformerPredicateHead(
        vocab_size=vocab.size,
    ).to(device)
    n_params = sum(p_.numel() for p_ in text_model.parameters())
    print(f"[ground-p]   tiny_tf factored head: {n_params:,} params")

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

    def split(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return logits[:, :NUM_COLORS], logits[:, NUM_COLORS:]

    def eval_split(tok, msk, c, t, z) -> dict:
        text_model.eval()
        with torch.no_grad():
            # Text → (color, type).
            t_logits = text_model(tok, msk)
            t_lc, t_lt = split(t_logits)
            t_pc = t_lc.argmax(-1)
            t_pt = t_lt.argmax(-1)
            text_c = float((t_pc == c).float().mean().item())
            text_t = float((t_pt == t).float().mean().item())
            text_joint = float(
                ((t_pc == c) & (t_pt == t)).float().mean().item()
            )
            # Readout from latent → (color, type).
            r_logits = readout(z)
            r_lc, r_lt = split(r_logits)
            r_pc = r_lc.argmax(-1)
            r_pt = r_lt.argmax(-1)
            r_c = float((r_pc == c).float().mean().item())
            r_t = float((r_pt == t).float().mean().item())
            r_joint = float(
                ((r_pc == c) & (r_pt == t)).float().mean().item()
            )
            # Agreement: T(text) == R(z).
            ag_c = float((t_pc == r_pc).float().mean().item())
            ag_t = float((t_pt == r_pt).float().mean().item())
            ag_joint = float(
                ((t_pc == r_pc) & (t_pt == r_pt)).float().mean().item()
            )
        text_model.train()
        return {
            "text_c": text_c, "text_t": text_t, "text_joint": text_joint,
            "read_c": r_c, "read_t": r_t, "read_joint": r_joint,
            "ag_c": ag_c, "ag_t": ag_t, "ag_joint": ag_joint,
        }

    # ---- train ----
    print(f"[ground-p] training {args.steps} steps, batch={args.batch_size}")
    for step in range(args.steps):
        idx = torch.randint(0, tr_tok.shape[0],
                            (args.batch_size,), device=device)
        logits = text_model(tr_tok[idx], tr_msk[idx])
        lc, lt = split(logits)
        loss = F.cross_entropy(lc, tr_c[idx]) + F.cross_entropy(lt, tr_t[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(text_model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            with torch.no_grad():
                pc = lc.argmax(-1)
                pt = lt.argmax(-1)
                tr_joint = float(
                    ((pc == tr_c[idx]) & (pt == tr_t[idx]))
                    .float().mean().item()
                )
            v = eval_split(va_tok, va_msk, va_c, va_t, va_z)
            h = eval_split(he_tok, he_msk, he_c, he_t, he_z)
            writer.add_scalar("train/loss", float(loss.item()), step)
            writer.add_scalar("val_id/text_joint", v["text_joint"], step)
            writer.add_scalar("val_held/text_joint", h["text_joint"], step)
            writer.add_scalar("val_held/agree_joint", h["ag_joint"], step)
            print(f"[step {step:5d}/{args.steps}]  "
                  f"loss={float(loss.item()):.3f}  "
                  f"tr={tr_joint*100:5.1f}%  "
                  f"id_text={v['text_joint']*100:.0f}  "
                  f"held_text={h['text_joint']*100:.0f}  "
                  f"held_read={h['read_joint']*100:.0f}  "
                  f"held_agree={h['ag_joint']*100:.0f}")

    # ---- final ----
    v = eval_split(va_tok, va_msk, va_c, va_t, va_z)
    h = eval_split(he_tok, he_msk, he_c, he_t, he_z)
    print("\n=== final results ===")
    print(f"  in-distribution (seen combos):")
    print(f"    text       color/type/joint: "
          f"{v['text_c']*100:.1f}% / {v['text_t']*100:.1f}% / "
          f"{v['text_joint']*100:.1f}%")
    print(f"    readout    color/type/joint: "
          f"{v['read_c']*100:.1f}% / {v['read_t']*100:.1f}% / "
          f"{v['read_joint']*100:.1f}%")
    print(f"    agreement  color/type/joint: "
          f"{v['ag_c']*100:.1f}% / {v['ag_t']*100:.1f}% / "
          f"{v['ag_joint']*100:.1f}%")
    print(f"  held-out (compositional, unseen combos):")
    print(f"    text       color/type/joint: "
          f"{h['text_c']*100:.1f}% / {h['text_t']*100:.1f}% / "
          f"{h['text_joint']*100:.1f}%")
    print(f"    readout    color/type/joint: "
          f"{h['read_c']*100:.1f}% / {h['read_t']*100:.1f}% / "
          f"{h['read_joint']*100:.1f}%")
    print(f"    agreement  color/type/joint: "
          f"{h['ag_c']*100:.1f}% / {h['ag_t']*100:.1f}% / "
          f"{h['ag_joint']*100:.1f}%  <<<  Stage 1 grounding signal")

    pass_text = h["text_joint"] >= 0.80
    pass_readout = h["read_joint"] >= 0.40
    pass_agreement = h["ag_joint"] >= 0.50
    print("\n=== verdict ===")
    if pass_text and pass_readout and pass_agreement:
        print("  PASS — text and latent-readout both ground to the same "
              "compositional (color, type) predicate. Stage 1.0-proper "
              "cleared.")
    else:
        reasons = []
        if not pass_text:
            reasons.append(
                f"text held-out joint {h['text_joint']*100:.1f}% < 80%")
        if not pass_readout:
            reasons.append(
                f"readout held-out joint {h['read_joint']*100:.1f}% < 40%")
        if not pass_agreement:
            reasons.append(
                f"agreement joint {h['ag_joint']*100:.1f}% < 50%")
        print(f"  PARTIAL/FAIL — {'; '.join(reasons)}")
        print("  Diagnostic:")
        print(f"    text alone     compositional? "
              f"{'yes' if h['text_joint']>=0.80 else 'no'}")
        print(f"    readout alone  compositional? "
              f"{'yes' if h['read_joint']>=0.40 else 'no'}")
        print(f"    they agree?    "
              f"{'yes' if h['ag_joint']>=0.50 else 'no'}")
        print("  Note: low T/R agreement may reflect that with random "
              "rollouts, z_last shows whatever the agent randomly ended "
              "up viewing — not the mission target. So agreement is "
              "bounded by the random policy's success rate at reaching "
              "the target.")

    torch.save(
        {
            "state_dict": text_model.state_dict(),
            "vocab_size": vocab.size,
            "args": vars(args),
        },
        out_dir / "grounding_predicate_final.pt",
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "id": v, "held": h,
            "pass_text": pass_text,
            "pass_readout": pass_readout,
            "pass_agreement": pass_agreement,
            "held_combos": sorted(list(held)),
        }, f, indent=2)
    print(f"\n[saved] {out_dir / 'grounding_predicate_final.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
