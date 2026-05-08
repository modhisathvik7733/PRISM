"""Phase 2 v0 — train and evaluate a linear predicate probe on the JEPA latent.

This is the first real Phase 2 falsifier. We freeze the JEPA encoder trained
in Phase 1, then train a single linear layer to map the latent z_t to the
96-d predicate vector (4 predicates × 4 object types × 6 colors).

Ground-truth predicates come from `compute_predicates(extract_slots(obs))` —
fully deterministic, no learning, parsed straight from the symbolic
partial-view obs.

What this tests:
  Does the JEPA latent linearly encode "the red ball is in front of me"?

  * If linear-probe accuracy >> chance → grounding is real, the latent
    contains the structured information operators need. We can build the
    planner / action refiner on top.

  * If accuracy ≈ chance → JEPA learned dynamics but not object-structured
    semantics. Need to add object-centric inductive bias (slot attention,
    DINO-style heads, etc.) before Phase 3+.

Usage:
    python -m scripts.train_predicate_probe \
        --jepa-checkpoint runs/jepa_BabyAI-GoToLocal-v0_seed0/jepa_final.pt \
        --steps 20_000 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  (registers BabyAI envs)
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from prism.envs.babyai import _encode_image
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.models.predicate_probe import make_probe
from prism.perception import (
    NUM_PREDICATES,
    NUM_TYPE_COLOR_PAIRS,
    PREDICATE_NAMES,
    compute_predicates,
    extract_slots,
)
from prism.utils.seed import set_global_seed


def collect_dataset(
    env_id: str,
    n_transitions: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll random policy, return (encoded_obs, predicate_targets) arrays.

    We need access to the *raw* uint8 image to extract slots, so we bypass
    our PrismImageOnlyWrapper here and apply `_encode_image` ourselves after
    capturing the raw obs for the slot extractor.
    """
    env = gym.make(env_id)  # raw env: obs dict has uint8 image
    obs_list = []
    pred_list = []

    obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
    while len(obs_list) < n_transitions:
        raw = obs["image"]                 # (7, 7, 3) uint8 codes
        slots = extract_slots(raw)
        preds = compute_predicates(slots)  # (96,) float32
        encoded = _encode_image(raw)       # (3, 7, 7) float32 normalized
        obs_list.append(encoded)
        pred_list.append(preds)

        a = int(rng.integers(env.action_space.n))
        obs, _r, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))

    return (
        np.stack(obs_list).astype(np.float32),
        np.stack(pred_list).astype(np.float32),
    )


def per_predicate_accuracy(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Return {predicate_name: accuracy} averaged across (type, color) pairs."""
    pred_bin = (torch.sigmoid(logits) > 0.5).float()
    correct = (pred_bin == targets).float()  # (B, 96)
    # Reshape into (B, NUM_PREDICATES, NUM_TYPE_COLOR_PAIRS)
    correct = correct.view(-1, NUM_PREDICATES, NUM_TYPE_COLOR_PAIRS)
    per_pred = correct.mean(dim=(0, 2))  # (NUM_PREDICATES,)
    return {name: float(per_pred[i].item()) for i, name in enumerate(PREDICATE_NAMES)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    parser.add_argument("--train-transitions", type=int, default=50_000)
    parser.add_argument("--eval-transitions", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-name", default=None)
    # Diagnostics ----------------------------------------------------------
    parser.add_argument(
        "--probe-hidden", type=int, default=0,
        help="0 = linear probe (default, the honest test). >0 = MLP probe with that "
             "hidden width — diagnostic for 'is the info present but tangled?'"
    )
    parser.add_argument(
        "--use-raw-obs", action="store_true",
        help="Skip the JEPA encoder; train the probe on flattened raw obs instead. "
             "Upper bound: tells us how much of the predicate task is solvable from "
             "raw obs at all."
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    run_name = args.run_name or f"probe_{Path(args.jepa_checkpoint).parent.name}"
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    print(f"[probe] writing to {out_dir}")

    # ------------------------------------------------------ load frozen JEPA
    # When --use-raw-obs is set, we skip the JEPA encoder entirely and train
    # the probe on flattened raw obs. This is the *upper-bound* sanity check.
    if args.use_raw_obs:
        # We still need cfg.embed_dim conceptually — set it to the flat obs dim.
        cfg = JepaConfig()  # only used for n_actions etc. defaults
        embed_dim = cfg.obs_channels * cfg.obs_h * cfg.obs_w  # 3*7*7 = 147
        jepa = None
        print(f"[probe] --use-raw-obs: probing on flattened raw obs (dim={embed_dim})")
    else:
        ckpt = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
        cfg = upgrade_config(ckpt["cfg"])  # backward compat for pre-encoder_type ckpts
        embed_dim = cfg.embed_dim
        jepa = JepaWorldModel(cfg).to(device)
        jepa.load_state_dict(ckpt["model"])
        jepa.eval()
        for p in jepa.parameters():
            p.requires_grad_(False)
        encoder_type = getattr(cfg, "encoder_type", "flat")
        print(
            f"[probe] loaded JEPA ({encoder_type} encoder): "
            f"{sum(p.numel() for p in jepa.parameters()):,} params (frozen), "
            f"embed_dim={embed_dim}"
        )

    # ------------------------------------------------------ data
    rng = np.random.default_rng(args.seed)
    print(f"[probe] collecting {args.train_transitions} train transitions...")
    train_obs, train_preds = collect_dataset(args.env_id, args.train_transitions, rng)
    print(f"[probe] collecting {args.eval_transitions} eval transitions...")
    eval_obs, eval_preds = collect_dataset(args.env_id, args.eval_transitions, rng)

    base_rate = float(train_preds.mean())  # fraction of "true" predicates
    print(f"[probe] train obs={train_obs.shape} preds={train_preds.shape} "
          f"positive-rate={base_rate:.3f}")

    # ------------------------------------------------------ probe
    probe = make_probe(embed_dim=embed_dim, hidden=args.probe_hidden).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    arch_label = "linear" if args.probe_hidden <= 0 else f"MLP(hidden={args.probe_hidden})"
    print(f"[probe] probe arch: {arch_label}, params: {sum(p.numel() for p in probe.parameters()):,}")

    train_obs_t = torch.from_numpy(train_obs).to(device)
    train_preds_t = torch.from_numpy(train_preds).to(device)
    eval_obs_t = torch.from_numpy(eval_obs).to(device)
    eval_preds_t = torch.from_numpy(eval_preds).to(device)

    # Pre-encode everything once. With --use-raw-obs we just flatten.
    @torch.no_grad()
    def encode_all(obs_tensor: torch.Tensor, batch: int = 1024) -> torch.Tensor:
        if jepa is None:
            return obs_tensor.reshape(obs_tensor.shape[0], -1)
        out = []
        for i in range(0, obs_tensor.shape[0], batch):
            out.append(jepa.encode(obs_tensor[i : i + batch]))
        return torch.cat(out, dim=0)

    print("[probe] pre-encoding train + eval obs...")
    train_z = encode_all(train_obs_t)
    eval_z = encode_all(eval_obs_t)
    print(f"[probe] z shapes: train={tuple(train_z.shape)} eval={tuple(eval_z.shape)}")

    # ------------------------------------------------------ train
    n_train = train_z.shape[0]
    for step in range(args.steps):
        idx = torch.randint(0, n_train, (args.batch_size,), device=device)
        z = train_z[idx]
        y = train_preds_t[idx]
        logits = probe(z)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == args.steps - 1:
            with torch.no_grad():
                eval_logits = probe(eval_z)
                eval_loss = F.binary_cross_entropy_with_logits(
                    eval_logits, eval_preds_t
                ).item()
                acc = per_predicate_accuracy(eval_logits, eval_preds_t)
                overall = float(np.mean(list(acc.values())))
                # F1-style: precision / recall on the positive class
                pred_bin = (torch.sigmoid(eval_logits) > 0.5).float()
                tp = (pred_bin * eval_preds_t).sum().item()
                fp = (pred_bin * (1 - eval_preds_t)).sum().item()
                fn = ((1 - pred_bin) * eval_preds_t).sum().item()
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-9)

            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("loss/eval", eval_loss, step)
            writer.add_scalar("acc/overall", overall, step)
            writer.add_scalar("metric/f1", f1, step)
            for name, v in acc.items():
                writer.add_scalar(f"acc/{name}", v, step)

            print(
                f"[step {step:5d}] train_loss={loss.item():.4f} "
                f"eval_loss={eval_loss:.4f} overall_acc={overall*100:.2f}% "
                f"f1={f1:.3f}  "
                + " ".join(f"{n}={v*100:.1f}%" for n, v in acc.items())
            )

    # ------------------------------------------------------ summary
    print("\n=== final eval ===")
    with torch.no_grad():
        eval_logits = probe(eval_z)
        acc = per_predicate_accuracy(eval_logits, eval_preds_t)
        overall = float(np.mean(list(acc.values())))
        pred_bin = (torch.sigmoid(eval_logits) > 0.5).float()
        tp = (pred_bin * eval_preds_t).sum().item()
        fp = (pred_bin * (1 - eval_preds_t)).sum().item()
        fn = ((1 - pred_bin) * eval_preds_t).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    chance_acc = max(base_rate, 1 - base_rate)
    print(f"  positive rate (chance acc) : {chance_acc*100:.2f}%")
    print(f"  overall accuracy           : {overall*100:.2f}%")
    print(f"  precision / recall / f1    : {precision:.3f} / {recall:.3f} / {f1:.3f}")
    for name, v in acc.items():
        print(f"  {name:9s} accuracy        : {v*100:.2f}%")

    # Phase 2 v0 falsifier: F1 is the real signal because positive class is
    # rare. A degenerate "always-0" probe gets high accuracy with F1 ≈ 0.
    # We grade on F1 and report accuracy alongside.
    pass_f1 = f1 > 0.5
    print(f"\n  pass (f1 > 0.5)            : {'YES' if pass_f1 else 'NO'}")
    if args.use_raw_obs:
        verdict = (
            "PASS — task is linearly solvable from raw obs"
            if pass_f1
            else "FAIL — task isn't even solvable from raw obs at this probe depth"
        )
    elif args.probe_hidden > 0:
        verdict = (
            "PASS — info is in the JEPA latent, just needs nonlinear readout"
            if pass_f1
            else "FAIL — info genuinely missing from JEPA latent (MLP can't recover)"
        )
    else:
        verdict = (
            "PASS — JEPA latent linearly encodes object structure"
            if pass_f1
            else "FAIL — JEPA latent doesn't linearly encode object structure"
        )
    print(f"\n  Phase 2 v0 verdict: {verdict}")

    ckpt_out = out_dir / "probe_final.pt"
    torch.save({"probe": probe.state_dict(), "embed_dim": embed_dim}, ckpt_out)
    print(f"\n[probe] saved {ckpt_out}")
    return 0 if pass_f1 else 2


if __name__ == "__main__":
    raise SystemExit(main())
