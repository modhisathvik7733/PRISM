"""Pretrain VanillaARModel from scratch on TinyStories — the matched-
param baseline for the Path B comparison.

Same data, same total compute as scripts/lang_pretrain/pretrain_structured.py.
The ONLY differences:
  - Model: VanillaARModel (decoder-only AR transformer, no encoder/middle)
  - Objective: next-token prediction (the natural objective for AR)

After both finish, scripts/lang_pretrain/finetune_babi.py fine-tunes
each on bAbI and the per-task accuracy difference is the JEPA-middle's
isolated contribution.

Usage:
    python -m scripts.lang_pretrain.pretrain_baseline \
        --steps 60000 --batch-size 32 --seq-len 256 \
        --run-name lang_pre_baseline_v0 --device cuda
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from prism.lang.config import LangConfig
from prism.lang_pretrain.vanilla_ar import VanillaARModel
from prism.utils.seed import set_global_seed
from scripts.lang_pretrain.data_tinystories import (
    TokenSampler, tokenize_and_cache,
)


def cosine_lr(step: int, total: int, peak_lr: float, warmup: int = 1000) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def make_ar_batch(token_window_batch: np.ndarray
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """For next-token prediction: input = window[:-1], target = window[1:].

    Caller passes (B, T+1) windows; we shift by 1 to get (B, T) input
    and (B, T) target."""
    x = token_window_batch[:, :-1]
    y = token_window_batch[:, 1:]
    return torch.from_numpy(x).long(), torch.from_numpy(y).long()


@torch.no_grad()
def quick_eval(
    model: VanillaARModel,
    val_sampler: TokenSampler,
    device: torch.device,
    n_batches: int = 20,
    batch_size: int = 16,
) -> float:
    model.eval()
    rng = np.random.default_rng(0)
    losses = []
    for _ in range(n_batches):
        windows = val_sampler.sample_batch(batch_size, rng)
        x, y = make_ar_batch(windows)
        out = model.loss(x.to(device), y.to(device))
        losses.append(float(out["ce_loss"].item()))
    return float(np.mean(losses))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="small",
                        choices=["tiny", "small", "medium"])
    parser.add_argument("--n-layers", type=int, default=12,
                        help="Number of AR transformer layers. 12 matches "
                             "the structured model's 4+6+4 = 14 effective "
                             "depth at d=256.")
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--save-every-steps", type=int, default=5000)
    parser.add_argument("--eval-every-steps", type=int, default=1000)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    cfg = LangConfig().scale_to(args.preset)
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len + 16)
    print(f"[pretrain-base] preset={args.preset} d={cfg.d_model} "
          f"n_layers={args.n_layers}")

    print("[pretrain-base] preparing TinyStories cache (one-time)…")
    train_path, val_path = tokenize_and_cache()
    train_sampler = TokenSampler(train_path, seq_len=args.seq_len + 1)
    val_sampler = TokenSampler(val_path, seq_len=args.seq_len + 1)
    print(f"[pretrain-base] train tokens={train_sampler.n_tokens:,} "
          f"val tokens={val_sampler.n_tokens:,}")

    model = VanillaARModel(cfg, n_layers=args.n_layers).to(device)
    print(f"[pretrain-base] params: {model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    print(f"[pretrain-base] writing to {out_dir}")

    rng = np.random.default_rng(args.seed)
    loss_window: deque[float] = deque(maxlen=200)

    for step in range(args.steps):
        windows = train_sampler.sample_batch(args.batch_size, rng)
        x, y = make_ar_batch(windows)
        x = x.to(device)
        y = y.to(device)

        model.train()
        lr = cosine_lr(step, args.steps, args.lr, warmup=args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        out = model.loss(x, y)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()

        loss_window.append(float(loss.item()))

        if step % args.log_every_steps == 0:
            mean_loss = float(np.mean(loss_window)) if loss_window else float("nan")
            writer.add_scalar("train/loss", float(loss.item()), step)
            writer.add_scalar("train/ce", float(out["ce_loss"].item()), step)
            writer.add_scalar("lr", lr, step)
            print(f"[step {step:6d}/{args.steps}] loss={float(loss.item()):.4f} "
                  f"mean200={mean_loss:.4f} lr={lr:.2e}")

        if (step + 1) % args.eval_every_steps == 0 or step == args.steps - 1:
            val_loss = quick_eval(model, val_sampler, device)
            writer.add_scalar("val/ce", val_loss, step)
            print(f"  [val @ step {step+1}] ce={val_loss:.4f} "
                  f"perplexity={math.exp(val_loss):.2f}")

        if (step + 1) % args.save_every_steps == 0 or step == args.steps - 1:
            ckpt_path = out_dir / f"model_step{step+1}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
                "n_layers": args.n_layers,
                "step": step + 1,
                "args": vars(args),
            }, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

    final = out_dir / "model_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": cfg, "n_layers": args.n_layers,
        "step": args.steps, "args": vars(args),
    }, final)
    val_loss = quick_eval(model, val_sampler, device, n_batches=50)
    print(f"[done] saved {final}")
    print(f"[done] final val_ce={val_loss:.4f} perplexity={math.exp(val_loss):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
