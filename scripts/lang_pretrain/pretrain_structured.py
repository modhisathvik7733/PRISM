"""Pretrain PrismLangModel (encoder + JEPA-middle + decoder) from
scratch on TinyStories using T5-style span corruption.

This is the thesis-pure pretraining: NO pretrained weights load. The
encoder, middle, and decoder all start from random init; everything
the model learns about English flows from gradient on span-corruption
loss.

The structured model is the SAME `prism.lang.model.PrismLangModel`
from v3.0 — no architectural changes. Only the training data + loop
are new.

Usage:
    python -m scripts.lang_pretrain.pretrain_structured \
        --steps 60000 --batch-size 32 --seq-len 256 \
        --run-name lang_pre_struct_v0 --device cuda
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
from prism.lang.model import PrismLangModel
from prism.lang_pretrain.corrupt import corrupt_batch
from prism.utils.seed import set_global_seed
from scripts.lang_pretrain.data_tinystories import (
    TokenSampler, tokenize_and_cache,
)


def cosine_lr(step: int, total: int, peak_lr: float, warmup: int = 1000) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def quick_eval(
    model: PrismLangModel,
    val_sampler: TokenSampler,
    cfg: LangConfig,
    device: torch.device,
    n_batches: int = 20,
    batch_size: int = 16,
    seq_len: int = 256,
    rng: np.random.Generator | None = None,
) -> float:
    """Mean validation loss over n_batches of corrupted windows."""
    model.eval()
    rng = rng or np.random.default_rng(0)
    losses = []
    for _ in range(n_batches):
        windows = val_sampler.sample_windows(batch_size, seq_len, rng)
        in_ids, t_in, t_out, t_mask = corrupt_batch(
            windows, rng=rng,
            pad_id=cfg.pad_token_id, bos_id=cfg.bos_token_id,
            eos_id=cfg.eos_token_id,
            max_in_len=seq_len, max_out_len=seq_len // 2,
        )
        out = model.loss(
            in_ids.to(device), t_in.to(device), t_out.to(device),
            target_mask=t_mask.to(device),
        )
        losses.append(float(out["ce_loss"].item()))
    return float(np.mean(losses))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="small",
                        choices=["tiny", "small", "medium"])
    parser.add_argument("--steps", type=int, default=60_000,
                        help="~60k @ batch 32 × seq 256 = ~500M tokens "
                             "(roughly 1 epoch over TinyStories).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=6e-4,
                        help="Higher LR than fine-tuning since we're "
                             "training from random init.")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mask-ratio", type=float, default=0.15,
                        help="Span corruption mask ratio (T5 default 0.15).")
    parser.add_argument("--avg-span-len", type=float, default=3.0,
                        help="Mean span length (T5 default 3).")
    parser.add_argument("--jepa-aux-weight", type=float, default=0.1)
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
    cfg.jepa_aux_weight = args.jepa_aux_weight
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len + 16)
    print(f"[pretrain-struct] preset={args.preset} d={cfg.d_model} "
          f"enc={cfg.n_enc_layers} mid_steps={cfg.n_thought_steps} "
          f"dec={cfg.n_dec_layers} jepa_aux={cfg.jepa_aux_weight}")

    # ---- data ----
    print("[pretrain-struct] preparing TinyStories cache (one-time, ~10 min)…")
    train_path, val_path = tokenize_and_cache()
    train_sampler = TokenSampler(train_path, seq_len=args.seq_len)
    val_sampler = TokenSampler(val_path, seq_len=args.seq_len)
    print(f"[pretrain-struct] train tokens={train_sampler.n_tokens:,} "
          f"val tokens={val_sampler.n_tokens:,}")

    # ---- model ----
    model = PrismLangModel(cfg).to(device)
    print(f"[pretrain-struct] params: {model.num_params() if hasattr(model, 'num_params') else sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    print(f"[pretrain-struct] writing to {out_dir}")

    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1)
    loss_window: deque[float] = deque(maxlen=200)

    for step in range(args.steps):
        # ---- sample + corrupt ----
        windows = train_sampler.sample_windows(args.batch_size, args.seq_len, rng)
        in_ids, t_in, t_out, t_mask = corrupt_batch(
            windows, rng=rng,
            pad_id=cfg.pad_token_id, bos_id=cfg.bos_token_id,
            eos_id=cfg.eos_token_id,
            max_in_len=args.seq_len, max_out_len=args.seq_len // 2,
            mask_ratio=args.mask_ratio, avg_span_len=args.avg_span_len,
        )
        in_ids = in_ids.to(device)
        t_in = t_in.to(device)
        t_out = t_out.to(device)
        t_mask = t_mask.to(device)

        # ---- step ----
        model.train()
        lr = cosine_lr(step, args.steps, args.lr, warmup=args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        out = model.loss(in_ids, t_in, t_out, target_mask=t_mask)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        model.middle.update_ema()

        loss_window.append(float(loss.item()))

        # ---- log ----
        if step % args.log_every_steps == 0:
            mean_loss = float(np.mean(loss_window)) if loss_window else float("nan")
            writer.add_scalar("train/loss", float(loss.item()), step)
            writer.add_scalar("train/ce", float(out["ce_loss"].item()), step)
            writer.add_scalar("train/aux", float(out["aux_loss"].item()), step)
            writer.add_scalar("lr", lr, step)
            print(f"[step {step:6d}/{args.steps}] loss={float(loss.item()):.4f} "
                  f"ce={float(out['ce_loss'].item()):.4f} "
                  f"aux={float(out['aux_loss'].item()):.4f} "
                  f"mean200={mean_loss:.4f} lr={lr:.2e}")

        # ---- eval ----
        if (step + 1) % args.eval_every_steps == 0 or step == args.steps - 1:
            val_loss = quick_eval(
                model, val_sampler, cfg, device, rng=eval_rng,
                seq_len=args.seq_len,
            )
            writer.add_scalar("val/ce", val_loss, step)
            print(f"  [val @ step {step+1}] ce={val_loss:.4f} "
                  f"perplexity={math.exp(val_loss):.2f}")

        # ---- save ----
        if (step + 1) % args.save_every_steps == 0 or step == args.steps - 1:
            ckpt_path = out_dir / f"model_step{step+1}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
                "step": step + 1,
                "args": vars(args),
            }, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

    final = out_dir / "model_final.pt"
    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg,
                "step": args.steps, "args": vars(args)}, final)
    val_loss = quick_eval(model, val_sampler, cfg, device, rng=eval_rng,
                          seq_len=args.seq_len, n_batches=50)
    print(f"[done] saved {final}")
    print(f"[done] final val_ce={val_loss:.4f} perplexity={math.exp(val_loss):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
