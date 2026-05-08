"""Train PrismLangV2 on GSM8K.

Stage 0 (this script's default) — full CoT supervision:
    target = " <reasoning>\\n#### <answer>"
    The model is trained to output BOTH the chain-of-thought AND the
    final answer. Easier than answer-only because there's a gradient
    signal at every reasoning step. The latent middle still gets
    gradient (it's in the forward path) but the task doesn't FORCE the
    middle to do the work — the decoder can lean on autoregressive CoT.

Stage 1 (--coconut-curriculum, future work) — Coconut-style curriculum:
    Gradually replace CoT tokens with continuous latent steps. Start at
    100% CoT supervision; over training, drop CoT tokens (replace with
    additional middle thinking steps). End at 0% CoT supervision —
    middle does ALL the reasoning. Not implemented yet; this script
    leaves the hook.

Usage:
    python -m scripts.lang_v2.train \
        --backbone gpt2 \
        --steps 5000 --batch-size 8 \
        --run-name lang_v2_gsm8k_v0 --device cuda
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from prism.lang.tokenizer import encode_batch, get_tokenizer
from prism.lang_v2.model import PrismLangV2
from prism.utils.seed import set_global_seed
from scripts.lang_v2.data_gsm8k import (
    format_input, format_target_answer_only, format_target_with_cot,
    is_correct, load_gsm8k, normalize_pred,
)


def encode_examples(
    examples: list[dict],
    *,
    cfg_pad_id: int,
    cfg_bos_id: int,
    cfg_eos_id: int,
    max_in_len: int,
    max_out_len: int,
    with_cot: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize a batch into (input_ids, target_in, target_out, target_mask).

    target_in  = [BOS, t_0, t_1, ..., t_{L-1}]  (decoder inputs)
    target_out = [t_0, t_1, ..., t_{L-1}, EOS]  (next-token targets)
    target_mask = 1 for non-PAD positions in target_out
    """
    tok = get_tokenizer("gpt2")
    inputs = [format_input(ex["question"]) for ex in examples]
    if with_cot:
        targets = [format_target_with_cot(ex["cot"], ex["answer"]) for ex in examples]
    else:
        targets = [format_target_answer_only(ex["answer"]) for ex in examples]

    input_ids, _ = encode_batch(inputs, max_len=max_in_len)

    # Tokenize targets individually so we can pad to the actual max len.
    tgt_ids_per = [tok(t, return_tensors="pt").input_ids[0] for t in targets]
    # +1 for BOS at front (target_in) and +1 for EOS at end (target_out).
    real_max = min(max_out_len, max(int(a.shape[0]) for a in tgt_ids_per) + 1)
    B = len(tgt_ids_per)
    target_in = torch.full((B, real_max), cfg_pad_id, dtype=torch.long)
    target_out = torch.full((B, real_max), cfg_pad_id, dtype=torch.long)
    target_mask = torch.zeros(B, real_max, dtype=torch.float)
    for i, a in enumerate(tgt_ids_per):
        L = min(int(a.shape[0]), real_max - 1)  # leave room for the trailing EOS
        target_in[i, 0] = cfg_bos_id
        target_in[i, 1:1 + L] = a[:L]
        target_out[i, :L] = a[:L]
        target_out[i, L] = cfg_eos_id
        target_mask[i, :L + 1] = 1.0
    return input_ids, target_in, target_out, target_mask


def cosine_lr(step: int, total: int, peak_lr: float, warmup: int = 200) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def quick_eval(
    model: PrismLangV2,
    examples: list[dict],
    device: torch.device,
    n: int = 100,
    max_new_tokens: int = 256,
) -> tuple[float, float]:
    """Generate answers for `n` examples; return (mean_loss_proxy, accuracy)."""
    model.eval()
    sample = examples[:n]
    tok = get_tokenizer("gpt2")
    inputs = [format_input(ex["question"]) for ex in sample]
    input_ids, _ = encode_batch(inputs, max_len=512)
    input_ids = input_ids.to(device)
    gen = model.generate(input_ids, max_new_tokens=max_new_tokens)
    correct = 0
    for i, ex in enumerate(sample):
        text = tok.decode(gen[i].tolist(), skip_special_tokens=True)
        if is_correct(text, ex["answer"]):
            correct += 1
    return 0.0, correct / max(1, len(sample))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="gpt2",
                        choices=["gpt2", "gpt2-medium"],
                        help="HF GPT-2 variant to load as the AR backbone.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip loading pretrained weights (random init). "
                             "Use only for ablation — defeats the whole point.")
    parser.add_argument("--n-thought-tokens", type=int, default=16)
    parser.add_argument("--n-thought-steps", type=int, default=6)
    parser.add_argument("--jepa-aux-weight", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Lower than from-scratch training because "
                             "we're fine-tuning pretrained weights.")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-in-len", type=int, default=512)
    parser.add_argument("--max-out-len", type=int, default=256)
    parser.add_argument("--with-cot", action="store_true", default=True,
                        help="Train with full CoT in target (Stage 0). "
                             "Use --no-with-cot for answer-only supervision.")
    parser.add_argument("--no-with-cot", dest="with_cot", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    print(f"[lang_v2-train] backbone={args.backbone} "
          f"pretrained={not args.no_pretrained} "
          f"K={args.n_thought_tokens} N={args.n_thought_steps} "
          f"jepa_aux={args.jepa_aux_weight} with_cot={args.with_cot}")

    print("[lang_v2-train] loading GSM8K…")
    train = load_gsm8k("train")
    test = load_gsm8k("test")
    print(f"[lang_v2-train] train={len(train)} test={len(test)}")

    model = PrismLangV2(
        backbone_name=args.backbone,
        n_thought_tokens=args.n_thought_tokens,
        n_thought_steps=args.n_thought_steps,
        jepa_aux_weight=args.jepa_aux_weight,
        load_pretrained=not args.no_pretrained,
    ).to(device)
    print(f"[lang_v2-train] total params: {model.num_params():,}")
    print(f"[lang_v2-train] middle params: {model.num_middle_params():,}  "
          f"(only piece trained from scratch)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    print(f"[lang_v2-train] writing to {out_dir}")

    rng = np.random.default_rng(args.seed)
    loss_window: deque[float] = deque(maxlen=100)

    for step in range(args.steps):
        idx = rng.integers(0, len(train), size=args.batch_size)
        batch = [train[int(i)] for i in idx]
        input_ids, target_in, target_out, target_mask = encode_examples(
            batch,
            cfg_pad_id=model.pad_token_id,
            cfg_bos_id=model.bos_token_id,
            cfg_eos_id=model.eos_token_id,
            max_in_len=args.max_in_len,
            max_out_len=args.max_out_len,
            with_cot=args.with_cot,
        )
        input_ids = input_ids.to(device)
        target_in = target_in.to(device)
        target_out = target_out.to(device)
        target_mask = target_mask.to(device)

        model.train()
        lr = cosine_lr(step, args.steps, args.lr, warmup=args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        out = model.loss(input_ids, target_in, target_out, target_mask=target_mask)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        model.middle.update_ema()

        loss_window.append(float(loss.item()))

        if step % 25 == 0:
            mean_loss = float(np.mean(loss_window)) if loss_window else float("nan")
            writer.add_scalar("loss/total", float(loss.item()), step)
            writer.add_scalar("loss/ce", float(out["ce_loss"].item()), step)
            writer.add_scalar("loss/aux", float(out["aux_loss"].item()), step)
            writer.add_scalar("lr", lr, step)
            print(f"[step {step:5d}/{args.steps}] loss={float(loss.item()):.4f} "
                  f"ce={float(out['ce_loss'].item()):.4f} "
                  f"aux={float(out['aux_loss'].item()):.4f} "
                  f"mean100={mean_loss:.4f} lr={lr:.2e}")

        if (step + 1) % args.eval_every_steps == 0 or step == args.steps - 1:
            _, acc = quick_eval(model, test, device, n=args.eval_n)
            writer.add_scalar("eval/acc", acc, step)
            print(f"  [eval @ step {step+1}] gsm8k_acc={acc*100:.1f}% "
                  f"(on {args.eval_n} test problems)")

        if (step + 1) % args.save_every_steps == 0 or step == args.steps - 1:
            ckpt_path = out_dir / f"model_step{step+1}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "backbone_name": args.backbone,
                    "n_thought_tokens": args.n_thought_tokens,
                    "n_thought_steps": args.n_thought_steps,
                    "jepa_aux_weight": args.jepa_aux_weight,
                },
                "step": step + 1,
                "args": vars(args),
            }, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

    # Final eval on more examples for a stable number.
    print("\n[lang_v2-train] final eval on 500 test problems…")
    _, final_acc = quick_eval(model, test, device, n=min(500, len(test)))
    print(f"[done] final gsm8k_acc={final_acc*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
