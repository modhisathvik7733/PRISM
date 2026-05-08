"""Train PrismLangModel on a bAbI task (or all 20 jointly).

Usage:
    # Phase 1 — single task
    python -m scripts.lang.train --task 1 --steps 8000 \
        --run-name lang_t1_v0 --device cuda

    # Phase 2 — all 20 tasks
    python -m scripts.lang.train --task all --steps 50000 \
        --run-name lang_all_v0 --device cuda

Mirrors the structure of scripts/train_jepa.py: thin loop, no fancy
schedulers beyond cosine LR + linear warmup, periodic eval, save every
N steps + at end. Designed to scale unchanged to bigger configs (just
pass `--preset medium` etc).
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
from prism.lang.tokenizer import encode_batch, get_tokenizer
from prism.utils.seed import set_global_seed
from scripts.lang.data_babi import (
    BABI_TASK_NAMES, format_input, format_target, load_babi,
)


def build_dataset(task_arg: str) -> tuple[list, list]:
    """Returns (train_examples, test_examples). task_arg = "1".."20" or "all"."""
    if task_arg == "all":
        train, test = [], []
        for tid in BABI_TASK_NAMES:
            train.extend(load_babi(tid, "train"))
            test.extend(load_babi(tid, "test"))
    else:
        tid = int(task_arg)
        train = load_babi(tid, "train")
        test = load_babi(tid, "test")
    return train, test


def encode_examples(examples: list[tuple[str, str, str]], cfg: LangConfig
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize (story, question, answer) triples into the four tensors
    the model needs: input_ids, target_in, target_out, target_mask."""
    tok = get_tokenizer(cfg.tokenizer_name)
    inputs = [format_input(s, q) for (s, q, _) in examples]
    targets = [format_target(a) for (_, _, a) in examples]
    input_ids, _ = encode_batch(inputs, max_len=cfg.max_seq_len)

    # Encode each target separately so we know its true length, then
    # build (B, T_max) tensors where T_max = max(answer_len) + 1 for BOS.
    ans_ids = [tok(t, return_tensors="pt").input_ids[0] for t in targets]
    max_t = max(int(a.shape[0]) for a in ans_ids) + 1
    B = len(ans_ids)
    target_in = torch.full((B, max_t), cfg.pad_token_id, dtype=torch.long)
    target_out = torch.full((B, max_t), cfg.pad_token_id, dtype=torch.long)
    target_mask = torch.zeros(B, max_t, dtype=torch.float)
    for i, a in enumerate(ans_ids):
        L = int(a.shape[0])
        target_in[i, 0] = cfg.bos_token_id
        target_in[i, 1:1 + L] = a
        target_out[i, :L] = a
        target_out[i, L] = cfg.eos_token_id
        target_mask[i, :L + 1] = 1.0
    return input_ids, target_in, target_out, target_mask


def cosine_lr(step: int, total: int, peak_lr: float, warmup: int = 200) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def quick_eval(
    model: PrismLangModel,
    examples: list[tuple[str, str, str]],
    cfg: LangConfig,
    device: torch.device,
    n: int = 200,
) -> tuple[float, float]:
    """Greedy-decode `n` examples; return (loss, accuracy)."""
    model.eval()
    sample = examples[:n]
    input_ids, target_in, target_out, target_mask = encode_examples(sample, cfg)
    input_ids = input_ids.to(device)
    target_in = target_in.to(device)
    target_out = target_out.to(device)
    target_mask = target_mask.to(device)
    out = model.loss(input_ids, target_in, target_out, target_mask=target_mask)
    loss_val = float(out["ce_loss"].item())

    gen_ids = model.generate(input_ids, max_new_tokens=8,
                             bos_id=cfg.bos_token_id,
                             eos_id=cfg.eos_token_id)
    tok = get_tokenizer(cfg.tokenizer_name)
    correct = 0
    for i, (_, _, gold_answer) in enumerate(sample):
        gen_text = tok.decode(gen_ids[i].tolist(), skip_special_tokens=True).strip()
        # bAbI answers are single words (or comma-separated for tasks 8/19);
        # we count an exact (case-insensitive) match on the first whitespace
        # token of the generation.
        first_tok = gen_text.split()[0] if gen_text.split() else ""
        if first_tok.lower() == gold_answer.lower():
            correct += 1
    return loss_val, correct / max(1, len(sample))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="1",
                        help='bAbI task id 1..20 or "all"')
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--preset", default="small",
                        choices=["tiny", "small", "medium", "gpt2-compat"])
    parser.add_argument("--jepa-aux-weight", type=float, default=0.1,
                        help="0 disables the JEPA-style predictive aux loss "
                             "(then we're pure Coconut-style end-to-end).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--save-every-steps", type=int, default=2000)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    cfg = LangConfig().scale_to(args.preset)
    cfg.jepa_aux_weight = args.jepa_aux_weight
    print(f"[lang-train] preset={args.preset} d={cfg.d_model} "
          f"enc={cfg.n_enc_layers} mid_steps={cfg.n_thought_steps} "
          f"dec={cfg.n_dec_layers} jepa_aux={cfg.jepa_aux_weight}")

    print(f"[lang-train] loading bAbI task={args.task}…")
    train, test = build_dataset(args.task)
    print(f"[lang-train] train={len(train)} test={len(test)}")

    model = PrismLangModel(cfg).to(device)
    print(f"[lang-train] model params: {model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    print(f"[lang-train] writing to {out_dir}")

    rng = np.random.default_rng(args.seed)
    loss_window: deque[float] = deque(maxlen=200)

    for step in range(args.steps):
        idx = rng.integers(0, len(train), size=args.batch_size)
        batch = [train[int(i)] for i in idx]
        input_ids, target_in, target_out, target_mask = encode_examples(batch, cfg)
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

        if step % 50 == 0:
            mean_loss = float(np.mean(loss_window)) if loss_window else float("nan")
            writer.add_scalar("loss/total", float(loss.item()), step)
            writer.add_scalar("loss/ce", float(out["ce_loss"].item()), step)
            writer.add_scalar("loss/aux", float(out["aux_loss"].item()), step)
            writer.add_scalar("lr", lr, step)
            print(f"[step {step:5d}/{args.steps}] loss={float(loss.item()):.4f} "
                  f"ce={float(out['ce_loss'].item()):.4f} "
                  f"aux={float(out['aux_loss'].item()):.4f} "
                  f"mean200={mean_loss:.4f} lr={lr:.2e}")

        if (step + 1) % args.eval_every_steps == 0 or step == args.steps - 1:
            eval_loss, acc = quick_eval(model, test, cfg, device)
            writer.add_scalar("eval/loss", eval_loss, step)
            writer.add_scalar("eval/acc", acc, step)
            print(f"  [eval @ step {step+1}] test_loss={eval_loss:.4f} "
                  f"test_acc={acc*100:.1f}%")

        if (step + 1) % args.save_every_steps == 0 or step == args.steps - 1:
            ckpt_path = out_dir / f"model_step{step+1}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
                "step": step + 1,
                "args": vars(args),
            }, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

    final_path = out_dir / "model_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": cfg,
        "step": args.steps,
        "args": vars(args),
    }, final_path)
    eval_loss, acc = quick_eval(model, test, cfg, device, n=min(1000, len(test)))
    print(f"[done] saved {final_path}")
    print(f"[done] final test_acc={acc*100:.2f}% on {min(1000, len(test))} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
