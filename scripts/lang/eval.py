"""Evaluate a PrismLangModel checkpoint on bAbI.

Reports per-task and overall exact-match accuracy. The checkpoint
already carries its `LangConfig` so you don't need to pass model dims.

Usage:
    python -m scripts.lang.eval \
        --checkpoint runs/lang_all_v0/model_final.pt \
        --task all --episodes 1000 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from prism.lang.config import LangConfig  # noqa: F401  (needed for torch.load)
from prism.lang.model import PrismLangModel
from prism.lang.tokenizer import encode_batch, get_tokenizer
from prism.utils.seed import set_global_seed
from scripts.lang.data_babi import (
    BABI_TASK_NAMES, format_input, format_target, load_babi,
)


@torch.no_grad()
def eval_one_task(
    model: PrismLangModel,
    cfg: LangConfig,
    device: torch.device,
    task_id: int,
    n_episodes: int,
    batch_size: int,
) -> tuple[float, list[tuple[str, str, str]]]:
    """Returns (accuracy, mistakes_list)."""
    examples = load_babi(task_id, "test")[:n_episodes]
    tok = get_tokenizer(cfg.tokenizer_name)

    inputs = [format_input(s, q) for (s, q, _) in examples]
    input_ids, _ = encode_batch(inputs, max_len=cfg.max_seq_len)

    correct = 0
    mistakes: list[tuple[str, str, str]] = []   # (input, gold, pred)
    for i in range(0, len(examples), batch_size):
        batch_ids = input_ids[i:i + batch_size].to(device)
        gen = model.generate(
            batch_ids, max_new_tokens=8,
            bos_id=cfg.bos_token_id, eos_id=cfg.eos_token_id,
        )
        for j, (_, _, gold) in enumerate(examples[i:i + batch_size]):
            text = tok.decode(gen[j].tolist(), skip_special_tokens=True).strip()
            first = text.split()[0] if text.split() else ""
            if first.lower() == gold.lower():
                correct += 1
            elif len(mistakes) < 5:
                inp_text = inputs[i + j]
                mistakes.append((inp_text, gold, first or "<empty>"))
    return correct / max(1, len(examples)), mistakes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="all",
                        help='task id 1..20 or "all"')
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--show-mistakes", action="store_true",
                        help="print up to 5 wrong predictions per task")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg: LangConfig = ckpt["cfg"]
    model = PrismLangModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[eval] loaded {args.checkpoint}")
    print(f"[eval] config preset-equiv: d={cfg.d_model} enc={cfg.n_enc_layers} "
          f"mid_steps={cfg.n_thought_steps} dec={cfg.n_dec_layers}")
    print(f"[eval] params: {model.num_params():,}")

    if args.task == "all":
        task_ids = list(BABI_TASK_NAMES.keys())
    else:
        task_ids = [int(args.task)]

    print()
    print(f"=== bAbI eval — {args.episodes} episodes per task ===")
    print(f"{'task':4s}  {'name':40s}  {'acc%':>6s}")
    accs: list[float] = []
    all_mistakes: dict[int, list] = {}
    for tid in task_ids:
        acc, mistakes = eval_one_task(
            model, cfg, device, tid, args.episodes, args.batch_size,
        )
        accs.append(acc)
        all_mistakes[tid] = mistakes
        print(f"{tid:>4d}  {BABI_TASK_NAMES[tid]:40s}  {acc*100:>5.1f}%")
    print(f"{'mean':>4s}  {'':40s}  {float(np.mean(accs))*100:>5.1f}%")

    if args.show_mistakes:
        print()
        print("=== sample mistakes (up to 5 per task) ===")
        for tid, mistakes in all_mistakes.items():
            if not mistakes:
                continue
            print(f"\n[task {tid} — {BABI_TASK_NAMES[tid]}]")
            for inp, gold, pred in mistakes:
                print(f"  input : {inp[:120]}{'…' if len(inp) > 120 else ''}")
                print(f"  gold  : {gold!r}")
                print(f"  pred  : {pred!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
