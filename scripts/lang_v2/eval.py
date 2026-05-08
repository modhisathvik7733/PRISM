"""Evaluate a PrismLangV2 checkpoint on GSM8K.

Reports exact-match accuracy on the test split. Optionally prints
sample mistakes (`--show-mistakes`) so you can see where the model
fails.

Usage:
    python -m scripts.lang_v2.eval \
        --checkpoint runs/lang_v2_gsm8k_v0/model_step5000.pt \
        --episodes 500 --device cuda
"""

from __future__ import annotations

import argparse

import torch

from prism.lang.tokenizer import encode_batch, get_tokenizer
from prism.lang_v2.model import PrismLangV2
from prism.utils.seed import set_global_seed
from scripts.lang_v2.data_gsm8k import (
    format_input, is_correct, load_gsm8k, normalize_pred,
)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--show-mistakes", action="store_true")
    parser.add_argument("--show-correct", action="store_true",
                        help="also show 3 correct examples for context")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    print(f"[eval] loading {args.checkpoint}…")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    # Re-build the model. We DON'T re-download pretrained weights because
    # the ckpt has them; pass load_pretrained=False to skip the HF call.
    model = PrismLangV2(
        backbone_name=config["backbone_name"],
        n_thought_tokens=config["n_thought_tokens"],
        n_thought_steps=config["n_thought_steps"],
        jepa_aux_weight=config["jepa_aux_weight"],
        load_pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[eval] params: {model.num_params():,}")

    print(f"[eval] loading GSM8K test split (using first {args.episodes})…")
    test = load_gsm8k("test")[:args.episodes]
    tok = get_tokenizer("gpt2")

    correct = 0
    mistakes: list[dict] = []
    correct_samples: list[dict] = []

    for i in range(0, len(test), args.batch_size):
        batch = test[i:i + args.batch_size]
        inputs = [format_input(ex["question"]) for ex in batch]
        input_ids, _ = encode_batch(inputs, max_len=512)
        input_ids = input_ids.to(device)
        gen = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        for j, ex in enumerate(batch):
            text = tok.decode(gen[j].tolist(), skip_special_tokens=True)
            pred_num = normalize_pred(text)
            ok = is_correct(text, ex["answer"])
            if ok:
                correct += 1
                if len(correct_samples) < 3:
                    correct_samples.append({
                        "q": ex["question"], "gold": ex["answer"],
                        "pred": pred_num, "full": text,
                    })
            elif len(mistakes) < 5:
                mistakes.append({
                    "q": ex["question"], "gold": ex["answer"],
                    "pred": pred_num, "full": text,
                })
        done = min(i + args.batch_size, len(test))
        if done % (args.batch_size * 10) == 0 or done == len(test):
            print(f"  [{done:4d}/{len(test)}] running_acc={correct/done*100:.1f}%")

    acc = correct / max(1, len(test))
    print()
    print(f"=== GSM8K eval — {len(test)} test problems ===")
    print(f"  exact-match accuracy: {acc*100:.2f}%")
    print(f"  references: GPT-2 small ~5%, GPT-3 175B ~15%, Coconut paper ~30%")

    if args.show_correct and correct_samples:
        print("\n=== sample CORRECT predictions ===")
        for s in correct_samples:
            print(f"  Q: {s['q'][:120]}{'…' if len(s['q']) > 120 else ''}")
            print(f"  gold: {s['gold']}   pred: {s['pred']}")
            print(f"  full: {s['full'][:200]}")
            print()

    if args.show_mistakes and mistakes:
        print("\n=== sample MISTAKES ===")
        for s in mistakes:
            print(f"  Q: {s['q'][:120]}{'…' if len(s['q']) > 120 else ''}")
            print(f"  gold: {s['gold']}   pred: {s['pred']}")
            print(f"  full: {s['full'][:200]}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
