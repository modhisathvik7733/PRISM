"""Interactive prompt tester for a trained PrismLangV2 checkpoint.

Like scripts/lang/ask.py but for v3.1 — supports free-form English
input (the encoder/decoder come from pretrained GPT-2 weights, so the
output vocab isn't locked to a tiny answer set).

Usage:
    python -m scripts.lang_v2.ask \
        --checkpoint runs/lang_v2_gsm8k_v0/model_step5000.pt \
        --device cuda
    > Question: If I have 5 apples and eat 2, how many are left?

    # batch from a file:
    python -m scripts.lang_v2.ask \
        --checkpoint runs/lang_v2_gsm8k_v0/model_step5000.pt \
        --file my_prompts.txt
"""

from __future__ import annotations

import argparse
import sys

import torch

from prism.lang.tokenizer import encode_batch, get_tokenizer
from prism.lang_v2.model import PrismLangV2


DEMO_PROMPTS: list[str] = [
    "Question: If I have 5 apples and eat 2, how many are left? Answer:",
    "Question: A train travels 60 miles per hour for 3 hours. How far does it go? Answer:",
    "Question: Sarah has $20. She buys a book for $7 and a pen for $3. How much does she have left? Answer:",
    "Question: There are 12 students in a class. Half are girls. How many boys are there? Answer:",
    "Question: A rectangle has length 8 and width 5. What is its area? Answer:",
    # Free-form, non-math
    "Question: What is the capital of France? Answer:",
    "Question: Write a one-sentence story about a dog. Answer:",
    # Out-of-distribution / nonsense
    "hi",
    "Once upon a time",
]


def load_model(checkpoint_path: str, device: torch.device) -> PrismLangV2:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = PrismLangV2(
        backbone_name=config["backbone_name"],
        n_thought_tokens=config["n_thought_tokens"],
        n_thought_steps=config["n_thought_steps"],
        jepa_aux_weight=config["jepa_aux_weight"],
        load_pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def ask_batch(model: PrismLangV2, device: torch.device, prompts: list[str],
              max_new_tokens: int = 128, temperature: float = 0.0,
              top_k: int | None = None) -> list[str]:
    if not prompts:
        return []
    input_ids, _ = encode_batch(prompts, max_len=512)
    input_ids = input_ids.to(device)
    gen = model.generate(input_ids, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    tok = get_tokenizer("gpt2")
    return [tok.decode(g.tolist(), skip_special_tokens=True) for g in gen]


def print_qa(prompt: str, answer: str) -> None:
    short_p = prompt if len(prompt) < 200 else prompt[:200] + "…"
    short_a = answer.strip() if len(answer) < 400 else answer.strip()[:400] + "…"
    print(f"  Q: {short_p}")
    print(f"  A: {short_a!r}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy. Try 0.7 for varied outputs.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--file", default=None)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[ask] loading {args.checkpoint}…")
    model = load_model(args.checkpoint, device)
    print(f"[ask] ready ({model.num_params():,} params)\n")

    if args.demo:
        print("=== curated demo ===\n")
        answers = ask_batch(model, device, DEMO_PROMPTS,
                            args.max_new_tokens, args.temperature, args.top_k)
        for p, a in zip(DEMO_PROMPTS, answers):
            print_qa(p, a)
        return 0

    if args.file:
        with open(args.file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        answers = ask_batch(model, device, prompts,
                            args.max_new_tokens, args.temperature, args.top_k)
        for p, a in zip(prompts, answers):
            print_qa(p, a)
        return 0

    if not sys.stdin.isatty():
        prompts = [line.strip() for line in sys.stdin if line.strip()]
        answers = ask_batch(model, device, prompts,
                            args.max_new_tokens, args.temperature, args.top_k)
        for p, a in zip(prompts, answers):
            print_qa(p, a)
        return 0

    # Interactive REPL
    print("=== interactive mode ===")
    print('type a prompt — model speaks GPT-2 English with JEPA-middle reasoning')
    print("blank line to exit, 'demo' to run the curated deck\n")
    try:
        while True:
            line = input("> ").strip()
            if not line:
                break
            if line == "demo":
                answers = ask_batch(model, device, DEMO_PROMPTS,
                                    args.max_new_tokens, args.temperature, args.top_k)
                for p, a in zip(DEMO_PROMPTS, answers):
                    print_qa(p, a)
                continue
            ans = ask_batch(model, device, [line], args.max_new_tokens,
                            args.temperature, args.top_k)[0]
            print(f"  → {ans.strip()!r}\n")
    except (EOFError, KeyboardInterrupt):
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
