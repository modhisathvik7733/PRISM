"""Interactive prompt tester for a trained PrismLangModel.

Loads a checkpoint once, then loops on stdin: type a story+question in
the bAbI format, get the model's answer. Useful for poking at where the
model succeeds and fails after training.

Usage:
    # interactive REPL
    python -m scripts.lang.ask --checkpoint runs/lang_all_v0/model_final.pt

    # pipe a single prompt
    echo "Story: Mary went to the kitchen. Question: Where is Mary? Answer:" \
        | python -m scripts.lang.ask --checkpoint runs/lang_all_v0/model_final.pt

    # batch from a file (one prompt per line)
    python -m scripts.lang.ask --checkpoint runs/lang_all_v0/model_final.pt \
        --file my_prompts.txt

    # quick smoke deck
    python -m scripts.lang.ask --checkpoint runs/lang_all_v0/model_final.pt \
        --demo
"""

from __future__ import annotations

import argparse
import sys

import torch

from prism.lang.config import LangConfig  # noqa: F401  (needed for torch.load)
from prism.lang.model import PrismLangModel
from prism.lang.tokenizer import encode_batch, get_tokenizer


# Curated bAbI-style demo prompts covering several reasoning patterns.
DEMO_PROMPTS: list[str] = [
    # Task 1 — single fact
    "Story: Mary went to the kitchen. John moved to the bedroom. Question: Where is Mary? Answer:",
    # Task 2 — two facts (object follows actor)
    "Story: Mary picked up the apple. Mary went to the garden. Question: Where is the apple? Answer:",
    # Task 4 — two-arg relations (directions)
    "Story: The bedroom is north of the kitchen. The garden is south of the kitchen. Question: What is north of the kitchen? Answer:",
    # Task 6 — yes/no
    "Story: John moved to the office. Question: Is John in the office? Answer:",
    # Task 11 — coreference
    "Story: Daniel went to the bathroom. After that he went to the office. Question: Where is Daniel? Answer:",
    # Task 13 — compound coreference
    "Story: Mary and Sandra went to the park. Then they went to the school. Question: Where is Mary? Answer:",
    # Task 14 — time reasoning (canonically hard)
    "Story: This morning Mary went to the school. Yesterday Mary went to the park. Question: Where was Mary before the school? Answer:",
    # Task 18 — size reasoning
    "Story: The football is bigger than the apple. The apple is bigger than the marble. Question: Is the football bigger than the marble? Answer:",
    # Task 19 — path-finding (canonically hardest)
    "Story: The kitchen is north of the bedroom. The garden is east of the kitchen. Question: How do you go from the bedroom to the garden? Answer:",
    # Distribution shift — same form, different name
    "Story: Sandra moved to the kitchen. Sandra went to the bedroom. Question: Where is Sandra? Answer:",
    # Out-of-distribution — out-of-vocab person
    "Story: Alice walked to the office. Question: Where is Alice? Answer:",
]


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: LangConfig = ckpt["cfg"]
    model = PrismLangModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def ask_batch(model: PrismLangModel, cfg: LangConfig, device: torch.device,
              prompts: list[str], max_new_tokens: int = 8) -> list[str]:
    if not prompts:
        return []
    input_ids, _ = encode_batch(prompts, max_len=cfg.max_seq_len)
    input_ids = input_ids.to(device)
    gen = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        bos_id=cfg.bos_token_id,
        eos_id=cfg.eos_token_id,
    )
    tok = get_tokenizer(cfg.tokenizer_name)
    return [tok.decode(g.tolist(), skip_special_tokens=True).strip() for g in gen]


def print_answer(prompt: str, answer: str) -> None:
    # Strip "Story: " prefix and "Answer:" suffix for cleaner display.
    short = prompt
    if short.startswith("Story: "):
        short = short[len("Story: "):]
    if short.endswith(" Answer:"):
        short = short[:-len(" Answer:")]
    print(f"  Q: {short}")
    print(f"  A: {answer!r}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--demo", action="store_true",
                        help="run the curated DEMO_PROMPTS deck")
    parser.add_argument("--file", default=None,
                        help="read prompts from a file (one per line)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[ask] loading {args.checkpoint}…")
    model, cfg = load_model(args.checkpoint, device)
    print(f"[ask] ready ({sum(p.numel() for p in model.parameters()):,} params)\n")

    if args.demo:
        print("=== curated demo prompts ===\n")
        answers = ask_batch(model, cfg, device, DEMO_PROMPTS, args.max_new_tokens)
        for p, a in zip(DEMO_PROMPTS, answers):
            print_answer(p, a)
        return 0

    if args.file:
        with open(args.file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        answers = ask_batch(model, cfg, device, prompts, args.max_new_tokens)
        for p, a in zip(prompts, answers):
            print_answer(p, a)
        return 0

    # If stdin is being piped, read all prompts at once.
    if not sys.stdin.isatty():
        prompts = [line.strip() for line in sys.stdin if line.strip()]
        answers = ask_batch(model, cfg, device, prompts, args.max_new_tokens)
        for p, a in zip(prompts, answers):
            print_answer(p, a)
        return 0

    # Interactive REPL.
    print("=== interactive mode ===")
    print('type a prompt in bAbI format ("Story: ... Question: ...? Answer:")')
    print("blank line to exit, 'demo' to run the curated deck\n")
    try:
        while True:
            line = input("> ").strip()
            if not line:
                break
            if line == "demo":
                answers = ask_batch(model, cfg, device, DEMO_PROMPTS, args.max_new_tokens)
                for p, a in zip(DEMO_PROMPTS, answers):
                    print_answer(p, a)
                continue
            # Auto-add "Answer:" if user forgot it.
            if not line.endswith("Answer:"):
                if not line.endswith(("?", ".")):
                    line = line + "?"
                line = line + " Answer:"
            ans = ask_batch(model, cfg, device, [line], args.max_new_tokens)[0]
            print(f"  → {ans!r}\n")
    except (EOFError, KeyboardInterrupt):
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
