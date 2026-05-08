"""GSM8K data loader.

GSM8K (Cobbe et al., 2021) is the canonical small math-word-problem
benchmark: 8.5k grade-school arithmetic problems with chain-of-thought
solutions. Hosted on HuggingFace as `gsm8k`.

We split each example into:
  - input_text  : "Question: {q} Answer:"
  - cot_text    : the full chain-of-thought reasoning (used for the
                   Coconut-style curriculum — replaced one sentence at
                   a time with continuous latent thoughts as training
                   progresses)
  - answer_text : the final numeric answer (after the "#### " marker
                   that GSM8K uses)

For Phase A (warm-start verification) we just train the model to
predict `answer_text` directly given `input_text`. For Phase B
(Coconut curriculum) we use the `cot_text` field to interpolate
discrete CoT tokens with continuous middle steps.
"""

from __future__ import annotations

import re
from typing import Iterator


def _split_gsm8k(answer_field: str) -> tuple[str, str]:
    """GSM8K stores reasoning + answer as one string with the literal
    "#### <number>" marker at the end. Split into (cot, answer)."""
    m = re.search(r"####\s*([+-]?\d[\d,\.]*)\s*$", answer_field)
    if m is None:
        # Fallback — strip trailing numbers from the answer text.
        return answer_field.strip(), answer_field.strip().split()[-1]
    answer = m.group(1).replace(",", "")
    cot = answer_field[:m.start()].strip()
    return cot, answer


def load_gsm8k(split: str = "train") -> list[dict]:
    """Returns list of {question, cot, answer} dicts.

    `split` is "train" or "test". Uses HF `datasets`."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be train or test, got {split}")
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    out: list[dict] = []
    for row in ds:
        cot, answer = _split_gsm8k(row["answer"])
        out.append({
            "question": row["question"].strip(),
            "cot": cot,
            "answer": answer,
        })
    return out


def format_input(question: str) -> str:
    """Canonical model input. Keep simple — we want the model to
    learn the format from data, not from prompt engineering."""
    return f"Question: {question} Answer:"


def format_target_answer_only(answer: str) -> str:
    """Phase A: target is the bare numeric answer. Leading space matches
    GPT-2 BPE conventions."""
    return f" {answer}"


def format_target_with_cot(cot: str, answer: str) -> str:
    """Phase B: target includes the full CoT. The Coconut curriculum
    will gradually replace early CoT tokens with latent thoughts."""
    # GSM8K CoT often contains arithmetic like "<<5+3=8>>" — keep as-is.
    return f" {cot}\n#### {answer}"


def iter_examples(split: str = "train",
                  with_cot: bool = False) -> Iterator[tuple[str, str, str]]:
    """Yield (input_text, target_text, gold_answer) tuples."""
    for ex in load_gsm8k(split):
        inp = format_input(ex["question"])
        gold = ex["answer"]
        if with_cot:
            tgt = format_target_with_cot(ex["cot"], ex["answer"])
        else:
            tgt = format_target_answer_only(ex["answer"])
        yield inp, tgt, gold


def normalize_pred(text: str) -> str:
    """Extract a numeric prediction from the model's free-form output.

    Looks for the "#### <num>" marker first (CoT-style); falls back to
    the LAST integer-or-decimal in the text. This matches GSM8K's
    standard eval convention."""
    m = re.search(r"####\s*([+-]?\d[\d,\.]*)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"[+-]?\d[\d,\.]*", text)
    if nums:
        return nums[-1].replace(",", "")
    return text.strip()


def is_correct(pred_text: str, gold_answer: str) -> bool:
    """Compare normalized prediction to gold. Both are stripped of
    commas; we accept either int-equivalent or float-equivalent
    matches (so "12" matches "12.0")."""
    p = normalize_pred(pred_text)
    g = gold_answer.replace(",", "")
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False
