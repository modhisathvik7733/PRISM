"""Smoke test for the PRISM-Lang stack.

Verifies (in order):
  1. transformers library is importable + GPT-2 tokenizer loads
  2. PrismLangModel constructs at the default config and prints param count
  3. A single forward pass on the offline bAbI sample produces logits of
     the expected shape and a finite loss
  4. The same model code constructs at the `medium` preset (~50M params)
     without architectural breakage — proves the design scales

Run BEFORE attempting any training:
    python -m scripts.lang.smoke_test
"""

from __future__ import annotations

import sys
import traceback

import torch
import torch.nn.functional as F


def check_tokenizer() -> bool:
    print("[1/4] loading GPT-2 tokenizer…", end=" ", flush=True)
    try:
        from prism.lang.tokenizer import encode_batch, get_tokenizer
        tok = get_tokenizer("gpt2")
        ids, mask = encode_batch(["hello world", "Mary went home"], max_len=16)
        assert ids.shape == (2, 16) and mask.shape == (2, 16)
        print(f"OK (vocab={tok.vocab_size}, sample shape {tuple(ids.shape)})")
        return True
    except Exception:
        print("FAILED")
        traceback.print_exc()
        print("\n[install] uv pip install transformers")
        return False


def check_model_construct() -> bool:
    print("[2/4] constructing PrismLangModel (small)…", end=" ", flush=True)
    try:
        from prism.lang.config import LangConfig
        from prism.lang.model import PrismLangModel
        cfg = LangConfig()
        model = PrismLangModel(cfg)
        n = model.num_params()
        print(f"OK ({n:,} params, d={cfg.d_model} enc={cfg.n_enc_layers} "
              f"mid_steps={cfg.n_thought_steps} dec={cfg.n_dec_layers})")
        return True
    except Exception:
        print("FAILED")
        traceback.print_exc()
        return False


def check_forward_loss() -> bool:
    print("[3/4] forward pass + loss on offline bAbI sample…", end=" ", flush=True)
    try:
        from prism.lang.config import LangConfig
        from prism.lang.model import PrismLangModel
        from prism.lang.tokenizer import encode_batch, get_tokenizer
        from scripts.lang.data_babi import (
            OFFLINE_SAMPLE, format_input, format_target,
        )
        cfg = LangConfig()
        model = PrismLangModel(cfg).eval()  # eval to skip dropout / aux
        tok = get_tokenizer(cfg.tokenizer_name)
        BOS = cfg.bos_token_id

        inputs = [format_input(s, q) for (s, q, _) in OFFLINE_SAMPLE]
        targets = [format_target(a) for (_, _, a) in OFFLINE_SAMPLE]

        input_ids, _ = encode_batch(inputs, max_len=64)
        # Build teacher-forced (in, out) for the answer part.
        # in = [BOS] + answer_tokens
        # out = answer_tokens + [EOS]
        ans_ids = [tok(t, return_tensors="pt").input_ids[0] for t in targets]
        max_t = max(int(a.shape[0]) for a in ans_ids) + 1  # +1 for BOS/EOS
        B = len(ans_ids)
        target_in = torch.full((B, max_t), cfg.pad_token_id, dtype=torch.long)
        target_out = torch.full((B, max_t), cfg.pad_token_id, dtype=torch.long)
        target_mask = torch.zeros(B, max_t, dtype=torch.float)
        for i, a in enumerate(ans_ids):
            L = int(a.shape[0])
            target_in[i, 0] = BOS
            target_in[i, 1:1 + L] = a
            target_out[i, :L] = a
            target_out[i, L] = cfg.eos_token_id
            target_mask[i, :L + 1] = 1.0

        out = model.loss(input_ids, target_in, target_out, target_mask=target_mask)
        loss_val = float(out["loss"].item())
        ce_val = float(out["ce_loss"].item())
        aux_val = float(out["aux_loss"].item())
        assert torch.isfinite(out["loss"]), "loss is non-finite"
        assert out["logits"].shape == (B, max_t, cfg.vocab_size)
        print(f"OK (loss={loss_val:.3f} ce={ce_val:.3f} aux={aux_val:.3f}, "
              f"logits shape {tuple(out['logits'].shape)})")
        return True
    except Exception:
        print("FAILED")
        traceback.print_exc()
        return False


def check_scaling() -> bool:
    print("[4/4] medium preset constructs + 1-step forward…", end=" ", flush=True)
    try:
        from prism.lang.config import LangConfig
        from prism.lang.model import PrismLangModel
        cfg = LangConfig().scale_to("medium")
        model = PrismLangModel(cfg).eval()
        n = model.num_params()
        # one fake forward to confirm shape compatibility
        B, T = 2, 32
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        target = torch.randint(0, cfg.vocab_size, (B, 4))
        out = model(ids, target)
        assert out["logits"].shape == (B, 4, cfg.vocab_size)
        print(f"OK ({n:,} params at medium preset; forward returns "
              f"{tuple(out['logits'].shape)})")
        return True
    except Exception:
        print("FAILED")
        traceback.print_exc()
        return False


def main() -> int:
    checks = [check_tokenizer, check_model_construct,
              check_forward_loss, check_scaling]
    for fn in checks:
        if not fn():
            print(f"\n[smoke] aborting after first failure ({fn.__name__})")
            return 1
    print("\n[smoke] all checks passed — ready to download bAbI and train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
