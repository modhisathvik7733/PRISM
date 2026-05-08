# PRISM-Lang v3.1 — pretrained AR backbone + JEPA-style latent middle

The v3.0 from-scratch experiment (`prism/lang/`) hit 57.3% on bAbI
1k variant but its decoder collapsed to bAbI's closed answer vocab —
free-form prompts ("hi", "where is the library") got mapped back to
~10 bAbI words. v3.1 fixes this by **swapping the from-scratch AR
edges for pretrained GPT-2 weights** while keeping the JEPA-style
latent middle as the only from-scratch component.

## Architecture

```
INPUT: any English text
   │
   ▼
[GPT-2 encoder]    pretrained, 12 layers, d=768
       (bidirectional — causal mask DROPPED)
   │
   ▼  context (T, 768)
[LatentMiddleV2]   from-scratch, K=16 thought tokens, N=6 thinking steps
       cross-attention to context, optional EMA-target JEPA aux loss
   │
   ▼  thoughts (16, 768)
[GPT-2 decoder]    pretrained self-attn weights + zero-init cross-attn modules
       causal self-attn + cross-attn to thoughts (NEVER to raw context)
   │
   ▼  logits over GPT-2 vocab
OUTPUT: free-form English (now actually English, not bAbI)
```

The decoder's cross-attention modules are added on top of pretrained
GPT-2 (which is decoder-only with no cross-attn). They're **zero-
initialized** so the decoder behaves exactly like vanilla GPT-2 at
step 0 — the cross-attn weights only learn to use the thoughts as
training proceeds. This means we never break GPT-2's English fluency
to learn reasoning.

## Param count

Backbone: gpt2 (124M params encoder + 124M params decoder, NOT shared)
+ LatentMiddleV2 (~10M, depends on K, N, share_weights)
**≈ 260M total** (roughly 2× a single GPT-2 small).

## Quickstart

```bash
# 0. one-time deps (already installed if you ran lang/v3.0)
uv pip install transformers datasets tensorboard

# 1. smoke test (~2 min — downloads gpt2 weights on first run)
python -m scripts.lang_v2.smoke_test

# 2. train on GSM8K (~24-48 hr on a single 16GB GPU)
python -m scripts.lang_v2.train \
    --backbone gpt2 \
    --steps 5000 --batch-size 8 \
    --jepa-aux-weight 0.1 \
    --run-name lang_v2_gsm8k_v0 --device cuda

# 3. eval (~10 min for 500 problems)
python -m scripts.lang_v2.eval \
    --checkpoint runs/lang_v2_gsm8k_v0/model_step5000.pt \
    --episodes 500 --device cuda --show-mistakes

# 4. interactive
python -m scripts.lang_v2.ask \
    --checkpoint runs/lang_v2_gsm8k_v0/model_step5000.pt
```

## Reference numbers on GSM8K (test set)

| System | Acc% | Notes |
|---|---:|---|
| Random | 0% | open-ended |
| GPT-2 small zero-shot | ~5% | the backbone we start from |
| Our PrismLangV2 untrained | ~3-7% | sanity baseline (after smoke test) |
| **PrismLangV2 fine-tuned (target)** | **15-30%** | what 24-48h of training should produce |
| GPT-3 175B zero-shot | ~15% | Cobbe et al., 2021 |
| Coconut (Llama-2 7B) | ~30% | the paper that motivated this design |
| GPT-4 zero-shot | ~92% | reference ceiling, very different scale |

We're targeting **the GPT-3 175B zero-shot band with a 260M model**.
That's only achievable IF the JEPA middle + Coconut-style training
buys real reasoning capability beyond fine-tuning alone.

## Why this is the right test of the bigger thesis

The user's broader hypothesis is **"weights are learning machinery,
not knowledge stores"**. v3.0 demonstrated the limit of pure
from-scratch: the decoder's LM head learned a closed answer vocab,
proving that small from-scratch models DO store knowledge in weights
when training data is narrow.

v3.1 separates the two roles cleanly:
- The **GPT-2 backbone** stores English knowledge (pretrained on web text)
- The **latent middle** does reasoning (trained from scratch on
  reasoning tasks)

If accuracy on GSM8K jumps significantly from "GPT-2 fine-tuned" to
"PrismLangV2 fine-tuned" at the same total training cost, the middle
is buying real inductive bias — supporting the thesis. If they're
the same, the middle is decorative.

## Files

| File | Purpose |
|---|---|
| `prism/lang_v2/gpt2_backbone.py` | GPT2Block / GPT2Stack + HF weight loader (handles Conv1D → Linear transposition) |
| `prism/lang_v2/middle.py` | LatentMiddleV2 — same recipe as v3.0 but uses GPT2Block at d=768 |
| `prism/lang_v2/model.py` | PrismLangV2 — encoder + middle + decoder + tied LM head + generate() |
| `scripts/lang_v2/data_gsm8k.py` | GSM8K loader; CoT splitting; numeric-answer extraction |
| `scripts/lang_v2/smoke_test.py` | 5 checks: deps / construct random / construct pretrained / forward+loss / generation |
| `scripts/lang_v2/train.py` | AdamW + cosine LR + periodic GSM8K eval |
| `scripts/lang_v2/eval.py` | Exact-match accuracy + sample correct/mistake printing |
| `scripts/lang_v2/ask.py` | Interactive REPL for free-form English prompts |

## Future work (out of scope this commit)

- **Coconut-style curriculum**: gradually replace CoT tokens with
  continuous middle steps. Right now Stage 0 trains with full CoT
  supervision; Stage 1 (the actual Coconut recipe) is a flag in
  `train.py` we haven't wired yet.
- **Bigger backbone**: `gpt2-medium` (355M), `gpt2-large` (774M),
  or TinyLlama-1.1B. The architecture supports these via
  `--backbone` if HF has the weights and they fit in memory.
- **Ablation against vanilla GPT-2 fine-tune**: clean comparison
  isolating the middle's contribution.
