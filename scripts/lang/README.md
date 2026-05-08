# PRISM-Lang — AR-edges + JEPA-style latent middle on bAbI

Tests whether PRISM's "structured latent middle, weights are not the
knowledge store" thesis ports from gridworld RL to language. AR
transformer encoder + decoder handle tokens at the language interface;
a recurrent latent middle does the actual reasoning over **K thought
tokens** for **N steps** — the Coconut paper's recipe with optional
JEPA-style EMA-target aux loss.

## Architecture

```
input_text → GPT-2 BPE tokenize
           → AR encoder (4 bidirectional transformer layers, d=256)
                  ↓ context (T, d)
           → LatentMiddle (8 thought tokens, 6 thinking steps,
                           shared-weight Block + JEPA aux loss)
                  ↓ thoughts (K, d)
           → AR decoder (4 causal layers, cross-attention to thoughts)
           → token-by-token AR generation → answer_text
```

All language understanding is forced through the K thought tokens —
the decoder never sees the encoder's raw output, only the middle's
"thought-out" condensation.

## Quickstart

```bash
# 0. install deps (one-time)
uv pip install transformers tensorboard

# 1. smoke test — verifies tokenizer, model, forward pass, scaling
python -m scripts.lang.smoke_test

# 2. Phase 1 — train on bAbI Task 1 (single supporting fact). ~5 min on a
#    single GPU. Expect test_acc > 90% by end.
python -m scripts.lang.train --task 1 --steps 8000 \
    --run-name lang_t1_v0 --device cuda

# 3. Phase 1 eval
python -m scripts.lang.eval \
    --checkpoint runs/lang_t1_v0/model_final.pt \
    --task 1 --episodes 1000 --device cuda

# 4. Phase 2 — all 20 tasks jointly. ~1 hr. Target mean acc > 80%.
python -m scripts.lang.train --task all --steps 50000 \
    --run-name lang_all_v0 --device cuda

python -m scripts.lang.eval \
    --checkpoint runs/lang_all_v0/model_final.pt \
    --task all --episodes 1000 --device cuda --show-mistakes
```

## Reference numbers

| System | bAbI Task 1 | bAbI all 20 (mean) | Notes |
|---|---:|---:|---|
| Random | ~5% | ~5% | 1/N answers |
| Vanilla LSTM (small) | ~50% | ~30% | Sukhbaatar 2015 |
| Memory Networks | ~99% | ~93% | structured external memory |
| **PRISM-Lang Phase 1 target** | **>90%** | — | proves pipeline |
| **PRISM-Lang Phase 2 target** | — | **>80%** | proves middle generalizes |

## Scaling without code changes

Every dimension is a `LangConfig` field. To bump from 22M (default) to
50M params for the Phase 2 multi-task run:

```bash
python -m scripts.lang.train --task all --preset medium \
    --steps 50000 --run-name lang_all_medium --device cuda
```

`--preset gpt2-compat` (768d, 12 layers — matches GPT-2 small) is the
configuration where you could load pretrained GPT-2 encoder/decoder
weights into our architecture. Out of scope for this iteration, but
the option is there.

## What we're NOT testing yet

- Pretrained encoder/decoder weights (clean experiment first)
- Multi-GPU / DDP (single GPU is plenty at 22M params)
- External / persistent memory beyond the K thought tokens
- Symbolic operators (the Coconut-style middle is purely continuous;
  adding operators is the next research step if Phase 2 succeeds)

## File map

| File | Purpose |
|---|---|
| `prism/lang/config.py` | LangConfig — every dim/depth/vocab as a hyperparam, plus `scale_to(preset)` helper |
| `prism/lang/transformer.py` | Reusable Block — pre-norm, MHA, GELU MLP, optional cross-attention |
| `prism/lang/tokenizer.py` | HF GPT-2 BPE wrapper (cached) |
| `prism/lang/encoder.py` | Bidirectional AR encoder = embedding + N self-attn Blocks |
| `prism/lang/middle.py` | LatentMiddle = K thought tokens × N recurrent thinking steps + EMA-target JEPA aux |
| `prism/lang/decoder.py` | Causal AR decoder cross-attending to thoughts + tied LM head |
| `prism/lang/model.py` | PrismLangModel composing encoder/middle/decoder + `loss()` + `generate()` |
| `scripts/lang/data_babi.py` | Download + parse bAbI; offline sample for smoke test |
| `scripts/lang/smoke_test.py` | 4 checks: tokenizer / model construct / forward+loss / medium-preset scaling |
| `scripts/lang/train.py` | Training loop — AdamW, cosine LR, periodic eval, checkpoints |
| `scripts/lang/eval.py` | Per-task and overall exact-match accuracy, optional mistake samples |
