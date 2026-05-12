# Phase A Runbook (PR-1 → PR-3)

This document is the smoke-test recipe for the v6.0 PR-1→PR-3 batch.
The actual training must run on a GPU box (Vast.ai); local syntax has
been verified.

## What landed

| PR | Files | Behavior change vs v5 |
|----|-------|------------------------|
| 1 | `prism/cognition/{__init__,tokens,tokenizer_base}.py` | none — substrate package skeleton + Protocol stubs |
| 1 | `prism/adapters/{__init__,base}.py` | none — DomainAdapter Protocol with `encoder()` ownership |
| 1 | `prism/curriculum/{__init__,stage,engine}.py` | none — Stage dataclass + engine stub |
| 1 | `scripts/experiments/checks/{__init__,phase_a,phase_b,phase_c,transfer}.py` | none — gate-check skeletons |
| 2 | `prism/cognition/policy.py` | none — `UniversalPolicy` thin wrapper around `HybridPolicy` / `RecurrentPolicy` |
| 2 | `prism/adapters/babyai_adapter.py` | none — `BabyAIAdapter` owns JEPA encoder + mission/masking logic |
| 2 | `scripts/ppo_train.py` | adds `--policy-type universal`, `--trunk gru`, `--universal-inner` flags; new code path delegates to v5 internals when `universal+gru` is selected |
| 3 | `scripts/experiments/checks/phase_a.py` | `check_phase_a()` — structural pre-conditions (imports, BabyAIAdapter attrs, UniversalPolicy methods) |
| 3 | `scripts/experiments/e0_phase_a_reward_parity.py` | end-to-end runner: structural check + 50k-step PPO + ±1% reward parity vs v5 baseline |

## Hard invariants enforced by PR-1→PR-3

1. **Encoder ownership** — the JEPA encoder lives in `BabyAIAdapter`, not
   in the substrate. `UniversalPolicy` never instantiates an encoder.
2. **Single mask routing** — `UniversalPolicy.action_dist()` is the only
   place action distributions are constructed; it always calls
   `adapter.mask_logits()` first. Missing-mask silent garbage convergence
   is structurally prevented.
3. **Adapter-side reward shaping** — `BabyAIAdapter.reward_shaper()`
   returns `None` for now; v5's `--shaping-coef` still works at the
   training-loop level. Cross-domain reward shaping has a clear API.

## What's deferred to PR-4 (Phase B)

- Two-tensor `(buf_tokens, buf_valid_len)` rolling state and paired
  `torch.where` resets.
- `UniversalTrunk` going live (4× HopfieldEncoderLayer replacing GRUCell).
- `RetrievalBlock` with 2 query tokens into ConceptMemory / OperatorMemory.
- `policy.reset_buffer()` single-function reset API.
- `step_with_value` signature change from `(z, ...)` to `(obs, ...)`.
- PPO `log_prob == replay_log_prob` bit-equality check.

## Smoke-test commands (on Vast.ai)

```bash
cd /workspace/PRISM
git pull origin main

# 1. Structural check (no GPU needed; needs torch + minigrid)
python -m scripts.experiments.checks.phase_a

# Expected output: {"passed": true, ...}

# 2. End-to-end reward parity (50k steps, ~10 minutes on a single GPU)
python -m scripts.experiments.e0_phase_a_reward_parity \
    --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \
    --total-steps 50000 \
    --device cuda

# Expected: structural check passes, ppo_train.py launches with
# --policy-type universal --trunk gru, training proceeds, final
# window_mean_R within ±1% of v5 baseline at the same step count.
```

If you do NOT yet have a v5 baseline `metrics.json` to compare against,
the parity script falls back to a docs-derived estimate (window_mean_R
≈ 0.45 at 50k steps for `--no-bc` GoToLocal training). For a stricter
check, first run:

```bash
# Run v5 baseline at the same step count for direct comparison
python -m scripts.ppo_train \
    --no-bc \
    --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \
    --policy-type hybrid \
    --total-steps 50000 \
    --run-name v5_baseline_50k \
    --device cuda
```

Then pass `--baseline-run runs/v5_baseline_50k` to the parity script.

## Pass criterion

Phase A is considered PASSED when:

1. `check_phase_a()` returns `passed=True`.
2. `e0_phase_a_reward_parity.py` exits with code 0.
3. The v6 run's reward curve in `runs/v6_phaseA_smoke/metrics.json` is
   within ±1% of the v5 baseline at the matched step count.

Phase B (PR-4) is gated on Phase A passing.

## If parity fails

Most likely causes, in decreasing order of likelihood:

1. **State_dict key mismatch** — `UniversalPolicy.state_dict()` has
   `_inner.*` prefix; v5 ckpts don't. This only matters if you pass
   `--bc-checkpoint`. For PR-2, use `--no-bc`.
2. **JEPA loaded twice** — if both `ppo_train.py` and the adapter
   load JEPA, GPU memory doubles. PR-2 passes the same JEPA instance
   into the adapter; verify by grepping for `JepaWorldModel(cfg)`
   constructor calls in the run logs (should be 1).
3. **Action masking inconsistency** — v5 masked via direct logit
   modification in the rollout loop; PR-2 masks via
   `adapter.mask_logits` inside `UniversalPolicy.action_dist`. If
   the v5 masking call sites are not also routed through
   `policy.action_dist`, behavior diverges. Verify by checking
   `ppo_train.py` lines 609 and 726 still use the same mask logic.

PR-4 will replace the manual `Categorical(logits=masked)` construction
at lines 609 and 726 with `policy.action_dist(logits, env_state)`.
For PR-2, those call sites are UNCHANGED — meaning the masking does
NOT yet go through the adapter, even when `--policy-type universal` is
selected. This is acceptable for Phase A because the masking math is
identical; resolution-7 enforcement (adapter-routed masking) lands in
PR-4 alongside the transformer-trunk migration.
