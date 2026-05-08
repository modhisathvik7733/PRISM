# Experiment Log — PRISM

Single-env (v1.x) RL results on `BabyAI-GoToLocal-v0`. Multi-env (v2.x) RL
results across the BabyAI go-to family. Language-domain (v3.x) results test
whether the structured-latent-middle thesis transfers from gridworld RL to
text reasoning. All RL evals run with
`scripts/eval_agent_cohorts.py --episodes 1000 --max-steps 128`; lang evals
run via `scripts/lang/eval.py --episodes 1000` unless noted.

## Summary — single-env (GoToLocal-v0)

| Version | Tag | Checkpoint | mean_R | adj | near | facing | visible | hidden | Notes |
|---------|-----|-----------|-------:|----:|-----:|-------:|--------:|-------:|-------|
| v1.0    | (none)                     | pre-budget BC + PPO       | ~0.60 | —     | —     | —     | —     | —     | Baseline before max_steps fix |
| v1.1    | `v1.1-extended-budget`     | ppo_v4 (max_steps=128)    | 0.682 | 0.622 | —     | —     | —     | —     | Extended episode budget; adjacent stuck at 49 steps |
| v1.2    | `v1.2-ppo-iter740`         | ppo_v5_long iter740       | 0.771 | 0.749 | 0.760 | 0.823 | 0.855 | 0.731 | 4× longer PPO; ep_steps fell 34→24 |
| **v1.3**| **`v1.3-pathB-iter400`**   | **ppo_v6_pathB iter400**  | **0.928** | **0.944** | **0.917** | **0.947** | **0.964** | **0.913** | **Path B: explicit memory features** |

## Summary — multi-env (go-to family)

Multi-env policies trained jointly on the 3 envs below; evaluated separately on each.

| Version | Policy | GoToLocal-v0 | GoTo-v0 | GoToObj-v0 | Notes |
|---------|--------|-------------:|--------:|-----------:|-------|
| v1.3 zero-shot (universal JEPA) | `ppo_v6_pathB iter400` | 0.570 (59.5%) | 0.163 (17.2%) | 0.898 (94.4%) | Same v1.3 weights, but using v2 universal JEPA — quantifies the JEPA-swap penalty (~−36 pt on GoToLocal) |
| **v2.0**| `v2_ppo_multienv policy_final` | **0.895 (94.6%)** | 0.178 (18.9%) | **0.965 (100%)** | Multi-env BC + multi-env PPO, 16 workers round-robin across 3 envs. Recipe transfers cleanly to GoToObj; GoTo's hidden-cohort exploration ceiling is unchanged from v1.3. |

## Summary — language reasoning (v3.x)

| Version | Model | Params | Task | mean_acc | Notes |
|---------|-------|-------:|------|---------:|-------|
| **v3.0**| `lang_t1_v1` (PRISM-Lang small, AR + JEPA-middle + AR) | 24M | Synthetic bAbI Task 1 (10k train) | **100.0%** | Architecture sanity check — proves the AR-edge + JEPA-middle stack can reach perfect accuracy on a controlled task. |
| **v3.0**| `lang_all_v0 step ~14k` (same arch, real bAbI) | 24M | bAbI 1k variant, all 20 tasks | **57.3%** | Beats vanilla LSTM band (~30%), below pretrained-fine-tune band (~75%). Per-task breakdown shows architecture handles condensation tasks (3 tasks ≥85%) but plateaus on multi-hop reasoning, exactly the literature-predicted ceiling for a from-scratch transformer on bAbI 1k. |

---

## v3.0 — PRISM-Lang on bAbI

**Date:** 2026-05-09
**Architecture:** AR transformer encoder (4 layers, d=256) →
LatentMiddle (8 thought tokens × 6 recurrent thinking steps + EMA-target
JEPA aux loss) → AR transformer decoder cross-attending to thoughts only
→ tied LM head over GPT-2 BPE vocab. 24,174,417 params.
**Tag (suggested):** `v3.0-lang-babi-step14k`

### Phase 1 — synthetic Task 1 (architecture validation)
**Checkpoint:** `runs/lang_t1_v1/model_final.pt`
**Final test_acc: 100.0% (1000 examples)**

Trajectory: random → 49% by step 500 → 73% by step 2500 → 92% by 3500 →
99.5% by 4500 → **100% from step 5000 onward**, held through step 8000.
Train CE stayed in the 0.005-0.05 range (rule-learning, not memorization).

This proves the architecture can reach the ceiling on a clean reasoning
task. Required two fixes from the v0 attempt: (1) bump synthetic data
1k → 10k so the 24M model can't trivially memorize; (2) mask-aware
pooling in the middle's `ctx_to_thought` projection (without it, ~85%
of the pool was PAD-position zeros at max_seq_len=256, drowning the
content-conditioned thought-init bias).

### Phase 2 — real bAbI all 20 tasks (1k variant)
**Checkpoint:** `runs/lang_all_v0/model_step14000.pt` (best-loss
checkpoint; the step-50000 final overfit hard, dropping to 50.5%).
**Final test_acc: 57.3% mean (1000 examples per task)**

Per-task breakdown:

| task | name | acc% |
|---:|---|---:|
| 20 | agents-motivations | **96.9** |
| 13 | compound-coreference | **92.1** |
| 18 | size-reasoning | **85.4** |
| 7 | counting | 73.3 |
| 11 | basic-coreference | 69.8 |
| 6 | yes-no-questions | 66.7 |
| 9 | simple-negation | 66.2 |
| 8 | lists-sets | 65.5 |
| 12 | conjunction | 64.5 |
| 4 | two-arg-relations | 64.2 |
| 10 | indefinite-knowledge | 57.7 |
| 17 | positional-reasoning | 56.1 |
| 15 | basic-deduction | 50.8 |
| 16 | basic-induction | 47.3 |
| 1 | single-supporting-fact | 46.9 |
| 5 | three-arg-relations | 45.9 |
| 14 | time-reasoning | 35.8 |
| 2 | two-supporting-facts | 32.9 |
| 3 | three-supporting-facts | 18.5 |
| 19 | path-finding | **8.6** |
| **mean** | | **57.3** |

### Headline conclusions
- **Architecture handles whole-story condensation tasks** (3 tasks
  ≥85%): tasks 13/18/20 are "synthesize the story → answer" patterns
  that the K=8 thought tokens × N=6 cross-attn iterations match cleanly.
- **Architecture plateaus on multi-hop sequential reasoning**: tasks 2
  (32.9%), 3 (18.5%), 19 (8.6%) need explicit multi-step chains the
  middle's recurrence at this scale cannot learn from 1k examples.
- **Result is in the literature-predicted band** for a 24M from-scratch
  transformer on bAbI 1k: above vanilla LSTM (~30%), below pretrained
  GPT-2 fine-tune (~75-85%), well below purpose-built Memory Networks
  (~93%).
- **Output vocabulary is locked to bAbI's answer set**. Free-form
  prompting reveals the LM head learned a closed answer distribution
  (~10 location words, ~5 object words). OOV inputs like "hi" or
  "Story: Alice walked to the library" get mapped to the nearest
  bAbI-vocab token (`bathroom`, `kitchen`, etc). The encoder/middle
  process the input fine — the decoder bottleneck collapses outputs
  to learned vocab. **This motivates v3.1**: pretrained encoder+decoder
  weights so the LM head is broad, with the middle layer trained from
  scratch on top.
- **Surprising weak point**: Task 1 at 46.9% (synthetic equivalent hit
  100%). Real Task 1 stories are longer than the 2-6-sentence synthetic
  variant; the model probably needs more thinking-step capacity for
  long-context recency tracking.

### What this proves about the JEPA-middle thesis
The middle is doing real work — different inputs produce different
(and sometimes correct) outputs across 20 distinct reasoning patterns.
The `--show-mistakes` traces confirm the model is recall-pattern-
matching rather than syntactic copying. But at 24M from-scratch, on 1k
examples per task, it cannot learn the multi-hop chains that Memory
Networks solve via task-specific architecture or that pretrained models
solve via scale. The hypothesis ("structured latent middle adds
inductive bias") is **partially supported** — a clean ablation against
a matched-param vanilla AR baseline would isolate the middle's
contribution; that's deferred to a follow-up.

### Out of scope (deferred to v3.1)
- Pretrained encoder/decoder backbones (the path to free-form English)
- Coconut-style continuous-thought CoT replacement curriculum
- Reasoning tasks with longer sequences (GSM8K math)
- Matched-param vanilla AR baseline for clean architecture comparison

---

## v2.0 — Multi-env transfer (go-to family)

**Date:** 2026-05-08
**Checkpoint:** `runs/v2_ppo_multienv/policy_final.pt`
**Commit:** `5aeb33c` (generalize fork + universal JEPA off-by-one fix)
**Capstone:** evaluated on 3 envs × 1000 episodes each.

### Setup
- BC dataset: 7890 episodes / 121k transitions across GoToLocal-v0 (3000),
  GoTo-v0 (1890), GoToObj-v0 (3000). Memory-mode teacher with
  `InjectingTeacher` wrapper (no-op for go-to family — kept for
  generality).
- Universal JEPA: same `categorical_spatial` config as v1.3, trained on
  100k optimizer steps with round-robin random-policy transitions across
  the 3 envs.
- Multi-env BC: 30k optimizer steps on the mixed dataset.
- Multi-env PPO: 2M env steps, 16 workers round-robin (4 each on
  GoToLocal/GoToObj, 8 on GoTo since the dataset was skewed).
- Same `mem_feat_dim=5` as v1.3 (PoseTracker v1 normalizations — see
  caveats).

### Per-env results (multi-env policy)

**BabyAI-GoToLocal-v0** — mean_R 0.895 (94.6% success, 14.3 mean steps)

| cohort | n | mean_R | success% | steps |
|---|---:|---:|---:|---:|
| adjacent | 31 | 0.889 | 93.5% | 14.9 |
| near | 56 | 0.917 | 96.4% | 11.3 |
| facing | 336 | 0.938 | 97.9% | 8.5 |
| visible | 54 | 0.950 | 98.1% | 6.9 |
| hidden | 523 | 0.859 | 92.0% | 19.0 |

**BabyAI-GoTo-v0** — mean_R 0.178 (18.9% success, 105.4 mean steps)

| cohort | n | mean_R | success% | steps |
|---|---:|---:|---:|---:|
| adjacent | 7 | 0.804 | 85.7% | 25.9 |
| near | 9 | 0.964 | 100.0% | 5.1 |
| facing | 65 | 0.931 | 96.9% | 9.7 |
| visible | 16 | 0.976 | 100.0% | 3.4 |
| hidden | 903 | **0.097** | **10.5%** | 115.7 |

**BabyAI-GoToObj-v0** — mean_R 0.965 (100% success, 5.0 mean steps)

| cohort | n | mean_R | success% | steps |
|---|---:|---:|---:|---:|
| adjacent | 27 | 0.963 | 100% | 5.3 |
| near | 56 | 0.966 | 100% | 4.8 |
| facing | 278 | 0.976 | 100% | 3.4 |
| visible | 55 | 0.979 | 100% | 3.0 |
| hidden | 584 | 0.958 | 100% | 6.0 |

### Headline conclusions
- **Multi-env transfer works for navigation envs the teacher can solve.**
  GoToObj jumped from 89.8% (v1.3 zero-shot) to **100%**; GoToLocal
  recovered to 94.6% despite the JEPA swap that cost v1.3 36 pts.
- **GoToObj-v0 is essentially solved** (100% across all cohorts).
- **GoTo's hidden cohort failure is the same bottleneck v1.3 has on
  GoToLocal hidden** — exploration ceiling in larger rooms. The
  `PoseTrackerV2` with widened normalizations was created for exactly
  this and is *not yet wired into PPO training* — that's the next
  obvious experiment.

### Caveats / scope
- **Pickup-v0 / Open-v0 dropped** because they're multi-room envs the
  memory teacher can't solve (2/5300 episodes for Pickup confirmed in
  phase 1). Adding them needs a stronger teacher (e.g. BabyAI's
  built-in `BabyAIBot`) — out of scope for this iteration.
- The multi-env PPO uses `PoseTracker v1` for memory features; wiring
  v2 norms requires a small `EnvWorker` shim (~30 min code, 1.5 hr
  retrain). Reserved for a possible v2.1.

---

## v1.3 — Path B: memory-augmented policy input

**Date:** 2026-05-08
**Checkpoint:** `runs/ppo_v6_pathB/policy_iter400.pt`
**Commit:** `ea94533` (phase 4 component 3)
**Capstone mean_R: 0.928 (n=1000)**

Per-cohort breakdown:

| cohort   | n   | mean_R | success% | mean_steps |
|----------|----:|-------:|---------:|-----------:|
| adjacent |  31 | 0.944  | 100.0%   |  7.9       |
| near     |  56 | 0.917  |  96.4%   | 11.3       |
| facing   | 336 | 0.947  |  99.1%   |  7.4       |
| visible  |  54 | 0.964  | 100.0%   |  5.2       |
| hidden   | 523 | 0.913  |  98.5%   | 12.2       |

Hidden-cohort exploration: 100% of episodes saw the target; mean steps to
first visible = 1.8.

**What changed vs v1.2:**
- Added `PoseTracker` (pose + visited/blocked + goal cache) running per env.
- 5-d feature vector (`n_visited`, `n_blocked`, `goal_seen`, `goal_fwd`,
  `goal_right`) projected as a zero-init residual onto the policy/value heads.
- Started PPO from v1.2 iter740. Zero-init meant step-0 behavior matched
  v1.2 exactly; the policy then learned to read the features.

**Training command:**
```bash
python -m scripts.ppo_train \
    --jepa-checkpoint $CKPT \
    --bc-checkpoint runs/ppo_v5_long/policy_iter740.pt \
    --mem-feat-dim 5 \
    --max-steps 128 --shaping-coef 0.1 \
    --total-steps 1000000 --run-name ppo_v6_pathB --device cuda
```

Training trajectory (window_R every ~50 iters):

| iter | window_R | ep_steps |
|-----:|---------:|---------:|
|    1 | 0.726    | 19.6     |
|  100 | 0.752    | 17.9     |
|  145 | 0.800    | 14.5     |
|  200 | 0.835    | 12.0     |
|  300 | 0.854    | 10.6     |
|  370 | 0.882    |  8.8     |
|  400 | 0.879    |  8.8     |
|  475 | 0.883    |  8.6     |
|  488 | 0.859    | 10.3     |

**Why it worked:** the GRU was implicitly fighting to encode pose / goal
location from sequences of partial 7×7 views. Lossless 5-d features
short-circuited that, freeing capacity for action decisions.

---

## v1.2 — Long PPO from BC

**Date:** 2026-05-07
**Checkpoint:** `runs/ppo_v5_long/policy_iter740.pt`
**Capstone mean_R: 0.771 (n=1000)**

| cohort   | n   | mean_R | success% | mean_steps |
|----------|----:|-------:|---------:|-----------:|
| adjacent |  31 | 0.749  |  80.6%   | 33.8       |
| near     |  56 | 0.760  |  82.1%   | 31.8       |
| facing   | 336 | 0.823  |  89.3%   | 23.8       |
| visible  |  54 | 0.855  |  94.4%   | 20.1       |
| hidden   | 523 | 0.731  |  83.6%   | 36.4       |

**Training command:**
```bash
python -m scripts.ppo_train \
    --jepa-checkpoint $CKPT \
    --bc-checkpoint runs/bc_recurrent_v0.9b/policy_final.pt \
    --max-steps 128 --shaping-coef 0.1 \
    --total-steps 2000000 --run-name ppo_v5_long --device cuda
```

**Notes:**
- Best mid-training window_R = 0.730 at iter945; saved checkpoints land at
  every 20 iters, so iter740 (window_R=0.700 at that point) was the best
  *saved* one. Final ckpt (0.640) was post-collapse and not used.
- Adjacent cohort stayed slow (33.8 steps for a 1-cell-away target) — the
  bottleneck Path B was designed to attack.

---

## v1.1 — Extended episode budget

**Tag:** `v1.1-extended-budget`
**Capstone mean_R: 0.682 (n=1000)**

Cohort breakdown not fully recorded. Adjacent ≈ 0.622, mean_steps ≈ 49.
The fix was raising `max_steps` from 64 → 128 in BabyAI-GoToLocal-v0
(commits a48d2d1, 9ca40ee, 3dad658, 1cdd8d7).

---

## v1.0 — Baseline (pre-budget fix)

mean_R ≈ 0.60. Hand-coded memory mode achieves 0.601 on the same env;
PPO on top of BC at this point did not consistently exceed it.
