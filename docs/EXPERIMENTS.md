# Experiment Log — BabyAI-GoToLocal-v0

Per-run results on `BabyAI-GoToLocal-v0` with the recurrent policy on top of the
frozen `categorical_spatial` JEPA. All evals run with
`scripts/eval_agent_cohorts.py --episodes 1000 --max-steps 128` unless noted.

## Summary

| Version | Tag | Checkpoint | mean_R | adj | near | facing | visible | hidden | Notes |
|---------|-----|-----------|-------:|----:|-----:|-------:|--------:|-------:|-------|
| v1.0    | (none)                     | pre-budget BC + PPO       | ~0.60 | —     | —     | —     | —     | —     | Baseline before max_steps fix |
| v1.1    | `v1.1-extended-budget`     | ppo_v4 (max_steps=128)    | 0.682 | 0.622 | —     | —     | —     | —     | Extended episode budget; adjacent stuck at 49 steps |
| v1.2    | `v1.2-ppo-iter740`         | ppo_v5_long iter740       | 0.771 | 0.749 | 0.760 | 0.823 | 0.855 | 0.731 | 4× longer PPO; ep_steps fell 34→24 |
| **v1.3**| **`v1.3-pathB-iter400`**   | **ppo_v6_pathB iter400**  | **0.928** | **0.944** | **0.917** | **0.947** | **0.964** | **0.913** | **Path B: explicit memory features** |

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
