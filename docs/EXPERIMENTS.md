# Experiment Log — PRISM

Single-env (v1.x) results on `BabyAI-GoToLocal-v0`. Multi-env (v2.x) results
across the BabyAI go-to family. All evals run with
`scripts/eval_agent_cohorts.py --episodes 1000 --max-steps 128` unless noted.

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
