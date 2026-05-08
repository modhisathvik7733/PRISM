# PRISM-v4 Phase 1 — Cognitive Core in Tiny Environments

Tests whether **causal/operator/counterfactual structure emerges** from
the frozen v1.3 JEPA world model, on top of which we add 4 thin
modules + 1 task scheduler. Five pass/fail emergence criteria gate
the whole upcoming developmental cognitive architecture.

## Governing principles (recap)

1. **World-model learning drives cognition** — language is interface, not cognition
2. **Developmental curriculum, never internet soup** — Phi-style curated content per stage

This phase implements **Stage 0 (pre-linguistic cognition)** only —
embodied object/action/causality in BabyAI gridworld. Language stages
(1-4) come later, AFTER Stage 0 emergence is verified.

## Components built (5 of 8)

| # | Module | What it does |
|---|---|---|
| 1 | `prism/cog_core/object_tracker.py` | Probe JEPA latents for persistent (type, color, pos) entities; greedy nearest-neighbor tracker maintains IDs across frames |
| 2 | `prism/cog_core/world_model_rollout.py` | Wraps frozen v1.3 JEPA for multi-step rollouts (`encode`, `step`, `rollout`, `predicates`, `latent_diff`) |
| 3 | `prism/cog_core/counterfactual.py` | Compares (state, actual_action) vs (state, cf_action) rollouts → divergence metrics + predicate flips |
| 4 | `prism/cog_core/operator_bank.py` | K-means clusters JEPA latent-deltas into operators; per-cluster purity + cross-env stability checks |
| 7 | `prism/cog_core/curriculum.py` | ALP-bandit (Akakzia 2021): `value(task) = (1 - sr) × |LP|` |

Components 5 (memory), 6 (curiosity), 8 (language grounding) are
explicitly **deferred** until Phase 1 emergence passes.

## Five emergence criteria (the gate to Phase 2)

| # | Test | Target |
|---|------|-------:|
| 1 | Object persistence: probe accuracy on held-out frames | ≥85% |
| 2 | World model: 1-step / 4-step latent cosine similarity | ≥0.95 / ≥0.85 |
| 3 | Counterfactual coherence: % swaps producing real divergence | ≥80% |
| 4 | Operator abstraction: # interpretable clusters (≥80% action purity) | ≥4 |
| 5 | Curriculum: ALP scheduler beats random by mean R | ≥+0.10 absolute |

If ALL 5 pass → Phase 2 (memory, curiosity, language grounding).
If any fail → re-architect the failing component, do NOT add more.

## Quickstart (~4-5 hours total compute)

```bash
# 0. Set the v1.3 JEPA path (already trained)
export V13_JEPA="runs/jepa_categorical_spatial_aux3_dist24_mix0.5_spat64_spatial_film_dyn3x256_BabyAI-GoToLocal-v0_seed0/jepa_final.pt"
export V13_POLICY="runs/ppo_v6_pathB/policy_iter400.pt"

# 1. Collect rollouts from v1.3 policy (~30 min)
python -m scripts.cog_core.collect_rollouts \
    --jepa-checkpoint $V13_JEPA \
    --policy-checkpoint $V13_POLICY \
    --envs BabyAI-GoToLocal-v0 BabyAI-GoTo-v0 BabyAI-GoToObj-v0 \
    --episodes-per-env 500 \
    --output runs/cog_core_phase1/rollouts.npz

# 2. Train object tracker probe (~1 hr)
python -m scripts.cog_core.train_object_tracker \
    --rollouts runs/cog_core_phase1/rollouts.npz \
    --steps 5000 --device cuda \
    --run-name cog_phase1_objects

# 3. Extract operators via K-means (~10 min, no GPU train)
python -m scripts.cog_core.extract_operators \
    --rollouts runs/cog_core_phase1/rollouts.npz \
    --n-clusters 8 --per-env \
    --output runs/cog_core_phase1/operators.npz

# 4a. Curriculum-driven PPO training (~1 hr) — ALP scheduler
python -m scripts.cog_core.train_curriculum \
    --jepa-checkpoint $V13_JEPA \
    --bc-checkpoint runs/v2_ppo_multienv/policy_final.pt \
    --envs BabyAI-GoToObj-v0 BabyAI-GoToLocal-v0 BabyAI-GoTo-v0 \
    --scheduler alp --total-steps 1000000 \
    --run-name cog_phase1_alp --device cuda

# 4b. Same compute, random scheduler (the control) (~1 hr)
python -m scripts.cog_core.train_curriculum \
    --jepa-checkpoint $V13_JEPA \
    --bc-checkpoint runs/v2_ppo_multienv/policy_final.pt \
    --envs BabyAI-GoToObj-v0 BabyAI-GoToLocal-v0 BabyAI-GoTo-v0 \
    --scheduler random --total-steps 1000000 \
    --run-name cog_phase1_random --device cuda

# 5. Run all 5 emergence tests (~5 min)
python -m scripts.cog_core.eval_emergence \
    --object-tracker runs/cog_phase1_objects/model_final.pt \
    --operators runs/cog_core_phase1/operators.npz \
    --rollouts runs/cog_core_phase1/rollouts.npz \
    --jepa-checkpoint $V13_JEPA \
    --alp-policy runs/cog_phase1_alp/policy_final.pt \
    --random-policy runs/cog_phase1_random/policy_final.pt \
    --output docs/EXPERIMENTS_phase1.md
```

Final stdout will print a 5-row PASS/FAIL table. The markdown report
at `docs/EXPERIMENTS_phase1.md` is meant to be appended to
`docs/EXPERIMENTS.md` as the v4.0 row.

## What this commit does NOT include

- **No JEPA retraining.** Reuses the frozen v1.3 JEPA exactly as-is.
- **No language at all.** Stage 0 is pre-linguistic; bAbI / TinyStories
  / GSM8K work is paused until emergence is verified.
- **No new RL algorithm.** ALP scheduler wraps the existing PPO loop;
  the underlying PPO code from `scripts/ppo_train.py` is reused unchanged.

## What happens AFTER Phase 1 passes

| Stage | Curated corpus | Component activated |
|---|---|---|
| 1 — Early grounded language | TinyStories + bAbI 1, 4, 6 | Language Grounding (8) |
| 2 — Compositional language | bAbI 2-20 + CLUTRR | Hierarchical Memory (5) |
| 3 — Math + reasoning | GSM8K + MATH-elementary | Active Inference (6) refines |
| 4 — Technical | code, science, philosophy | Full system runs end-to-end |

Each stage is a separate phase. None starts until the previous has
passed its own emergence criteria. The curriculum scheduler from
Phase 1 controls the stage transitions in later phases.

## Reused (no edits to existing code)

- `prism/models/jepa.py` — frozen v1.3 JEPA
- `prism/perception/slots.py` — supervision signal for object tracker
- `prism/agents/grounded_agent.py` — recurrent policy interface
- `prism/agents/pose_tracker.py` — initialization seed (informational)
- `prism/envs/babyai.py` — env wrappers
- `scripts/ppo_train.py` — `EnvWorker`, `compute_gae`, `make_action_mask`,
  `latent_dim_for_cfg` (imported by `train_curriculum.py`)
- `runs/ppo_v6_pathB/policy_iter400.pt` — v1.3 policy for rollout source
- `runs/v2_ppo_multienv/policy_final.pt` — v2.0 multi-env policy as
  curriculum BC warmstart

## Literature alignment

- Object/event persistence: **Causal-JEPA** (arXiv 2602.11389, 2026),
  **V-JEPA 2** (Meta 2025) — object permanence emerges from JEPA-style
  pretraining; we test if it emerges in our small JEPA via probing
- Operator extraction: **Mixture-of-World-Models** (arXiv 2602.01270,
  2026) — gradient-based clustering for task heads. We start with
  K-means and only escalate if K-means is messy.
- Curriculum: **CURIOUS** (Colas et al. 2018) refined by **Akakzia
  et al. (2021)** — value = `(1 - sr) × |LP|` (the Akakzia formula).
- Phi philosophy for later stages: **Phi-1** (arXiv 2306.11644,
  Microsoft 2023) "Textbooks Are All You Need", **Phi-4-reasoning**
  (April 2025) — small models match big with curated data.
