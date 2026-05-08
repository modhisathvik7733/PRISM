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

## Components built (5 of 8 + Path B developmental trainer)

| # | Module | What it does |
|---|---|---|
| 1 | `prism/cog_core/object_tracker.py` | Probe JEPA latents for persistent (type, color, pos) entities; greedy nearest-neighbor tracker maintains IDs across frames |
| 2 | `prism/cog_core/world_model_rollout.py` | Wraps a frozen JEPA for multi-step rollouts (`encode`, `step`, `rollout`, `predicates`, `latent_diff`) |
| 3 | `prism/cog_core/counterfactual.py` | Compares (state, actual_action) vs (state, cf_action) rollouts → divergence metrics + predicate flips |
| 4 | `prism/cog_core/operator_bank.py` | K-means clusters JEPA latent-deltas into operators; per-cluster purity + cross-env stability checks |
| 7 | `prism/cog_core/curriculum.py` | Per-task ALP-bandit (Akakzia 2021): `value(task) = (1 - sr) × |LP|` — used at PPO fine-tune level |
| **JEPA-curriculum** | **`prism/cog_core/dev_curriculum.py`** | **Stage definitions (0a → 0b → 0c → 0d) + competence-gated transition logic for the JEPA itself** |

Components 5 (memory), 6 (curiosity), 8 (language grounding) are
explicitly **deferred** until Phase 1 emergence passes.

## Path A vs Path B

We deliberately built BOTH paths so the developmental-curriculum
principle can be tested cleanly:

| Path | What | Why |
|---|---|---|
| **A** (v1.3 reuse) | Reuse the v1.3 JEPA frozen, run all 5 emergence tests on it | Fast diagnostic (~4-5 hr). Tests whether emergence happens WITHOUT explicit curriculum (was random-rollout trained). |
| **B** (this commit) | Re-train JEPA from scratch with strict stage progression | Honors the principle "treat as person who knows nothing." Tests whether developmental ordering produces stronger emergence than random ordering. |

The **comparison between A and B** is the actual test of the
"developmental curriculum is critical" claim. If B's emergence
metrics are noticeably stronger than A's, the principle is empirically
supported. If they're tied, the principle is decorative at this
scale (and we still have a working Stage 0 substrate either way).

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

## Path B — train JEPA developmentally (the principled foundation)

Train the JEPA itself with strict stage progression. Each stage has a
competence-gate: don't graduate until 1-step latent cosine ≥ threshold
on held-out data of THAT stage.

```bash
# Stages defined in prism/cog_core/dev_curriculum.DEFAULT_STAGES:
#   0a  BabyAI-OneRoomS8-v0     basic movement + walls in 8x8 empty room
#   0b  BabyAI-GoToObj-v0       single-object permanence
#   0c  BabyAI-GoToLocal-v0     multi-object discrimination, small room
#   0d  BabyAI-OneRoomS16-v0    larger spatial complexity (16x16)

python -m scripts.cog_core.train_jepa_developmental \
    --total-steps 80000 --batch-size 128 \
    --encoder-type categorical_spatial --spatial-channels 64 \
    --dynamics-type spatial_film --dynamics-hidden 256 --dynamics-layers 3 \
    --aux-predicate-weight 3.0 --aux-distance-dim 24 --aux-distance-weight 0.5 \
    --run-name jepa_dev_v0 --device cuda
```

Compute: ~3-5 hr on the 5070 Ti. Each stage logs cosine-sim every
500 steps; stage transitions are auto-saved as
`runs/jepa_dev_v0/jepa_after_<stage>.pt`. Final model at
`runs/jepa_dev_v0/jepa_final.pt`.

After this finishes, **re-run all the Phase 1 emergence tests pointing
at the dev JEPA instead of v1.3**:

```bash
export DEV_JEPA="runs/jepa_dev_v0/jepa_final.pt"

# Re-collect rollouts using DEV JEPA (we still need a v1.3-style policy
# for action-generation; either reuse v1.3 or train a quick BC policy
# on the dev-JEPA latents). For first comparison just reuse v1.3 policy:
python -m scripts.cog_core.collect_rollouts \
    --jepa-checkpoint $DEV_JEPA \
    --policy-checkpoint $V13_POLICY \
    --envs BabyAI-GoToLocal-v0 BabyAI-GoTo-v0 BabyAI-GoToObj-v0 \
    --episodes-per-env 500 \
    --output runs/cog_core_phase1_devB/rollouts.npz

# All other steps from Path A's quickstart, but pointed at the new
# rollouts + the new JEPA. This produces the second column of
# emergence numbers for the head-to-head comparison.
```

**The headline result is the comparison table:**

| Emergence test | Path A (v1.3 JEPA) | Path B (dev JEPA) |
|---|---:|---:|
| Object persistence accuracy | TBD | TBD |
| World model 1-step / 4-step cos | TBD | TBD |
| Counterfactual coherence | TBD | TBD |
| # interpretable operators | TBD | TBD |
| ALP curriculum lift | TBD | TBD |

If Path B numbers beat Path A by ≥10%, **the developmental principle
is empirically validated**. That's the v4.0 result for `EXPERIMENTS.md`.

---

## Quickstart (~4-5 hours total compute) — Path A (v1.3 emergence test)

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

## Stage 1 sketch (next phase, after Phase 1 emergence verified ✓)

Phase 1 verified the cognitive substrate. Stage 1 adds **early grounded
language** on top — the model speaks via AR I/O but its reasoning is
grounded in the cognitive substrate, not in token statistics.

### The architecture for Stage 1

```
INPUT TEXT → AR encoder ──► language tokens
                              │
                              ▼
                        ┌──────────────────────────────┐
                        │  GROUNDED MIDDLE             │
                        │ ───────────────────────────  │
                        │ K thought tokens that:       │
                        │  • cross-attend to token ctx │
                        │  • READ from object_tracker  │
                        │    (current entity beliefs)  │
                        │  • READ from operator_bank   │
                        │    (apply ops in latent)     │
                        │  • UPDATE entity beliefs as  │
                        │    new sentences arrive      │
                        └──────────────┬───────────────┘
                                       ▼
                              AR decoder ──► OUTPUT TEXT
```

The middle is the same K-thought-tokens × N-steps recipe from v3.0,
but each thinking step has **read access to the cognitive substrate
state** (object tracker beliefs, operator outputs). Words bind to
operators by training on (sentence, before_state, after_state)
triples — bAbI provides this naturally.

### Stage 1 corpus (curated, never internet soup)

| Source | What it teaches | Size |
|---|---|---|
| TinyStories (synthetic) | Basic English grammar + short narratives | ~470M tokens |
| bAbI tasks 1, 4, 6 (curated subset) | "Mary moved to X. Where is Mary?" — single-fact grounding, two-arg relations, yes/no | ~3k examples each |
| Procedural BabyAI captions | Auto-generated sentences describing observed transitions in BabyAI rollouts (e.g. "agent moved forward into kitchen") — direct sentence ↔ operator binding | ~10k auto-generated |

**No web scraping.** Same Phi philosophy as before. The procedural
BabyAI captions are the new piece that physically binds words to the
cognitive substrate's operators.

### Stage 1 emergence criteria (the Stage 2 gate)

| Test | Target |
|------|-------:|
| bAbI Task 1 accuracy (the tasks the v3.0 24M model solved at 100%) | ≥95% |
| TinyStories perplexity on held-out | ≤5 (grammatically coherent) |
| **Free-form generalization**: "Alice walked to the library. Where is Alice?" → answers "library" (NOT a memorized bAbI vocab word) | answer must be in input text |
| Operator-binding probe: given (sentence, world-state-before), predict the operator that produced (world-state-after) | ≥85% |

The third criterion is **the test of the bigger thesis**: does the
model output words that come from understanding, not from memorized
vocab? v3.0 failed this — it always output bAbI vocab. Stage 1
should pass it because the cognitive substrate forces grounded
representations.

### Files Stage 1 will need (deferred to future commit)

```
prism/lang_grounded/
├── __init__.py
├── grounded_middle.py       # Middle that reads from object_tracker + operator_bank
├── caption_generator.py     # Procedural BabyAI sentence generator
└── stage1_model.py          # Composes encoder + grounded middle + decoder

scripts/lang_grounded/
├── data_stage1_corpus.py    # Curate + tokenize TinyStories + bAbI 1/4/6 + captions
├── train_stage1.py          # Train with cognitive-substrate access
├── eval_stage1.py           # 4 emergence criteria
└── ask_stage1.py            # Interactive probing (free-form prompts, OOV names)
```

Plus reuse the existing `prism/lang_pretrain/` scaffolding (vocab,
tokenizer, span-corruption objective, vanilla AR baseline) — that
work was paused, not deleted, exactly for this moment.

### Compute budget for Stage 1

- Curate corpus + tokenize: ~10 min one-time
- Train Stage 1 model (similar size to v3.0, ~50-100M params): ~24 hr
- Eval + interactive probing: ~1 hr

**Total Stage 1: ~1-2 days** of compute. Engineering is ~2-3 days.

### Why this is the test that matters

If Stage 1 passes the third criterion (free-form generalization with
out-of-vocab words), you've **empirically shown that grounding language
in a JEPA-style cognitive substrate produces real semantic
understanding** — not just memorized token correlations. That's the
strongest possible result this whole project could produce at small
scale, and it directly addresses your "treat the model like a child"
principle.

If Stage 1 fails the third criterion, the failure mode tells us
exactly what's missing (likely: the cognitive substrate isn't
expressive enough, or the binding signal is too sparse), and we
iterate the substrate architecture before scaling.

---

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
