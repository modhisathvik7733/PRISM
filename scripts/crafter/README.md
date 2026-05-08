# Crafter port — extending PRISM beyond gridworld

Tests whether the v2.0 PRISM stack (frozen JEPA + recurrent policy + PPO)
scales beyond BabyAI's 7×7 categorical view to Crafter's 64×64 RGB
procedural worlds. Crafter is the standard "small but real" benchmark for
world-model RL.

## Reference numbers
- **Random policy:** ~1.0 (geometric-mean achievement score, %)
- **PPO from scratch:** ~5.0 ([Crafter paper, 2021](https://arxiv.org/pdf/2109.06780))
- **DreamerV3:** ~12.1 ([Hafner et al., 2024](https://danijar.com/project/crafter/))
- **PRISM target:** unknown — that's what we're measuring.

## Roadmap (3 commits)

### Commit 1 (this one) — env + smoke test + scaffolding
- `prism/crafter/env_wrapper.py` — gymnasium wrapper that encodes obs to
  (3, 64, 64) float32 and tracks unlocked achievements per episode.
- `prism/crafter/cnn_encoder.py` — Impala-style 4-conv CNN, 64×64 → 256-d.
- `scripts/crafter/smoke_test.py` — verifies install + env + encoder.

**Verify before proceeding:**
```bash
pip install crafter
python -m scripts.crafter.smoke_test
```
Expect 4/4 checks to pass with random-episode achievement count of 0-2
(random policy unlocks `collect_wood` occasionally).

### Commit 2 — PPO baseline (no JEPA)
- `scripts/crafter/ppo_train_baseline.py` — RecurrentPolicy with
  end-to-end CNN encoder + GRU + PPO. No mission, no pose features.
- `scripts/crafter/eval.py` — geometric-mean achievement score across
  N episodes.

Goal: hit the published ~5% baseline. Validates the env + RL infra
before we layer in the world model.

### Commit 3 — RGB JEPA + PPO with frozen latent
- `prism/crafter/jepa_rgb.py` — JEPA model with `CrafterCNN` encoder +
  the existing dynamics MLP from `prism.models.jepa`. No aux predicate
  loss (Crafter has no slot extractor); just predictive + EMA.
- `scripts/crafter/train_jepa_rgb.py` — train on 100k random rollouts.
- `scripts/crafter/ppo_train.py` — PPO with frozen JEPA latent, mirrors
  the BabyAI `scripts/ppo_train.py` structure.

Goal: beat the PPO-from-scratch baseline. If PRISM hits 8-10%, the
recipe scales. If it stalls at 5%, we know the categorical_spatial
encoder + slot-based aux losses were doing more work than we thought.

## Architecture differences from the BabyAI stack

| Component | BabyAI (v1.3 / v2.0) | Crafter |
|-----------|---------------------|---------|
| Observation | 7×7×3 categorical | 64×64×3 RGB float |
| Encoder | `categorical_spatial` (per-cell embedding lookup + conv) | Impala CNN (4 conv layers) |
| Action space | 7 (BabyAI) | 17 (4 move + 13 interact) |
| Mission | 24-d one-hot | none — drop the projection |
| Aux predicate loss | 96-d binary + 24-d distance | none — slot extractor only works for BabyAI |
| Pose tracker memory features | 5-d (Path B) | none — Crafter scrolls around the agent |
| BC warmstart | memory-mode teacher | random policy → JEPA, then PPO from scratch |
| Reward | sparse (1 on success) | dense (+1 per achievement, ±0.1 per health) |
