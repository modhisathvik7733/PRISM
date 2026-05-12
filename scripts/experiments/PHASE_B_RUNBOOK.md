# Phase B Runbook — v6 substrate success-rate validation

This is the runbook for the v6 Phase B exit gate: confirm that the
substrate-as-built (transformer trunk + Hopfield memory via
RetrievalBlock, no curriculum) matches v5 on BabyAI's standard envs.

Plan criterion: **GoToObj window_mean_R ≥ 0.85 AND GoToLocal
window_mean_R within 5pp of v5.0 at the same step count**.

50k-step smoke runs already passed this within seed noise:
v5 hybrid at 50k → 0.462, v6 transformer+retrieval at 50k → 0.420
(−4.2pp, within tolerance). The 500k run confirms the substrate scales
to plan-stated training budgets without divergence or regression.

## What this runbook covers

| Run | Purpose | Wall-clock (single GPU) |
|---|---|---|
| 1 | v6 on GoToLocal, 500k steps | ~2-3h |
| 2 | v6 on GoToObj, 500k steps | ~2-3h |
| 3 (optional) | v5 baseline on GoToLocal, 500k steps | ~2-3h |
| 4 (optional) | v5 baseline on GoToObj, 500k steps | ~2-3h |

Runs 3 and 4 are optional: the gate script falls back to docs-derived
v5 numbers (0.55 GoToLocal, 0.90 GoToObj) when matched baselines aren't
provided. For a publishable comparison, run them.

## Pre-flight

```bash
cd /workspace/PRISM
git pull origin main
# Confirm the substrate's standalone smokes still pass on this commit.
python -m prism.cognition.memory_bank
python -m prism.cognition.policy
python -m prism.curriculum.engine
```

If any smoke fails, do NOT proceed — diagnose first.

## Launch commands

Run these in sequence (or in parallel on multiple Vast.ai instances).
The four runs are independent; the JEPA checkpoint is read-only.

### 1. v6 substrate, GoToLocal

```bash
python -m scripts.ppo_train --no-bc \
    --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \
    --policy-type universal --trunk transformer \
    --env-id BabyAI-GoToLocal-v0 \
    --total-steps 500000 \
    --run-name v6_phaseB_GoToLocal_500k \
    --device cuda
```

### 2. v6 substrate, GoToObj

```bash
python -m scripts.ppo_train --no-bc \
    --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \
    --policy-type universal --trunk transformer \
    --env-id BabyAI-GoToObj-v0 \
    --total-steps 500000 \
    --run-name v6_phaseB_GoToObj_500k \
    --device cuda
```

### 3. v5 baseline, GoToLocal (optional, for a matched comparison)

```bash
python -m scripts.ppo_train --no-bc \
    --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \
    --policy-type hybrid \
    --env-id BabyAI-GoToLocal-v0 \
    --total-steps 500000 \
    --run-name v5_phaseB_GoToLocal_500k \
    --device cuda
```

### 4. v5 baseline, GoToObj (optional)

```bash
python -m scripts.ppo_train --no-bc \
    --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \
    --policy-type hybrid \
    --env-id BabyAI-GoToObj-v0 \
    --total-steps 500000 \
    --run-name v5_phaseB_GoToObj_500k \
    --device cuda
```

## Evaluate the gate

Once both v6 runs complete, the gate is auto-evaluable from
`metrics.json`:

```bash
# Without matched v5 baselines (uses docs fallback):
python -m scripts.experiments.checks.phase_b_success_gate \
    --v6-gotolocal runs/v6_phaseB_GoToLocal_500k \
    --v6-gotoobj   runs/v6_phaseB_GoToObj_500k

# With matched v5 baselines:
python -m scripts.experiments.checks.phase_b_success_gate \
    --v6-gotolocal runs/v6_phaseB_GoToLocal_500k \
    --v6-gotoobj   runs/v6_phaseB_GoToObj_500k \
    --v5-gotolocal runs/v5_phaseB_GoToLocal_500k \
    --v5-gotoobj   runs/v5_phaseB_GoToObj_500k
```

Exit codes:
- `0` → both gates pass; Phase B PASSES.
- `3` → at least one env below threshold; Phase B FAILS.
- `4` → a metrics.json is missing; re-run or fix paths.

## What to do if Phase B fails

Most likely causes, in decreasing order of likelihood:

1. **GoToObj missions don't parse correctly.** v5's mission parser
   (`prism.agents.grounded_agent.allowed_actions_for_spec`) was
   designed for GoToLocal's templates. GoToObj uses "go to <type>"
   without color; check that `goal_predicates_for_mission()` returns
   a non-empty allowed-action set. Symptom: `window_R` stays near 0
   for the entire run, episodes time out.

2. **Substrate over-regularizes.** Hopfield normalization in
   `_TransformerInner` may smooth out the harder rooms in GoToObj
   where strong locale-specific features matter. Mitigation: lower
   `concept_scaling` from 1.0 to 0.5; that softens the retrieval and
   lets the trunk's local computation dominate. Add to from_adapter
   kwargs if confirmed.

3. **Buffer length too short for multi-step bindings.** Default L=16
   may be insufficient for some GoToObj rooms. Increase to L=32 and
   re-run — keep the rest of the substrate config locked.

4. **Optimizer state mismatch from BC checkpoint.** If
   --bc-checkpoint is passed, missing keys for the new
   Concept/Operator banks load with random init; the value head also
   needs warmup. Run with --no-bc for the Phase B gate.

## What this does NOT validate

The Phase B gate confirms the substrate can LEARN on individual envs at
the plan-stated training scale. It does NOT cover:

- Cross-stage retention (E2) — needs curriculum mode + freeze test.
- Curriculum ordering (E1) — needs per-stage env_factories (PR-6b).
- Cross-env transfer (E3) — needs multi-env training pipeline.
- Cross-domain transfer (E5) — needs a non-BabyAI adapter.

Those are the experiments downstream of Phase B passing.
