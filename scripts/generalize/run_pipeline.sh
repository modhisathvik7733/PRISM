#!/usr/bin/env bash
# End-to-end pipeline for the generalization experiment.
#
# Stops between every phase so you can inspect intermediates. Set the four
# variables at the top, then `bash scripts/generalize/run_pipeline.sh phase1`,
# `phase2`, etc. Or `bash scripts/generalize/run_pipeline.sh all` to chain
# everything.
#
# Pre-reqs:
#   - $V13_JEPA points at the v1.3 GoToLocal JEPA (used only by the BC teacher
#     for predicate signal during data collection).
#   - $V13_POLICY points at runs/ppo_v6_pathB/policy_iter400.pt (for the
#     zero-shot baseline).

set -euo pipefail

V13_JEPA="${V13_JEPA:-runs/jepa_categorical_spatial_aux3_dist24_mix0.5_spat64_spatial_film_dyn3x256_BabyAI-GoToLocal-v0_seed0/jepa_final.pt}"
V13_POLICY="${V13_POLICY:-runs/ppo_v6_pathB/policy_iter400.pt}"
DEVICE="${DEVICE:-cuda}"
# Path A scope — go-to family only. Pickup/Open are multi-room and the
# memory teacher's frontier exploration can't navigate through doors
# (phase1 confirmed: 2/5300 episodes for Pickup-v0). Use the navigation
# family where the teacher is competent: GoToLocal (v1.3 baseline, small
# room), GoTo (single room + distractors), GoToObj (simplest, no
# distractors). Story: "recipe transfers across the go-to family."
ENVS=(BabyAI-GoToLocal-v0 BabyAI-GoTo-v0 BabyAI-GoToObj-v0)
RUN_TAG="${RUN_TAG:-v2}"

phase0_zeroshot() {
    # The "did v1.3 memorize?" baseline. Eval the v1.3 policy on each target
    # env without retraining. Should hit ~0.92 on GoToLocal and degrade on
    # the others — the magnitude of degradation is the headline.
    echo "=== phase0: v1.3 zero-shot baseline across ${#ENVS[@]} envs ==="
    python -m scripts.generalize.ppo_eval_multi \
        --jepa-checkpoint "$V13_JEPA" \
        --policy-checkpoint "$V13_POLICY" \
        --envs "${ENVS[@]}" \
        --episodes 1000 --max-steps 128 --device "$DEVICE" \
        --per-cohort
}

phase1_bc_data() {
    echo "=== phase1: collect mixed BC data across ${#ENVS[@]} envs ==="
    python -m scripts.generalize.collect_bc_multienv \
        --jepa-checkpoint "$V13_JEPA" \
        --envs "${ENVS[@]}" \
        --episodes-per-env 3000 \
        --max-steps 128 \
        --reward-threshold 0.55 \
        --output "runs/${RUN_TAG}_multienv_bc/bc_data.npz" \
        --device "$DEVICE"
    echo
    echo "[verify] action histogram per env:"
    python -c "
import numpy as np
d = np.load('runs/${RUN_TAG}_multienv_bc/bc_data.npz')
acts = d['action_seqs']; lens = d['ep_lengths']; envs = d['env_ids']
for e in sorted(set(envs.tolist())):
    mask = envs == e
    counts = np.zeros(7, dtype=int)
    for i in np.where(mask)[0]:
        for a in acts[i, :lens[i]]:
            counts[int(a)] += 1
    print(f'  {e:30s} {counts.tolist()}')
"
}

phase2_universal_jepa() {
    echo "=== phase2: train universal JEPA on mixed envs ==="
    python -m scripts.generalize.train_jepa_universal \
        --envs "${ENVS[@]}" \
        --steps 100000 --batch-size 128 --rollout-size 10000 \
        --encoder-type categorical_spatial --spatial-channels 64 \
        --dynamics-type spatial_film --dynamics-hidden 256 --dynamics-layers 3 \
        --aux-predicate-weight 3.0 --aux-distance-dim 24 --aux-distance-weight 0.5 \
        --run-name "${RUN_TAG}_jepa_universal" --device "$DEVICE"
}

phase3_bc_train() {
    # Reuse the existing BC trainer — the multi-env BC dataset has the same
    # field layout as the v1.3 single-env one, plus env_ids which it ignores.
    echo "=== phase3: BC-train recurrent policy on the mixed dataset ==="
    python -m scripts.train_recurrent_policy \
        --jepa-checkpoint "runs/${RUN_TAG}_jepa_universal/jepa_final.pt" \
        --bc-data "runs/${RUN_TAG}_multienv_bc/bc_data.npz" \
        --steps 30000 --batch-size 64 \
        --run-name "${RUN_TAG}_bc_multienv" --device "$DEVICE"
}

phase4_ppo_per_env() {
    # Per-env PPO from the multi-env BC warmstart. Two runs (GoToLocal is
    # already v1.3 = ppo_v6_pathB, no need to re-train it here). Each
    # writes to runs/${RUN_TAG}_ppo_<env-suffix>.
    echo "=== phase4: PPO per env from multi-env BC warmstart ==="
    for env in BabyAI-GoTo-v0 BabyAI-GoToObj-v0; do
        suffix="${env#BabyAI-}"; suffix="${suffix%-v0}"
        echo "  --- PPO on $env (out: runs/${RUN_TAG}_ppo_${suffix}) ---"
        python -m scripts.ppo_train \
            --jepa-checkpoint "runs/${RUN_TAG}_jepa_universal/jepa_final.pt" \
            --bc-checkpoint "runs/${RUN_TAG}_bc_multienv/policy_final.pt" \
            --env-id "$env" \
            --mem-feat-dim 5 \
            --max-steps 128 --shaping-coef 0.1 \
            --total-steps 1000000 \
            --run-name "${RUN_TAG}_ppo_${suffix}" --device "$DEVICE"
    done
}

phase4b_ppo_multienv() {
    # One PPO run, 16 workers round-robin across all 4 envs. Per AMAGO-2 /
    # BabyAI++ this is the variant that actually generalizes across levels.
    # Each env gets 16/4 = 4 workers, so 32 envsteps/iter come from each.
    echo "=== phase4b: multi-env PPO from multi-env BC warmstart ==="
    python -m scripts.generalize.ppo_train_multienv \
        --jepa-checkpoint "runs/${RUN_TAG}_jepa_universal/jepa_final.pt" \
        --bc-checkpoint "runs/${RUN_TAG}_bc_multienv/policy_final.pt" \
        --envs "${ENVS[@]}" \
        --mem-feat-dim 5 \
        --max-steps 128 --shaping-coef 0.1 \
        --total-steps 2000000 \
        --run-name "${RUN_TAG}_ppo_multienv" --device "$DEVICE"
}

phase5_capstone() {
    # For each per-env PPO checkpoint AND the multi-env one, run the
    # multi-env eval so we see both in-domain performance AND cross-env
    # transfer in a single comparable table.
    echo "=== phase5: capstone — eval each policy across all envs ==="
    for env in BabyAI-GoTo-v0 BabyAI-GoToObj-v0; do
        suffix="${env#BabyAI-}"; suffix="${suffix%-v0}"
        ckpt="runs/${RUN_TAG}_ppo_${suffix}/policy_final.pt"
        if [[ -f "$ckpt" ]]; then
            echo "  --- eval $ckpt across all envs ---"
            python -m scripts.generalize.ppo_eval_multi \
                --jepa-checkpoint "runs/${RUN_TAG}_jepa_universal/jepa_final.pt" \
                --policy-checkpoint "$ckpt" \
                --envs "${ENVS[@]}" \
                --episodes 1000 --max-steps 128 --device "$DEVICE" \
                --per-cohort
        fi
    done
    # Also eval the v1.3 policy (already a "GoToLocal" expert) on the
    # broader env set so the capstone table has a per-env row for v1.3 too.
    if [[ -f "$V13_POLICY" ]]; then
        echo "  --- eval $V13_POLICY (v1.3 baseline) across all envs ---"
        python -m scripts.generalize.ppo_eval_multi \
            --jepa-checkpoint "runs/${RUN_TAG}_jepa_universal/jepa_final.pt" \
            --policy-checkpoint "$V13_POLICY" \
            --envs "${ENVS[@]}" \
            --episodes 1000 --max-steps 128 --device "$DEVICE" \
            --per-cohort
    fi
    multi_ckpt="runs/${RUN_TAG}_ppo_multienv/policy_final.pt"
    if [[ -f "$multi_ckpt" ]]; then
        echo "  --- eval $multi_ckpt (multi-env) across all envs ---"
        python -m scripts.generalize.ppo_eval_multi \
            --jepa-checkpoint "runs/${RUN_TAG}_jepa_universal/jepa_final.pt" \
            --policy-checkpoint "$multi_ckpt" \
            --envs "${ENVS[@]}" \
            --episodes 1000 --max-steps 128 --device "$DEVICE" \
            --per-cohort
    fi
}

case "${1:-help}" in
    phase0|zeroshot) phase0_zeroshot ;;
    phase1|bc_data) phase1_bc_data ;;
    phase2|jepa) phase2_universal_jepa ;;
    phase3|bc_train) phase3_bc_train ;;
    phase4|ppo) phase4_ppo_per_env ;;
    phase4b|ppo_multi) phase4b_ppo_multienv ;;
    phase5|capstone) phase5_capstone ;;
    all)
        phase0_zeroshot
        phase1_bc_data
        phase2_universal_jepa
        phase3_bc_train
        phase4_ppo_per_env
        phase4b_ppo_multienv
        phase5_capstone
        ;;
    *)
        echo "usage: $0 {phase0|phase1|phase2|phase3|phase4|phase4b|phase5|all}"
        echo "  phase0   zero-shot eval of v1.3 across new envs"
        echo "  phase1   collect mixed BC data (~30 min)"
        echo "  phase2   train universal JEPA (~30-60 min)"
        echo "  phase3   BC-train recurrent policy on mixed data (~30 min)"
        echo "  phase4   per-env PPO from multi-env BC (3 runs, ~3 hr)"
        echo "  phase4b  multi-env PPO (one run, ~1.5 hr) — recommended"
        echo "  phase5   multi-env capstone eval per policy"
        exit 1
        ;;
esac
