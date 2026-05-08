# Generalization fork — port v1.3 to Pickup, GoTo, Open

Tests whether the v1.3 stack (frozen JEPA + recurrent policy + 5-d pose
features + PPO) generalizes beyond `BabyAI-GoToLocal-v0`. None of the code
under `prism/` or `scripts/` (other than this directory and
`prism/generalize/`) is modified.

## Why a separate fork?

`v1.3-pathB-iter400` hits 98.7% success on GoToLocal, matching the BabyAI
99% ceiling. Open question: did the recipe transfer or memorize? Three new
envs answer this — Pickup needs action 3, Open needs action 5, GoTo adds
distractors. If the same JEPA + memory-augmented policy hits comparable
numbers on those, the recipe is real.

## What's new

| File | Purpose |
|------|---------|
| `prism/generalize/teacher_inject.py` | Wraps `GroundedAgent` memory mode to emit pickup (3) / toggle (5) when adjacent + facing the goal — solves the BC-teacher gap. |
| `scripts/generalize/collect_bc_multienv.py` | Forked BC collector that iterates over a list of envs and uses `InjectingTeacher`. |
| `scripts/generalize/train_jepa_universal.py` | Trains one JEPA on round-robin data from all target envs. |
| `scripts/generalize/ppo_eval_multi.py` | Loads one policy ckpt and evals it across all target envs sequentially → single comparison table. |
| `scripts/generalize/run_pipeline.sh` | Phase orchestrator (phase0…phase5). |

## What it reuses unchanged

- `prism/agents/grounded_agent.py` — memory teacher
- `prism/agents/pose_tracker.py` — Path B 5-d features
- `prism/models/{jepa.py, recurrent_policy.py}` — same architectures
- `prism/perception/{slots.py, predicates.py}` — env-agnostic
- `prism/envs/babyai.py` — env wrapper
- `scripts/train_recurrent_policy.py` — BC trainer (consumes the new .npz directly, ignores extra `env_ids` field)
- `scripts/ppo_train.py` — PPO (already supports `--env-id`)
- `scripts/eval_agent_cohorts.py` — single-env capstone eval

## Quickstart

```bash
# 1. zero-shot baseline (does v1.3 work on new envs as-is?)
bash scripts/generalize/run_pipeline.sh phase0

# 2. mixed BC data (~30 min)
bash scripts/generalize/run_pipeline.sh phase1

# 3. universal JEPA (~30-60 min)
bash scripts/generalize/run_pipeline.sh phase2

# 4. BC train (~30 min)
bash scripts/generalize/run_pipeline.sh phase3

# 5. per-env PPO from BC warmstart (3 runs, ~3 hr)
bash scripts/generalize/run_pipeline.sh phase4

# 6. multi-env capstone — eval each per-env policy across all envs
bash scripts/generalize/run_pipeline.sh phase5
```

Override defaults via env vars:
```bash
V13_JEPA=runs/<other>/jepa_final.pt RUN_TAG=v2b DEVICE=cuda \
    bash scripts/generalize/run_pipeline.sh all
```

## Verification checkpoints

After phase1, the script auto-prints a per-env action histogram. Confirm:
- GoToLocal / GoTo: actions 0/1/2 dominate, no 3 or 5
- Pickup: action 3 appears (~3-15% of steps)
- Open: action 5 appears (~3-15% of steps)

If 3 or 5 are missing for those envs, the InjectingTeacher isn't firing —
likely because the memory teacher's `p_adjacent` / `p_facing` predicates are
mis-calibrated on the new env's distribution. Inspect a few episodes
manually before training a JEPA on the bad data.

## Known caveats

- **Same-env retraining is not the only baseline.** A more interesting
  control is "v1.3 policy on the universal JEPA" — i.e., swap only the JEPA
  and see if the policy still works. That's a one-line change to phase0:
  pass the universal JEPA instead of the v1.3 one.
- **PPO is per-env, not multi-env.** The plan calls for 3 separate PPO runs
  (one per target env) from the same BC warmstart. Training one PPO across
  mixed envs would require modifying `EnvWorker` to sample env_id per
  worker — left for a follow-up since it touches existing code.
- **Mission encoding stays 24-d.** All four target envs use single-object
  missions, so the existing `(type, color)` one-hot suffices. PutNextLocal
  would need a wider encoding and is intentionally out of scope.
