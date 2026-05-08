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
| `prism/generalize/mission_parser_v2.py` | Extended parser. Phase 0 zero-shot showed 75% of Open-v0 missions failing on the v1 regex (`"open a door"`, `"open the locked door"`, etc.). Wraps the v1 parser and adds tolerant Open patterns; falls through unchanged for everything else. |
| `prism/generalize/pose_tracker_v2.py` | PoseTracker with constructor-tunable normalizations. v1's `n_visited / 30.0` saturates in the larger GoTo / Open rooms — v2 defaults to `80 / 30 / 12` for those envs. Same 5-d feature layout, drop-in compatible. |
| `scripts/generalize/collect_bc_multienv.py` | Forked BC collector that iterates over a list of envs and uses `InjectingTeacher` + `goal_predicates_for_mission_ext`. |
| `scripts/generalize/train_jepa_universal.py` | Trains one JEPA on round-robin data from all target envs. |
| `scripts/generalize/ppo_train_multienv.py` | PPO with per-worker round-robin env assignment. AMAGO-2 / BabyAI++ recipe — one policy across mixed envs beats per-env runs on transfer metrics. |
| `scripts/generalize/ppo_eval_multi.py` | Loads one policy ckpt and evals it across all target envs sequentially → single comparison table. Uses `run_episode_ext` which calls the v2 parser, so Open-v0 isn't dropped on parse fail. |
| `scripts/generalize/run_pipeline.sh` | Phase orchestrator (phase0…phase5, plus phase4b for multi-env PPO). |

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
#    With the v2 parser wired in, Open should now parse ~100% (vs 24%
#    pre-fix). Pickup/Open will still fail at 0% because the policy
#    doesn't know action 3 / 5 — that's what phases 1–4 fix.
bash scripts/generalize/run_pipeline.sh phase0

# 2. mixed BC data (~30 min)
bash scripts/generalize/run_pipeline.sh phase1

# 3. universal JEPA (~30-60 min)
bash scripts/generalize/run_pipeline.sh phase2

# 4. BC train (~30 min)
bash scripts/generalize/run_pipeline.sh phase3

# 5a. per-env PPO from BC warmstart (3 runs, ~3 hr) — the per-env baseline
bash scripts/generalize/run_pipeline.sh phase4

# 5b. (recommended) one multi-env PPO run, ~1.5 hr — the AMAGO-2 / BabyAI++
#     recipe. 16 workers round-robin across 4 envs.
bash scripts/generalize/run_pipeline.sh phase4b

# 6. multi-env capstone — eval each policy (per-env + multi-env) across all envs
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
- **Mission encoding stays 24-d.** All four target envs use single-object
  missions, so the existing `(type, color)` one-hot suffices. PutNextLocal
  would need a wider encoding and is intentionally out of scope.
- **`pose_tracker_v2` is not yet wired into PPO training.** The
  `EnvWorker` in `scripts/ppo_train.py` (and the multi-env trainer that
  imports it) instantiates the v1 `PoseTracker`. To use v2 norms during
  training, the cleanest path is a small subclass of `EnvWorker` that
  swaps in `PoseTrackerV2` — left as a follow-up. The eval path already
  uses v2-compatible logic via `goal_predicates_for_mission_ext`.
