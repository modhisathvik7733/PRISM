# PRISM

**An adaptive cognition runtime that grounds language to action compositionally.**

PRISM is a small, end-to-end-trained cognition stack: a JEPA-style latent
world model, a behavioral-anchor operator bank with anti-drift, a learned
predicate readout, and a policy that takes language-predicted goals and
acts in an environment. Every component is designed to be **domain-general**;
games are the first proving ground, not the destination.

The headline scientific claim — *language understanding emerges from
operators bound to predicted state transitions, with compositional
generalization at every layer of the stack* — is now empirically
defended on BabyAI. Full per-experiment write-up:
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

---

## v4.x results — the compositional grounding stack

| Layer | What was tested | Result |
|---|---|---:|
| **v4.0** | JEPA dev-curriculum; cross-env operator stability (K-means baseline) | mean cosine 0.45-0.56 |
| **v4.1** | OperatorBankV3 with EMA + anchors + replay | cross-env stability 0.50 → **0.80** |
| **v4.1.1** | JEPA factored-aux loss; compositional perception (policy-confounded) | held-out joint 6.9% → 22.5% |
| **v4.1.2** | Re-measured with `--require-goal-visible` filter | **53.6%** held-out joint, no compositional gap |
| **v4.1.3** | PPO trained with `--goal-source lang` (language-predicted goals) | **bit-identical** to rule-parser baseline |
| **v4.1.4** | PPO with 4 (color, type) combos held out from training | **47.5%** held-out vs **57.9%** ID — **82% (PASS)** |

The full stack composes: a 749k-param JEPA + 39k-param text head +
725k-param recurrent policy, ~$1 of compute total, language→action
compositional generalization on BabyAI go-to missions.

## What PRISM is (and isn't)

PRISM is **a domain-general adaptive cognition runtime**, not game-specific
AI. Games are the first interactive, measurable, easy-to-ground environment
where the architecture can be validated. The same cognition substrate is
intended to operate across:

- Games (current proving ground; BabyAI go-to envs validated)
- Coding environments (observation = code graphs / compiler output; action = edit / refactor / test)
- Simulations & robotics (observation = sensors / spatial state; action = movement / manipulation)
- Workflow / interactive agents (observation = app state; action = UI interactions)

Domain changes modify the environment interface, not the cognition system.
The core primitives — memory, operators, planning, prediction, semantic
grounding, continual learning, abstraction formation — stay domain-agnostic.

What PRISM is *not*:
- A scaled-up next-token language model
- A game-AI middleware
- An NPC framework

What it is:
- A grounded world-model architecture
- A continual-learning, neurosymbolic-flavored cognition substrate
- A research-quality implementation that fits on a single 24-48 GB GPU
  and trains in minutes-to-tens-of-minutes per layer

Full design rationale, phased build order, and falsifiable success criteria:
[`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## Phase status

| Phase | Goal                                              | Status      |
| ----- | ------------------------------------------------- | ----------- |
| 0     | Substrate + sanity baseline (BabyAI + PPO)        | done        |
| 1     | JEPA + factored aux + counterfactual              | **done (v4.1.1 / v4.1.2)** |
| 2     | Operators with cross-env stability (V3 + anti-drift) | **done (v4.1)** |
| 3     | Episodic ⇄ semantic memory (predictive coding)    | scaffolded; deferred |
| 4     | Language→policy compositional generalization      | **done (v4.1.3 / v4.1.4)** |
| 5     | Transfer to video / 3D sim-body                   | not started — next major axis |
| 6     | Cross-domain adapter (code, workflow, robotics)   | not started — long-term |

Phases 1, 2, and 4 together close the loop on the v4.x scientific
question: can a small, end-to-end-trained cognition stack learn
language→action compositional generalization on a structured environment?
**Yes.** Phases 5 and 6 are scale + transfer experiments, separate from the
compositional-grounding thesis.

---

## Reproducing v4.1.4 end-to-end (~30 minutes on a single 24 GB GPU)

```bash
# 1. Train dev-curriculum JEPA with factored aux (~6 min)
python -m scripts.cog_core.train_jepa_developmental \
    --aux-factored-weight 1.0 --bf16 --compile \
    --run-name jepa_dev_v1_factored --device cuda

# 2. Collect rollouts via JEPA + random policy (~5 min)
python -m scripts.cog_core.collect_rollouts \
    --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \
    --random-policy --episodes-per-env 1000 \
    --output runs/cog_core_phase1_factored/rollouts.npz --device cuda

# 3. Train text → (color, type) classifier (~30 sec)
python -m scripts.lang.train_grounding_floor \
    --rollouts runs/cog_core_phase1_factored/rollouts.npz \
    --kind tiny_tf --run-name grounding_floor_tt_clean --device cuda

# 4. Train PPO with language-predicted goals, 4 combos held out (~15 min)
python -m scripts.ppo_train \
    --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \
    --no-bc --goal-source lang \
    --lang-checkpoint runs/grounding_floor_tt_clean/grounding_floor_final.pt \
    --vocab-checkpoint runs/grounding_floor_tt_clean/vocab.pt \
    --held-out-combos 0,1 3,1 3,3 5,2 \
    --total-steps 500000 \
    --run-name ppo_stage1_3_lang_heldout --device cuda

# 5. Eval compositional generalization (~3 min)
python -m scripts.eval_lang_policy_compositional \
    --policy-checkpoint runs/ppo_stage1_3_lang_heldout/policy_final.pt \
    --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \
    --lang-checkpoint runs/grounding_floor_tt_clean/grounding_floor_final.pt \
    --vocab-checkpoint runs/grounding_floor_tt_clean/vocab.pt \
    --held-out-combos 0,1 3,1 3,3 5,2 \
    --episodes-per-combo 30 --device cuda
```

Expected output of step 5: held-out aggregate ~47%, ID ~58%, **PASS verdict**.

---

## Running on Vast.ai

PRISM is designed to be developed locally and run on a rented Vast.ai GPU.
The local repo is purely source code — no environment is installed locally.

### 1. Pick a Vast.ai instance

Phase 0–4 (BabyAI gridworld) is *tiny*. Don't overpay:

| Phase | Recommended GPU                              | Why                                              |
| ----- | -------------------------------------------- | ------------------------------------------------ |
| 0–4   | RTX 3090 / 4090 (24 GB)                      | BabyAI obs is 7×7×3; total params ~30–50M.       |
| 5     | RTX 5090 (32 GB) or H100 (80 GB)             | V-JEPA video / 3D sim needs the headroom.        |

Use a PyTorch-ready Ubuntu 22.04 image — most Vast.ai "PyTorch" templates qualify.

### 2. Clone + setup

```bash
git clone https://github.com/<your-user>/PRISM.git
cd PRISM
bash setup.sh                  # auto-detects GPU, picks the right CUDA wheel
# or force a CUDA wheel:
CUDA=cu128 bash setup.sh       # for RTX 5090 (Blackwell)
CUDA=cu124 bash setup.sh       # for RTX 4090 / A100
```

`setup.sh` will:
1. Install `uv` (if missing).
2. Create `.venv` with Python 3.11.
3. Install base deps (`minigrid`, `gymnasium`, `tensorboard`, …).
4. Install `torch` from the right CUDA index.
5. Run a sanity check.

### 3. Phase 0: smoke test + baseline

After `setup.sh`, activate the project venv (so commands resolve to it
instead of any pre-existing `/venv/main` that the Vast.ai image may have
shipped with):

```bash
source .venv/bin/activate
```

Then:

```bash
# Smoke test — confirms env loads, GPU works, JEPA forward pass runs.
python -m scripts.smoke_test

# Phase 0 baseline — vanilla PPO on the simplest BabyAI level.
# Falsifier: mean reward stays ~0 after 1M steps → harness is broken.
python -m scripts.train_ppo_baseline \
    --env-id BabyAI-GoToLocal-v0 \
    --total-timesteps 1_000_000 \
    --n-envs 8 \
    --device cuda
```

TensorBoard is written under `runs/<run-name>/tb`.

### 4. Phase 1: JEPA pretraining

```bash
python -m scripts.train_jepa \
    --env-id BabyAI-GoToLocal-v0 \
    --steps 200_000 \
    --batch-size 128 \
    --device cuda
```

Checkpoints saved every 10k steps under `runs/<run-name>/`.

### Troubleshooting: "ModuleNotFoundError: No module named 'torch'"

Means `uv` installed torch into a different venv than the one your script is
running from. Fix:

```bash
unset VIRTUAL_ENV
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio
```

(Use `cu124` instead of `cu128` for RTX 30/40-series; `cu126` for H100.)

---

## Repo layout

```
PRISM/
├── README.md
├── setup.sh                    # one-shot Vast.ai bootstrap
├── pyproject.toml              # uv-managed (torch installed separately)
├── docs/
│   └── ROADMAP.md              # full plan + falsifiers per phase
├── prism/
│   ├── envs/
│   │   └── babyai.py           # MiniGrid/BabyAI env factory + obs wrappers
│   ├── models/
│   │   ├── jepa.py             # encoder + EMA target + latent dynamics
│   │   ├── counterfactual.py   # counterfactual prediction loss (Phase 1)
│   │   ├── operators.py        # 12 seeded operators (Phase 2)
│   │   └── memory.py           # episodic + semantic stubs (Phase 3)
│   ├── losses/
│   │   └── consistency.py      # self-model consistency (Phase 1)
│   └── utils/
│       └── seed.py
└── scripts/
    ├── smoke_test.py           # Phase 0 sanity
    ├── train_ppo_baseline.py   # Phase 0 PPO baseline
    └── train_jepa.py           # Phase 1 JEPA pretraining
```

---

## Design principles

These are codified in `docs/ROADMAP.md`; read it first if you're contributing.

1. **Latent prediction, not pixel prediction.** JEPA-style: predict embeddings,
   not images. Pixel reconstruction wastes capacity on irrelevant detail.
2. **Counterfactuals from day one.** `predict_counterfactual(z_t, a' ≠ a_t)`
   trained alongside factual prediction. Forces causal structure, not
   trajectory fitting.
3. **Self-model consistency.** Predicates can't flip across time unless an
   executed operator licensed it. Strong inductive bias toward grounded
   semantics.
4. **Seeded operators, then discovery.** 12 hand-defined primitives bootstrap
   the symbolic layer. Refinement / merging / discovery layered on top —
   never starting from zero.
5. **Active curiosity, not just passive prediction.** Active-inference EFE
   with the epistemic term *on*: the agent prefers experiences that reduce
   uncertainty over latents and operator effects.
6. **Falsifiable per phase.** Every phase has a result that, if it fires,
   means we stop and rethink rather than push through.

---

## License

TBD.
