# PRISM

**Tiny grounded predictive semantic system.**

Don't start by scaling language. Start by building a small grounded predictive
semantic system correctly: a JEPA-style latent world model + episodic→semantic
memory + neurosymbolic operator abstraction, trained under a BabyAI-style
language curriculum, in an active-inference action loop.

Full design rationale, phased build order, and falsifiable success criteria:
[`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## Phase status

| Phase | Goal                                              | Status      |
| ----- | ------------------------------------------------- | ----------- |
| 0     | Substrate + sanity baseline (BabyAI + PPO)        | scaffolded  |
| 1     | JEPA + counterfactual + consistency               | scaffolded  |
| 2     | Seeded operators → refinement → discovery         | seed defined |
| 3     | Episodic ⇄ semantic memory (predictive coding)    | stub        |
| 4     | Curriculum + active inference + curiosity         | not started |
| 5     | Transfer to video / 3D sim-body                   | not started |

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
