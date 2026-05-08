---
plan: tiny-grounded-predictive-semantic-system
goal: runnable prototype, months-scale, gridworld → video sequence
---

# A Better Roadmap: Tiny Grounded Predictive Semantic System

## Context

Your original framing is right but under-specified. The core thesis — *don't start by scaling language; start by building a small grounded predictive semantic system correctly* — is now actually defensible (not just contrarian) because of three things that converged in 2024–2026:

1. **JEPA / V-JEPA 2 (Meta, 2025)** showed that predicting in *latent embedding space* rather than pixel space is a workable substrate for world models that can plan in physical environments. ([V-JEPA 2 announcement](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/))
2. **LeWorldModel (2026)** showed JEPA can be trained stably end-to-end at **~15M parameters on a single GPU in hours**, planning 48× faster than foundation-model world models. ([LeWorldModel](https://le-wm.github.io/)) — this is the existence proof that "tiny, grounded, predictive" is viable.
3. **Predictive-coding models of episodic→semantic memory (arxiv 2509.01987, 2025)** give a biologically plausible mechanism for *consolidation* — the missing piece between "the agent had an experience" and "the agent now knows something general."

So a sharper version of your bet:

> Combine a **JEPA-style latent world model** (LeWorldModel scale) with an **episodic→semantic predictive-coding memory** and **neurosymbolic operator abstraction** (VisualPredicator-style), trained under a **BabyAI-style language curriculum**, in an **active-inference action loop**. Language understanding emerges from operators bound to predicted state transitions — not from imitating a corpus.

The original outline lists six pillars side-by-side. The better plan is to recognize they form a **stack** with a build order, and that several pieces have existing implementations to borrow from rather than invent.

## Critique of the Original Six Pillars

| Pillar (yours)              | Keep / Sharpen / Drop                                                                                                                                                       |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Grounded meaning            | **Keep.** But specify the substrate: BabyAI gridworld first, then transfer to video / MIMo-v2 sim-body. ([BabyAI](https://github.com/mila-iqia/babyai), [MIMo](https://www.tandfonline.com/doi/full/10.1080/01691864.2023.2225232)) |
| Semantic memory             | **Sharpen.** It's not one thing — it's a *consolidation pipeline* from episodic traces (sparse, hippocampal-style) to dense semantic embeddings, learned bidirectionally per the 2025 predictive-coding-of-neocortex result. ([arxiv 2509.01987](https://arxiv.org/pdf/2509.01987)) |
| Curriculum learning         | **Keep but borrow.** BabyAI already ships 19 levels of compositional language difficulty. Don't design a curriculum from scratch — use it and extend at the top end.        |
| Predictive world model      | **Sharpen.** Specifically a JEPA in latent space, not a pixel-space world model. Pixel reconstruction wastes capacity on irrelevant detail (this is the V-JEPA argument).   |
| Operator abstraction        | **Sharpen + scaffold.** Do *not* attempt fully unsupervised operator discovery from day one — that path eats months. Seed with **10–20 manually-defined primitive operators** (`move`, `touch`, `push`, `contain`, `transfer`, `break`, `increase`, `decrease`, `pickup`, `drop`, …) and let the system *refine, merge, abstract, and eventually discover* new operators on top of that scaffold. VisualPredicator (ICLR 2025) is still the closest prior art for the discovery phase. ([VisualPredicator](https://openreview.net/forum?id=QOfswj7hij)) |
| Language understanding (vs imitation) | **Keep, and make falsifiable.** "Understanding" must be operationalized: agent must succeed on novel compositions of known operators, not just memorized commands. BabyAI's compositional-generalization split is the lever. |

What was **missing**:

- **Action-selection mechanism.** A predictive world model isn't an agent — you need a controller. Active inference (free-energy minimization over predicted trajectories) is the principled fit and has tutorial-grade implementations. ([Active Inference tutorial](https://www.sciencedirect.com/science/article/pii/S0022249621000973))
- **Counterfactual prediction, from the start.** Not "what *will* happen if I do `a_t`" but *also* "what *would* have happened if I had done `a'_t` instead?" This is the lever for causal structure, planning, abstraction, and the difference between correlation-fitting and understanding. Humans rely on counterfactuals heavily; current world-model literature mostly doesn't. We add this in Phase 1, not later.
- **Self-model consistency loss.** A coherent agent must maintain a persistent world state: if `cup inside box` is asserted at `t`, then at `t+k` the cup cannot simultaneously be elsewhere unless an operator moved it. We add an explicit consistency penalty over the predicate readout across time — a strong inductive bias toward grounded semantics rather than free-floating embeddings.
- **Active curiosity / intrinsic motivation.** Passive prediction trains a good observer, not a good learner. The controller should *prefer* trajectories that reduce posterior uncertainty over latents/operators and that expand the set of predicates it can confidently read out. This is already the epistemic-value half of active inference's expected free energy — we just have to actually use it instead of zeroing it out.
- **Falsifiable milestones.** "Understands language" is unfalsifiable as stated. Below, every phase gets a quantitative pass/fail.
- **Transfer claim.** The whole point of building tiny is to test whether the *same* agent core transfers from gridworld to a 3D / video setting. If it doesn't, the architecture is wrong, not just under-scaled.

## Recommended Architecture (Concrete)

```
                  ┌─────────────────────────────────────────┐
                  │  Language curriculum (BabyAI 19 levels) │
                  └────────────────┬────────────────────────┘
                                   │ instructions
                  ┌────────────────▼────────────────┐
                  │  Operator/predicate layer       │  ← VisualPredicator-style
                  │  (symbolic plan in latent       │     joint learning of
                  │   abstract state)               │     predicates + ops
                  └──────┬─────────────────▲────────┘
                         │ subgoal         │ predicate readout
                         ▼                 │
        ┌───────────────────────────────────┴─────┐
        │  JEPA latent world model (LeWM-style)   │
        │  s_t → ŝ_{t+1} | a_t,  in embedding     │
        │  space; ~15M params                     │
        └──────┬───────────────────────▲──────────┘
               │ predicted latents     │ encoded obs
               ▼                       │
        ┌──────────────────────────────┴──────────┐
        │  Episodic buffer  ⇄  Semantic memory    │
        │  (sparse traces)     (predictive-coding │
        │                       consolidation)     │
        └──────────┬──────────────────────────────┘
                   │
                   ▼
        ┌─────────────────────────────┐
        │  Active-inference controller│
        │  (EFE minimization over     │
        │   predicted trajectories)   │
        └──────────┬──────────────────┘
                   │ action a_t
                   ▼
              Environment (BabyAI → MIMo / V-JEPA video)
```

**Why this composition and not just scaling an LLM:** every arrow above is a mechanism with formal grounding (predictive coding, free energy, compositional planning). An LLM trained to imitate text has none of these arrows — it has one giant function from tokens to tokens, and the things you want (causal reasoning, novel-composition generalization, action-grounded semantics) are exactly what it doesn't reliably do, per the 2025 Frontiers review on multimodal LLMs and deep understanding. ([Frontiers 2025](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2025.1683133/full))

## Build Order (Phased, Months-Scale)

Each phase has: **deliverable**, **falsifier** (what result would mean the phase failed and we should rethink), and a rough time band. Treat the bands as commitment devices, not estimates.

### Phase 0 — Substrate (week 1)
- Stand up BabyAI + MiniGrid; reproduce a published baseline (PPO on `BabyAI-GoToLocal`) to confirm the harness works.
- Stand up logging/eval pipeline (sample-efficiency curves on BabyAI's standard splits).
- **Falsifier:** can't reproduce a known baseline → the harness is wrong before we add anything.

### Phase 1 — JEPA world model + counterfactual head + consistency (weeks 2–5)
- Implement a LeWorldModel-style JEPA: encode partial observations into latents, predict next latent from `(z_t, a_t)`. Base losses: next-embedding prediction + Gaussian regularizer on latents.
- **Counterfactual head.** Same predictor, but train it to also produce `ẑ_{t+1} | (z_t, a')` for sampled alternative actions `a' ≠ a_t`. Cheap during training (extra forward passes), and it forces the dynamics model to encode *action-conditioned causal structure* rather than just the realized trajectory.
- **Self-model consistency loss.** Reserve a small set of slot-style predicate readouts (`contains(x,y)`, `at(x,loc)`, etc.). Penalize predicate flips across time that aren't licensed by an executed operator. Initially these readouts use the seeded primitives from Phase 2's operator scaffold; consistency pressure couples Phase 1 and Phase 2 from the start.
- Train on random-policy rollouts; evaluate next-state prediction error, **counterfactual prediction error** on held-out alternative actions, and **rollout consistency at horizon ≥ 16**.
- **Falsifier:** rollout drift dominates by horizon 4, *or* counterfactual error collapses to mean-prediction → JEPA losses aren't sufficient at this scale (may need VICReg-style regularization), or the counterfactual head is shortcut-learning the marginal action distribution.

### Phase 2 — Seeded operators → discovery (weeks 6–9)
- **Start with a hand-defined operator library of 10–20 primitives** chosen to span the BabyAI verb space and physical-interaction core: `move`, `touch`, `push`, `pickup`, `drop`, `open`, `close`, `contain`, `transfer`, `break`, `increase`, `decrease`, plus a handful of relational predicates (`at`, `near`, `holding`, `inside`). Each operator gets a typed precondition/effect signature in predicate space.
- Bind operators to BabyAI's instruction templates (`go to X` ↔ `move(agent, X)`, `pick up X` ↔ `pickup(agent, X)`, …). This is the language-grounding entry point — instructions don't need to be parsed by an LLM; they index operators.
- Plan in predicate space using these seeded operators; refine plan to actions via a small policy net (the neuro-symbolic imitation-learning split).
- **Only after** the seeded system reaches baseline competence: enable *refinement* (operator parameter learning), *merging* (collapsing operators that share preconditions+effects), and finally *discovery* (proposing new operators when the planner repeatedly fails). The operator-count regularizer (arxiv 2503.21406) prevents discovery from exploding.
- **Falsifier:** seeded system can't even solve BabyAI's lower levels → the predicate readouts from Phase 1 don't carry enough state, revisit the encoder/predicate-head design before touching discovery. Discovery itself is judged by whether merged/discovered operators *survive* over training and *transfer* in Phase 5 — discovery that produces ephemeral operators is just overfitting.

### Phase 3 — Episodic→semantic memory (weeks 10–13)
- Episodic buffer: sparse, pattern-separated traces of `(z_t, a_t, z_{t+1}, instruction)`.
- Semantic store: dense, slowly consolidated via a predictive-coding update rule (the bidirectional version from the 2025 neocortex paper).
- During action selection, controller can query semantic memory for "what usually happens when…" and episodic for "have I been here before?"
- **Falsifier:** semantic memory adds nothing on top of the JEPA latent — i.e., performance unchanged on tasks requiring recall of rare events from earlier in training. If so, the consolidation rule is wrong or the JEPA is already memorizing too much.

### Phase 4 — Language curriculum + active inference + curiosity (weeks 14–18)
- Walk BabyAI's 19 levels with active-inference action selection: minimize expected free energy (EFE) over JEPA-predicted trajectories conditioned on the current instruction.
- **Active curiosity is not optional.** Use the full EFE = (pragmatic value) + (epistemic value), with the epistemic term explicitly rewarding trajectories that reduce posterior uncertainty over latents *and* over operator effects (which preconditions actually fire, which discovered operators are still uncertain). This is what makes the agent prefer experiences that *expand operator understanding* rather than grinding the same successful trajectory.
- Counterfactual rollouts from Phase 1 are reused at planning time: at each decision point the controller evaluates `EFE(a)` over candidate actions using the counterfactual head, not just the realized policy. This couples planning and counterfactual reasoning.
- **Falsifier (the big one):** agent solves training-distribution levels but **fails BabyAI's compositional-generalization split** — i.e., it's still imitating, not understanding. If this happens, either operators aren't being learned/used as compositional units (revisit Phase 2's discovery rules), or the curiosity term has collapsed and the agent never explored novel operator compositions (check epistemic-value contribution to EFE in logs).

### Phase 5 — Transfer to video / 3D (weeks 19–24)
- Swap the gridworld encoder for a V-JEPA-2-style video encoder, or wire the agent core into MIMo-v2's simulated body. **Architecture stays the same** — only the perceptual front-end changes.
- Test: does the predicate/operator layer trained in gridworld provide *any* speedup on a small set of analogous tasks in the new substrate?
- **Falsifier:** zero positive transfer → the predicates we learned were gridworld artifacts, not abstractions. Either (a) the gridworld curriculum was too narrow, or (b) predicates need to be conditioned on the encoder, not absolute. This is the most informative possible negative result — it tells us where grounding actually lives.

## What Success Looks Like (Falsifiable)

The prototype is a success if and only if:

1. **Sample efficiency:** matches or beats published BabyAI baselines on the lower 12 levels with comparable or fewer environment steps. Curiosity-driven exploration should make this *better*, not worse — if it doesn't, the epistemic term isn't doing real work.
2. **Compositional generalization:** ≥ 70% on BabyAI's held-out compositional split (vs. typical ~30–50% for imitation-trained agents at this scale).
3. **Counterfactual accuracy:** held-out counterfactual prediction error within 1.5× of factual prediction error. If the gap is much larger, the model is fitting trajectories rather than dynamics.
4. **Consistency:** measurable drop in spurious predicate flips (cup-not-licensed-to-have-moved-yet-moved) when the consistency loss is enabled vs. ablated.
5. **Recall:** semantic memory measurably helps on a held-out task that requires information seen 100k+ steps earlier.
6. **Transfer (the real test):** non-trivial positive transfer of the operator layer to the video / sim-body substrate.

If we hit (1)–(5) but fail (6), we have a useful gridworld result, not a theory of grounded semantics. Be honest about that.

## Risks / Open Questions

- **Predicate discovery is the hard part.** Most published neurosymbolic work either pre-specifies predicates or discovers very few. Joint discovery from scratch may need careful inductive bias.
- **Active inference at scale is finicky.** Tutorials work in cartpole-scale POMDPs; scaling EFE computation to longer horizons typically requires amortization tricks. Budget time for this.
- **"Tiny" might still be too small.** LeWorldModel is 15M params for a controlled-physics setting. BabyAI is structurally simpler but has language; budget for ~30–50M total across all components.
- **Measurement risk:** BabyAI compositional splits have known ceilings under standard methods. If we hit the ceiling, that's evidence *for* the approach, not just a number.

## Critical References

- [LeWorldModel (le-wm.github.io)](https://le-wm.github.io/) — the architectural template for the JEPA piece.
- [V-JEPA 2 / I-JEPA blog](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/) — the underlying philosophy and evidence base.
- [VisualPredicator (ICLR 2025)](https://openreview.net/forum?id=QOfswj7hij) — joint predicate+operator learning, the Phase 2 template.
- [Neuro-symbolic imitation learning (arxiv 2503.21406)](https://arxiv.org/html/2503.21406) — operator-count regularization that prevents predicate collapse.
- [Semantic+episodic memory in predictive-coding neocortex (arxiv 2509.01987)](https://arxiv.org/pdf/2509.01987) — Phase 3 mechanism.
- [BabyAI platform](https://github.com/mila-iqia/babyai) — Phases 0, 1, 4 substrate.
- [MIMo developmental sim-body](https://www.tandfonline.com/doi/full/10.1080/01691864.2023.2225232) — Phase 5 transfer target option.
- [Active inference tutorial (Smith et al.)](https://www.sciencedirect.com/science/article/pii/S0022249621000973) — Phase 4 controller, with worked code.
- [Multimodal LLMs and deep understanding (Frontiers 2025)](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2025.1683133/full) — the negative case for the "just scale language" alternative; useful framing for any writeup.

## Verification (How We Know Each Phase Worked End-to-End)

- Phase 0: `python -m babyai.scripts.train_rl` on `BabyAI-GoToLocal` reproduces published reward curve within noise.
- Phase 1: held-out next-latent prediction MSE plotted vs. rollout horizon; compare to a pixel-space baseline of equal capacity.
- Phase 2: ablate the symbolic layer, expect drop on tasks that require multi-step planning.
- Phase 3: ablate semantic memory, expect drop only on tasks requiring long-range recall.
- Phase 4: full-system run through BabyAI levels 1–12; held-out compositional split as the primary headline number.
- Phase 5: zero-shot and few-shot transfer numbers in the new substrate, with the operator layer frozen vs. fine-tuned.

End every phase with a one-page writeup: what was tried, what worked, what falsifier (if any) fired, and what the next phase actually needs to look like in light of the result. Resist the urge to skip ahead.
