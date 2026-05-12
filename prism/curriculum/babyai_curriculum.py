"""BabyAI developmental curriculum — concrete Stage list for E1.

Three BabyAI levels in increasing cognitive complexity:

  Stage 1 — sensorimotor          BabyAI-GoToObj-v0
    Walk to a single object in an empty-ish room. No distractors,
    no color discrimination. The substrate learns basic
    look-and-move.

  Stage 2 — object recognition    BabyAI-GoToLocal-v0
    Walk to a specific (color, type) object among distractors. The
    substrate must distinguish objects by attributes. Builds on the
    motor primitives from Stage 1.

  Stage 3 — action composition    BabyAI-PickupLoc-v0
    Pick up a specific object at a specific location. Adds the
    "holding" predicate and pickup action (3). Builds on object
    recognition from Stage 2.

These are the same env family (BabyAI / MiniGrid), same observation
shape (3×7×7 grid), same action space (7 discrete), same mission
encoding (color, type one-hot). Only the task structure changes —
which is exactly what an E1 ordering ablation needs to isolate the
"developmental order" signal from confounds.

The probe set is always collected on the CANONICAL env (default:
BabyAI-GoToLocal-v0) so E4 stability measurements are comparable
across curriculum-order arms (forward / reverse / shuffled). Without
this, different orderings would be measured on different probe
distributions, making the comparison meaningless.
"""

from __future__ import annotations

from typing import Any, Callable

from prism.curriculum.stage import Stage


# Canonical probe env for BabyAI curricula. Probe set is collected
# once on this env (with a random policy on a fixed seed) and reused
# across all curriculum-order arms.
BABYAI_PROBE_ENV_ID = "BabyAI-GoToLocal-v0"


def _make_env_factory(env_id: str) -> Callable[[], Any]:
    """Build a closure that constructs a fresh env for that level.
    The closure is what stage.env_factory holds; the trainer calls it
    once per worker on stage entry.
    """
    def _factory():
        from prism.envs.babyai import make_env_with_max_steps
        return make_env_with_max_steps(env_id, max_steps=64)
    return _factory


def build_babyai_developmental_curriculum(
    per_stage_env_steps: int = 167_000,
    expand_slots_per_transition: int = 0,
) -> list[Stage]:
    """Three-stage forward curriculum on BabyAI levels.

    Parameters
    ----------
    per_stage_env_steps : how many env steps each stage gets.
        Matched across stages so position-in-curriculum doesn't
        confound budget. Default 167k × 3 = ~500k total, matching
        the Phase B compute budget.
    expand_slots_per_transition : if > 0, bank.expand(N) fires
        before each stage AFTER the first. Default 0 (no growth
        for E1's first pass — keeps the comparison clean).

    Returns the forward-ordered list. Use `reorder_curriculum` to
    produce reverse / shuffled variants for E1's ablation arms.
    """
    stages = [
        Stage(
            name="sensorimotor",
            env_factory=_make_env_factory("BabyAI-GoToObj-v0"),
            max_env_steps=per_stage_env_steps,
            freeze_after=True,
            expand_slots_before=0,
            extra={"env_id": "BabyAI-GoToObj-v0", "cognitive_level": "motor"},
        ),
        Stage(
            name="object_recognition",
            env_factory=_make_env_factory("BabyAI-GoToLocal-v0"),
            max_env_steps=per_stage_env_steps,
            freeze_after=True,
            expand_slots_before=expand_slots_per_transition,
            extra={"env_id": "BabyAI-GoToLocal-v0", "cognitive_level": "perceptual"},
        ),
        Stage(
            name="action_composition",
            env_factory=_make_env_factory("BabyAI-PickupLoc-v0"),
            max_env_steps=per_stage_env_steps,
            # Final stage: no freeze (no next stage to gate).
            freeze_after=False,
            expand_slots_before=expand_slots_per_transition,
            extra={"env_id": "BabyAI-PickupLoc-v0", "cognitive_level": "goal"},
        ),
    ]
    return stages


def reorder_curriculum(stages: list[Stage], order: str) -> list[Stage]:
    """Apply a permutation to the forward stage list.

    `order` ∈ {"forward", "reverse", "shuffled"}:
      - "forward": identity.
      - "reverse": flip the list. Forces a structurally-later stage
        to be trained first.
      - "shuffled": deterministic Fisher-Yates with seed=0. Same
        permutation across runs so the comparison is reproducible.

    Critically: the final stage's `freeze_after` is forced to False
    in the original construction, but after reordering the new final
    stage might have `freeze_after=True`. We re-clear it here so the
    last stage never tries to freeze (no next stage exists).
    """
    if order == "forward":
        new_stages = list(stages)
    elif order == "reverse":
        new_stages = list(reversed(stages))
    elif order == "shuffled":
        import random
        rng = random.Random(0)
        new_stages = list(stages)
        rng.shuffle(new_stages)
    else:
        raise ValueError(
            f"unknown curriculum order {order!r}; "
            f"expected 'forward', 'reverse', or 'shuffled'"
        )
    # Re-clear final stage's freeze_after (dataclass is frozen, must
    # rebuild via dataclasses.replace).
    from dataclasses import replace
    if new_stages[-1].freeze_after:
        new_stages[-1] = replace(new_stages[-1], freeze_after=False)
    return new_stages


# Registry of named curricula. New curricula register here; the
# trainer looks them up by `--curriculum` flag.
CURRICULUM_REGISTRY: dict[str, Callable[[], list[Stage]]] = {
    "babyai_developmental": build_babyai_developmental_curriculum,
}


def get_curriculum(name: str, **kwargs) -> list[Stage]:
    """Look up a curriculum builder by name. kwargs are forwarded
    to the builder (e.g., `per_stage_env_steps`)."""
    if name not in CURRICULUM_REGISTRY:
        raise ValueError(
            f"unknown curriculum {name!r}; "
            f"registered: {list(CURRICULUM_REGISTRY.keys())}"
        )
    return CURRICULUM_REGISTRY[name](**kwargs)


if __name__ == "__main__":
    # Standalone smoke test: build curriculum, reorder it, inspect.
    # Run with: `python -m prism.curriculum.babyai_curriculum`
    import sys as _sys

    fwd = build_babyai_developmental_curriculum(per_stage_env_steps=10_000)
    if len(fwd) != 3:
        print(f"FAIL: expected 3 stages, got {len(fwd)}")
        _sys.exit(1)
    print(f"[bcurr] forward: {[s.name for s in fwd]}")

    rev = reorder_curriculum(fwd, "reverse")
    if [s.name for s in rev] != ["action_composition", "object_recognition", "sensorimotor"]:
        print(f"FAIL: reverse order wrong: {[s.name for s in rev]}")
        _sys.exit(1)
    if rev[-1].freeze_after:
        print(f"FAIL: reverse curriculum final stage has freeze_after=True")
        _sys.exit(1)
    print(f"[bcurr] reverse: {[s.name for s in rev]} "
          f"(final freeze_after={rev[-1].freeze_after})")

    shuf = reorder_curriculum(fwd, "shuffled")
    if shuf[-1].freeze_after:
        print(f"FAIL: shuffled curriculum final stage has freeze_after=True")
        _sys.exit(1)
    print(f"[bcurr] shuffled: {[s.name for s in shuf]}")

    # Determinism: same shuffle seed → identical order.
    shuf2 = reorder_curriculum(fwd, "shuffled")
    if [s.name for s in shuf] != [s.name for s in shuf2]:
        print(f"FAIL: shuffled order not deterministic across calls")
        _sys.exit(1)
    print(f"[bcurr] shuffled is deterministic across calls")

    # Registry lookup.
    c = get_curriculum("babyai_developmental", per_stage_env_steps=5_000)
    if c[0].max_env_steps != 5_000:
        print(f"FAIL: kwargs not forwarded through registry")
        _sys.exit(1)
    print(f"[bcurr] get_curriculum() forwards kwargs correctly")

    # env_factory should be callable but we don't actually run it (no env deps).
    if not callable(fwd[0].env_factory):
        print(f"FAIL: env_factory not callable")
        _sys.exit(1)
    print(f"[bcurr] env_factory callables present")

    print("[bcurr] all smoke checks passed")
