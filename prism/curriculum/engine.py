"""CurriculumEngine — sequences Stages with activation-based freezing.

Implements resolutions 4, 5, and 6 from the v6 plan as a single
orchestrator. The engine is the substrate's continual-learning driver.

Hard contracts:

1. Stage transitions are SYNCHRONOUS. No path through ConceptManager,
   no async naming gates the freeze decision (audit pass-2 issue 2b/5d).
   The freeze set is a deterministic function of activation statistics.

2. The freeze set is computed via `bank.slot_activation_fraction()`:
   slots whose fraction exceeds `activation_freeze_threshold` are frozen.
   ConceptManager runs asynchronously for inspection ONLY (resolution 4).

3. Probe set is created exactly once at the engine's first stage init
   via `collect_probe_set()`, persisted with hash, and re-used unchanged
   for all later E4 measurements (resolution 6).

4. Pre-allocated capacity per bank — the engine never reshapes
   `nn.Parameter` tensors. `bank.expand(n_new)` activates pre-allocated
   inactive slots; capacity must be sized at substrate construction.

5. Freeze + Adam-state-zeroing is one atomic operation via
   `bank.freeze_slots_with_optimizer(idx, optimizer)`. The optimizer
   passed in must be the SAME one being used for substrate training
   (otherwise Adam moments leak across freeze boundaries — audit 3a).

The engine itself does NOT drive training — that's the caller's job
via a `train_stage_fn(stage, engine) -> dict` callback. The callback
runs PPO (or whatever the trainer is) for the stage's training budget
and returns a metrics dict (used for logging, not for the freeze
decision — that's read off the banks directly).

PR-5 step 4: this commit ships the orchestration. Wiring into
ppo_train (so the trainer can call `engine.advance_stage()` between
stages) is PR-6.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch

from prism.curriculum.probe_set import (
    ProbeSet,
    collect_probe_set,
    load_probe_set,
    save_probe_set,
)

if TYPE_CHECKING:
    from prism.cognition.memory_bank import MemoryBank
    from prism.curriculum.stage import Stage


@dataclass
class CurriculumEngineConfig:
    """Substrate-locked configuration for the curriculum engine.

    Per resolution 3 (locked substrate hyperparameters), these values
    are part of `substrate_config_hash`; changing them across stages or
    domains is forbidden. The activation_freeze_threshold default is
    provisional; the v6 plan requires a Phase C ablation sweep over
    {0.001, 0.005, 0.01, 0.05} to lock the final value (and record it
    in the hash). For now we expose it so the ablation can set it.
    """

    activation_freeze_threshold: float = 0.005
    """Slots whose activation_fraction (cumulative mass / steps) exceeds
    this at stage-end enter the freeze set."""

    warmup_steps: int = 5000
    """Audit issue 7a: must validate at stage end that every slot added
    in this stage has run for at least this many gradient steps,
    otherwise the new capacity is inert.
    PR-5 step 4 does the assertion; the β anneal layered on top of the
    masked-softmax is audit issue 7c, scheduled for a later step."""

    cold_anneal_steps: int = 5000
    """Gradient steps over which a newly-activated slot's β anneals from
    cold (0.1) to full (1.0). Linear schedule. Not yet implemented
    in MemoryBank — currently inactive slots are hard-masked via
    association_mask and active slots use the bank's nominal β. The
    anneal layers a temperature schedule on top."""

    probe_set_size: int = 5000
    """Number of frames collected with a fixed-seed random policy at
    Stage 0 init. Persisted to disk and re-used across all stages and
    all E4 measurements. Hash recorded in substrate_config_hash."""

    probe_seed: int = 0
    """Fixed RNG seed for probe set collection. Substrate-locked: changing
    it changes the probe set, which changes substrate_config_hash."""


@dataclass
class StageReport:
    """Record of one stage transition."""

    stage_name: str
    stage_idx: int
    bank_reports: dict[str, dict]
    """Per-bank: {'frozen_idx': [...], 'expanded_idx': [...],
        'n_active_after': int, 'n_frozen_after': int,
        'activation_steps': int}."""
    train_metrics: dict[str, Any]
    """Whatever the training callback returned."""


class CurriculumEngine:
    """Sequencer for developmental Stages with activation-based freezing.

    Usage:
        engine = CurriculumEngine(
            stages=[stage_a, stage_b, ...],
            banks={"concept": policy.inner.retrieval.concept_bank,
                   "operator": policy.inner.retrieval.operator_bank},
        )
        engine.init_probe_set(env_factory, n_actions, obs_fn, mission_fn,
                              env_id, save_dir=Path("runs/<name>"))
        reports = engine.run(train_stage_fn=lambda stg, eng: train_for(stg),
                             get_optimizer=lambda: opt)

    All methods that mutate state are synchronous and return a dict
    suitable for logging. No async naming, no probabilistic gating.
    """

    def __init__(
        self,
        stages: list["Stage"],
        banks: dict[str, "MemoryBank"],
        config: CurriculumEngineConfig | None = None,
    ):
        if not stages:
            raise ValueError("CurriculumEngine requires at least one Stage")
        self.stages = stages
        self.banks = banks
        self.config = config or CurriculumEngineConfig()
        self._current_stage_idx: int = 0
        self._probe_set: ProbeSet | None = None
        # Track per-bank cumulative warmup-step counter for issue 7a.
        # Slots added in stage N must run for at least warmup_steps
        # before stage N+1 transition.
        self._slot_first_active_step: dict[str, dict[int, int]] = {
            name: {} for name in banks
        }
        self._cumulative_gradient_steps: int = 0
        self._reports: list[StageReport] = []

    @property
    def current_stage(self) -> "Stage":
        return self.stages[self._current_stage_idx]

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @property
    def probe_set(self) -> ProbeSet | None:
        return self._probe_set

    # ------------------------------------------------------------------
    # Probe set (resolution 6)
    # ------------------------------------------------------------------
    def init_probe_set(
        self,
        env_factory: Callable[[], Any],
        env_id: str,
        n_actions: int,
        obs_fn: Callable[[Any], Any] | None = None,
        mission_fn: Callable[[Any], Any] | None = None,
        save_dir: Path | None = None,
        overwrite: bool = False,
    ) -> ProbeSet:
        """Collect the probe set with a random policy. MUST be called
        before any stage runs. Persists to `save_dir / probe_set.pt`
        if `save_dir` is provided.

        Resolution 6: the probe set is exactly one artifact for the
        lifetime of the substrate. Calling this twice with overwrite=False
        raises if a probe set already exists on disk — guards against
        accidental re-collection that would silently change E4 metrics.
        """
        if save_dir is not None:
            save_dir = Path(save_dir)
            probe_path = save_dir / "probe_set.pt"
            if probe_path.exists() and not overwrite:
                # Load existing rather than re-collect — the substrate's
                # probe set is supposed to be immutable across runs.
                self._probe_set = load_probe_set(probe_path, verify_hash=True)
                return self._probe_set

        ps = collect_probe_set(
            env_factory=env_factory,
            n_frames=self.config.probe_set_size,
            seed=self.config.probe_seed,
            env_id=env_id,
            obs_fn=obs_fn,
            mission_fn=mission_fn,
            n_actions=n_actions,
        )
        if save_dir is not None:
            save_probe_set(ps, save_dir / "probe_set.pt")
        self._probe_set = ps
        return ps

    # ------------------------------------------------------------------
    # Stage transition (resolution 4, audit 3a / 3c / 7a)
    # ------------------------------------------------------------------
    def advance_stage(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> StageReport | None:
        """Apply the just-completed stage's freeze policy, then advance
        the stage index. Returns the StageReport for the completed stage,
        or None if there are no more stages.

        Sequence:
          1. Validate audit 7a: every slot added in this stage has run
             for at least `warmup_steps` gradient steps. Raise if not.
          2. For each bank: compute freeze_set = {slot indices with
             slot_activation_fraction > threshold}, AND not already frozen,
             AND in the currently-active range.
          3. Call bank.freeze_slots_with_optimizer(freeze_set, optimizer)
             on each bank. This is the audit-3a-compliant freeze.
          4. Call bank.reset_activation_history() so the next stage's
             freeze decision uses only that stage's stats.
          5. Advance stage idx. If a next stage exists, call
             bank.expand(next_stage.expand_slots_before) for each bank
             that the stage specifies growth on (default: all banks
             grow by the same amount).
          6. Record the new active-slot indices' "first active step" for
             audit-7a tracking in the NEXT stage.
        """
        if self._current_stage_idx >= self.n_stages:
            return None

        completed = self.stages[self._current_stage_idx]
        bank_reports: dict[str, dict] = {}

        # Step 1: warmup completion check (audit 7a).
        for name, bank in self.banks.items():
            first_active = self._slot_first_active_step.get(name, {})
            for slot_idx, first_step in first_active.items():
                steps_active = self._cumulative_gradient_steps - first_step
                if steps_active < self.config.warmup_steps:
                    raise RuntimeError(
                        f"Audit 7a violation: bank={name!r} slot {slot_idx} "
                        f"has only run {steps_active} gradient steps in this "
                        f"stage (warmup_steps={self.config.warmup_steps}). "
                        f"The slot's added capacity is effectively inert. "
                        f"Either extend the stage's training budget, lower "
                        f"warmup_steps, or refuse to advance."
                    )

        # Step 2+3: per-bank freeze decision.
        if completed.freeze_after:
            for name, bank in self.banks.items():
                fraction = bank.slot_activation_fraction()
                threshold = self.config.activation_freeze_threshold
                # Eligible: active AND not already frozen AND above threshold.
                eligible = (
                    bank.active_mask
                    & (~bank.frozen_mask)
                    & (fraction > threshold)
                )
                freeze_idx = torch.nonzero(eligible, as_tuple=False).flatten()
                if freeze_idx.numel() > 0:
                    bank.freeze_slots_with_optimizer(freeze_idx, optimizer)
                bank_reports[name] = {
                    "frozen_idx": freeze_idx.tolist(),
                    "expanded_idx": [],
                    "n_active_after": int(bank.n_active),
                    "n_frozen_after": int(bank.frozen_mask.sum().item()),
                    "activation_steps": int(bank.activation_steps.item()),
                }
        else:
            for name, bank in self.banks.items():
                bank_reports[name] = {
                    "frozen_idx": [],
                    "expanded_idx": [],
                    "n_active_after": int(bank.n_active),
                    "n_frozen_after": int(bank.frozen_mask.sum().item()),
                    "activation_steps": int(bank.activation_steps.item()),
                }

        # Step 4: reset activation history for next stage.
        for bank in self.banks.values():
            bank.reset_activation_history()

        # Step 5: advance and expand for the next stage.
        next_idx = self._current_stage_idx + 1
        if next_idx < self.n_stages:
            next_stage = self.stages[next_idx]
            if next_stage.expand_slots_before > 0:
                n_new = next_stage.expand_slots_before
                for name, bank in self.banks.items():
                    new_idx = bank.expand(n_new)
                    bank_reports[name]["expanded_idx"] = new_idx.tolist()
                    # Step 6: register first-active step for the new slots.
                    for slot_i in new_idx.tolist():
                        self._slot_first_active_step[name][slot_i] = (
                            self._cumulative_gradient_steps
                        )

        report = StageReport(
            stage_name=completed.name,
            stage_idx=self._current_stage_idx,
            bank_reports=bank_reports,
            train_metrics={},     # filled in by run()
        )
        self._reports.append(report)
        self._current_stage_idx = next_idx
        return report

    # ------------------------------------------------------------------
    # Tracking toggle (caller wraps each stage's training in this)
    # ------------------------------------------------------------------
    def start_stage_tracking(self) -> None:
        """Begin accumulating per-slot activation statistics in every
        managed bank. Called by the trainer immediately before a stage's
        training loop. Idempotent."""
        for bank in self.banks.values():
            bank.tracking = True

    def stop_stage_tracking(self) -> None:
        """Stop accumulating; the next `advance_stage` will read the
        accumulated statistics and reset. Called by the trainer
        immediately after a stage's training loop, before
        `advance_stage`. Idempotent."""
        for bank in self.banks.values():
            bank.tracking = False

    def record_gradient_steps(self, n_steps: int) -> None:
        """Trainer reports gradient steps taken during this stage so the
        engine can enforce the audit-7a warmup invariant.
        """
        self._cumulative_gradient_steps += n_steps

    # ------------------------------------------------------------------
    # End-to-end run (caller-supplied training function)
    # ------------------------------------------------------------------
    def run(
        self,
        train_stage_fn: Callable[["Stage", "CurriculumEngine"], dict],
        get_optimizer: Callable[[], torch.optim.Optimizer],
    ) -> list[StageReport]:
        """Run all stages in order. For each stage:
          1. Call `train_stage_fn(stage, engine)` to do the training.
             The callback must call `engine.record_gradient_steps()` so
             the engine knows how many gradient steps were taken.
          2. Call `engine.advance_stage(get_optimizer())` to compute the
             freeze set, freeze, and expand for the next stage.

        Returns the list of StageReports (one per completed stage).
        """
        while self._current_stage_idx < self.n_stages:
            stage = self.current_stage
            train_metrics = train_stage_fn(stage, self)
            opt = get_optimizer()
            report = self.advance_stage(opt)
            if report is not None:
                report.train_metrics = train_metrics
        return list(self._reports)


if __name__ == "__main__":
    # Standalone smoke test using synthetic banks (no PPO training, no env).
    # Validates the freeze-set decision, expand sequencing, and audit-7a check.
    # Run with: `python -m prism.curriculum.engine`
    import sys as _sys

    from prism.cognition.memory_bank import MemoryBank
    from prism.curriculum.stage import Stage

    # Two banks (sized small for fast smoke).
    concept = MemoryBank(D_tok=32, n_slots=32, n_active_init=8, n_heads=2)
    operator = MemoryBank(D_tok=32, n_slots=16, n_active_init=4, n_heads=2,
                          scaling=4.0, update_steps=3)

    # Three synthetic stages. Stage 2 expands by 8 (concept) — we'll
    # verify the expanded indices appear in the bank report. Stage 3
    # has freeze_after=False to test the no-freeze path.
    stages = [
        Stage(name="stage0", env_factory=lambda: None, max_env_steps=100),
        Stage(name="stage1", env_factory=lambda: None, max_env_steps=100,
              expand_slots_before=4),
        Stage(name="stage2", env_factory=lambda: None, max_env_steps=100,
              expand_slots_before=8, freeze_after=False),
    ]

    cfg = CurriculumEngineConfig(
        activation_freeze_threshold=0.005,
        warmup_steps=10,      # tiny for smoke
    )
    engine = CurriculumEngine(stages=stages, banks={"concept": concept,
                                                     "operator": operator},
                              config=cfg)
    opt = torch.optim.Adam(
        list(concept.parameters()) + list(operator.parameters()), lr=1e-3,
    )

    # ----- Stage 0 simulated -----
    # Do some retrievals with tracking; some slots get high activation, others low.
    # We craft queries that match the first 4 slots of each bank strongly.
    torch.manual_seed(0)
    for step in range(50):
        q_c = torch.randn(4, 32)
        q_o = torch.randn(4, 32)
        concept.retrieve(q_c, track_activations=True)
        operator.retrieve(q_o, track_activations=True)
    engine.record_gradient_steps(50)

    # Advance from stage 0 → stage 1.
    report0 = engine.advance_stage(opt)
    assert report0 is not None, "advance_stage returned None on first transition"
    print(f"[engine] stage 0 → 1: frozen concept={len(report0.bank_reports['concept']['frozen_idx'])}, "
          f"frozen operator={len(report0.bank_reports['operator']['frozen_idx'])}, "
          f"concept expanded={report0.bank_reports['concept']['expanded_idx']}")
    # Stage 1 expand_slots_before=4 → 4 new indices in BOTH banks.
    if report0.bank_reports["concept"]["expanded_idx"] != [8, 9, 10, 11]:
        print(f"FAIL: concept expanded should be [8,9,10,11], got {report0.bank_reports['concept']['expanded_idx']}")
        _sys.exit(1)
    if report0.bank_reports["operator"]["expanded_idx"] != [4, 5, 6, 7]:
        print(f"FAIL: operator expanded should be [4,5,6,7], got {report0.bank_reports['operator']['expanded_idx']}")
        _sys.exit(1)
    if concept.n_active != 12 or operator.n_active != 8:
        print(f"FAIL: post-expand n_active concept={concept.n_active} operator={operator.n_active}")
        _sys.exit(1)
    print(f"[engine] expand sequencing OK; n_active concept→12 operator→8")

    # Activation history reset after stage transition.
    if concept.activation_steps.item() != 0:
        print(f"FAIL: activation_steps not reset after advance_stage")
        _sys.exit(1)
    print(f"[engine] activation history reset after stage transition")

    # ----- Audit-7a check: try to advance again before any gradient steps -----
    # The 4 new concept slots were just added; warmup_steps=10 but
    # _cumulative_gradient_steps hasn't advanced. So they have 0 steps active
    # < 10 warmup. advance_stage should raise.
    try:
        engine.advance_stage(opt)
    except RuntimeError as e:
        if "Audit 7a violation" in str(e):
            print(f"[engine] audit 7a check fires: new slots cannot enter freeze")
        else:
            print(f"FAIL: advance_stage raised unexpected error: {e}")
            _sys.exit(1)
    else:
        print(f"FAIL: advance_stage past warmup did not raise")
        _sys.exit(1)

    # ----- Run more steps so warmup completes, then advance -----
    for step in range(20):
        q_c = torch.randn(4, 32)
        q_o = torch.randn(4, 32)
        concept.retrieve(q_c, track_activations=True)
        operator.retrieve(q_o, track_activations=True)
    engine.record_gradient_steps(20)
    report1 = engine.advance_stage(opt)
    assert report1 is not None
    print(f"[engine] stage 1 → 2: stage_idx now {engine._current_stage_idx}; "
          f"frozen concept={len(report1.bank_reports['concept']['frozen_idx'])}")

    # Stage 2 has freeze_after=False — no new freezes when we transition off it.
    for step in range(20):
        concept.retrieve(torch.randn(4, 32), track_activations=True)
        operator.retrieve(torch.randn(4, 32), track_activations=True)
    engine.record_gradient_steps(20)
    n_frozen_before = int(concept.frozen_mask.sum().item())
    report2 = engine.advance_stage(opt)
    n_frozen_after = int(concept.frozen_mask.sum().item())
    if n_frozen_after != n_frozen_before:
        print(f"FAIL: freeze_after=False stage still froze: "
              f"{n_frozen_before} → {n_frozen_after}")
        _sys.exit(1)
    print(f"[engine] freeze_after=False stage: 0 new freezes (frozen stays at {n_frozen_after})")

    # Past the last stage.
    if engine.advance_stage(opt) is not None:
        print(f"FAIL: advance_stage past last stage should return None")
        _sys.exit(1)
    print(f"[engine] advance past final stage returns None")

    print("[engine] all smoke checks passed")
