"""Phase A gate — refactor with no behavioral change.

Pre-conditions for Phase A to be considered passed:
  1. The new substrate packages import without error.
  2. A `BabyAIAdapter` exists and satisfies the `DomainAdapter` Protocol.
  3. `UniversalPolicy.from_adapter(adapter)` builds a policy whose
     `step_with_value` signature is backwards-compatible with v5's
     HybridPolicy.
  4. A short PPO training run (~50k env steps) on
     `BabyAI-GoToLocal-v0` reaches reward within ±1% of the v5 baseline
     at the same step count.

This module exposes a single entry point, `check_phase_a()`, that runs
all checks and returns `(passed, report)`. PR-3 implements the actual
checks; PR-1 (this file) provides the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhaseACheckResult:
    passed: bool
    report: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def check_imports() -> tuple[bool, dict]:
    """Verify the substrate / adapter / curriculum packages import cleanly."""
    failures = []
    try:
        import prism.cognition  # noqa: F401
        import prism.cognition.tokens  # noqa: F401
        import prism.cognition.tokenizer_base  # noqa: F401
    except Exception as e:
        failures.append(f"prism.cognition import failed: {e!r}")
    try:
        import prism.adapters  # noqa: F401
        import prism.adapters.base  # noqa: F401
    except Exception as e:
        failures.append(f"prism.adapters import failed: {e!r}")
    try:
        import prism.curriculum  # noqa: F401
        import prism.curriculum.stage  # noqa: F401
        import prism.curriculum.engine  # noqa: F401
    except Exception as e:
        failures.append(f"prism.curriculum import failed: {e!r}")
    return (len(failures) == 0, {"failures": failures})


def check_babyai_adapter_present() -> tuple[bool, dict]:
    """Verify BabyAIAdapter exists and implements the Protocol.

    PR-3 implementation; PR-1 returns NotImplemented.
    """
    try:
        from prism.adapters.babyai_adapter import BabyAIAdapter  # noqa: F401
        from prism.adapters.base import DomainAdapter
        adapter_cls = BabyAIAdapter
        # Protocol structural check is runtime via isinstance because
        # DomainAdapter is @runtime_checkable.
        # We can't instantiate without an encoder checkpoint at check
        # time; just verify the class declares the required attributes.
        required_attrs = (
            "name", "latent_dim", "mission_dim_max", "n_obs_tokens",
            "encoder", "tokenize", "action_head", "mask_logits",
            "reward_shaper",
        )
        missing = [a for a in required_attrs if not hasattr(adapter_cls, a)]
        ok = len(missing) == 0
        return ok, {"missing_attrs": missing}
    except ImportError as e:
        return False, {"error": f"BabyAIAdapter not yet present (PR-2): {e!r}"}


def check_universal_policy_present() -> tuple[bool, dict]:
    """Verify UniversalPolicy exists with expected entry points.

    PR-2 implementation; PR-1 returns NotImplemented.
    """
    try:
        from prism.cognition.policy import UniversalPolicy
        required_methods = ("init_hidden", "step_with_value", "from_adapter")
        missing = [m for m in required_methods if not hasattr(UniversalPolicy, m)]
        ok = len(missing) == 0
        return ok, {"missing_methods": missing}
    except ImportError as e:
        return False, {"error": f"UniversalPolicy not yet present (PR-2): {e!r}"}


def check_phase_a() -> PhaseACheckResult:
    """Run all Phase A pre-conditions and return a structured result.

    Note: the reward-parity check (item 4 in the docstring above) is
    NOT in this function — it runs after a training experiment completes
    and reads the run's `metrics.json`. See
    `scripts/experiments/e0_phase_a_reward_parity.py` (PR-3) for the
    end-to-end runner.
    """
    failures: list[str] = []
    report: dict = {}

    ok_imports, r_imports = check_imports()
    report["imports"] = r_imports
    if not ok_imports:
        failures.append("imports")

    ok_adapter, r_adapter = check_babyai_adapter_present()
    report["babyai_adapter"] = r_adapter
    if not ok_adapter:
        failures.append("babyai_adapter")

    ok_policy, r_policy = check_universal_policy_present()
    report["universal_policy"] = r_policy
    if not ok_policy:
        failures.append("universal_policy")

    return PhaseACheckResult(
        passed=(len(failures) == 0),
        report=report,
        failures=failures,
    )


if __name__ == "__main__":
    import json
    import sys

    result = check_phase_a()
    print(json.dumps(
        {
            "passed": result.passed,
            "failures": result.failures,
            "report": result.report,
        },
        indent=2,
        default=str,
    ))
    sys.exit(0 if result.passed else 1)
