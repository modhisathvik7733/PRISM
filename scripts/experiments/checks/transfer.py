"""Transfer-experiment gate checks (used by E3 and E5).

These are the resolution-7 pre-conditions that prevent the composite
false-positive failure mode in cross-environment / cross-domain transfer:

  1. Every arm uses a freshly-initialized encoder trained in the target
     domain (encoder-as-adapter, resolution 1). Source-domain encoder
     weights MUST NOT appear in any arm.
  2. `substrate_config_hash` matches between source and target runs.
     Locked-hyperparameter invariant (resolution 3).
  3. Mission tokenization for the target domain produces no truncation
     events on the evaluation set.

PR-1: stubs. Implemented as gates by PR-8 (E3) and PR-9+ (E5).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TransferCheckResult:
    passed: bool
    report: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def check_fresh_encoders_across_arms(arm_checkpoints: dict[str, str]) -> tuple[bool, dict]:
    """Verify each arm's encoder weights differ from the source checkpoint.

    Computes encoder-weight hashes for the source checkpoint and each arm's
    final checkpoint. Asserts no arm shares an encoder hash with the source.

    NOT IMPLEMENTED in PR-1. PR-8 implements the hash comparison.
    """
    raise NotImplementedError("check_fresh_encoders_across_arms: PR-8")


def check_substrate_config_hash_match(source_ckpt: str, target_run: str) -> tuple[bool, dict]:
    """Verify the substrate-locked hyperparameters did not change between
    the source-domain training and the target-domain transfer run.

    NOT IMPLEMENTED in PR-1. Implemented alongside `substrate_config_hash`
    in PR-5/PR-6.
    """
    raise NotImplementedError("check_substrate_config_hash_match: PR-5/PR-6")


def check_no_mission_truncation(eval_set_path: str, adapter_mission_max: int) -> tuple[bool, dict]:
    """Verify no mission in the evaluation set was truncated by the
    adapter's `mission_dim_max` cap. Catches the BabyAI-leakage failure
    mode where an inadequate cap silently sanitizes inputs.

    NOT IMPLEMENTED in PR-1. PR-8 (E3) implements with a tokenization
    dry-run over the eval set.
    """
    raise NotImplementedError("check_no_mission_truncation: PR-8")
