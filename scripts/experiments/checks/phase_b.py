"""Phase B gate — transformer trunk goes live.

Pre-conditions for Phase B to be considered passed:
  1. PR-4 has landed: `UniversalTrunk` exists with two-tensor buffer.
  2. The two `torch.where` reset calls (tokens + valid_len) are paired
     via the single `policy.reset_buffer` API. No call site resets one
     without the other.
  3. PPO `log_prob(rollout) == log_prob(replay)` bit-exactly on a
     2-epoch update over a fixed mini-batch (catches replay-buffer
     corruption per audit pass-2 issue 4a).
  4. Action-masking pre-condition: `adapter.mask_logits` produces at
     least one `-inf` entry on >5% of eval steps (catches missing
     masking, resolution 7).
  5. PR-4 success-rate gate: BabyAI-GoToObj ≥0.85, GoToLocal within 5pp
     of v5 baseline.

NOT IMPLEMENTED in PR-1. PR-4 will fill these in.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhaseBCheckResult:
    passed: bool
    report: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def check_phase_b(*args, **kwargs) -> PhaseBCheckResult:
    raise NotImplementedError("check_phase_b: implemented in PR-4")
