"""Machine-checkable pre-conditions for each phase gate.

Every PR-7+ experiment script calls into this directory before recording
a result. Each gate is a small Python module exposing `check_<phase>()`
that returns `(passed: bool, report: dict)`. The harness aborts the
experiment if `passed is False`.

This is where the composite false-positive failure mode (third-pass
audit) is mechanically prevented. The checks correspond to the
"Must Resolve Before Phase X" items in the plan:

  phase_a.py: action masking applied; mission length within adapter cap;
              ppo log_prob bit-equal between rollout and replay (smoke).
  phase_b.py: transformer trunk live; tensor-buffer reset paired;
              v5 success-rate parity.
  phase_c.py: probe set exists with verified hash; warmup completed
              for every expanded slot; frozen-row checksums bit-equal
              after stage transitions.
  transfer.py (used by E3, E5): every arm uses a fresh encoder; source
              and target substrate_config_hash match; mission tokenization
              produces no truncation events.
"""

from __future__ import annotations
