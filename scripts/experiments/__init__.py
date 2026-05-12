"""scripts.experiments — validation experiments E1-E5 and phase gates.

Each experiment script is a self-contained falsifier for one claim:
  e1_curriculum_ordering: does developmental order matter beyond curriculum?
  e2_freeze_retention:    does memory persist without overwrite?
  e3_intra_game_transfer: does the substrate transfer within games?
  e4_slot_stability:      are emerged concepts inspectable and stable?
  e5_code_transfer:       does the substrate transfer across domains?

`checks/` contains machine-readable pre-conditions that gate each phase.
Every experiment script calls the relevant check module before recording
results — if a check fails, the experiment aborts loudly rather than
silently producing a misleading positive.
"""

from __future__ import annotations
