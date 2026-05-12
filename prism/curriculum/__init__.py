"""prism.curriculum — domain-agnostic developmental curriculum engine.

The engine sequences Stages, freezes slot subsets between transitions,
and persists the probe set used by E4. Stage *content* is domain-specific
(curriculum stages for BabyAI vs code editing have different success
criteria, different env levels) — Stage instances are constructed by
the adapter or experiment script. The engine itself is domain-free.

Hard invariants (from plan):
  * Freeze set is determined by activation statistics, not by
    ConceptManager naming (resolution 4). Naming runs async and is
    inspection-only.
  * Probe set is a persisted artifact created once at Stage 0 with a
    random policy, hashed into substrate_config_hash (resolution 6).
  * Stage transitions block on the bank applying freeze_slots; there is
    no async path that bypasses freezing.
"""

from __future__ import annotations
