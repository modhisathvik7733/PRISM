"""prism.cognition — domain-agnostic cognition substrate.

This package is the SUBSTRATE: every component here must work identically
across games, code editing, robotics, workflows. Domain-specific code lives
in `prism.adapters`. Domain-agnostic curriculum logic lives in
`prism.curriculum`.

Substrate contract (load-bearing across all domains):
  * Universal token interface (`tokens.TokenStream`)
  * Transformer-with-Hopfield-memory trunk (`trunk.UniversalTrunk`)
  * Growable Hopfield K/V memory banks (`memory_bank.GrowableHopfieldBank`)
  * Sparse, slot-localized continual updates
  * Cycle-consistent language readout
  * Curriculum-driven stage transitions with activation-based freezing

Substrate-locked hyperparameters live in `SubstrateConfig` and are
checkpoint-hashed; they cannot change across domains. Adapter-side
hyperparameters (encoder dim, action-decoder type, mission tokenization)
are free to vary per adapter.

See: /Users/chintu/.claude/plans/you-know-my-goal-warm-hopcroft.md
"""

from __future__ import annotations
