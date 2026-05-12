"""prism.adapters — domain interface layer.

Each domain (BabyAI, code editing, robotics, workflows) provides exactly
one adapter implementing `DomainAdapter`. The adapter owns:

  * An observation encoder (the only place per-domain perceptual structure
    lives — JEPA-style for BabyAI images, token-embedding for code, etc.)
  * A tokenizer that converts raw env obs + mission into a TokenStream.
  * An action decoder providing a uniform `.dist()` API to PPO.
  * Action masking, if the env disallows actions in some states.
  * Reward shaping, if any.

The substrate (prism.cognition) NEVER touches domain-specific data. All
domain knowledge flows through the adapter. This is enforced by the
checkpoint invariant: substrate-config-hash is locked across domains;
adapter-config-hash varies.

Hard invariant (resolution 1 in plan): the JEPA encoder is part of the
adapter, NOT the substrate. Cross-domain transfer experiments train a
fresh encoder per domain; substrate transfer is measured on the
post-encoder portion only.
"""

from __future__ import annotations
