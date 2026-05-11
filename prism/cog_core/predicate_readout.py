"""PredicateReadout — map a JEPA latent to a vector of predicate logits.

Domain-general by design: the cognition core stores a readout from
latent → predicate space. The *semantics* of each predicate slot is
defined by the supervisory signal that env adapters provide (slot
extractors, sensor parsers, compiler outputs, etc.). The readout itself
is just a function `z → R^P`.

Two roles:

* **Training**: env adapter labels each frame with a target predicate
  vector (e.g., from BabyAI's slot extraction, or from a robotics
  proprioception parser, or from compiler output parsing). The readout
  is supervised toward that target.

* **Inference**: cognition reads predicates from latents to answer
  "what's true about the current state?" without re-running the
  env-specific extractor. This is what enables language grounding,
  planning, predicate-conditioned policies, etc.

Architecture is a small MLP. The intentional smallness keeps it from
"learning" predicates that aren't really in the latent — if a target
predicate can be linearly or near-linearly read out, it's encoded in z.
If not, the JEPA's representation lacks that signal (and that's a
useful diagnostic — fix the JEPA, not the readout).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PredicateReadout(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_predicates: int,
        hidden: int = 512,
        n_layers: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_predicates = n_predicates
        self.hidden = hidden
        self.n_layers = n_layers

        # `hidden = 0` (or n_layers = 0) is interpreted as a pure linear
        # probe: Linear(latent_dim, n_predicates) with no hidden layer or
        # nonlinearity. Useful for testing whether features are linearly
        # decodable from z, separate from MLP capacity.
        if hidden == 0 or n_layers == 0:
            layers: list[nn.Module] = [nn.Linear(latent_dim, n_predicates)]
        else:
            layers = [nn.Linear(latent_dim, hidden), nn.GELU()]
            for _ in range(max(0, n_layers - 1)):
                layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
            layers.append(nn.Linear(hidden, n_predicates))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) or (B, ...). Returns logits (B, n_predicates)."""
        if z.dim() > 2:
            z = z.flatten(1)
        return self.net(z)

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "latent_dim": self.latent_dim,
                "n_predicates": self.n_predicates,
                "hidden": self.hidden,
                "n_layers": self.n_layers,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        device: torch.device,
        *,
        hidden: int | None = None,
        n_layers: int | None = None,
    ) -> "PredicateReadout":
        """Load a saved readout.

        If `hidden` / `n_layers` are None, they're read from the checkpoint
        so the constructed module exactly matches the saved state dict.
        Pass explicit overrides only if loading an old checkpoint that
        didn't save these fields.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd = ckpt["state_dict"]
        # Auto-detect hidden/n_layers from state_dict for old checkpoints
        # that didn't save these fields. A pure linear probe has exactly two
        # keys: net.0.weight and net.0.bias. Anything else is an MLP.
        if hidden is None or n_layers is None:
            if "hidden" in ckpt and "n_layers" in ckpt:
                ck_hidden = ckpt["hidden"]
                ck_layers = ckpt["n_layers"]
            else:
                keys = sorted(sd.keys())
                # Pure linear probe: only one Linear layer (idx 0).
                if set(keys) == {"net.0.weight", "net.0.bias"}:
                    ck_hidden = 0
                    ck_layers = 0
                else:
                    # Fall back to the original defaults; if state-dict
                    # mismatch persists, the caller will see a clear error.
                    ck_hidden = 512
                    ck_layers = 2
            if hidden is not None:
                ck_hidden = hidden
            if n_layers is not None:
                ck_layers = n_layers
        else:
            ck_hidden = hidden
            ck_layers = n_layers
        m = cls(
            latent_dim=ckpt["latent_dim"],
            n_predicates=ckpt["n_predicates"],
            hidden=ck_hidden,
            n_layers=ck_layers,
        )
        m.load_state_dict(sd)
        m.to(device)
        return m
