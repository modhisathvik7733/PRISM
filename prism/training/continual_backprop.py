"""ContinualBackprop — random reinit of dead neural units (Sutton 2024 Nature).

Paper: "Loss of plasticity in continual learning" (Dohare, Sutton et al.,
Nature 2024). Reframes catastrophic forgetting research: plain SGD/Adam
networks not only forget — they progressively LOSE the ability to learn
new things in long task streams. Many units become permanently inactive.

Mechanism: track utility of each unit. Periodically reinitialize the
least-useful units to random weights. This restores plasticity without
disturbing useful representations.

Apply this to PRISM's JEPA encoder (and other deep nets) but NOT to the
Hopfield memory (which has its own anti-drift via SparseHopfieldOptimizer).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContinualBackpropHook:
    """Wraps a nn.Linear or nn.Conv2d to track unit utility and reinit dead ones.

    Utility: exponential moving average of |activation| * |outgoing weight|.
    Units with utility below threshold for too long get reinitialized.
    """

    def __init__(
        self,
        layer: nn.Module,
        replacement_rate: float = 1e-4,        # fraction of units to consider reinit per step
        decay_rate: float = 0.99,              # EMA decay for utility
        maturity_threshold: int = 100,         # don't reinit until unit has been seen N times
        unit_dim: int = 0,                     # which dim of output is the "unit" dim
        protected_mask: torch.Tensor | None = None,
    ):
        self.layer = layer
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.unit_dim = unit_dim
        # Audit pass-2 issue 3c: when this hook wraps weights backed by a
        # MemoryBank (frozen-slot semantics), the bank's frozen_mask must
        # be passed here so reinit cannot re-write rows the curriculum
        # froze. `protected_mask` is a (n_units,) bool tensor: True at
        # indices that must never be reinitialized. Callers that wrap a
        # bank's K/V tensors should pass `bank.frozen_mask` here and
        # update it via `set_protected_mask` whenever the curriculum
        # engine freezes new slots.
        self.protected_mask: torch.Tensor | None = protected_mask

        # Get number of units (output dim).
        if isinstance(layer, nn.Linear):
            self.n_units = layer.out_features
        elif isinstance(layer, nn.Conv2d):
            self.n_units = layer.out_channels
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

        device = next(layer.parameters()).device
        self.utility = torch.zeros(self.n_units, device=device)
        self.age = torch.zeros(self.n_units, dtype=torch.long, device=device)

        # Register forward hook to track activations.
        self._handle = layer.register_forward_hook(self._hook)
        self._last_input_norm: float | None = None

    def _hook(self, module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        # out shape: (B, n_units) for Linear, (B, C, H, W) for Conv2d.
        with torch.no_grad():
            if out.dim() == 2:
                # Linear: (B, n_units)
                act = out.abs().mean(dim=0)
            elif out.dim() == 4:
                # Conv2d: (B, C, H, W) → per-channel mean of abs activation
                act = out.abs().mean(dim=(0, 2, 3))
            else:
                return

            # Outgoing weight magnitude (||W|| for the unit).
            if isinstance(module, nn.Linear):
                outgoing = module.weight.detach().abs().mean(dim=1)  # (out_features,)
            elif isinstance(module, nn.Conv2d):
                outgoing = module.weight.detach().abs().mean(dim=(1, 2, 3))
            else:
                outgoing = torch.ones_like(act)

            unit_utility = act * outgoing
            self.utility = self.decay_rate * self.utility + (1 - self.decay_rate) * unit_utility
            self.age += 1

    @torch.no_grad()
    def set_protected_mask(self, mask: torch.Tensor) -> None:
        """Replace the protected-units mask. Called by trainers that
        share a bank's frozen_mask: as the CurriculumEngine freezes new
        slots, the trainer must propagate the updated frozen_mask into
        every ContinualBackpropHook wrapping that bank's weights. Audit
        pass-2 issue 3c: missing this propagation lets reinit silently
        re-write rows the curriculum froze."""
        if mask.shape != (self.n_units,):
            raise ValueError(
                f"protected_mask shape {tuple(mask.shape)} != ({self.n_units},)"
            )
        self.protected_mask = mask

    def reinit_dead_units(self, optimizer: torch.optim.Optimizer | None = None) -> int:
        """Reinitialize the lowest-utility mature units. Returns count of reinit."""
        # Only consider units past maturity threshold AND not protected.
        mature = self.age >= self.maturity_threshold
        if self.protected_mask is not None:
            mature = mature & (~self.protected_mask)
        if mature.sum() == 0:
            return 0

        n_to_reinit = max(1, int(self.replacement_rate * mature.sum().item()))
        if n_to_reinit == 0:
            return 0

        # Pick the n_to_reinit lowest-utility mature units.
        mature_indices = torch.where(mature)[0]
        mature_utility = self.utility[mature_indices]
        _, low_indices_in_mature = mature_utility.topk(
            n_to_reinit, largest=False
        )
        units_to_reinit = mature_indices[low_indices_in_mature]

        # Reinit those rows in the layer's weight matrix.
        if isinstance(self.layer, nn.Linear):
            for u in units_to_reinit:
                u = int(u)
                nn.init.kaiming_uniform_(
                    self.layer.weight[u : u + 1, :], a=5**0.5
                )
                if self.layer.bias is not None:
                    nn.init.zeros_(self.layer.bias[u : u + 1])
                self.utility[u] = 0
                self.age[u] = 0
                # Also reset optimizer state for that unit if Adam.
                if optimizer is not None:
                    self._reset_optimizer_state(optimizer, self.layer.weight, u)
                    if self.layer.bias is not None:
                        self._reset_optimizer_state(optimizer, self.layer.bias, u)

        elif isinstance(self.layer, nn.Conv2d):
            for u in units_to_reinit:
                u = int(u)
                nn.init.kaiming_uniform_(
                    self.layer.weight[u : u + 1], a=5**0.5
                )
                if self.layer.bias is not None:
                    nn.init.zeros_(self.layer.bias[u : u + 1])
                self.utility[u] = 0
                self.age[u] = 0

        return n_to_reinit

    @staticmethod
    def _reset_optimizer_state(
        optimizer: torch.optim.Optimizer,
        param: nn.Parameter,
        unit_idx: int,
    ) -> None:
        """Reset Adam-style optimizer state for a single unit row."""
        if param not in optimizer.state:
            return
        state = optimizer.state[param]
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state:
                tensor = state[key]
                if tensor.dim() >= 1 and tensor.size(0) > unit_idx:
                    tensor[unit_idx].zero_()

    def detach(self) -> None:
        """Remove the forward hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class ContinualBackpropManager:
    """Manages ContinualBackpropHook for multiple layers in a model.

    Usage:
        manager = ContinualBackpropManager()
        manager.attach_to_model(jepa_encoder)

        # Each PPO update step:
        manager.maybe_reinit(step, optimizer)
    """

    def __init__(self, reinit_every: int = 100):
        self.reinit_every = reinit_every
        self.hooks: list[ContinualBackpropHook] = []

    def attach_to_model(
        self,
        model: nn.Module,
        replacement_rate: float = 1e-4,
        decay_rate: float = 0.99,
        maturity_threshold: int = 100,
    ) -> None:
        """Attach hooks to all eligible layers in a model."""
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = ContinualBackpropHook(
                    module,
                    replacement_rate=replacement_rate,
                    decay_rate=decay_rate,
                    maturity_threshold=maturity_threshold,
                )
                self.hooks.append(hook)

    def maybe_reinit(
        self,
        step: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> int:
        if step % self.reinit_every != 0:
            return 0
        total = 0
        for hook in self.hooks:
            total += hook.reinit_dead_units(optimizer)
        return total

    def detach_all(self) -> None:
        for hook in self.hooks:
            hook.detach()
        self.hooks.clear()


if __name__ == "__main__":
    # Standalone smoke test for audit-3c: a protected_mask prevents
    # reinit from touching the masked rows even when their utility is
    # the lowest in the layer. Run with:
    #   `python -m prism.training.continual_backprop`
    import sys as _sys

    torch.manual_seed(0)
    layer = nn.Linear(8, 16)          # 16 units; will protect indices [0, 5, 11]
    protected = torch.zeros(16, dtype=torch.bool)
    protected[[0, 5, 11]] = True

    hook = ContinualBackpropHook(
        layer=layer,
        replacement_rate=1.0,         # try to reinit every mature unit
        maturity_threshold=1,
        protected_mask=protected,
    )

    # Drive enough forward passes to mature all units.
    for _ in range(5):
        x = torch.randn(4, 8)
        _ = layer(x)
    # Manually nudge utility: force protected units to look "dead"
    # (lowest utility) so the naive selector would pick them first.
    hook.utility[:] = 1.0
    hook.utility[[0, 5, 11]] = 0.0

    # Snapshot weights for the protected rows.
    before_protected = layer.weight[[0, 5, 11], :].detach().clone()
    n_reinit = hook.reinit_dead_units()
    after_protected = layer.weight[[0, 5, 11], :].detach().clone()

    if not torch.equal(before_protected, after_protected):
        print(f"FAIL: protected rows were modified by reinit_dead_units")
        _sys.exit(1)
    print(f"[cbp] reinit touched {n_reinit} units; "
          f"protected rows [0, 5, 11] are bit-identical pre/post")

    # set_protected_mask should accept a new mask and apply on next reinit.
    new_protected = torch.zeros(16, dtype=torch.bool)
    new_protected[[2, 7]] = True
    hook.set_protected_mask(new_protected)
    hook.utility[:] = 1.0
    hook.utility[[2, 7]] = 0.0
    before_new = layer.weight[[2, 7], :].detach().clone()
    n_reinit2 = hook.reinit_dead_units()
    after_new = layer.weight[[2, 7], :].detach().clone()
    if not torch.equal(before_new, after_new):
        print(f"FAIL: set_protected_mask did not apply to next reinit_dead_units")
        _sys.exit(1)
    print(f"[cbp] set_protected_mask: new mask applied; "
          f"protected rows [2, 7] bit-identical pre/post")

    # Wrong-shape mask raises.
    try:
        hook.set_protected_mask(torch.zeros(15, dtype=torch.bool))
    except ValueError:
        print(f"[cbp] set_protected_mask rejects wrong-shape mask")
    else:
        print("FAIL: set_protected_mask accepted wrong-shape mask")
        _sys.exit(1)

    print("[cbp] all smoke checks passed")
