"""Self-model consistency loss.

A coherent agent maintains a persistent world state. If `contains(cup, box)`
is asserted at time t, then at t+k it cannot simultaneously hold
`at(cup, table)` unless an executed operator licensed the transition
(e.g. `take(cup, box)` removed `contains(cup, box)`).

We operationalize this as a soft penalty on *unlicensed* predicate flips
across consecutive timesteps.

Implementation:
  predicate_logits_t  : (B, T, P)   per-timestep predicate readout from JEPA latents
  operator_effects    : (B, T, P)   for each timestep the +/- effect mask of the
                                    operator that was *actually executed*
                                    (+1 = predicate is asserted, -1 = retracted,
                                     0 = unaffected)

The loss penalizes the difference between observed predicate change and the
change licensed by the executed operator's effects. Predicates *should not flip*
unless an operator licenses it.

In Phase 1 we don't yet have operators executing, so the licensed-change mask
is all-zero — the loss simply penalizes any predicate flip across time. This
already provides a strong inductive bias: the encoder must produce temporally
stable predicate readouts. In Phase 2 the executed-operator mask becomes
informative.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def consistency_loss(
    predicate_logits: torch.Tensor,        # (B, T, P)
    licensed_change: torch.Tensor | None,  # (B, T-1, P) signed effect mask, or None
) -> torch.Tensor:
    """Penalty on predicate flips not licensed by an executed operator.

    Args:
        predicate_logits: pre-sigmoid predicate readouts across a trajectory.
        licensed_change: signed mask in {-1, 0, +1} per (timestep transition,
            predicate). +1 means the executed operator asserted this predicate,
            -1 means it retracted it, 0 means untouched. Pass None during Phase 1
            (treated as all zeros — no flips allowed).

    Returns:
        scalar loss tensor.
    """
    p = torch.sigmoid(predicate_logits)            # (B, T, P)
    delta = p[:, 1:] - p[:, :-1]                   # (B, T-1, P) observed change

    if licensed_change is None:
        target = torch.zeros_like(delta)
    else:
        # +1 -> target change to "fully on" (delta = +1.0 wrt prev), etc.
        target = licensed_change.float()

    # Soft penalty: squared error between observed change and licensed change.
    # We don't want to over-penalize small noise — clip target band.
    return F.mse_loss(delta, target)
