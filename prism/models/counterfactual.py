"""Counterfactual training objective.

Per Phase 1 of the roadmap: at each transition we *also* train the dynamics
network to predict the latent that *would have* arrived under an alternative
action.

This is cheap (extra forward passes, no extra rollouts) and forces the
dynamics model to encode action-conditioned causal structure rather than
just the realized policy's marginal trajectory distribution.

Falsifier: if `loss_cf` collapses to ~ Var(target) — the model is predicting
the marginal next-state regardless of action — the counterfactual head is
short-circuited. That likely means the action embedding is too small or the
realized policy is too narrow (use random rollouts to broaden coverage).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from prism.models.jepa import JepaWorldModel


def counterfactual_loss(
    model: JepaWorldModel,
    obs_t: torch.Tensor,           # (B, C, H, W)
    action_t: torch.Tensor,        # (B,) the realized action
    obs_tp1: torch.Tensor,         # (B, C, H, W) realized next obs
    sim_step_fn,                   # callable: (env_state, a_prime) -> obs_tp1_cf
    env_states_t,                  # opaque list of env snapshots at time t
    n_samples: int = 1,
) -> dict[str, torch.Tensor]:
    """Counterfactual prediction loss.

    Requires a *resettable* env so we can roll forward from `env_states_t` with
    an alternative action `a' != action_t` to produce ground-truth counterfactual
    next observations. BabyAI/MiniGrid is fully resettable from internal state,
    so this is feasible.

    For Phase 1 we sample one alternative action per transition; this is enough
    to break the mean-prediction shortcut. Increase `n_samples` later if the
    counterfactual error is high-variance.
    """
    n_actions = model.cfg.n_actions
    B = obs_t.shape[0]
    device = obs_t.device

    # Sample alt actions uniformly from {0..n_actions-1} \ {action_t[i]}.
    rand = torch.randint(0, n_actions - 1, (B, n_samples), device=device)
    rand = rand + (rand >= action_t.unsqueeze(1)).long()  # skip the realized action

    # Roll the env forward from the saved states under each alt action.
    # `sim_step_fn` returns a tensor (B, n_samples, C, H, W) of CF next-obs.
    obs_tp1_cf = sim_step_fn(env_states_t, rand)  # provided by the trainer

    z_t = model.encode(obs_t)                                  # (B, D)
    losses_cf = []
    for s in range(n_samples):
        z_pred_cf = model.predict_counterfactual(z_t, rand[:, s])  # (B, D)
        with torch.no_grad():
            z_target_cf = model.encode_target(obs_tp1_cf[:, s])
        losses_cf.append(F.mse_loss(z_pred_cf, z_target_cf))
    l_cf = torch.stack(losses_cf).mean()

    # Sanity: predicting realized for comparison (lets us monitor the gap).
    z_pred = model.predict(z_t, action_t)
    with torch.no_grad():
        z_target = model.encode_target(obs_tp1)
    l_factual = F.mse_loss(z_pred, z_target)

    return {
        "loss_cf": l_cf,
        "loss_factual": l_factual.detach(),
        "cf_to_factual_ratio": (l_cf.detach() / (l_factual.detach() + 1e-8)),
    }
