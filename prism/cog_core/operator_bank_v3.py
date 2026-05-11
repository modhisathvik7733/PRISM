"""OperatorBankV3 — anti-drift operator bank.

V2's failure mode: operators are MLPs whose weights drift under continual
training. The names "push", "move", etc. are just labels in code — there
is no mechanism that anchors operator k's *behavior* across data shifts.

V3 keeps V2's MoE structure (routing net + K dynamics MLPs) and adds:

1. **EMA target bank.** A slow-moving twin of the online bank, updated by
   exponential moving average each step. The online bank pays a small
   consistency cost when its predictions disagree with the EMA bank's.
   This is the same trick JEPA uses for the target encoder and the same
   trick Cola DLM uses for its reference VAE.

2. **Behavioral anchor buffer.** A small per-operator buffer of canonical
   (z_t, a, z_{t+1}) tuples. After seeding (e.g. after a first stable
   training phase), these tuples are *frozen* and the model must continue
   to predict them. This binds operator identity to behavior, not to
   weights.

3. **Straight-through hard routing.** Replace softmax with Gumbel-argmax
   with straight-through gradients. Each transition picks one operator;
   gradients still flow through routing logits.

4. **Stop-gradient on anchor target.** When computing the anchor-consistency
   loss, the target z_{t+1} comes from the *initial* stored anchor batch,
   never from a current encoder pass — anchors are immutable.

The three losses add up to:
    L = recon_mse + lambda_ema * ema_consistency + lambda_anchor * anchor_loss
        - entropy_coef * H(routing)

When `anchor_loss` is enabled but no anchors have been seeded yet, that
term is zero — V3 reduces to V2 during the first training phase. After
seeding, the anchor term kicks in and prevents drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OperatorV3Stats:
    op_id: int
    activation_rate: float
    dominant_action: int
    purity: float
    action_distribution: dict[int, float]
    anchor_valid: bool
    anchor_mse: float


class OperatorBankV3(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_actions: int,
        *,
        n_ops: int = 8,
        hidden: int = 256,
        action_emb_dim: int = 16,
        entropy_coef: float = 0.01,
        ema_tau: float = 0.995,
        lambda_ema: float = 0.1,
        lambda_anchor: float = 1.0,
        lambda_load_balance: float = 0.1,
        lambda_sharpness: float = 0.05,
        anchor_size: int = 64,
        hard_routing: bool = False,
        gumbel_tau: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self.n_ops = n_ops
        self.entropy_coef = entropy_coef
        self.ema_tau = ema_tau
        self.lambda_ema = lambda_ema
        self.lambda_anchor = lambda_anchor
        self.lambda_load_balance = lambda_load_balance
        self.lambda_sharpness = lambda_sharpness
        self.anchor_size = anchor_size
        self.hard_routing = hard_routing
        self.gumbel_tau = gumbel_tau

        self.action_emb = nn.Embedding(n_actions, action_emb_dim)
        in_dim = latent_dim + action_emb_dim

        self.routing = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_ops),
        )
        # Break symmetry: small per-op bias spread so routing logits are
        # differentiated at step 0 instead of converging to the same value
        # for every op.
        with torch.no_grad():
            final = self.routing[-1]
            final.bias.copy_(torch.linspace(-0.5, 0.5, n_ops))

        self.ops = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, latent_dim),
            )
            for _ in range(n_ops)
        ])

        # --- EMA target bank (parameter-only mirror, no separate forward graph)
        # We register target parameters as buffers (no grad), and copy initial
        # values from the online params. They update via EMA in `ema_step`.
        self._ema_init = False
        self._ema_params: dict[str, torch.Tensor] = {}

        # --- Behavioral anchor buffer (per operator).
        # Shape: (n_ops, anchor_size, latent_dim) for z_t and z_tp1.
        # `anchor_valid` marks which slots have been seeded.
        self.register_buffer(
            "anchor_z_t",
            torch.zeros(n_ops, anchor_size, latent_dim),
        )
        self.register_buffer(
            "anchor_a",
            torch.zeros(n_ops, anchor_size, dtype=torch.long),
        )
        self.register_buffer(
            "anchor_z_tp1",
            torch.zeros(n_ops, anchor_size, latent_dim),
        )
        self.register_buffer(
            "anchor_valid",
            torch.zeros(n_ops, dtype=torch.bool),
        )
        # Python-level flag to avoid a CUDA sync (anchor_valid.any().item())
        # in the hot path. Updated only when seed_anchors is called.
        self._has_anchors: bool = False

    # ------------------------------------------------------------------
    # core forward
    # ------------------------------------------------------------------
    def _flatten(self, z: torch.Tensor) -> torch.Tensor:
        return z.flatten(1) if z.dim() > 2 else z

    def _route_components(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (routing_used, routing_soft).

        `routing_used` is what the forward pass uses for blending op outputs
        (hard one-hot via Gumbel if hard_routing+training, else softmax).
        `routing_soft` is always the softmax — used for load-balancing /
        sharpness losses and diagnostics."""
        logits = self.routing(x)
        soft = F.softmax(logits, dim=-1)
        if self.hard_routing and self.training:
            hard = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True, dim=-1)
            return hard, soft
        return soft, soft

    def forward(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_flat = self._flatten(z_t)
        a_emb = self.action_emb(action)
        x = torch.cat([z_flat, a_emb], dim=-1)

        routing_used, _routing_soft = self._route_components(x)
        op_preds = torch.stack([op(x) for op in self.ops], dim=1)
        delta_pred = (routing_used.unsqueeze(-1) * op_preds).sum(dim=1)
        # Return routing_used as the second element to keep backward
        # compatibility with callers; soft version available via routing_soft.
        return delta_pred, routing_used, op_preds

    def _forward_with_soft(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Variant returning soft probs too — used inside `loss`."""
        z_flat = self._flatten(z_t)
        a_emb = self.action_emb(action)
        x = torch.cat([z_flat, a_emb], dim=-1)

        routing_used, routing_soft = self._route_components(x)
        op_preds = torch.stack([op(x) for op in self.ops], dim=1)
        delta_pred = (routing_used.unsqueeze(-1) * op_preds).sum(dim=1)
        return delta_pred, routing_used, routing_soft, op_preds

    # ------------------------------------------------------------------
    # EMA target bank
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _init_ema(self) -> None:
        for name, p in self.named_parameters():
            self._ema_params[name] = p.detach().clone()
        self._ema_init = True

    @torch.no_grad()
    def ema_step(self) -> None:
        if not self._ema_init:
            self._init_ema()
            return
        # Fused EMA update across all params in a single kernel.
        # lerp_(start, end, w) -> start = (1-w)*start + w*end
        # We want ema = tau*ema + (1-tau)*online, so w = (1-tau).
        online_list: list[torch.Tensor] = []
        target_list: list[torch.Tensor] = []
        for name, p in self.named_parameters():
            online_list.append(p.detach())
            target_list.append(self._ema_params[name])
        torch._foreach_lerp_(target_list, online_list, 1.0 - self.ema_tau)

    @torch.no_grad()
    def _ema_forward(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the EMA target bank. Returns delta_pred only.

        Uses fused _foreach_copy_ to batch the param swap into ~4 CUDA
        kernel launches instead of 90+ individual launches.
        """
        if not self._ema_init:
            self._init_ema()
        # Collect param tensors in one list (deterministic order).
        names = [n for n, _ in self.named_parameters()]
        live = [p.data for _, p in self.named_parameters()]
        ema = [self._ema_params[n] for n in names]
        # Backup live params (single fused clone-like op).
        backup = [t.clone() for t in live]
        # Swap in EMA.
        torch._foreach_copy_(live, ema)
        try:
            was_training = self.training
            self.eval()
            delta_pred, _, _ = self.forward(z_t, action)
            if was_training:
                self.train()
        finally:
            # Restore live params.
            torch._foreach_copy_(live, backup)
        return delta_pred.detach()

    # ------------------------------------------------------------------
    # Behavioral anchors
    # ------------------------------------------------------------------
    @torch.no_grad()
    def seed_anchors(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        z_tp1: torch.Tensor,
    ) -> dict[int, int]:
        """Use current routing to assign transitions to operators, then
        store up to `anchor_size` per operator in the anchor buffer.

        Returns a dict of op_id -> number of anchors stored.
        """
        z_t_flat = self._flatten(z_t)
        z_tp1_flat = self._flatten(z_tp1)
        a_emb = self.action_emb(action)
        x = torch.cat([z_t_flat, a_emb], dim=-1)
        logits = self.routing(x)
        hard = logits.argmax(dim=-1)              # (B,)

        stored: dict[int, int] = {}
        for k in range(self.n_ops):
            mask = hard == k
            n = int(mask.sum())
            if n == 0:
                continue
            take = min(n, self.anchor_size)
            idx_in_batch = mask.nonzero(as_tuple=True)[0][:take]
            self.anchor_z_t[k, :take] = z_t_flat[idx_in_batch]
            self.anchor_a[k, :take] = action[idx_in_batch]
            self.anchor_z_tp1[k, :take] = z_tp1_flat[idx_in_batch]
            # Zero out unused slots so we never train on stale data.
            if take < self.anchor_size:
                self.anchor_z_t[k, take:] = 0.0
                self.anchor_z_tp1[k, take:] = 0.0
            self.anchor_valid[k] = True
            stored[k] = take
        self._has_anchors = True
        return stored

    def _anchor_loss(self) -> torch.Tensor:
        """For every seeded operator, the bank must still predict its
        canonical effect on its anchor batch.

        We force the routing to operator k for anchor[k] (no gradient through
        routing for this term — the point is the *dynamics head k* must keep
        producing the right delta)."""
        if not bool(self.anchor_valid.any().item()):
            return torch.tensor(0.0, device=self.anchor_z_t.device)

        device = self.anchor_z_t.device
        total = torch.tensor(0.0, device=device)
        n_valid = 0
        for k in range(self.n_ops):
            if not bool(self.anchor_valid[k].item()):
                continue
            z_t = self.anchor_z_t[k]          # (anchor_size, latent_dim)
            a = self.anchor_a[k]              # (anchor_size,)
            z_tp1 = self.anchor_z_tp1[k]
            a_emb = self.action_emb(a)
            x = torch.cat([z_t, a_emb], dim=-1)
            # Direct call to dynamics head k — bypasses routing.
            delta_pred_k = self.ops[k](x)
            delta_true = z_tp1 - z_t
            total = total + F.mse_loss(delta_pred_k, delta_true)
            n_valid += 1
        return total / max(n_valid, 1)

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------
    def loss(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        z_tp1: torch.Tensor,
        use_ema: bool = True,
        use_anchor: bool = True,
    ) -> dict:
        z_t_flat = self._flatten(z_t)
        z_tp1_flat = self._flatten(z_tp1)
        delta_actual = z_tp1_flat - z_t_flat

        delta_pred, routing_used, routing_soft, _ = self._forward_with_soft(
            z_t, action,
        )
        mse = F.mse_loss(delta_pred, delta_actual)

        # ---- routing diagnostics on soft probs ---
        per_sample_entropy = -(
            routing_soft * (routing_soft + 1e-9).log()
        ).sum(dim=-1).mean()
        avg_routing = routing_soft.mean(dim=0)                # (K,)
        batch_entropy = -(avg_routing * (avg_routing + 1e-9).log()).sum()

        # ---- Switch-Transformer-style load-balancing loss ----
        # f_k = fraction of samples whose argmax is op k (hard usage)
        # P_k = mean softmax prob for op k over the batch
        # aux = K * sum(f_k * P_k). Minimized when uniform usage AND sharp.
        with torch.no_grad():
            hard_idx = routing_soft.argmax(dim=-1)
            f = F.one_hot(hard_idx, num_classes=self.n_ops).float().mean(dim=0)
        P = avg_routing
        load_balance_loss = self.n_ops * (f * P).sum()

        # Sharpness term: minimize per-sample entropy (sharp routing).
        sharpness_loss = per_sample_entropy

        if use_ema and self._ema_init:
            delta_ema = self._ema_forward(z_t, action)
            ema_loss = F.mse_loss(delta_pred, delta_ema)
        else:
            ema_loss = torch.tensor(0.0, device=mse.device)

        if use_anchor:
            anchor_loss = self._anchor_loss()
        else:
            anchor_loss = torch.tensor(0.0, device=mse.device)

        total = (
            mse
            + self.lambda_ema * ema_loss
            + self.lambda_anchor * anchor_loss
            + self.lambda_load_balance * load_balance_loss
            + self.lambda_sharpness * sharpness_loss
        )
        return {
            "loss": total,
            "mse": mse.detach(),
            "ema_loss": ema_loss.detach(),
            "anchor_loss": anchor_loss.detach(),
            "load_balance": load_balance_loss.detach(),
            "sharpness": sharpness_loss.detach(),
            "per_sample_entropy": per_sample_entropy.detach(),
            "batch_entropy": batch_entropy.detach(),
            "routing": routing_used.detach(),
        }

    # ------------------------------------------------------------------
    # eval helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def assign(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        _, routing, _ = self.forward(z_t, action)
        return routing.argmax(dim=-1)

    @torch.no_grad()
    def routing_for(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        _, routing, _ = self.forward(z_t, action)
        if was_training:
            self.train()
        return routing

    @torch.no_grad()
    def anchor_mse_per_op(self) -> dict[int, float]:
        """Current MSE of dynamics-head k against its stored anchor batch.
        Use this to *measure* drift directly."""
        result: dict[int, float] = {}
        for k in range(self.n_ops):
            if not bool(self.anchor_valid[k].item()):
                continue
            z_t = self.anchor_z_t[k]
            a = self.anchor_a[k]
            z_tp1 = self.anchor_z_tp1[k]
            a_emb = self.action_emb(a)
            x = torch.cat([z_t, a_emb], dim=-1)
            pred = self.ops[k](x)
            actual = z_tp1 - z_t
            result[k] = float(F.mse_loss(pred, actual).item())
        return result

    @torch.no_grad()
    def analyze(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
    ) -> list[OperatorV3Stats]:
        routing = self.routing_for(z_t, action)
        hard = routing.argmax(dim=-1).cpu().numpy()
        actions = action.cpu().numpy()
        anchor_mse = self.anchor_mse_per_op()

        stats: list[OperatorV3Stats] = []
        for k in range(self.n_ops):
            mask = hard == k
            n = int(mask.sum())
            mean_prob = float(routing[:, k].mean().item())
            if n == 0:
                stats.append(OperatorV3Stats(
                    op_id=k, activation_rate=mean_prob,
                    dominant_action=-1, purity=0.0,
                    action_distribution={},
                    anchor_valid=bool(self.anchor_valid[k].item()),
                    anchor_mse=anchor_mse.get(k, float("nan")),
                ))
                continue
            assigned = actions[mask]
            unique, counts = np.unique(assigned, return_counts=True)
            ad = {int(a): float(c / counts.sum()) for a, c in zip(unique, counts)}
            dominant = int(unique[counts.argmax()])
            purity = float(counts.max() / counts.sum())
            stats.append(OperatorV3Stats(
                op_id=k, activation_rate=mean_prob,
                dominant_action=dominant, purity=purity,
                action_distribution=ad,
                anchor_valid=bool(self.anchor_valid[k].item()),
                anchor_mse=anchor_mse.get(k, float("nan")),
            ))
        return stats

    @torch.no_grad()
    def cross_env_stability(
        self,
        per_env_data: dict[str, tuple[torch.Tensor, torch.Tensor]],
        threshold: float = 0.8,
    ) -> dict:
        env_ids = sorted(per_env_data.keys())
        env_matrices: dict[str, np.ndarray] = {}
        for env_id, (latents, actions) in per_env_data.items():
            routing = self.routing_for(latents, actions).cpu().numpy()
            actions_np = actions.cpu().numpy()
            mat = np.zeros((self.n_actions, self.n_ops))
            counts = np.zeros(self.n_actions)
            for i, a in enumerate(actions_np):
                mat[a] += routing[i]
                counts[a] += 1
            counts = np.maximum(counts, 1)
            env_matrices[env_id] = mat / counts[:, None]

        results: dict = {"per_env_action_op_matrix":
                         {k: v.tolist() for k, v in env_matrices.items()}}
        if len(env_ids) >= 2:
            pair_results = []
            for i in range(len(env_ids)):
                for j in range(i + 1, len(env_ids)):
                    e1, e2 = env_ids[i], env_ids[j]
                    m1 = env_matrices[e1].flatten()
                    m2 = env_matrices[e2].flatten()
                    cos = float((m1 @ m2) / (
                        (np.linalg.norm(m1) + 1e-9) *
                        (np.linalg.norm(m2) + 1e-9)
                    ))
                    pair_results.append({
                        "env1": e1, "env2": e2,
                        "matrix_cosine_sim": cos,
                        "pass": cos >= threshold,
                    })
            results["pairwise"] = pair_results
            results["mean_cosine"] = float(np.mean(
                [p["matrix_cosine_sim"] for p in pair_results]
            ))
            results["all_pass"] = all(p["pass"] for p in pair_results)
        return results

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            "model_state_dict": self.state_dict(),
            "ema_params": self._ema_params,
            "ema_init": self._ema_init,
            "latent_dim": self.latent_dim,
            "n_actions": self.n_actions,
            "n_ops": self.n_ops,
            "anchor_size": self.anchor_size,
        }, path)

    @classmethod
    def load(
        cls,
        path: str,
        device: torch.device,
        *,
        hidden: int = 256,
        action_emb_dim: int = 16,
    ) -> "OperatorBankV3":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        bank = cls(
            latent_dim=ckpt["latent_dim"],
            n_actions=ckpt["n_actions"],
            n_ops=ckpt["n_ops"],
            anchor_size=ckpt.get("anchor_size", 64),
            hidden=hidden,
            action_emb_dim=action_emb_dim,
        )
        bank.load_state_dict(ckpt["model_state_dict"])
        bank._ema_init = bool(ckpt.get("ema_init", False))
        bank._ema_params = {
            k: v.to(device) for k, v in ckpt.get("ema_params", {}).items()
        }
        # Restore Python anchor flag from the buffer state (no sync needed
        # at load time, but avoids a per-step .any().item() during training).
        bank._has_anchors = bool(bank.anchor_valid.any().item())
        bank.to(device)
        return bank
