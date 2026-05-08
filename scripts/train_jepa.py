"""Phase 1 — JEPA pretraining on random rollouts.

Trains the JEPA latent world model (encoder + EMA target + dynamics) on
random-policy rollouts in BabyAI. Optionally adds:

  * counterfactual loss (requires a resettable env — BabyAI is)
  * consistency loss (requires predicate readouts; stubbed for now)

This script is intentionally minimal — no wandb, no fancy schedulers. Once it
runs cleanly we'll layer in counterfactual and consistency, then graduate to
Phase 2.

Run:
    uv run python -m scripts.train_jepa \
        --env-id BabyAI-GoToLocal-v0 \
        --steps 200_000 \
        --batch-size 128 \
        --device cuda
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import minigrid  # noqa: F401  (registers BabyAI envs)

from prism.envs.babyai import _encode_image
from prism.models.jepa import JepaConfig, JepaWorldModel
from prism.perception import compute_predicates, extract_slots
from prism.utils.seed import set_global_seed


def collect_random_transitions(
    env_id: str,
    n: int,
    rng: np.random.Generator,
    *,
    with_predicates: bool = False,
):
    """Collect n one-step transitions under random policy.

    We bypass the PrismImageOnlyWrapper here so we can keep the raw uint8
    obs around for slot extraction (only when with_predicates=True).

    Returns:
        obs_t:          (n, 3, 7, 7) float32 normalized
        actions:        (n,) int64
        obs_tp1:        (n, 3, 7, 7) float32 normalized
        predicates_t:   (n, 96) float32, or None if with_predicates=False
        predicates_tp1: (n, 96) float32, or None if with_predicates=False
    """
    env = gym.make(env_id)
    obs_t_list, act_list, obs_tp1_list = [], [], []
    pred_t_list, pred_tp1_list = [], []
    obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
    while len(obs_t_list) < n:
        raw_t = obs["image"]                      # (7, 7, 3) uint8
        a = int(rng.integers(env.action_space.n))
        next_obs, _r, term, trunc, _ = env.step(a)
        raw_tp1 = next_obs["image"]

        obs_t_list.append(_encode_image(raw_t))
        act_list.append(a)
        obs_tp1_list.append(_encode_image(raw_tp1))
        if with_predicates:
            pred_t_list.append(compute_predicates(extract_slots(raw_t)))
            pred_tp1_list.append(compute_predicates(extract_slots(raw_tp1)))

        if term or trunc:
            obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        else:
            obs = next_obs
    return (
        np.stack(obs_t_list).astype(np.float32),
        np.array(act_list, dtype=np.int64),
        np.stack(obs_tp1_list).astype(np.float32),
        np.stack(pred_t_list).astype(np.float32) if with_predicates else None,
        np.stack(pred_tp1_list).astype(np.float32) if with_predicates else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    parser.add_argument("--steps", type=int, default=200_000, help="optimizer steps")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--collect-every", type=int, default=1000,
                        help="refresh rollout buffer every N optimizer steps")
    parser.add_argument("--rollout-size", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--encoder-type", default="categorical", choices=["flat", "categorical"],
        help="categorical (default) uses per-cell type/color/state embeddings — "
             "preserves object identity for downstream linear predicate readout. "
             "flat is the legacy continuous-input encoder kept for backward compat."
    )
    parser.add_argument(
        "--aux-predicate-weight", type=float, default=0.0,
        help="Weight on the auxiliary predicate-supervised BCE loss. >0 attaches "
             "a small predicate readout head and forces the encoder to preserve "
             "object-typed information that pure next-state prediction discards. "
             "Try 1.0 as a starting point."
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    aux_tag = f"_aux{args.aux_predicate_weight:g}" if args.aux_predicate_weight > 0 else ""
    run_name = (
        args.run_name
        or f"jepa_{args.encoder_type}{aux_tag}_{args.env_id}_seed{args.seed}"
    )
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    print(f"[train] writing to {out_dir}")

    # Probe the env once to get n_actions (we don't need a persistent env handle —
    # collect_random_transitions creates its own each refresh).
    _probe_env = gym.make(args.env_id)
    n_actions = _probe_env.action_space.n
    _probe_env.close()

    cfg = JepaConfig(
        n_actions=n_actions,
        encoder_type=args.encoder_type,
        aux_predicate_weight=args.aux_predicate_weight,
    )
    model = JepaWorldModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(
        f"[model] encoder={args.encoder_type} "
        f"aux_predicate_weight={args.aux_predicate_weight} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    rng = np.random.default_rng(args.seed)
    obs_t_buf = act_buf = obs_tp1_buf = pred_t_buf = pred_tp1_buf = None
    loss_window = deque(maxlen=200)
    use_aux = args.aux_predicate_weight > 0.0

    for step in range(args.steps):
        if step % args.collect_every == 0:
            (
                obs_t_buf, act_buf, obs_tp1_buf, pred_t_buf, pred_tp1_buf
            ) = collect_random_transitions(
                args.env_id, args.rollout_size, rng, with_predicates=use_aux
            )

        idx = rng.integers(0, args.rollout_size, size=args.batch_size)
        obs_t = torch.from_numpy(obs_t_buf[idx]).to(device)
        a_t = torch.from_numpy(act_buf[idx]).to(device)
        obs_tp1 = torch.from_numpy(obs_tp1_buf[idx]).to(device)
        preds_t = (
            torch.from_numpy(pred_t_buf[idx]).to(device) if use_aux else None
        )
        preds_tp1 = (
            torch.from_numpy(pred_tp1_buf[idx]).to(device) if use_aux else None
        )

        out = model.loss(
            obs_t, a_t, obs_tp1,
            predicates_t=preds_t, predicates_tp1=preds_tp1,
        )
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.update_target()

        loss_window.append(loss.item())

        if step % 100 == 0:
            mean_loss = float(np.mean(loss_window)) if loss_window else float("nan")
            writer.add_scalar("loss/total", loss.item(), step)
            writer.add_scalar("loss/pred", out["loss_pred"].item(), step)
            writer.add_scalar("loss/reg", out["loss_reg"].item(), step)
            writer.add_scalar("loss/total_mean200", mean_loss, step)
            aux_str = ""
            if "loss_aux_t" in out:
                aux_t_val = out["loss_aux_t"].item()
                writer.add_scalar("loss/aux_t", aux_t_val, step)
                aux_str += f" aux_t={aux_t_val:.4f}"
            if "loss_aux_tp1" in out:
                aux_tp1_val = out["loss_aux_tp1"].item()
                writer.add_scalar("loss/aux_tp1", aux_tp1_val, step)
                aux_str += f" aux_tp1={aux_tp1_val:.4f}"
            print(
                f"[step {step:6d}] loss={loss.item():.4f} "
                f"pred={out['loss_pred'].item():.4f} reg={out['loss_reg'].item():.4f}"
                f"{aux_str} mean200={mean_loss:.4f}"
            )

        if step > 0 and step % 10_000 == 0:
            ckpt = out_dir / f"jepa_step{step}.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, ckpt)
            print(f"[ckpt] saved {ckpt}")

    final = out_dir / "jepa_final.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": args.steps}, final)
    print(f"[done] saved {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
