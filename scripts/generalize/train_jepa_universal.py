"""Train a single universal JEPA on transitions sampled from multiple BabyAI
levels, so the same world model can serve all downstream policies.

This is a thin reformulation of `scripts/train_jepa.py` — same encoder
config, same loss, same optimizer — but the per-iteration rollout buffer
draws round-robin from a list of envs instead of one. The output checkpoint
is drop-in compatible with every downstream consumer (BC trainer, PPO,
eval): they all just read `cfg` from the checkpoint and load `model`.

Usage:
    python -m scripts.generalize.train_jepa_universal \
        --envs BabyAI-GoToLocal-v0 BabyAI-Pickup-v0 BabyAI-GoTo-v0 BabyAI-Open-v0 \
        --steps 100000 --batch-size 128 \
        --encoder-type categorical_spatial --spatial-channels 64 \
        --dynamics-type spatial_film --dynamics-hidden 256 --dynamics-layers 3 \
        --aux-predicate-weight 3.0 --aux-distance-dim 24 --aux-distance-weight 0.5 \
        --run-name jepa_universal --device cuda
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from prism.envs.babyai import _encode_image
from prism.models.jepa import JepaConfig, JepaWorldModel
from prism.perception import (
    compute_augmented_predicates,
    compute_predicates,
    extract_slots,
)
from prism.utils.seed import set_global_seed


def collect_random_transitions_multi(
    env_ids: list[str],
    n_total: int,
    rng: np.random.Generator,
    *,
    with_predicates: bool,
    augmented: bool,
):
    """Collect `n_total` random-policy transitions, split round-robin across
    `env_ids`. Each env gets its own gym.make + per-call seed so episodes
    don't interleave state. Returns the same tuple shape as the single-env
    version in `scripts/train_jepa.py`."""
    n_per_env = n_total // len(env_ids)
    obs_t_lists, act_lists, obs_tp1_lists = [], [], []
    pred_t_lists, pred_tp1_lists = [], []

    for env_id in env_ids:
        env = gym.make(env_id)
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        kept = 0
        while kept < n_per_env:
            raw_t = obs["image"]
            a = int(rng.integers(env.action_space.n))
            next_obs, _r, term, trunc, _ = env.step(a)
            raw_tp1 = next_obs["image"]

            obs_t_lists.append(_encode_image(raw_t))
            act_lists.append(a)
            obs_tp1_lists.append(_encode_image(raw_tp1))
            if with_predicates:
                pred_fn = compute_augmented_predicates if augmented else compute_predicates
                pred_t_lists.append(pred_fn(extract_slots(raw_t)))
                pred_tp1_lists.append(pred_fn(extract_slots(raw_tp1)))

            kept += 1
            if term or trunc:
                obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
            else:
                obs = next_obs
        env.close()

    return (
        np.stack(obs_t_lists).astype(np.float32),
        np.array(act_lists, dtype=np.int64),
        np.stack(obs_tp1_lists).astype(np.float32),
        np.stack(pred_t_lists).astype(np.float32) if with_predicates else None,
        np.stack(pred_tp1_lists).astype(np.float32) if with_predicates else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+",
                        default=["BabyAI-GoToLocal-v0", "BabyAI-Pickup-v0",
                                 "BabyAI-GoTo-v0", "BabyAI-Open-v0"])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--collect-every", type=int, default=1000)
    parser.add_argument("--rollout-size", type=int, default=10_000,
                        help="total transitions per refresh — split round-robin "
                             "across --envs (so per-env rollout = rollout_size / N).")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-name", default="jepa_universal")
    parser.add_argument("--encoder-type", default="categorical_spatial",
                        choices=["flat", "categorical", "categorical_spatial"])
    parser.add_argument("--aux-predicate-weight", type=float, default=3.0)
    parser.add_argument("--aux-distance-dim", type=int, default=24)
    parser.add_argument("--aux-distance-weight", type=float, default=0.5)
    parser.add_argument("--dynamics-hidden", type=int, default=256)
    parser.add_argument("--dynamics-layers", type=int, default=3)
    parser.add_argument("--dynamics-type", default="spatial_film",
                        choices=["mlp", "film", "spatial_film"])
    parser.add_argument("--spatial-channels", type=int, default=64)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    print(f"[universal-jepa] writing to {out_dir}")
    print(f"[universal-jepa] envs: {args.envs}")

    # All BabyAI envs share the same 7-action space; verify on the first one
    # so the JepaConfig is consistent and inconsistencies surface loudly.
    n_actions_set = set()
    for env_id in args.envs:
        e = gym.make(env_id)
        n_actions_set.add(e.action_space.n)
        e.close()
    if len(n_actions_set) != 1:
        raise SystemExit(f"envs disagree on n_actions: {n_actions_set}")
    n_actions = next(iter(n_actions_set))

    cfg = JepaConfig(
        n_actions=n_actions,
        encoder_type=args.encoder_type,
        aux_predicate_weight=args.aux_predicate_weight,
        aux_distance_dim=args.aux_distance_dim,
        aux_distance_weight=args.aux_distance_weight,
        dynamics_hidden_dim=args.dynamics_hidden,
        dynamics_layers=args.dynamics_layers,
        dynamics_type=args.dynamics_type,
        spatial_channels=args.spatial_channels,
    )
    model = JepaWorldModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(
        f"[universal-jepa] encoder={args.encoder_type} dyn={args.dynamics_type} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    rng = np.random.default_rng(args.seed)
    obs_t_buf = act_buf = obs_tp1_buf = pred_t_buf = pred_tp1_buf = None
    loss_window = deque(maxlen=200)
    use_aux = args.aux_predicate_weight > 0.0
    use_distance = args.aux_distance_dim > 0
    if use_distance:
        print(f"[universal-jepa] distance head dim={args.aux_distance_dim} "
              f"weight={args.aux_distance_weight}")

    for step in range(args.steps):
        if step % args.collect_every == 0:
            (
                obs_t_buf, act_buf, obs_tp1_buf, pred_t_buf, pred_tp1_buf
            ) = collect_random_transitions_multi(
                args.envs, args.rollout_size, rng,
                with_predicates=use_aux, augmented=use_distance,
            )

        idx = rng.integers(0, args.rollout_size, size=args.batch_size)
        obs_t = torch.from_numpy(obs_t_buf[idx]).to(device)
        a_t = torch.from_numpy(act_buf[idx]).to(device)
        obs_tp1 = torch.from_numpy(obs_tp1_buf[idx]).to(device)
        preds_t = torch.from_numpy(pred_t_buf[idx]).to(device) if use_aux else None
        preds_tp1 = torch.from_numpy(pred_tp1_buf[idx]).to(device) if use_aux else None

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
            for key in ("loss_aux_t", "loss_aux_tp1", "loss_dist_t", "loss_dist_tp1"):
                if key in out:
                    val = out[key].item()
                    writer.add_scalar(f"loss/{key.replace('loss_', '')}", val, step)
                    aux_str += f" {key.replace('loss_', '')}={val:.4f}"
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
