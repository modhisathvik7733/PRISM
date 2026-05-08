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

from prism.envs import make_babyai_env
from prism.models.jepa import JepaConfig, JepaWorldModel
from prism.utils.seed import set_global_seed


def collect_random_transitions(env, n: int, rng: np.random.Generator):
    """Collect n one-step transitions under random policy. Returns CHW float32 arrays."""
    obs_t_list, act_list, obs_tp1_list = [], [], []
    obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
    while len(obs_t_list) < n:
        a = int(rng.integers(env.action_space.n))
        next_obs, _r, term, trunc, _ = env.step(a)
        obs_t_list.append(obs["image"] if isinstance(obs, dict) else obs)
        act_list.append(a)
        obs_tp1_list.append(next_obs["image"] if isinstance(next_obs, dict) else next_obs)
        if term or trunc:
            obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        else:
            obs = next_obs
    return (
        np.stack(obs_t_list).astype(np.float32),
        np.array(act_list, dtype=np.int64),
        np.stack(obs_tp1_list).astype(np.float32),
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
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    run_name = args.run_name or f"jepa_{args.env_id}_seed{args.seed}"
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    print(f"[train] writing to {out_dir}")

    env = make_babyai_env(args.env_id, include_mission=False)
    cfg = JepaConfig(n_actions=env.action_space.n)
    model = JepaWorldModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}")

    rng = np.random.default_rng(args.seed)
    obs_t_buf = act_buf = obs_tp1_buf = None
    loss_window = deque(maxlen=200)

    for step in range(args.steps):
        if step % args.collect_every == 0:
            obs_t_buf, act_buf, obs_tp1_buf = collect_random_transitions(
                env, args.rollout_size, rng
            )

        idx = rng.integers(0, args.rollout_size, size=args.batch_size)
        obs_t = torch.from_numpy(obs_t_buf[idx]).to(device)
        a_t = torch.from_numpy(act_buf[idx]).to(device)
        obs_tp1 = torch.from_numpy(obs_tp1_buf[idx]).to(device)

        out = model.loss(obs_t, a_t, obs_tp1)
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
            print(
                f"[step {step:6d}] loss={loss.item():.4f} "
                f"pred={out['loss_pred'].item():.4f} reg={out['loss_reg'].item():.4f} "
                f"mean200={mean_loss:.4f}"
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
