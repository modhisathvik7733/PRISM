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
        "--encoder-type", default="categorical",
        choices=["flat", "categorical", "categorical_spatial"],
        help="categorical (default) uses per-cell embeddings then flattens to "
             "embed_dim. categorical_spatial preserves spatial structure — "
             "encoder outputs (B, C, H, W) so a convolutional dynamics can "
             "express rotations as approximate spatial permutations. Use with "
             "--dynamics-type spatial_film for the LeWorldModel-style stack. "
             "flat is the legacy continuous-input encoder kept for back-compat."
    )
    parser.add_argument(
        "--aux-predicate-weight", type=float, default=0.0,
        help="Weight on the auxiliary predicate-supervised BCE loss. >0 attaches "
             "a small predicate readout head and forces the encoder to preserve "
             "object-typed information that pure next-state prediction discards. "
             "Try 1.0 as a starting point."
    )
    parser.add_argument(
        "--agent-data-path", default=None,
        help="Optional path to a .npz of goal-directed transitions collected by "
             "scripts/collect_agent_data.py. If set, each training batch is built "
             "by mixing random rollouts and agent transitions per --agent-data-mix. "
             "This is the policy-iteration step — random rollouts under-represent "
             "trajectories that actually approach targets, so the dynamics model "
             "is degraded on the transitions the agent needs at inference."
    )
    parser.add_argument(
        "--agent-data-mix", type=float, default=0.5,
        help="Fraction of each batch drawn from agent data (0.0 = none, 1.0 = "
             "agent only). Default 0.5 = balanced mix. Ignored if --agent-data-path "
             "is not set."
    )
    parser.add_argument(
        "--dynamics-hidden", type=int, default=256,
        help="LatentDynamics MLP hidden width. Default 256 = original. Use 512 "
             "to test whether rotation-action prediction (turn_left/turn_right "
             "F1 ~0.55 in eval_dynamics_predicates) is capacity-bound."
    )
    parser.add_argument(
        "--dynamics-layers", type=int, default=2,
        help="Number of (Linear+GELU) blocks in LatentDynamics before the "
             "output projection. Default 2 = original. Try 4 alongside "
             "--dynamics-hidden 512."
    )
    parser.add_argument(
        "--dynamics-type", default="mlp", choices=["mlp", "film", "spatial_film"],
        help="mlp = concat(z, action_embed) → MLP. film = flat-latent FiLM. "
             "spatial_film = convolutional FiLM dynamics over a spatial latent "
             "(requires --encoder-type categorical_spatial). Use spatial_film "
             "to address the rotation-prediction failure that flat-latent "
             "architectures cannot fix (turn F1 capped at ~0.55)."
    )
    parser.add_argument(
        "--spatial-channels", type=int, default=64,
        help="Channel count C for the spatial encoder/dynamics latent "
             "(B, C, H, W). Only used when encoder-type=categorical_spatial. "
             "Default 64."
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    aux_tag = f"_aux{args.aux_predicate_weight:g}" if args.aux_predicate_weight > 0 else ""
    mix_tag = (
        f"_mix{args.agent_data_mix:g}" if args.agent_data_path is not None else ""
    )
    dyn_tag = (
        f"_dyn{args.dynamics_layers}x{args.dynamics_hidden}"
        if (args.dynamics_hidden != 256 or args.dynamics_layers != 2)
        else ""
    )
    if args.dynamics_type != "mlp":
        dyn_tag = f"_{args.dynamics_type}{dyn_tag}"
    if args.encoder_type == "categorical_spatial":
        dyn_tag = f"_spat{args.spatial_channels}{dyn_tag}"
    run_name = (
        args.run_name
        or f"jepa_{args.encoder_type}{aux_tag}{mix_tag}{dyn_tag}_{args.env_id}_seed{args.seed}"
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
        dynamics_hidden_dim=args.dynamics_hidden,
        dynamics_layers=args.dynamics_layers,
        dynamics_type=args.dynamics_type,
        spatial_channels=args.spatial_channels,
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

    # Optional agent data — pre-collected goal-directed trajectories from
    # scripts/collect_agent_data.py. We sample a fraction of every batch
    # from this pool to expose the dynamics model to "approach target"
    # transitions that random rollouts under-represent.
    agent_data = None
    n_agent_per_batch = 0
    n_random_per_batch = args.batch_size
    if args.agent_data_path is not None:
        print(f"[train] loading agent data from {args.agent_data_path}")
        npz = np.load(args.agent_data_path)
        agent_data = {
            "obs_t":         npz["obs_t"].astype(np.float32),
            "actions":       npz["actions"].astype(np.int64),
            "obs_tp1":       npz["obs_tp1"].astype(np.float32),
            "predicates_t":  npz["predicates_t"].astype(np.float32),
            "predicates_tp1": npz["predicates_tp1"].astype(np.float32),
        }
        n_agent_total = agent_data["obs_t"].shape[0]
        n_agent_per_batch = int(round(args.batch_size * args.agent_data_mix))
        n_random_per_batch = args.batch_size - n_agent_per_batch
        print(
            f"[train] agent data: {n_agent_total} transitions; "
            f"per batch: {n_random_per_batch} random + {n_agent_per_batch} agent"
        )

    for step in range(args.steps):
        if step % args.collect_every == 0:
            (
                obs_t_buf, act_buf, obs_tp1_buf, pred_t_buf, pred_tp1_buf
            ) = collect_random_transitions(
                args.env_id, args.rollout_size, rng, with_predicates=use_aux
            )

        # ------------------------------------------------ build batch
        if n_random_per_batch > 0:
            r_idx = rng.integers(0, args.rollout_size, size=n_random_per_batch)
            r_obs_t = obs_t_buf[r_idx]
            r_act = act_buf[r_idx]
            r_obs_tp1 = obs_tp1_buf[r_idx]
            r_pred_t = pred_t_buf[r_idx] if use_aux else None
            r_pred_tp1 = pred_tp1_buf[r_idx] if use_aux else None
        else:
            r_obs_t = r_act = r_obs_tp1 = r_pred_t = r_pred_tp1 = None

        if agent_data is not None and n_agent_per_batch > 0:
            n_agent_total = agent_data["obs_t"].shape[0]
            a_idx = rng.integers(0, n_agent_total, size=n_agent_per_batch)
            a_obs_t = agent_data["obs_t"][a_idx]
            a_act = agent_data["actions"][a_idx]
            a_obs_tp1 = agent_data["obs_tp1"][a_idx]
            a_pred_t = agent_data["predicates_t"][a_idx] if use_aux else None
            a_pred_tp1 = agent_data["predicates_tp1"][a_idx] if use_aux else None
            obs_t_np = (
                np.concatenate([r_obs_t, a_obs_t]) if r_obs_t is not None else a_obs_t
            )
            act_np = np.concatenate([r_act, a_act]) if r_act is not None else a_act
            obs_tp1_np = (
                np.concatenate([r_obs_tp1, a_obs_tp1])
                if r_obs_tp1 is not None else a_obs_tp1
            )
            pred_t_np = (
                np.concatenate([r_pred_t, a_pred_t])
                if (use_aux and r_pred_t is not None) else (a_pred_t if use_aux else None)
            )
            pred_tp1_np = (
                np.concatenate([r_pred_tp1, a_pred_tp1])
                if (use_aux and r_pred_tp1 is not None)
                else (a_pred_tp1 if use_aux else None)
            )
        else:
            obs_t_np, act_np, obs_tp1_np = r_obs_t, r_act, r_obs_tp1
            pred_t_np, pred_tp1_np = r_pred_t, r_pred_tp1

        obs_t = torch.from_numpy(obs_t_np).to(device)
        a_t = torch.from_numpy(act_np).to(device)
        obs_tp1 = torch.from_numpy(obs_tp1_np).to(device)
        preds_t = torch.from_numpy(pred_t_np).to(device) if use_aux else None
        preds_tp1 = torch.from_numpy(pred_tp1_np).to(device) if use_aux else None

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
