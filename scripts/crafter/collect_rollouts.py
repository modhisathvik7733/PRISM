"""Collect (obs_t, action, obs_tp1) transitions from a trained baseline policy.

Stores observations as uint8 to keep the file manageable:
  200K transitions × (3×64×64 × 2 obs + 1 action byte) ≈ 4.7 GB uncompressed.
  npz compressed is typically 1.5-2× smaller.

Terminal transitions (episode-end steps) are skipped because the dynamics
model must not be trained across episode boundaries.

Usage:
    python -m scripts.crafter.collect_rollouts \\
        --checkpoint runs/crafter_ppo_baseline/policy_final.pt \\
        --out data/crafter_rollouts.npz \\
        --n-transitions 200000 --n-envs 8 --device cuda

Shapes written to .npz:
  obs_t    (N, 3, 64, 64)  uint8   [0, 255]
  actions  (N,)            uint8   [0, 16]
  obs_tp1  (N, 3, 64, 64)  uint8   [0, 255]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from prism.crafter.env_wrapper import make_crafter_env
from prism.crafter.policy import CrafterPolicy
from prism.utils.seed import set_global_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/crafter_ppo_baseline/policy_final.pt")
    p.add_argument("--out", default="data/crafter_rollouts.npz")
    p.add_argument("--n-transitions", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=5_000_000,
                   help="Base seed disjoint from training (2.1M) and eval (4242).")
    p.add_argument("--state-dim", type=int, default=12,
                   help="Structured game-state dims to save alongside obs (0 = skip).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device(args.device)

    # Load policy.
    ckpt = torch.load(args.checkpoint, map_location=device)
    policy = CrafterPolicy(
        n_actions=ckpt.get("n_actions", 17),
        embed_dim=ckpt.get("embed_dim", 256),
        hidden_dim=ckpt.get("hidden_dim", 256),
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    print(f"[collect] loaded policy from {args.checkpoint}")

    B = args.n_envs
    N = args.n_transitions

    # Separate env instances (not CrafterEnvWorker) so we can access the raw
    # obs_tp1 before any reset overwrites it.
    envs = [
        make_crafter_env(seed=args.seed + i * 1_000_003)
        for i in range(B)
    ]
    obs_np = np.stack([env.reset()[0] for env in envs])  # (B, 3, 64, 64) float32
    hidden = policy.init_hidden(B, device)               # (B, 256)
    prev_actions = torch.full((B,), -1, dtype=torch.long, device=device)

    # Pre-allocate output buffers (uint8 to keep memory manageable).
    obs_t_buf   = np.empty((N, 3, 64, 64), dtype=np.uint8)
    act_buf     = np.empty((N,),           dtype=np.uint8)
    obs_tp1_buf = np.empty((N, 3, 64, 64), dtype=np.uint8)
    save_states = args.state_dim > 0
    if save_states:
        game_states_t = np.empty((N, args.state_dim), dtype=np.float32)

    collected = 0
    steps_taken = 0
    print(f"[collect] collecting {N} non-terminal transitions "
          f"from {B} envs  (seed={args.seed})")

    while collected < N:
        obs_t_np = obs_np.copy()                                    # (B, 3, 64, 64)
        obs_t_t  = torch.from_numpy(obs_np).to(device)

        with torch.no_grad():
            logits, _, h_next = policy.step_with_value(obs_t_t, prev_actions, hidden)
        actions = torch.distributions.Categorical(logits=logits).sample()  # (B,)
        actions_list = actions.cpu().tolist()

        dones = torch.zeros(B, dtype=torch.bool, device=device)

        for i, env in enumerate(envs):
            if collected >= N:
                break
            a = int(actions_list[i])
            # Capture game state BEFORE the step (pre-transition state).
            if save_states:
                pre_state = env.get_game_state()
            obs_tp1, _reward, term, trunc, _info = env.step(a)
            done = term or trunc
            dones[i] = done

            if not done:
                # Non-terminal: valid (obs_t, action, obs_tp1) tuple.
                obs_t_buf[collected]   = (obs_t_np[i] * 255.0).astype(np.uint8)
                act_buf[collected]     = a
                obs_tp1_buf[collected] = (obs_tp1 * 255.0).astype(np.uint8)
                if save_states:
                    game_states_t[collected] = pre_state
                collected += 1
                obs_np[i] = obs_tp1
            else:
                obs_np[i], _ = env.reset()

        steps_taken += B
        # Reset hidden and prev_actions for envs that finished.
        zero_h = policy.init_hidden(B, device)
        hidden = torch.where(dones.unsqueeze(1), zero_h, h_next)
        prev_actions = torch.where(dones, torch.full_like(actions, -1), actions)

        if collected % 10_000 < B:
            terminal_frac = 1.0 - collected / max(steps_taken, 1)
            print(f"[collect] {collected:>7d}/{N}  "
                  f"(steps={steps_taken}, skip_frac={terminal_frac:.3f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[collect] saving {out_path} ...")
    save_kwargs = dict(obs_t=obs_t_buf, actions=act_buf, obs_tp1=obs_tp1_buf)
    if save_states:
        save_kwargs["game_states_t"] = game_states_t
    np.savez_compressed(out_path, **save_kwargs)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[collect] done — {N} transitions, {size_mb:.0f} MB  →  {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
