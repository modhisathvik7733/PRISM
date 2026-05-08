"""Phase 0 smoke test.

Confirms the BabyAI substrate loads and steps cleanly under a random policy,
and that PyTorch sees the GPU. No training. Run on Vast.ai after `setup.sh`.

    uv run python -m scripts.smoke_test
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from prism.envs import make_babyai_env
from prism.utils.seed import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="PRISM Phase 0 smoke test")
    parser.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_global_seed(args.seed)

    # ---- env ---------------------------------------------------------------
    env = make_babyai_env(args.env_id, seed=args.seed, include_mission=True)
    print(f"[env] {args.env_id}")
    print(f"[env] obs space: {env.observation_space}")
    print(f"[env] act space: {env.action_space}")

    rng = np.random.default_rng(args.seed)

    total_steps = 0
    successes = 0
    t0 = time.time()
    last_mission = None

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        if obs["mission"] != last_mission:
            print(f"[ep {ep:02d}] mission: {obs['mission']!r}")
            last_mission = obs["mission"]

        ep_reward = 0.0
        for step in range(args.max_steps):
            action = int(rng.integers(env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            total_steps += 1
            if terminated or truncated:
                break
        if ep_reward > 0:
            successes += 1
        print(f"[ep {ep:02d}] steps={step+1:3d} reward={ep_reward:.3f}")

    dt = time.time() - t0
    sps = total_steps / max(dt, 1e-6)
    print(
        f"[summary] episodes={args.episodes} success={successes}/{args.episodes} "
        f"steps={total_steps} sps={sps:.0f} wall={dt:.2f}s"
    )

    # ---- torch / cuda ------------------------------------------------------
    try:
        import torch

        print(f"[torch] version={torch.__version__}")
        print(f"[torch] cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[torch] device_count={torch.cuda.device_count()}")
            print(f"[torch] device_0={torch.cuda.get_device_name(0)}")
            # quick GPU op
            x = torch.randn(1024, 1024, device="cuda")
            y = (x @ x.T).sum().item()
            print(f"[torch] gpu matmul ok (sum={y:.2e})")
    except ImportError:
        print("[torch] not installed")
        return 1

    # ---- jepa quick check --------------------------------------------------
    try:
        import torch

        from prism.models.jepa import JepaConfig, JepaWorldModel

        cfg = JepaConfig()
        model = JepaWorldModel(cfg)
        if torch.cuda.is_available():
            model = model.cuda()
        B = 8
        obs_t = torch.rand(B, cfg.obs_channels, cfg.obs_h, cfg.obs_w,
                           device=next(model.parameters()).device)
        obs_tp1 = torch.rand_like(obs_t)
        a = torch.randint(0, cfg.n_actions, (B,), device=obs_t.device)
        out = model.loss(obs_t, a, obs_tp1)
        print(f"[jepa] loss={out['loss'].item():.4f} pred={out['loss_pred'].item():.4f} "
              f"reg={out['loss_reg'].item():.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"[jepa] FAILED: {e}")
        return 1

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
