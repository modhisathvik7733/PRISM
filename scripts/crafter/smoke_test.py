"""Smoke test for the Crafter port.

Verifies (in order):
  1. crafter package is importable
  2. CrafterPrismWrapper constructs and reset/step work end-to-end
  3. Random policy runs to episode termination, reports achievement count
  4. CrafterCNN encoder takes a real obs and produces an embed_dim vector

Run on the GPU box BEFORE writing any training script. If any check
fails, the rest of the port is dead in the water.

    python -m scripts.crafter.smoke_test
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import torch


def check_import() -> bool:
    print("[1/4] importing crafter…", end=" ")
    try:
        import crafter  # noqa: F401
        print("OK")
        return True
    except Exception:
        print("FAILED")
        traceback.print_exc()
        print("\n[install] pip install crafter")
        return False


def check_env() -> bool:
    print("[2/4] constructing CrafterPrismWrapper…", end=" ")
    try:
        from prism.crafter.env_wrapper import make_crafter_env
        env = make_crafter_env(seed=0)
        obs, info = env.reset(seed=0)
        assert obs.shape == (3, 64, 64), f"obs shape {obs.shape}"
        assert obs.dtype == np.float32, f"obs dtype {obs.dtype}"
        assert env.action_space.n == 17, f"expected 17 actions, got {env.action_space.n}"
        next_obs, r, term, trunc, info = env.step(0)
        assert next_obs.shape == (3, 64, 64)
        env.close()
        print(f"OK (obs={obs.shape} {obs.dtype}, n_actions={env.action_space.n})")
        return True
    except Exception:
        print("FAILED")
        traceback.print_exc()
        return False


def check_random_episode() -> bool:
    print("[3/4] running 1 random episode…", end=" ", flush=True)
    try:
        from prism.crafter.env_wrapper import CRAFTER_ACHIEVEMENTS, make_crafter_env
        env = make_crafter_env(seed=0)
        obs, info = env.reset(seed=0)
        rng = np.random.default_rng(0)
        total_R = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(rng.integers(env.action_space.n))
            obs, r, terminated, truncated, info = env.step(action)
            total_R += r
            steps += 1
            if steps > 5000:
                break
        unlocked = info.get("achievements_unlocked", set())
        env.close()
        print(
            f"OK (steps={steps}, total_R={total_R:.2f}, "
            f"achievements_unlocked={len(unlocked)}/{len(CRAFTER_ACHIEVEMENTS)})"
        )
        if unlocked:
            print(f"      unlocked: {sorted(unlocked)}")
        return True
    except Exception:
        print("FAILED")
        traceback.print_exc()
        return False


def check_encoder() -> bool:
    print("[4/4] CrafterCNN forward pass…", end=" ")
    try:
        from prism.crafter.cnn_encoder import CrafterCNN
        from prism.crafter.env_wrapper import make_crafter_env
        env = make_crafter_env(seed=0)
        obs, _ = env.reset(seed=0)
        env.close()
        cnn = CrafterCNN(embed_dim=256)
        x = torch.from_numpy(obs).unsqueeze(0)  # (1, 3, 64, 64)
        z = cnn(x)
        assert z.shape == (1, 256), f"expected (1,256), got {z.shape}"
        n_params = sum(p.numel() for p in cnn.parameters())
        print(f"OK (latent {z.shape}, {n_params:,} params)")
        return True
    except Exception:
        print("FAILED")
        traceback.print_exc()
        return False


def main() -> int:
    checks = [check_import, check_env, check_random_episode, check_encoder]
    results = []
    for fn in checks:
        results.append(fn())
        if not results[-1]:
            print(f"\n[smoke] aborting after first failure ({fn.__name__})")
            return 1
    print("\n[smoke] all checks passed — ready to write the PPO baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
