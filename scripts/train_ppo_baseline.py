"""Phase 0 — PPO baseline on BabyAI-GoToLocal-v0.

This is the *sanity baseline*: a vanilla PPO with a small CNN policy on the
image-only observation (no mission conditioning). Goal is to confirm that:

  (a) the harness works end-to-end (env → policy → optimizer → eval),
  (b) the simplest BabyAI level is solvable by a standard agent in <~1M steps.

If this fails, fix the harness *before* layering JEPA / operators / memory
on top.

Falsifier (Phase 0): mean episode return stays near zero after 1M env steps
on `BabyAI-GoToLocal-v0` → harness is broken or env wiring is wrong.

Run:
    uv run python -m scripts.train_ppo_baseline \
        --env-id BabyAI-GoToLocal-v0 \
        --total-timesteps 1_000_000 \
        --n-envs 8 \
        --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from prism.envs import make_babyai_env
from prism.utils.seed import set_global_seed


def make_env_fn(env_id: str, seed: int):
    def _thunk():
        env = make_babyai_env(env_id, seed=seed, include_mission=False)
        return env
    return _thunk


def main() -> int:
    parser = argparse.ArgumentParser(description="PRISM Phase 0 PPO baseline")
    parser.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--vec", default="dummy", choices=["dummy", "subproc"])
    args = parser.parse_args()

    set_global_seed(args.seed)

    run_name = args.run_name or f"ppo_baseline_{args.env_id.replace('/', '_')}_seed{args.seed}"
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] writing to {out_dir}")

    VecEnv = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    venv = VecEnv([make_env_fn(args.env_id, seed=args.seed + i) for i in range(args.n_envs)])

    # 7x7x3 = 147 features — too small for SB3's NatureCNN (which requires
    # ≥36x36 images). MlpPolicy auto-flattens via FlattenExtractor and is the
    # right baseline at this scale anyway.
    #
    # Hyperparameters tuned for BabyAI's sparse reward + short episodes:
    #   * n_steps=256 — bigger rollout so each PPO update sees more completed
    #     episodes (BabyAI episodes cap at ~64 steps).
    #   * ent_coef=0.001 — at 0.01 entropy was actually rising during training
    #     (agent stayed too random). Lower so the policy commits.
    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        device=args.device,
        seed=args.seed,
        n_steps=256,
        batch_size=512,
        learning_rate=3e-4,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [128, 128]},
        tensorboard_log=str(out_dir / "tb"),
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        tb_log_name="ppo",
    )

    ckpt = out_dir / "ppo_final.zip"
    model.save(ckpt)
    print(f"[train] saved {ckpt}")

    # ---- quick eval --------------------------------------------------------
    eval_env = make_babyai_env(args.env_id, seed=args.seed + 9999, include_mission=False)
    n_eval = 20
    rewards = []
    for ep in range(n_eval):
        obs, _ = eval_env.reset(seed=args.seed + 10000 + ep)
        done = False
        ep_r = 0.0
        steps = 0
        while not done and steps < 256:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = eval_env.step(int(action))
            ep_r += float(r)
            steps += 1
            done = term or trunc
        rewards.append(ep_r)
    mean_r = sum(rewards) / len(rewards)
    print(f"[eval] env={args.env_id} mean_reward={mean_r:.3f} over {n_eval} episodes")

    # Phase 0 falsifier: BabyAI-GoToLocal returns ~0.95 for solved, 0 for unsolved.
    if mean_r < 0.1:
        print("[eval] WARNING: near-zero reward — Phase 0 falsifier may have fired.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
