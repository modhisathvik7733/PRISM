"""Per-env state for the Crafter PPO trainer.

One worker = one CrafterPrismEnv + per-env bookkeeping (current obs,
episode reward, steps, unlocked-achievement set). Mirrors the layout of
`scripts.ppo_train.EnvWorker` but adapted for Crafter:
  - No mission, no goal predicates, no allowed-actions mask (all 17
    Crafter actions are always available).
  - No reward shaping (Crafter's native reward is dense).
  - Achievement set is reported in the episode summary so the trainer
    can log per-episode achievement counts.
"""

from __future__ import annotations

import numpy as np

from prism.crafter.env_wrapper import CRAFTER_ACHIEVEMENTS, make_crafter_env


class CrafterEnvWorker:
    def __init__(self, base_seed: int, worker_id: int, reward_mode: str = "reward"):
        self.base_seed = base_seed
        self.worker_id = worker_id
        self.episode_idx = 0
        self.env = make_crafter_env(reward_mode=reward_mode,
                                    seed=base_seed + worker_id * 1_000_003)
        self._reset_episode()

    def _reset_episode(self):
        seed = self.base_seed + self.worker_id * 1_000_003 + self.episode_idx * 7919
        self.episode_idx += 1
        try:
            obs, _ = self.env.reset(seed=seed)
        except TypeError:
            obs, _ = self.env.reset()
        self.obs = obs                   # (3, 64, 64) float32
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.prev_action = -1
        self.achievements: set[str] = set()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        next_obs, reward, term, trunc, info = self.env.step(int(action))
        done = term or trunc
        self.episode_reward += reward
        self.episode_steps += 1
        self.prev_action = int(action)
        unlocked = info.get("achievements_unlocked", set())
        # Track new achievements as they're unlocked. The set is also
        # available on the wrapper, but copying here means the summary
        # we emit on episode end is self-contained.
        self.achievements |= unlocked

        if done:
            ep_summary = {
                "ep_reward": self.episode_reward,
                "ep_steps": self.episode_steps,
                "achievements": set(self.achievements),
                "n_achievements": len(self.achievements),
            }
            self._reset_episode()
            return self.obs, float(reward), True, ep_summary
        else:
            self.obs = next_obs
            return self.obs, float(reward), False, {}


def aggregate_achievement_score(per_episode_unlocks: list[set[str]]) -> tuple[float, dict[str, float]]:
    """Geometric-mean achievement score (the Crafter paper's metric).

    Score = exp(1/N * Σ ln(1 + s_i)) - 1, in percent — where s_i is the
    success RATE (in %) for achievement i across all episodes.
    Returns (score_percent, per_achievement_rates).
    """
    if not per_episode_unlocks:
        return 0.0, {}
    n = len(per_episode_unlocks)
    rates: dict[str, float] = {}
    for ach in CRAFTER_ACHIEVEMENTS:
        unlocked = sum(1 for s in per_episode_unlocks if ach in s)
        rates[ach] = 100.0 * unlocked / n
    log_terms = [np.log1p(rates[a]) for a in CRAFTER_ACHIEVEMENTS]
    score = float(np.exp(np.mean(log_terms)) - 1)
    return score, rates
