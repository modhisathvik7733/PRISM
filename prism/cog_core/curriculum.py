"""ALPCurriculum — task scheduler using Absolute Learning Progress
bandit.

Implements the Akakzia et al. (2021) refinement of CURIOUS
(Colas et al. 2018):

    value(task) = (1 - success_rate(task)) * |learning_progress(task)|

This favors tasks the agent is MAKING PROGRESS on AND has not yet
mastered — exactly what a developmental scheduler should pick.

  - If success_rate is near 1 (task mastered) → value → 0 → don't pick
  - If success_rate is near 0 AND |LP| is high → value high → pick
  - If success_rate is near 0 AND |LP| is ~0 → don't pick (stuck task,
    not learnable now; come back later)

Used in two ways:
  1. Within Phase 1: pick which BabyAI env to train PPO on next, per
     episode, based on per-env success-rate evolution
  2. In future phases: pick which Stage-1+ corpus chunk to train on
     next (same algorithm, different "tasks")
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Hashable

import numpy as np


@dataclass
class TaskStats:
    task_id: Hashable
    n_attempts: int = 0
    recent_success: float = 0.5      # EMA, fast (alpha=0.1 default)
    older_success: float = 0.5       # EMA, slow (lags recent_success)


class ALPCurriculum:
    """Multi-armed bandit on absolute learning progress.

    Args:
        task_ids: hashable list/tuple of task identifiers.
        alpha_recent: EMA rate for recent success (default 0.1 — last
                       ~10 attempts dominate)
        alpha_older: EMA rate for the lagging baseline (default 0.02 —
                       last ~50 attempts dominate)
        eps: epsilon-greedy probability of picking a random task
              instead of the argmax-value task (default 0.1)
    """

    def __init__(
        self,
        task_ids: list[Hashable],
        *,
        alpha_recent: float = 0.1,
        alpha_older: float = 0.02,
        eps: float = 0.1,
        seed: int = 0,
    ):
        self.task_ids = list(task_ids)
        self.alpha_recent = alpha_recent
        self.alpha_older = alpha_older
        self.eps = eps
        self._rng = random.Random(seed)
        self.stats: dict[Hashable, TaskStats] = {
            t: TaskStats(task_id=t) for t in task_ids
        }
        # Scheduling history for analysis: list of (task_id, success_bool, value).
        self.history: list[tuple[Hashable, bool, float]] = []

    def update(self, task_id: Hashable, success: bool) -> None:
        """Call after each episode completes. `success` is bool (1/0)."""
        s = self.stats[task_id]
        s.n_attempts += 1
        # Update older first (lags) using the CURRENT recent value as input.
        s.older_success = (
            (1 - self.alpha_older) * s.older_success
            + self.alpha_older * s.recent_success
        )
        # Then update recent with the new observation.
        s.recent_success = (
            (1 - self.alpha_recent) * s.recent_success
            + self.alpha_recent * float(success)
        )
        self.history.append((task_id, bool(success), self.value(task_id)))

    def value(self, task_id: Hashable) -> float:
        """Akakzia: (1 - success_rate) × |LP|."""
        s = self.stats[task_id]
        sr = s.recent_success
        lp = abs(s.recent_success - s.older_success)
        return (1.0 - sr) * lp

    def propose(self) -> Hashable:
        """Pick the next task. Epsilon-greedy on values."""
        if self._rng.random() < self.eps:
            return self._rng.choice(self.task_ids)
        values = [(self.value(t), t) for t in self.task_ids]
        # Tie-break randomly so we don't always grab the first task with
        # value 0 at startup.
        max_val = max(v for v, _ in values)
        if max_val == 0:
            return self._rng.choice(self.task_ids)
        candidates = [t for v, t in values if v == max_val]
        return self._rng.choice(candidates)

    def snapshot(self) -> dict:
        """Current per-task state — for logging / Phase 1 emergence eval."""
        return {
            t: {
                "n_attempts": s.n_attempts,
                "recent_success": s.recent_success,
                "older_success": s.older_success,
                "value": self.value(t),
            }
            for t, s in self.stats.items()
        }

    def schedule_summary(self, last_n: int | None = None) -> dict:
        """How often each task was picked. last_n limits the window."""
        h = self.history if last_n is None else self.history[-last_n:]
        if not h:
            return {t: 0.0 for t in self.task_ids}
        counts: dict[Hashable, int] = {t: 0 for t in self.task_ids}
        for tid, _, _ in h:
            counts[tid] += 1
        total = len(h)
        return {t: counts[t] / total for t in self.task_ids}


# ---------------------------------------------------- random-baseline scheduler
class RandomScheduler:
    """Uniform-random task picker — used as the control in the Phase 1
    'curriculum beats random' emergence test."""

    def __init__(self, task_ids: list[Hashable], seed: int = 0):
        self.task_ids = list(task_ids)
        self._rng = random.Random(seed)
        self.stats = {t: TaskStats(task_id=t) for t in task_ids}
        self.history: list[tuple[Hashable, bool, float]] = []

    def propose(self) -> Hashable:
        return self._rng.choice(self.task_ids)

    def update(self, task_id: Hashable, success: bool) -> None:
        s = self.stats[task_id]
        s.n_attempts += 1
        s.recent_success = 0.95 * s.recent_success + 0.05 * float(success)
        self.history.append((task_id, bool(success), 0.0))
