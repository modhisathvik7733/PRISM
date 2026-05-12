"""ParallelEnvWorkers — subprocess-parallel env stepping for ppo_train.

The bottleneck in the serial EnvWorker path is that 32 BabyAI envs step
sequentially in Python per rollout step. Each env.step is sub-ms but
the loop dominates wall-clock and the GPU sits idle (~30% utilization).

This module forks N subprocesses on construction. Each subprocess
owns one EnvWorker. The main process sends actions to all workers in
one batch, then reads back state in another batch — IPC happens
twice per rollout step regardless of N, while the env stepping
parallelizes across CPU cores.

Realistic speedup is 2-3× on BabyAI at n_envs=32 (limited by
core count, pickle overhead, and the GPU phase that's already fast).

API:
    pool = ParallelEnvWorkers(env_id, n_envs, **worker_kwargs)
    state = pool.current_state()          # {'obs_encoded', 'mission_oh',
                                          #  'allowed_lists', 'mem_feat'}
    state, rewards, dones, infos = pool.step_all(actions)
    pool.close()

Workers are forked from the parent process, so the parent's already-
imported modules (gym, minigrid, prism) are inherited — no per-worker
re-import. Fork is the default mp start method on Linux; we assert it
on construction so macOS-spawn or Windows-spawn don't silently fall
through to slow paths.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

import numpy as np


# Sentinel commands sent on the pipe.
_CMD_STEP = "step"
_CMD_RESET = "reset"
_CMD_CLOSE = "close"


def _worker_loop(conn, worker_kwargs: dict) -> None:
    """Subprocess body. Constructs an EnvWorker, loops on commands from
    `conn`, sends results back.

    Result dict for step/reset: {
        'obs_encoded': np.ndarray (3, 7, 7) int8,
        'mission_oh': np.ndarray (mission_dim,) float32,
        'allowed': tuple[int, ...],
        'mem_feat': np.ndarray | None,
        'reward': float,         # 0.0 for reset
        'done': bool,             # False for reset
        'info': dict,             # {} for reset; ep_reward/ep_steps on done
    }
    """
    # Import inside the subprocess so the EnvWorker class is fully
    # constructible. Under fork on Linux, scripts.ppo_train is already
    # imported in the parent; under spawn this re-imports it.
    from scripts.ppo_train import EnvWorker

    worker = EnvWorker(**worker_kwargs)

    def _state_dict(reward: float = 0.0, done: bool = False, info: dict | None = None) -> dict:
        return {
            "obs_encoded": worker.obs_encoded,
            "mission_oh": worker.mission_oh,
            "allowed": worker.allowed,
            "mem_feat": worker.mem_feat,
            "reward": reward,
            "done": done,
            "info": info or {},
        }

    try:
        while True:
            cmd = conn.recv()
            if cmd[0] == _CMD_STEP:
                action = int(cmd[1])
                _obs, r, d, info = worker.step(action)
                conn.send(_state_dict(reward=float(r), done=bool(d), info=info or {}))
            elif cmd[0] == _CMD_RESET:
                # EnvWorker auto-resets internally; we expose its current state.
                conn.send(_state_dict())
            elif cmd[0] == _CMD_CLOSE:
                conn.send(("closed",))
                break
            else:
                conn.send(("error", f"unknown cmd {cmd[0]!r}"))
                break
    except Exception as e:
        # Propagate exception to parent rather than dying silently.
        import traceback
        conn.send(("error", f"worker {worker_kwargs.get('worker_id', '?')} died: "
                            f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    finally:
        conn.close()


class ParallelEnvWorkers:
    """Subprocess pool that mirrors the list-of-EnvWorker API in batched form.

    Construct with the same kwargs you'd pass to a single EnvWorker, plus
    `n_envs`. Worker IDs are 0..n_envs-1 (override via worker_id_offset
    if you need disjoint seeds across pools, e.g. across env swaps).
    """

    def __init__(self, env_id: str, n_envs: int, base_seed: int,
                 mission_dim: int, n_actions: int,
                 max_steps: int = 64, shaping_coef: float = 0.0,
                 use_pose_tracker: bool = False,
                 goal_provider=None,
                 held_out_combos: set[tuple[int, int]] | None = None):
        # Fork is required on Linux for fast spawn + parent-import inheritance.
        # Spawn would re-import everything per worker (slow) and require
        # goal_provider to be picklable (it might not be).
        ctx = mp.get_context("fork")
        if goal_provider is not None:
            raise NotImplementedError(
                "ParallelEnvWorkers does not yet support goal_provider; "
                "the language head is not safely shared across processes. "
                "Use serial workers when goal_provider != None."
            )

        self.n_envs = n_envs
        self._closed = False
        self._procs: list[mp.process.BaseProcess] = []
        self._conns: list = []

        common_kwargs = dict(
            env_id=env_id,
            base_seed=base_seed,
            mission_dim=mission_dim,
            n_actions=n_actions,
            max_steps=max_steps,
            shaping_coef=shaping_coef,
            use_pose_tracker=use_pose_tracker,
            goal_provider=None,
            held_out_combos=held_out_combos,
        )

        for i in range(n_envs):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            worker_kwargs = {**common_kwargs, "worker_id": i}
            proc = ctx.Process(
                target=_worker_loop,
                args=(child_conn, worker_kwargs),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # parent only uses parent_conn
            self._procs.append(proc)
            self._conns.append(parent_conn)

        # Prime the cache by asking every worker for its current state.
        self._cached_state = self._gather_state(initial=True)

    def _gather_state(self, initial: bool = False) -> dict:
        """Collect state from all workers. On `initial`, send a RESET
        command first. Otherwise this is called right after step_all
        which already sent STEP commands."""
        if initial:
            for c in self._conns:
                c.send((_CMD_RESET,))

        n = self.n_envs
        results: list[dict] = [None] * n  # type: ignore
        for i, c in enumerate(self._conns):
            msg = c.recv()
            if isinstance(msg, tuple) and msg and msg[0] == "error":
                raise RuntimeError(f"ParallelEnvWorkers error: {msg[1]}")
            results[i] = msg

        # Batch into arrays.
        obs_encoded = np.stack([r["obs_encoded"] for r in results], axis=0)
        mission_oh = np.stack([r["mission_oh"] for r in results], axis=0)
        allowed = [r["allowed"] for r in results]
        if results[0]["mem_feat"] is None:
            mem_feat = None
        else:
            mem_feat = np.stack([r["mem_feat"] for r in results], axis=0)
        rewards = np.array([r["reward"] for r in results], dtype=np.float32)
        dones = np.array([r["done"] for r in results], dtype=np.bool_)
        infos = [r["info"] for r in results]

        return {
            "obs_encoded": obs_encoded,
            "mission_oh": mission_oh,
            "allowed_lists": allowed,
            "mem_feat": mem_feat,
            "rewards": rewards,
            "dones": dones,
            "infos": infos,
        }

    def current_state(self) -> dict:
        """Return the batched state from the last step (or initial reset)."""
        return self._cached_state

    def step_all(self, actions: list[int] | np.ndarray) -> dict:
        """Send actions to all workers, gather updated state. Returns
        the same dict shape as current_state(), with rewards/dones/
        infos populated for the just-completed step.
        """
        if self._closed:
            raise RuntimeError("ParallelEnvWorkers is closed")
        actions = list(actions)
        if len(actions) != self.n_envs:
            raise ValueError(
                f"expected {self.n_envs} actions, got {len(actions)}"
            )
        # Send all actions first (pipelined), then read all responses.
        for c, a in zip(self._conns, actions):
            c.send((_CMD_STEP, int(a)))
        self._cached_state = self._gather_state(initial=False)
        return self._cached_state

    def close(self) -> None:
        """Terminate all subprocesses cleanly."""
        if self._closed:
            return
        for c in self._conns:
            try:
                c.send((_CMD_CLOSE,))
            except Exception:
                pass
        for c in self._conns:
            try:
                c.close()
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
