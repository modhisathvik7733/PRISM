"""ProbeSet — persisted random-policy rollout fixed at Stage 0 init.

Resolution 6 from the v6 plan: the probe set is a load-bearing
experimental artifact with strict lifecycle rules:

1. Created exactly once at the moment a substrate is initialized for
   any curriculum run.
2. Collected via random-policy rollouts on a fixed seed in the source
   environment.
3. Persisted to `runs/<run_name>/probe_set.pt`. Content hash recorded
   in `substrate_config_hash`.
4. Not re-collected per stage / per adapter / per seed. ONE probe set
   per substrate-instance, for the lifetime of that substrate.

Audit pass-2 issue 7d (probe-set policy-dependent / circular metric):
the probe set MUST be collected with a random policy, not the
current-policy. A current-policy probe co-evolves with training and
produces tautological E4 metrics.

This module is domain-agnostic: the caller passes an `env_factory`
callable so the same code works for BabyAI, code-editing, and beyond.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


@dataclass
class ProbeSet:
    """Immutable record of probe observations.

    Once `save_probe_set` writes this to disk, the hash is the
    authoritative identifier. Any attempt to compute E4 against a
    probe set whose hash doesn't match the substrate checkpoint's
    `probe_set_hash` field must be rejected.
    """

    obs: torch.Tensor                            # (N, *obs_shape)
    missions: torch.Tensor | None                # (N, *mission_shape) or None
    env_id: str
    seed: int
    n_frames: int
    hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_probe_set_hash(
    obs: torch.Tensor,
    missions: torch.Tensor | None,
) -> str:
    """Deterministic hex hash of the probe tensors.

    SHA256 of the bytes of (obs, missions) in tensor-canonical order
    (contiguous, CPU, fp32 / int64 as native). Stable across machines
    since we serialize via numpy after .cpu().contiguous().
    """
    h = hashlib.sha256()
    obs_np = obs.detach().cpu().contiguous().numpy()
    h.update(obs_np.tobytes())
    h.update(str(obs_np.shape).encode())
    h.update(str(obs_np.dtype).encode())
    if missions is not None:
        m_np = missions.detach().cpu().contiguous().numpy()
        h.update(m_np.tobytes())
        h.update(str(m_np.shape).encode())
        h.update(str(m_np.dtype).encode())
    else:
        h.update(b"no_missions")
    return h.hexdigest()


def collect_probe_set(
    env_factory: Callable[[], Any],
    n_frames: int,
    seed: int,
    env_id: str,
    obs_fn: Callable[[Any], np.ndarray] | None = None,
    mission_fn: Callable[[Any], np.ndarray] | None = None,
    n_actions: int = 7,
    metadata: dict[str, Any] | None = None,
) -> ProbeSet:
    """Collect `n_frames` (obs, mission) pairs via a random policy on a
    fixed seed. The env_factory is called once to construct a single env
    instance; we step it with uniform-random actions, resetting on done.

    Parameters
    ----------
    env_factory : callable returning a Gym-like env with .reset() and
        .step(a) -> (obs, reward, terminated, truncated, info).
    n_frames : total number of frames to collect.
    seed : fixed RNG seed; same value reproduces the same probe set.
    env_id : the env identifier (recorded in the ProbeSet for sanity).
    obs_fn : extracts the obs tensor from the env's raw reset/step
        return — for BabyAI, this is e.g. `obs["image"]`. Defaults to
        identity if obs is already a numpy array.
    mission_fn : extracts the mission tensor. None if env has no mission.
    n_actions : random-policy uniform action space size.
    metadata : extra info stored alongside (e.g., adapter name).

    Returns
    -------
    ProbeSet with `hash` set to the SHA256 of the collected tensors.
    """
    rng = np.random.default_rng(seed)
    env = env_factory()

    obs_list: list[np.ndarray] = []
    mission_list: list[np.ndarray] = [] if mission_fn is not None else []

    raw_obs, info = env.reset(seed=seed)
    frames_collected = 0

    while frames_collected < n_frames:
        obs_arr = obs_fn(raw_obs) if obs_fn is not None else np.asarray(raw_obs)
        obs_list.append(obs_arr)
        if mission_fn is not None:
            mission_list.append(mission_fn(raw_obs))
        frames_collected += 1

        action = int(rng.integers(0, n_actions))
        step_out = env.step(action)
        # Gymnasium returns 5-tuple, Gym 0.21 returns 4-tuple. Handle both.
        if len(step_out) == 5:
            raw_obs, _reward, terminated, truncated, _info = step_out
            done = bool(terminated or truncated)
        else:
            raw_obs, _reward, done, _info = step_out
        if done:
            raw_obs, _info = env.reset()

    obs = torch.from_numpy(np.stack(obs_list, axis=0))
    missions = (
        torch.from_numpy(np.stack(mission_list, axis=0))
        if mission_fn is not None else None
    )
    h = compute_probe_set_hash(obs, missions)
    return ProbeSet(
        obs=obs,
        missions=missions,
        env_id=env_id,
        seed=seed,
        n_frames=n_frames,
        hash=h,
        metadata=metadata or {},
    )


def save_probe_set(probe_set: ProbeSet, path: str | Path) -> str:
    """Persist probe set to disk. Returns the hash (same as
    `probe_set.hash`). The file format is a pickle-safe torch.save dict
    so the probe set is platform-portable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "obs": probe_set.obs,
        "missions": probe_set.missions,
        "env_id": probe_set.env_id,
        "seed": probe_set.seed,
        "n_frames": probe_set.n_frames,
        "hash": probe_set.hash,
        "metadata": probe_set.metadata,
    }
    torch.save(payload, path)
    return probe_set.hash


def load_probe_set(path: str | Path, verify_hash: bool = True) -> ProbeSet:
    """Load probe set from disk. By default, recomputes the hash from
    the tensors and verifies it matches the stored hash — guards against
    silent disk corruption or hand-edited tensors.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ps = ProbeSet(
        obs=payload["obs"],
        missions=payload["missions"],
        env_id=payload["env_id"],
        seed=payload["seed"],
        n_frames=payload["n_frames"],
        hash=payload["hash"],
        metadata=payload.get("metadata", {}),
    )
    if verify_hash:
        recomputed = compute_probe_set_hash(ps.obs, ps.missions)
        if recomputed != ps.hash:
            raise ValueError(
                f"ProbeSet hash mismatch: stored={ps.hash} "
                f"recomputed={recomputed}. The probe-set file at {path} "
                f"has been tampered with or corrupted; refusing to load."
            )
    return ps


if __name__ == "__main__":
    # Standalone smoke test using a trivial fake env. Real-env collection
    # is exercised when ppo_train wires this in (PR-5 step 4).
    # Run with: `python -m prism.curriculum.probe_set`
    import sys as _sys

    class _FakeEnv:
        """Minimal Gymnasium-style env. obs is a 3x7x7 float array
        (BabyAI-shaped). Mission is a 24-d one-hot. Episodes are 8 steps."""
        def __init__(self):
            self.t = 0
            self._rng = np.random.default_rng(0)

        def reset(self, seed: int | None = None):
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self.t = 0
            self._obs = self._rng.standard_normal((3, 7, 7)).astype(np.float32)
            self._mission = np.zeros(24, dtype=np.float32)
            self._mission[self._rng.integers(0, 24)] = 1.0
            return {"image": self._obs, "mission": self._mission}, {}

        def step(self, _action: int):
            self.t += 1
            self._obs = self._rng.standard_normal((3, 7, 7)).astype(np.float32)
            done = self.t >= 8
            return ({"image": self._obs, "mission": self._mission},
                    0.0, done, False, {})

    ps = collect_probe_set(
        env_factory=_FakeEnv,
        n_frames=100,
        seed=42,
        env_id="FakeEnv-v0",
        obs_fn=lambda o: o["image"],
        mission_fn=lambda o: o["mission"],
        n_actions=7,
    )
    print(f"[probe] collected {ps.n_frames} frames; "
          f"obs.shape={tuple(ps.obs.shape)} missions.shape={tuple(ps.missions.shape)} "
          f"hash={ps.hash[:16]}…")
    assert ps.obs.shape == (100, 3, 7, 7)
    assert ps.missions.shape == (100, 24)
    assert len(ps.hash) == 64  # SHA256 hex
    print("[probe] shape contract OK")

    # Determinism: same seed should produce bit-identical probe set.
    ps2 = collect_probe_set(
        env_factory=_FakeEnv, n_frames=100, seed=42, env_id="FakeEnv-v0",
        obs_fn=lambda o: o["image"], mission_fn=lambda o: o["mission"],
        n_actions=7,
    )
    if ps2.hash != ps.hash:
        print(f"FAIL: same seed → different hash (ps={ps.hash[:16]}, ps2={ps2.hash[:16]})")
        _sys.exit(1)
    print(f"[probe] determinism OK: same seed → same hash")

    # Different seed → different hash.
    ps3 = collect_probe_set(
        env_factory=_FakeEnv, n_frames=100, seed=43, env_id="FakeEnv-v0",
        obs_fn=lambda o: o["image"], mission_fn=lambda o: o["mission"],
        n_actions=7,
    )
    if ps3.hash == ps.hash:
        print(f"FAIL: different seed → same hash (collision or RNG bug)")
        _sys.exit(1)
    print(f"[probe] seed-sensitivity OK: different seed → different hash")

    # Round-trip via disk.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = Path(f.name)
    h = save_probe_set(ps, path)
    if h != ps.hash:
        print(f"FAIL: save_probe_set returned wrong hash")
        _sys.exit(1)
    loaded = load_probe_set(path, verify_hash=True)
    if loaded.hash != ps.hash:
        print(f"FAIL: loaded probe set hash differs from original")
        _sys.exit(1)
    if not torch.equal(loaded.obs, ps.obs):
        print(f"FAIL: loaded obs differs from original")
        _sys.exit(1)
    print(f"[probe] save/load round-trip OK; hash preserved")

    # Tamper detection: edit a tensor, ensure load() rejects.
    payload = torch.load(path, weights_only=False)
    payload["obs"][0, 0, 0, 0] += 1.0   # one-element edit
    torch.save(payload, path)
    try:
        load_probe_set(path, verify_hash=True)
    except ValueError as e:
        if "hash mismatch" in str(e).lower():
            print(f"[probe] tamper detection OK: load_probe_set raises on hash mismatch")
        else:
            print(f"FAIL: load raised unexpected error: {e}")
            _sys.exit(1)
    else:
        print(f"FAIL: tampered probe set loaded without complaint")
        _sys.exit(1)
    path.unlink()

    print("[probe] all smoke checks passed")
