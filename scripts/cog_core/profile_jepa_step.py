"""Sandbox profiler for the JEPA dev-curriculum training loop.

Goal: find out where the wall-time of a single training iteration is
actually going. We've fixed several plausible bottlenecks already
(GPU-resident buffers, fused EMA, ignore_index, GPU loss accumulator,
N parallel envs in collection) and the user is still seeing low GPU
util — so it's time to stop guessing.

This script:
  1. Builds the same JEPA model used by `train_jepa_developmental.py`
     (same config: encoder_type=categorical_spatial, dyn=spatial_film,
     aux_predicate_weight=3.0, aux_distance_dim=24, aux_factored_weight=1.0).
  2. Generates synthetic data of the same shape as a real batch.
  3. Times each phase of one training step in isolation, using
     `torch.cuda.synchronize()` before+after each block so the wall
     timings reflect actual GPU work, not async queue depth.
  4. Repeats for a warmup window then averages over 50 iters.
  5. Separately times the collection path (n_envs=1, 8) on the real
     BabyAI env to confirm what fraction of total wall time is
     collection vs training.

Usage:
    python -m scripts.cog_core.profile_jepa_step --device cuda
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager

import numpy as np
import torch

from prism.models.jepa import JepaConfig, JepaWorldModel


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed(name: str, store: dict, device: torch.device, n_repeats: int = 1):
    """Context manager that times a block of GPU work."""
    cuda_sync(device)
    t0 = time.perf_counter()
    yield
    cuda_sync(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / n_repeats
    store.setdefault(name, []).append(elapsed_ms)


def make_dummy_batch(batch_size: int, device: torch.device, *, with_aux: bool,
                     with_factored: bool):
    """Synthetic batch matching the real loss() signature."""
    # BabyAI obs after _encode_image: (B, 3, 7, 7) float32 in [0, 1].
    obs_t = torch.rand(batch_size, 3, 7, 7, device=device)
    obs_tp1 = torch.rand(batch_size, 3, 7, 7, device=device)
    actions = torch.randint(0, 7, (batch_size,), device=device)

    predicates_t = predicates_tp1 = None
    if with_aux:
        # 96-d binary + 24-d distance = 120
        predicates_t = (torch.rand(batch_size, 120, device=device) > 0.5).float()
        predicates_tp1 = (torch.rand(batch_size, 120, device=device) > 0.5).float()

    col_t = typ_t = col_tp1 = typ_tp1 = None
    if with_factored:
        col_t = torch.randint(0, 6, (batch_size,), device=device)
        typ_t = torch.randint(0, 4, (batch_size,), device=device)
        col_tp1 = torch.randint(0, 6, (batch_size,), device=device)
        typ_tp1 = torch.randint(0, 4, (batch_size,), device=device)
    return obs_t, obs_tp1, actions, predicates_t, predicates_tp1, \
        col_t, typ_t, col_tp1, typ_tp1


def profile_training_step(args):
    device = torch.device(args.device)
    cfg = JepaConfig(
        n_actions=7,
        encoder_type="categorical_spatial",
        aux_predicate_weight=3.0,
        aux_distance_dim=24,
        aux_distance_weight=0.5,
        aux_factored_weight=1.0,
        dynamics_hidden_dim=256,
        dynamics_layers=3,
        dynamics_type="spatial_film",
        spatial_channels=64,
    )
    model = JepaWorldModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[profile] model params: {n_params:,}")
    print(f"[profile] device:       {device}")
    print(f"[profile] bf16:         {args.bf16}")
    print(f"[profile] compile:      {args.compile}")
    print(f"[profile] batch_size:   {args.batch_size}")
    print()

    # `loss_fn` mirrors what train_jepa_developmental.py wires up.
    loss_fn = model.loss
    if args.compile:
        print("[profile] compiling model.loss with mode='reduce-overhead' …")
        loss_fn = torch.compile(
            model.loss, mode="reduce-overhead", fullgraph=False,
        )

    # IMPORTANT: BabyAI categorical encoder expects the raw obs in a specific
    # range. The dummy data uses rand which may not exercise the embedding
    # lookups the same way. For *timing*, it's still representative.

    obs_t, obs_tp1, actions, preds_t, preds_tp1, col_t, typ_t, col_tp1, typ_tp1 = \
        make_dummy_batch(args.batch_size, device,
                         with_aux=True, with_factored=True)

    store: dict[str, list[float]] = {}
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.bf16 and device.type == "cuda"
        else None
    )

    def loss_call():
        if autocast_ctx is not None:
            with autocast_ctx:
                return model.loss(
                    obs_t, actions, obs_tp1,
                    predicates_t=preds_t, predicates_tp1=preds_tp1,
                    color_label_t=col_t, type_label_t=typ_t,
                    color_label_tp1=col_tp1, type_label_tp1=typ_tp1,
                )
        return model.loss(
            obs_t, actions, obs_tp1,
            predicates_t=preds_t, predicates_tp1=preds_tp1,
            color_label_t=col_t, type_label_t=typ_t,
            color_label_tp1=col_tp1, type_label_tp1=typ_tp1,
        )

    # Warmup
    print("[profile] warmup (5 iters)...")
    for _ in range(5):
        out = loss_call()
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.update_target()
    cuda_sync(device)

    print(f"[profile] timing {args.n_iters} training iters (per-op average ms)...")
    for _ in range(args.n_iters):
        with timed("01_forward_loss", store, device):
            out = loss_call()
        loss = out["loss"]
        with timed("02_zero_grad", store, device):
            opt.zero_grad(set_to_none=True)
        with timed("03_backward", store, device):
            loss.backward()
        with timed("04_clip_grad_norm", store, device):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        with timed("05_opt_step", store, device):
            opt.step()
        with timed("06_ema_update", store, device):
            model.update_target()

    print()
    print("=== per-iter timing (averaged) ===")
    total = 0.0
    for name in sorted(store.keys()):
        times = store[name]
        mean = float(np.mean(times))
        std = float(np.std(times))
        total += mean
        print(f"  {name:20s}  {mean:7.3f} ms  (± {std:5.3f})")
    print(f"  {'TOTAL':20s}  {total:7.3f} ms")
    print(f"  Theoretical max throughput: {1000.0 / max(total, 1e-6):.1f} steps/sec")
    print()


def profile_collection(args):
    """Time the env collection loop for n_envs=1 and n_envs=8."""
    from prism.envs.babyai import _encode_image
    from prism.perception import compute_augmented_predicates, extract_slots
    from scripts.cog_core.train_jepa_developmental import (
        collect_random_transitions, primary_object_label,
    )

    env_id = "BabyAI-OneRoomS8-v0"
    n_transitions = args.collect_n
    print(f"[profile] collecting {n_transitions} transitions from {env_id}")
    rng = np.random.default_rng(0)

    # ---- Time n_envs=1 ----
    print(f"\n[profile] n_envs=1 ...")
    t0 = time.perf_counter()
    _ = collect_random_transitions(
        env_id, n_transitions, rng,
        with_predicates=True, augmented=True, n_envs=1,
    )
    t1 = time.perf_counter()
    elapsed_1 = t1 - t0
    per_trans_1 = elapsed_1 * 1000.0 / n_transitions
    print(f"  total: {elapsed_1:6.2f}s  per-transition: {per_trans_1:6.3f}ms")

    # ---- Time n_envs=8 ----
    print(f"\n[profile] n_envs=8 ...")
    t0 = time.perf_counter()
    _ = collect_random_transitions(
        env_id, n_transitions, rng,
        with_predicates=True, augmented=True, n_envs=8,
    )
    t1 = time.perf_counter()
    elapsed_8 = t1 - t0
    per_trans_8 = elapsed_8 * 1000.0 / n_transitions
    print(f"  total: {elapsed_8:6.2f}s  per-transition: {per_trans_8:6.3f}ms")

    speedup = elapsed_1 / max(elapsed_8, 1e-6)
    print(f"\n  n_envs=8 speedup over n_envs=1: {speedup:.2f}×")

    # ---- Decompose: how much of collection is env.step vs slot/predicate? ----
    print(f"\n[profile] decomposing per-transition cost (n_envs=1, n=2000)...")
    import gymnasium as gym
    env = gym.make(env_id)
    obs, _ = env.reset(seed=0)
    raw = obs["image"]

    times = {"env_step": [], "encode_image": [], "extract_slots": [],
             "compute_predicates": [], "primary_object_label": []}
    for _ in range(2000):
        t0 = time.perf_counter()
        nxt, _, term, trunc, _ = env.step(int(rng.integers(7)))
        t1 = time.perf_counter()
        times["env_step"].append(t1 - t0)

        raw_t = obs["image"]
        t0 = time.perf_counter()
        _ = _encode_image(raw_t)
        t1 = time.perf_counter()
        times["encode_image"].append(t1 - t0)

        t0 = time.perf_counter()
        slots = extract_slots(raw_t)
        t1 = time.perf_counter()
        times["extract_slots"].append(t1 - t0)

        t0 = time.perf_counter()
        _ = compute_augmented_predicates(slots)
        t1 = time.perf_counter()
        times["compute_predicates"].append(t1 - t0)

        t0 = time.perf_counter()
        _ = primary_object_label(slots)
        t1 = time.perf_counter()
        times["primary_object_label"].append(t1 - t0)

        if term or trunc:
            obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        else:
            obs = nxt

    print(f"  {'phase':25s}  {'mean us':>10s}  {'p50 us':>10s}  {'total ms':>10s}")
    grand_total = 0.0
    for k, v in times.items():
        arr_us = np.array(v) * 1e6
        total_ms = arr_us.sum() / 1000.0
        grand_total += total_ms
        print(f"  {k:25s}  {arr_us.mean():10.2f}  {np.median(arr_us):10.2f}  "
              f"{total_ms:10.2f}")
    print(f"  {'TOTAL (2000 transitions)':25s}  "
          f"{'':10s}  {'':10s}  {grand_total:10.2f}")
    env.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", action="store_false", dest="bf16")
    p.add_argument("--n-iters", type=int, default=50)
    p.add_argument("--collect-n", type=int, default=2000,
                   help="transitions per collection profile run")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--skip-collection", action="store_true")
    args = p.parse_args()

    if not args.skip_training:
        profile_training_step(args)
    if not args.skip_collection:
        profile_collection(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
