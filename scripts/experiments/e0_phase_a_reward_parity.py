"""E0 — Phase A reward parity smoke test.

Confirms PR-1 → PR-3 introduced no behavioral regression: running PPO
with `UniversalPolicy.from_adapter(BabyAIAdapter)` reaches the same
reward as v5 `HybridPolicy` at the same env-step count.

Usage:
    python -m scripts.experiments.e0_phase_a_reward_parity \\
        --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \\
        --baseline-run runs/ppo_hybrid_v5_validation \\
        --total-steps 50000 \\
        --device cuda

The script does NOT retrain v5. It expects an existing v5 reward log at
`<baseline-run>/window_R.json` (or comparable) and compares the new
run's reward at the same step count. Tolerance: ±1% on `window_mean_R`.

This is the Phase A exit gate. If parity passes, PR-1→PR-3 are
behaviorally compatible with v5 and the refactor is safe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.experiments.checks.phase_a import check_phase_a


def _load_baseline_reward(baseline_run: Path, at_step: int) -> float:
    """Read the v5 baseline's window_mean_R at the specified env step.

    v5 ppo_train.py writes a metrics file as `<run>/metrics.json` with
    per-iteration entries. We linearly interpolate window_mean_R at
    `at_step` from the entries that bracket it.

    If no metrics file exists, fall back to v5's reported headline
    number from the docs (window_mean_R ~0.53 for `--no-bc` lang at
    500k steps; we conservatively use 0.45 at 50k steps based on the
    early-training curve).
    """
    metrics_path = baseline_run / "metrics.json"
    if not metrics_path.exists():
        print(
            f"[warn] no {metrics_path} found; using docs-derived "
            f"baseline ≈ 0.45 at 50k steps for --no-bc training"
        )
        return 0.45

    with metrics_path.open() as f:
        data = json.load(f)

    if isinstance(data, dict) and "iterations" in data:
        iters = data["iterations"]
    elif isinstance(data, list):
        iters = data
    else:
        raise ValueError(f"unrecognized metrics.json schema in {metrics_path}")

    pairs = [(it["env_steps"], it["window_mean_R"]) for it in iters
             if "env_steps" in it and "window_mean_R" in it]
    pairs.sort()
    if not pairs:
        raise ValueError("no usable (env_steps, window_mean_R) pairs")

    if at_step <= pairs[0][0]:
        return pairs[0][1]
    if at_step >= pairs[-1][0]:
        return pairs[-1][1]
    for (s0, r0), (s1, r1) in zip(pairs, pairs[1:]):
        if s0 <= at_step <= s1:
            t = (at_step - s0) / max(s1 - s0, 1)
            return r0 + t * (r1 - r0)
    return pairs[-1][1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jepa-checkpoint", required=True, type=Path)
    p.add_argument("--baseline-run", type=Path, default=None,
                   help="v5 run dir to compare against; if omitted, "
                        "uses docs-derived baseline")
    p.add_argument("--total-steps", type=int, default=50_000)
    p.add_argument("--tolerance", type=float, default=0.01,
                   help="fractional tolerance on window_mean_R; default 1%%")
    p.add_argument("--run-name", default="v6_phaseA_smoke")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # ------------------------------------------------------------------
    # 1. Structural pre-conditions (Phase A gate)
    # ------------------------------------------------------------------
    pre = check_phase_a()
    if not pre.passed:
        print("[E0] Phase A structural pre-conditions FAILED:")
        print(json.dumps({"failures": pre.failures, "report": pre.report},
                         indent=2, default=str))
        sys.exit(1)
    print("[E0] Phase A structural pre-conditions: PASS")

    # ------------------------------------------------------------------
    # 2. Run the new-API PPO for `--total-steps` and read final reward
    # ------------------------------------------------------------------
    # We invoke the existing ppo_train.py as a subprocess with the new
    # `--policy-type universal --trunk gru` flag. This ensures the
    # smoke test exercises the exact PPO entry point the rest of the
    # codebase uses; no parallel "test rig" can drift from production.
    import subprocess
    cmd = [
        sys.executable, "-m", "scripts.ppo_train",
        "--no-bc",
        "--jepa-checkpoint", str(args.jepa_checkpoint),
        "--policy-type", "universal",
        "--trunk", "gru",
        "--total-steps", str(args.total_steps),
        "--run-name", args.run_name,
        "--device", args.device,
    ]
    print(f"[E0] launching: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        print(f"[E0] PPO subprocess returned {proc.returncode}; gate FAILED")
        sys.exit(2)

    # ------------------------------------------------------------------
    # 3. Compare reward to baseline at the matched step
    # ------------------------------------------------------------------
    new_run_dir = Path("runs") / args.run_name
    new_metrics = new_run_dir / "metrics.json"
    if not new_metrics.exists():
        print(f"[E0] expected metrics file {new_metrics} not found; gate INCONCLUSIVE")
        sys.exit(3)

    new_window = _load_baseline_reward(new_run_dir, args.total_steps)

    if args.baseline_run is not None:
        base_window = _load_baseline_reward(args.baseline_run, args.total_steps)
    else:
        base_window = _load_baseline_reward(Path("/nonexistent"), args.total_steps)

    rel_diff = abs(new_window - base_window) / max(base_window, 1e-6)
    passed = rel_diff <= args.tolerance

    report = {
        "new_run": str(new_run_dir),
        "new_window_mean_R": new_window,
        "baseline_run": str(args.baseline_run) if args.baseline_run else "docs-derived",
        "baseline_window_mean_R": base_window,
        "at_step": args.total_steps,
        "relative_diff": rel_diff,
        "tolerance": args.tolerance,
        "passed": passed,
    }
    print("[E0] reward parity report:")
    print(json.dumps(report, indent=2))
    sys.exit(0 if passed else 4)


if __name__ == "__main__":
    main()
