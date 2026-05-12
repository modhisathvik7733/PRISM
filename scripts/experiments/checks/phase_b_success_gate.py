"""Phase B success gate — does the v6 substrate match v5 on BabyAI?

Plan exit criterion for Phase B (transformer trunk + retrieval, no
curriculum): "Match v5.0 success on GoToObj (≥0.85) and within 5pp of
v5.0 on GoToLocal."

This script consumes one or more completed `runs/<name>/metrics.json`
files and renders a pass/fail decision. It does NOT run training — that
happens on Vast.ai via `python -m scripts.ppo_train ...`.

Usage:

    # Single-run check against docs-derived v5 baselines.
    python -m scripts.experiments.checks.phase_b_success_gate \\
        --v6-gotolocal runs/v6_phaseB_GoToLocal_500k \\
        --v6-gotoobj runs/v6_phaseB_GoToObj_500k

    # Stricter: compare against your own v5 baselines.
    python -m scripts.experiments.checks.phase_b_success_gate \\
        --v6-gotolocal runs/v6_phaseB_GoToLocal_500k \\
        --v6-gotoobj runs/v6_phaseB_GoToObj_500k \\
        --v5-gotolocal runs/v5_phaseB_GoToLocal_500k \\
        --v5-gotoobj runs/v5_phaseB_GoToObj_500k

Exit code:
  0 = Phase B passes (both gates met).
  3 = window_mean_R below the required threshold on at least one env.
  4 = a required metrics.json is missing or malformed.

When `--v5-*` flags are omitted, the script falls back to docs-derived
v5 estimates:
  GoToLocal @ 500k:  ~0.55 (--no-bc training, v5 HybridPolicy).
  GoToObj   @ 500k:  ~0.90 (easier env, sparser action space).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Docs-derived v5 baselines for the no-bc, single-env, no-shaping config.
# Used as fallbacks when the user doesn't supply matched v5 runs.
DOCS_V5_GOTOLOCAL_500K = 0.55
DOCS_V5_GOTOOBJ_500K = 0.90

# Phase B gate thresholds from the v6 plan.
GOTOLOCAL_DELTA_PP = 0.05     # within 5pp of v5
GOTOOBJ_ABS_THRESHOLD = 0.85  # absolute floor


def _load_window_R_at_end(metrics_path: Path) -> float:
    """Read the final window_mean_R from a run's metrics.json.

    Schema (written by ppo_train.py at run end):
        {"iterations": [{"iter": int, "env_steps": int,
                         "window_mean_R": float, ...}, ...]}
    """
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json missing at {metrics_path}")
    with metrics_path.open() as f:
        data = json.load(f)
    if "iterations" not in data or not data["iterations"]:
        raise ValueError(f"metrics.json at {metrics_path} has no iterations")
    last = data["iterations"][-1]
    if "window_mean_R" not in last:
        raise ValueError(f"metrics.json at {metrics_path} missing window_mean_R")
    return float(last["window_mean_R"])


def _check_one(
    v6_run: Path,
    v5_run: Path | None,
    docs_fallback: float,
    rule: str,
) -> dict:
    """Apply the success rule to one env. Returns a report dict."""
    metrics = v6_run / "metrics.json"
    try:
        v6_R = _load_window_R_at_end(metrics)
    except (FileNotFoundError, ValueError) as e:
        return {"ok": False, "error": str(e), "v6_run": str(v6_run)}

    if v5_run is not None:
        try:
            v5_R = _load_window_R_at_end(v5_run / "metrics.json")
            v5_source = "matched run"
        except (FileNotFoundError, ValueError):
            v5_R = docs_fallback
            v5_source = f"docs-derived fallback (matched v5 missing at {v5_run})"
    else:
        v5_R = docs_fallback
        v5_source = "docs-derived fallback"

    report: dict = {
        "v6_run": str(v6_run),
        "v6_window_R": v6_R,
        "v5_baseline": v5_R,
        "v5_source": v5_source,
        "rule": rule,
    }

    if rule == "absolute_floor":
        threshold = GOTOOBJ_ABS_THRESHOLD
        report["threshold"] = threshold
        report["ok"] = v6_R >= threshold
        report["pass_text"] = f"v6={v6_R:.3f} ≥ {threshold:.2f}" if report["ok"] \
            else f"v6={v6_R:.3f} < {threshold:.2f} (gap = {threshold - v6_R:.3f})"
    elif rule == "within_5pp_of_v5":
        delta_floor = v5_R - GOTOLOCAL_DELTA_PP
        report["threshold"] = delta_floor
        report["ok"] = v6_R >= delta_floor
        report["pass_text"] = (
            f"v6={v6_R:.3f} ≥ v5−5pp = {delta_floor:.3f}" if report["ok"]
            else f"v6={v6_R:.3f} < v5−5pp = {delta_floor:.3f} (gap = {delta_floor - v6_R:.3f})"
        )
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v6-gotolocal", type=Path, required=True,
                   help="Path to the v6 BabyAI-GoToLocal run dir.")
    p.add_argument("--v6-gotoobj", type=Path, required=True,
                   help="Path to the v6 BabyAI-GoToObj run dir.")
    p.add_argument("--v5-gotolocal", type=Path, default=None,
                   help="Optional: matched v5 baseline run dir for "
                        "GoToLocal. If omitted, the docs-derived 0.55 "
                        "fallback is used.")
    p.add_argument("--v5-gotoobj", type=Path, default=None,
                   help="Optional: matched v5 baseline run dir for "
                        "GoToObj. If omitted, the docs-derived 0.90 "
                        "fallback is used.")
    args = p.parse_args()

    gotolocal_report = _check_one(
        v6_run=args.v6_gotolocal,
        v5_run=args.v5_gotolocal,
        docs_fallback=DOCS_V5_GOTOLOCAL_500K,
        rule="within_5pp_of_v5",
    )
    gotoobj_report = _check_one(
        v6_run=args.v6_gotoobj,
        v5_run=args.v5_gotoobj,
        docs_fallback=DOCS_V5_GOTOOBJ_500K,
        rule="absolute_floor",
    )

    # Render the decision.
    print("=" * 64)
    print("Phase B success gate")
    print("=" * 64)
    for env, r in [("BabyAI-GoToLocal", gotolocal_report),
                    ("BabyAI-GoToObj", gotoobj_report)]:
        if "error" in r:
            status = "MISSING"
        else:
            status = "PASS" if r["ok"] else "FAIL"
        print(f"\n[{env}] {status}")
        for k, v in r.items():
            print(f"  {k}: {v}")

    if any("error" in r for r in (gotolocal_report, gotoobj_report)):
        print("\n[gate] some metrics.json files missing — re-run training "
              "or fix paths, then re-evaluate.")
        sys.exit(4)
    all_pass = gotolocal_report["ok"] and gotoobj_report["ok"]
    if all_pass:
        print("\n[gate] Phase B: PASS — v6 substrate meets the plan's exit criterion.")
        sys.exit(0)
    print("\n[gate] Phase B: FAIL — at least one environment did not meet "
          "the threshold. See per-env detail above.")
    sys.exit(3)


if __name__ == "__main__":
    main()
