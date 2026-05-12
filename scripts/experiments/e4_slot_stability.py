"""E4 — slot stability between two substrate snapshots.

Operationalizes resolution 5 / audit issue 3d (the v6 plan's most-
likely long-term failure mode): a slot's K/V can be bit-identical
across stages while the trainable query MLP re-routes which slot fires
for a given input. Weight-stability checksums pass; slot *function*
drifts; the inspection metric "stable abstractions" is satisfied
tautologically. The substrate is silently broken.

Operational definition of stability (resolution 5, interpretation b):
  1. Weight bit-equality on frozen rows (gates via PR-5 step 1).
  2. Top-K activating probe frames Jaccard ≥ 0.6 between snapshots.
  3. JS divergence of per-slot attention ≤ 0.1 between snapshots.

This script implements measurements (2) and (3). It does NOT train —
it consumes two completed policy checkpoints and a persisted probe
set, runs the probe set through both substrates' Hopfield banks,
and reports per-bank stability statistics.

Usage:

    # Compare a mid-training checkpoint to the final.
    python -m scripts.experiments.e4_slot_stability \\
        --checkpoint-a runs/v6_phaseB_GoToLocal_500k/policy_iter60.pt \\
        --checkpoint-b runs/v6_phaseB_GoToLocal_500k/policy_final.pt \\
        --probe-set runs/v6_phaseB_GoToLocal_500k/probe_set.pt \\
        --jepa-checkpoint runs/jepa_dev_v1_factored/jepa_final.pt \\
        --top-k 50

    # If no probe set exists yet, collect one first:
    python -m scripts.experiments.e4_slot_stability \\
        --checkpoint-a ... --checkpoint-b ... \\
        --jepa-checkpoint ... \\
        --collect-probe-set runs/probe_sets/babyai_gotolocal.pt \\
        --probe-env BabyAI-GoToLocal-v0 --probe-size 2000

Exit codes:
  0 = both banks pass the stability gates.
  3 = at least one bank's Jaccard or JS gate failed.
  4 = required input file missing.

What "passes" means:
  - For each bank: median top-K Jaccard ≥ 0.6 across slots that were
    active in both snapshots. Per-slot Jaccard <0.4 signals routing
    drift for that slot.
  - For each bank: median JS divergence of slot attention ≤ 0.1.
  - Frozen slots specifically: K and V parameter rows must be bit-equal
    (checksum). Any drift on a frozen row indicates the freeze mask
    leaked (audit 3a/3c — different failure mode than 3d).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from prism.adapters.babyai_adapter import BabyAIAdapter
from prism.cognition.policy import UniversalPolicy
from prism.curriculum.probe_eval import (
    js_divergence,
    per_slot_jaccard,
    top_k_frames_per_slot,
)
from prism.curriculum.probe_set import (
    ProbeSet,
    collect_probe_set,
    load_probe_set,
    save_probe_set,
)


JACCARD_PASS_THRESHOLD = 0.6
JS_PASS_THRESHOLD = 0.1
JACCARD_DRIFT_FLAG = 0.4   # below this on a single slot = routing drift


def _build_policy(jepa_checkpoint: Path, device: torch.device) -> UniversalPolicy:
    """Reconstruct the UniversalPolicy with the same shape ppo_train builds.
    We don't load policy weights here — the caller loads them per snapshot.
    """
    adapter = BabyAIAdapter.from_jepa_checkpoint(jepa_checkpoint, device=device)
    policy = UniversalPolicy.from_adapter(
        adapter,
        trunk="transformer",
        D_tok=128, L=16,
        n_trunk_layers=4, n_trunk_heads=4, trunk_ffn_dim=512,
        concept_n_slots=1024, operator_n_slots=64,
    ).to(device)
    return policy


def _load_policy_weights(policy: UniversalPolicy, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["policy_state_dict"]
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[E4] WARN: unexpected state_dict keys when loading "
              f"{ckpt_path.name}: {unexpected[:5]}...")
    # Missing keys are normal: the inner _TransformerInner has its own
    # bank parameters etc. We strict=False so v5/v6 ckpts both load.


@torch.no_grad()
def _collect_per_frame_attention(
    policy: UniversalPolicy,
    probe_set: ProbeSet,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, torch.Tensor]:
    """Run every probe frame through both retrieval banks, return
    {'concept': (N, n_slots), 'operator': (N, n_slots)} attention tensors.

    We use the SAME query path the substrate uses at training time:
    obs → adapter encode → input token → RetrievalBlock's
    concept_cond / operator_cond → bank.retrieve_with_attention.
    This way the metric measures what the substrate actually attends
    to, not some abstraction-of-the-attention.
    """
    policy.eval()
    inner = policy.inner
    if not hasattr(inner, "retrieval"):
        raise RuntimeError("policy.inner has no retrieval block; "
                           "must be trunk=transformer with use_retrieval=True")
    retrieval = inner.retrieval

    N = probe_set.obs.shape[0]
    n_concept = retrieval.concept_bank.n_slots
    n_operator = retrieval.operator_bank.n_slots
    concept_attn = torch.zeros(N, n_concept)
    operator_attn = torch.zeros(N, n_operator)

    obs_dev = probe_set.obs.to(device)
    mission_dev = probe_set.missions.to(device) if probe_set.missions is not None else None

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        obs_batch = obs_dev[start:end]
        mission_batch = (
            mission_dev[start:end] if mission_dev is not None
            else torch.zeros(end - start, inner.mission_dim, device=device)
        )

        # Build the input token (same path as step_with_value).
        z = policy.adapter.encode_obs(obs_batch.float())
        prev_a = torch.zeros(end - start, dtype=torch.long, device=device)
        new_token = inner._build_input_token(z, prev_a, mission_batch, mem_feat=None)

        # Apply the cond MLPs to build queries, then retrieve with attention.
        B = new_token.size(0)
        cq = retrieval.concept_base.expand(B, -1) + retrieval.concept_cond(new_token)
        oq = retrieval.operator_base.expand(B, -1) + retrieval.operator_cond(new_token)
        _, c_attn = retrieval.concept_bank.retrieve_with_attention(cq)
        _, o_attn = retrieval.operator_bank.retrieve_with_attention(oq)
        concept_attn[start:end] = c_attn.cpu()
        operator_attn[start:end] = o_attn.cpu()

    return {"concept": concept_attn, "operator": operator_attn}


def _checksum_bank_weights(bank) -> dict[str, str]:
    """Hex SHA256 over the bank's K and V parameters. Used for the
    weight-stability gate (resolution 5 part 1)."""
    import hashlib
    h_k = hashlib.sha256(bank.keys.detach().cpu().numpy().tobytes()).hexdigest()
    h_v = hashlib.sha256(bank.values.detach().cpu().numpy().tobytes()).hexdigest()
    return {"K": h_k, "V": h_v}


def _frozen_rows_changed(
    bank_a, bank_b, frozen_mask: torch.Tensor
) -> tuple[bool, dict]:
    """For each frozen slot, check K[i] and V[i] are bit-equal across
    snapshots. Audit issue 3a/3c (different from 3d): if frozen rows
    drift, the freeze mask leaked. This must be 0; any non-zero count
    is a substrate bug, not a stability question.
    """
    if not frozen_mask.any():
        return False, {"n_frozen": 0, "k_diffs": 0, "v_diffs": 0}
    idx = torch.nonzero(frozen_mask, as_tuple=False).flatten()
    k_a = bank_a.keys[:, idx, :].detach().cpu()
    k_b = bank_b.keys[:, idx, :].detach().cpu()
    v_a = bank_a.values[:, idx, :].detach().cpu()
    v_b = bank_b.values[:, idx, :].detach().cpu()
    k_diff = int((k_a != k_b).any(dim=-1).sum().item())
    v_diff = int((v_a != v_b).any(dim=-1).sum().item())
    return (k_diff > 0 or v_diff > 0), {
        "n_frozen": int(frozen_mask.sum().item()),
        "k_diffs": k_diff, "v_diffs": v_diff,
    }


def _summarize_jaccard(j: torch.Tensor, name: str) -> dict:
    """Format Jaccard distribution into a report block."""
    n = int(j.numel())
    drift_count = int((j < JACCARD_DRIFT_FLAG).sum().item())
    return {
        "bank": name,
        "n_slots_compared": n,
        "median": float(j.median().item()),
        "mean": float(j.mean().item()),
        "frac_above_0.6": float((j >= 0.6).float().mean().item()),
        "n_below_0.4_drift_flag": drift_count,
        "min": float(j.min().item()),
        "max": float(j.max().item()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-a", type=Path, required=True,
                   help="First policy snapshot (e.g. earlier in training).")
    p.add_argument("--checkpoint-b", type=Path, required=True,
                   help="Second policy snapshot (e.g. final).")
    p.add_argument("--jepa-checkpoint", type=Path, required=True,
                   help="JEPA encoder (shared between snapshots).")
    p.add_argument("--probe-set", type=Path, default=None,
                   help="Path to a persisted probe_set.pt. If omitted, "
                        "use --collect-probe-set to make one.")
    p.add_argument("--collect-probe-set", type=Path, default=None,
                   help="Path to write a new probe set to. Requires "
                        "--probe-env and --probe-size.")
    p.add_argument("--probe-env", default="BabyAI-GoToLocal-v0")
    p.add_argument("--probe-size", type=int, default=2000)
    p.add_argument("--probe-seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=50,
                   help="Top-K activating frames per slot for Jaccard.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    device = torch.device(args.device)

    # --- Probe set ---
    if args.probe_set is not None and args.probe_set.exists():
        probe_set = load_probe_set(args.probe_set, verify_hash=True)
        print(f"[E4] loaded probe set: {probe_set.n_frames} frames, "
              f"hash={probe_set.hash[:16]}…")
    elif args.collect_probe_set is not None:
        # Build env factory for BabyAI.
        from prism.envs.babyai import make_env_with_max_steps
        def env_factory():
            return make_env_with_max_steps(args.probe_env, max_steps=64)
        def obs_fn(raw):
            from scripts.ppo_train import _goal_distance_from_raw_obs  # noqa
            from prism.envs.babyai import _encode_image
            return _encode_image(raw["image"])
        def mission_fn(raw):
            # Build the (color, type) one-hot the trainer uses.
            from prism.agents import goal_predicates_for_mission
            from prism.perception.slots import NUM_COLORS, OBJECT_TYPES
            mission_dim = len(OBJECT_TYPES) * NUM_COLORS
            v = np.zeros(mission_dim, dtype=np.float32)
            try:
                preds = goal_predicates_for_mission(raw["mission"], None)
                if preds and preds[0][1] is not None and preds[0][2] is not None:
                    col, typ = preds[0][1], preds[0][2]
                    if col in range(NUM_COLORS) and typ in range(len(OBJECT_TYPES)):
                        v[typ * NUM_COLORS + col] = 1.0
            except Exception:
                pass
            return v
        probe_set = collect_probe_set(
            env_factory=env_factory,
            n_frames=args.probe_size,
            seed=args.probe_seed,
            env_id=args.probe_env,
            obs_fn=obs_fn,
            mission_fn=mission_fn,
            n_actions=7,
        )
        save_probe_set(probe_set, args.collect_probe_set)
        print(f"[E4] collected + saved probe set: {probe_set.n_frames} frames, "
              f"hash={probe_set.hash[:16]}…")
    else:
        print("[E4] FAIL: must provide either --probe-set (existing) or "
              "--collect-probe-set + --probe-env (new). Exiting.")
        sys.exit(4)

    # --- Build policy + load two snapshots, collect attention each ---
    print(f"[E4] building substrate (transformer trunk + retrieval) …")
    policy = _build_policy(args.jepa_checkpoint, device)

    if not args.checkpoint_a.exists():
        print(f"[E4] FAIL: --checkpoint-a not found: {args.checkpoint_a}")
        sys.exit(4)
    if not args.checkpoint_b.exists():
        print(f"[E4] FAIL: --checkpoint-b not found: {args.checkpoint_b}")
        sys.exit(4)

    print(f"[E4] snapshot A: {args.checkpoint_a.name}")
    _load_policy_weights(policy, args.checkpoint_a)
    bank_a_checksums = {
        "concept": _checksum_bank_weights(policy.inner.retrieval.concept_bank),
        "operator": _checksum_bank_weights(policy.inner.retrieval.operator_bank),
    }
    bank_a_frozen = {
        "concept": policy.inner.retrieval.concept_bank.frozen_mask.clone().cpu(),
        "operator": policy.inner.retrieval.operator_bank.frozen_mask.clone().cpu(),
    }
    bank_a_keys = {
        "concept": policy.inner.retrieval.concept_bank.keys.detach().clone().cpu(),
        "operator": policy.inner.retrieval.operator_bank.keys.detach().clone().cpu(),
    }
    bank_a_values = {
        "concept": policy.inner.retrieval.concept_bank.values.detach().clone().cpu(),
        "operator": policy.inner.retrieval.operator_bank.values.detach().clone().cpu(),
    }
    attn_a = _collect_per_frame_attention(policy, probe_set, device, args.batch_size)
    print(f"[E4]   attention collected: concept {tuple(attn_a['concept'].shape)} "
          f"operator {tuple(attn_a['operator'].shape)}")

    print(f"[E4] snapshot B: {args.checkpoint_b.name}")
    _load_policy_weights(policy, args.checkpoint_b)
    bank_b_checksums = {
        "concept": _checksum_bank_weights(policy.inner.retrieval.concept_bank),
        "operator": _checksum_bank_weights(policy.inner.retrieval.operator_bank),
    }
    attn_b = _collect_per_frame_attention(policy, probe_set, device, args.batch_size)
    print(f"[E4]   attention collected.")

    # Synthetic mini-banks to feed _frozen_rows_changed (it just reads keys/values).
    class _FakeBank:
        def __init__(self, keys, values):
            self.keys = keys
            self.values = values
    fake_a_concept = _FakeBank(bank_a_keys["concept"], bank_a_values["concept"])
    fake_a_operator = _FakeBank(bank_a_keys["operator"], bank_a_values["operator"])
    fake_b_concept = _FakeBank(
        policy.inner.retrieval.concept_bank.keys.detach().cpu(),
        policy.inner.retrieval.concept_bank.values.detach().cpu(),
    )
    fake_b_operator = _FakeBank(
        policy.inner.retrieval.operator_bank.keys.detach().cpu(),
        policy.inner.retrieval.operator_bank.values.detach().cpu(),
    )

    # --- Compute per-bank metrics ---
    reports: dict[str, dict] = {}
    for bank_name in ("concept", "operator"):
        a = attn_a[bank_name]
        b = attn_b[bank_name]
        top_a = top_k_frames_per_slot(a, k=args.top_k)
        top_b = top_k_frames_per_slot(b, k=args.top_k)
        jacc = per_slot_jaccard(top_a, top_b)

        # Average attention distribution over the probe set per slot.
        avg_a = a.mean(dim=0)
        avg_b = b.mean(dim=0)
        js = float(js_divergence(avg_a.unsqueeze(0), avg_b.unsqueeze(0)).item())

        # Frozen-row drift check.
        fa = fake_a_concept if bank_name == "concept" else fake_a_operator
        fb = fake_b_concept if bank_name == "concept" else fake_b_operator
        frz_violated, frz_info = _frozen_rows_changed(
            fa, fb, bank_a_frozen[bank_name]
        )

        reports[bank_name] = {
            "jaccard": _summarize_jaccard(jacc, bank_name),
            "js_divergence_avg_attn": js,
            "weight_checksum_a": bank_a_checksums[bank_name],
            "weight_checksum_b": bank_b_checksums[bank_name],
            "frozen_rows_violated": frz_violated,
            "frozen_rows_info": frz_info,
        }

    # --- Pass/fail decision ---
    print("=" * 70)
    print("E4 — slot stability between snapshots")
    print("=" * 70)
    all_pass = True
    for bank_name, r in reports.items():
        jacc_med = r["jaccard"]["median"]
        js = r["js_divergence_avg_attn"]
        jacc_pass = jacc_med >= JACCARD_PASS_THRESHOLD
        js_pass = js <= JS_PASS_THRESHOLD
        frz_pass = not r["frozen_rows_violated"]
        bank_pass = jacc_pass and js_pass and frz_pass
        all_pass = all_pass and bank_pass
        status = "PASS" if bank_pass else "FAIL"
        print(f"\n[{bank_name}] {status}")
        print(f"  jaccard.median = {jacc_med:.3f} (gate ≥ {JACCARD_PASS_THRESHOLD})  "
              f"→ {'PASS' if jacc_pass else 'FAIL'}")
        print(f"  jaccard.mean   = {r['jaccard']['mean']:.3f}")
        print(f"  jaccard.frac_above_0.6 = {r['jaccard']['frac_above_0.6']:.2%}")
        print(f"  jaccard slots flagged drift (<0.4) = {r['jaccard']['n_below_0.4_drift_flag']}/"
              f"{r['jaccard']['n_slots_compared']}")
        print(f"  js_divergence  = {js:.4f} (gate ≤ {JS_PASS_THRESHOLD}) "
              f"→ {'PASS' if js_pass else 'FAIL'}")
        print(f"  frozen rows: n={r['frozen_rows_info']['n_frozen']} "
              f"k_diffs={r['frozen_rows_info']['k_diffs']} "
              f"v_diffs={r['frozen_rows_info']['v_diffs']} "
              f"→ {'PASS' if frz_pass else 'FAIL (freeze leaked!)'}")
        wa_k = r["weight_checksum_a"]["K"][:12]
        wb_k = r["weight_checksum_b"]["K"][:12]
        print(f"  weight K hash A={wa_k}… B={wb_k}… "
              f"({'identical' if wa_k == wb_k else 'differ — expected if not all rows frozen'})")

    if all_pass:
        print("\n[E4] PASS — substrate slot stability holds across both snapshots.")
        sys.exit(0)
    print("\n[E4] FAIL — see per-bank detail above. Audit issue 3d (routing drift) "
          "or 3a/3c (frozen-mask leak) may be in play.")
    sys.exit(3)


if __name__ == "__main__":
    main()
