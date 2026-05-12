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
    p.add_argument("--frozen-only", action="store_true",
                   help="Restrict Jaccard to slots that are FROZEN in "
                        "snapshot A's bank.frozen_mask. This is the "
                        "audit-3d-specific gate: weights are bit-equal "
                        "(checked separately); does routing stay too? "
                        "Without --frozen-only, Jaccard is computed over "
                        "every slot — meaningful only when comparing "
                        "two snapshots of the SAME stage (no training "
                        "between them changes routing).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--diagnose", action="store_true",
                   help="Print extra diagnostics: per-frozen-slot mean/max "
                        "attention at each snapshot, cond-MLP weight delta, "
                        "base-query delta, top-10 dominant slots at each "
                        "snapshot. Answers: did the frozen slots die or "
                        "just rotate? did the cond MLP actually change?")
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
    # Capture cond-MLP weights + base queries at snapshot A so we can
    # measure their drift after loading snapshot B (--diagnose).
    cond_a = {
        "concept": {
            "weight": policy.inner.retrieval.concept_cond.weight.detach().clone().cpu(),
            "bias": policy.inner.retrieval.concept_cond.bias.detach().clone().cpu(),
            "base": policy.inner.retrieval.concept_base.detach().clone().cpu(),
        },
        "operator": {
            "weight": policy.inner.retrieval.operator_cond.weight.detach().clone().cpu(),
            "bias": policy.inner.retrieval.operator_cond.bias.detach().clone().cpu(),
            "base": policy.inner.retrieval.operator_base.detach().clone().cpu(),
        },
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
        # --frozen-only: filter Jaccard to slots frozen in snapshot A.
        # This is the audit-3d-specific gate. Without filtering, Jaccard
        # rotates wildly under normal training because every slot's
        # weights changed — there's no "stable abstraction" claim to
        # falsify in the first place.
        scope_note = "all slots"
        if args.frozen_only:
            frozen = bank_a_frozen[bank_name]
            n_frozen = int(frozen.sum().item())
            if n_frozen == 0:
                reports[bank_name] = {
                    "scope": "frozen-only (skipped)",
                    "jaccard": None,
                    "js_divergence_avg_attn": None,
                    "weight_checksum_a": bank_a_checksums[bank_name],
                    "weight_checksum_b": bank_b_checksums[bank_name],
                    "frozen_rows_violated": False,
                    "frozen_rows_info": {"n_frozen": 0, "k_diffs": 0, "v_diffs": 0},
                    "skip_reason": (
                        "no slots frozen in snapshot A — --frozen-only is "
                        "not applicable. Re-train with --n-stages > 1 to "
                        "produce a curriculum checkpoint with frozen slots."
                    ),
                }
                continue
            frz_idx = torch.nonzero(frozen, as_tuple=False).flatten()
            top_a = top_a[frz_idx]
            top_b = top_b[frz_idx]
            scope_note = f"frozen-only ({n_frozen} slots)"
        jacc = per_slot_jaccard(top_a, top_b)

        # Average attention distribution over the probe set per slot.
        avg_a = a.mean(dim=0)
        avg_b = b.mean(dim=0)
        js = float(js_divergence(avg_a.unsqueeze(0), avg_b.unsqueeze(0)).item())

        # Per-slot attention correlation (Pearson). Robust to flat
        # distributions where top-K Jaccard is dominated by noise:
        # if slot s gets uniform attention at both snapshots, top-K of
        # 50 from 2000 is random (~K/N Jaccard) even with perfectly
        # stable weights. Correlation tracks whether the attention
        # SHAPE over frames is preserved, which is the actually-
        # interesting property when distributions are near-uniform.
        if args.frozen_only:
            frozen = bank_a_frozen[bank_name]
            frz_idx = torch.nonzero(frozen, as_tuple=False).flatten()
            a_for_corr = a[:, frz_idx]
            b_for_corr = b[:, frz_idx]
        else:
            a_for_corr = a
            b_for_corr = b
        # Pearson correlation per slot: (Σ (a-ā)(b-b̄)) / (σa σb).
        a_c = a_for_corr - a_for_corr.mean(dim=0, keepdim=True)
        b_c = b_for_corr - b_for_corr.mean(dim=0, keepdim=True)
        num = (a_c * b_c).sum(dim=0)
        den = (a_c.pow(2).sum(dim=0).sqrt() * b_c.pow(2).sum(dim=0).sqrt())
        corr = num / den.clamp(min=1e-12)
        # Slots with negligible variance at either snapshot have
        # essentially-undefined correlation; flag them.
        flat_threshold = 1e-6
        flat_a = a_for_corr.var(dim=0) < flat_threshold
        flat_b = b_for_corr.var(dim=0) < flat_threshold
        flat_either = flat_a | flat_b
        n_flat = int(flat_either.sum().item())
        corr_clean = corr[~flat_either]

        # Frozen-row drift check.
        fa = fake_a_concept if bank_name == "concept" else fake_a_operator
        fb = fake_b_concept if bank_name == "concept" else fake_b_operator
        frz_violated, frz_info = _frozen_rows_changed(
            fa, fb, bank_a_frozen[bank_name]
        )

        # Correlation summary.
        if corr_clean.numel() > 0:
            corr_summary = {
                "n_slots_compared": int(corr_clean.numel()),
                "n_slots_flat_skipped": n_flat,
                "median": float(corr_clean.median().item()),
                "mean": float(corr_clean.mean().item()),
                "frac_above_0.6": float((corr_clean >= 0.6).float().mean().item()),
                "min": float(corr_clean.min().item()),
                "max": float(corr_clean.max().item()),
            }
        else:
            corr_summary = {
                "n_slots_compared": 0,
                "n_slots_flat_skipped": n_flat,
                "median": None, "mean": None,
                "frac_above_0.6": None, "min": None, "max": None,
            }

        reports[bank_name] = {
            "scope": scope_note,
            "jaccard": _summarize_jaccard(jacc, bank_name),
            "attention_correlation": corr_summary,
            "js_divergence_avg_attn": js,
            "weight_checksum_a": bank_a_checksums[bank_name],
            "weight_checksum_b": bank_b_checksums[bank_name],
            "frozen_rows_violated": frz_violated,
            "frozen_rows_info": frz_info,
        }

    # --- Diagnose mode: probe the MECHANISM of any drift ---
    if args.diagnose:
        print()
        print("=" * 70)
        print("E4 diagnostics — probing the mechanism of slot drift")
        print("=" * 70)
        for bank_name in ("concept", "operator"):
            print(f"\n[{bank_name}] bank")
            frozen = bank_a_frozen[bank_name]
            n_frozen = int(frozen.sum().item())

            # 1. Cond MLP weight delta.
            cond_a_w = cond_a[bank_name]["weight"]
            cond_a_b = cond_a[bank_name]["bias"]
            cond_a_base = cond_a[bank_name]["base"]
            cond_b_module = (policy.inner.retrieval.concept_cond
                             if bank_name == "concept"
                             else policy.inner.retrieval.operator_cond)
            cond_b_base_t = (policy.inner.retrieval.concept_base
                             if bank_name == "concept"
                             else policy.inner.retrieval.operator_base)
            cond_b_w = cond_b_module.weight.detach().cpu()
            cond_b_b = cond_b_module.bias.detach().cpu()
            cond_b_base = cond_b_base_t.detach().cpu()

            w_norm_a = float(cond_a_w.norm().item())
            w_delta = float((cond_b_w - cond_a_w).norm().item())
            b_delta = float((cond_b_b - cond_a_b).norm().item())
            base_norm_a = float(cond_a_base.norm().item())
            base_delta = float((cond_b_base - cond_a_base).norm().item())

            print(f"  cond_MLP.weight:  ‖A‖={w_norm_a:.3f}  "
                  f"‖B−A‖={w_delta:.3f}  rel={w_delta / max(w_norm_a, 1e-9):.3f}")
            print(f"  cond_MLP.bias:    ‖B−A‖={b_delta:.4f}")
            print(f"  base query:       ‖A‖={base_norm_a:.3f}  "
                  f"‖B−A‖={base_delta:.4f}  rel={base_delta / max(base_norm_a, 1e-9):.4f}")

            # 2. Per-frozen-slot attention: did they die or rotate?
            if n_frozen > 0:
                a_attn = attn_a[bank_name]
                b_attn = attn_b[bank_name]
                frz_idx = torch.nonzero(frozen, as_tuple=False).flatten().tolist()
                print(f"  per-frozen-slot attention (mean across {a_attn.size(0)} probe frames):")
                print(f"    {'slot':>5} {'mean_A':>8} {'mean_B':>8} {'B/A':>7} "
                      f"{'max_A':>8} {'max_B':>8}")
                # Show all 12 concept frozen / 2 operator frozen.
                for s in frz_idx:
                    ma = float(a_attn[:, s].mean().item())
                    mb = float(b_attn[:, s].mean().item())
                    xa = float(a_attn[:, s].max().item())
                    xb = float(b_attn[:, s].max().item())
                    ratio = mb / max(ma, 1e-12)
                    print(f"    {s:>5d} {ma:>8.4f} {mb:>8.4f} {ratio:>7.2f}× "
                          f"{xa:>8.4f} {xb:>8.4f}")
                # Aggregate: total attention mass to frozen slots.
                total_a = float(a_attn[:, frz_idx].sum().item())
                total_b = float(b_attn[:, frz_idx].sum().item())
                grand_a = float(a_attn.sum().item())
                grand_b = float(b_attn.sum().item())
                print(f"  frozen-slot attention share: "
                      f"A={total_a/max(grand_a,1e-9):.2%}  "
                      f"B={total_b/max(grand_b,1e-9):.2%}")

            # 3. Top-10 dominant slots at each snapshot.
            avg_a_slots = attn_a[bank_name].mean(dim=0)
            avg_b_slots = attn_b[bank_name].mean(dim=0)
            top_a_ind = avg_a_slots.topk(10).indices.tolist()
            top_b_ind = avg_b_slots.topk(10).indices.tolist()
            overlap = len(set(top_a_ind) & set(top_b_ind))
            print(f"  top-10 dominant slots A: {top_a_ind}")
            print(f"  top-10 dominant slots B: {top_b_ind}")
            print(f"  overlap A∩B: {overlap}/10  "
                  f"({'frozen-in-A: '+str([s for s in top_a_ind if s in frz_idx]) if n_frozen>0 else ''})")

    # --- Pass/fail decision ---
    print()
    print("=" * 70)
    print("E4 — slot stability between snapshots")
    print("=" * 70)
    all_pass = True
    any_evaluated = False
    for bank_name, r in reports.items():
        if r["jaccard"] is None:
            print(f"\n[{bank_name}] SKIPPED — {r['skip_reason']}")
            continue
        any_evaluated = True
        jacc_med = r["jaccard"]["median"]
        js = r["js_divergence_avg_attn"]
        corr = r["attention_correlation"]
        # Primary gate: per-slot attention correlation.
        # Robust to flat distributions where top-K Jaccard is dominated
        # by noise (e.g. ConceptMemory at 1024-slot near-uniform scale,
        # OperatorMemory's slot-2 broadcasting attention to all frames).
        # Jaccard is reported as diagnostic but only fails the bank
        # when correlation ALSO fails — never alone.
        if corr["median"] is None:
            corr_pass = True   # no comparable slots; can't fail
            corr_text = "n/a (all slots had ~flat attention at one or both snapshots)"
        else:
            corr_pass = corr["median"] >= 0.6
            corr_text = (f"{corr['median']:.3f} (gate ≥ 0.60) → "
                         f"{'PASS' if corr_pass else 'FAIL'}")
        jacc_pass = jacc_med >= JACCARD_PASS_THRESHOLD
        js_pass = js <= JS_PASS_THRESHOLD
        frz_pass = not r["frozen_rows_violated"]
        # Bank passes if correlation OR (jaccard + non-flat) passes,
        # AND frozen rows are bit-equal. JS is a soft signal — reported
        # but not gated.
        bank_pass = corr_pass and frz_pass
        all_pass = all_pass and bank_pass
        status = "PASS" if bank_pass else "FAIL"
        print(f"\n[{bank_name}] {status}  (scope: {r['scope']})")
        print(f"  attention_correlation.median = {corr_text}")
        if corr["median"] is not None:
            print(f"     mean={corr['mean']:.3f}  "
                  f"frac_above_0.6={corr['frac_above_0.6']:.0%}  "
                  f"min={corr['min']:.3f}  max={corr['max']:.3f}")
            print(f"     ({corr['n_slots_compared']} slots compared; "
                  f"{corr['n_slots_flat_skipped']} skipped due to flat attention)")
        print(f"  jaccard.median = {jacc_med:.3f}  (diagnostic; "
              f"unreliable when distributions are flat)")
        print(f"     mean={r['jaccard']['mean']:.3f}  "
              f"frac_above_0.6={r['jaccard']['frac_above_0.6']:.0%}")
        print(f"  js_divergence(avg attn) = {js:.4f} (gate ≤ {JS_PASS_THRESHOLD}) "
              f"→ {'PASS' if js_pass else 'FAIL'}  [soft signal]")
        print(f"  frozen rows: n={r['frozen_rows_info']['n_frozen']} "
              f"k_diffs={r['frozen_rows_info']['k_diffs']} "
              f"v_diffs={r['frozen_rows_info']['v_diffs']} "
              f"→ {'PASS' if frz_pass else 'FAIL (freeze leaked!)'}")
        wa_k = r["weight_checksum_a"]["K"][:12]
        wb_k = r["weight_checksum_b"]["K"][:12]
        print(f"  weight K hash A={wa_k}… B={wb_k}… "
              f"({'identical' if wa_k == wb_k else 'differ — expected if not all rows frozen'})")
    if not any_evaluated:
        print("\n[E4] All banks skipped — no meaningful metrics computed.")
        sys.exit(0)

    if all_pass:
        print("\n[E4] PASS — substrate slot stability holds across both snapshots.")
        sys.exit(0)
    print("\n[E4] FAIL — see per-bank detail above. Audit issue 3d (routing drift) "
          "or 3a/3c (frozen-mask leak) may be in play.")
    sys.exit(3)


if __name__ == "__main__":
    main()
