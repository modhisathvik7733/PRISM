"""PPO training for the recurrent policy on top of frozen JEPA.

Phase 4 — closes the BC plateau. Hand-coded memory mode hits 0.601, BC
plateau is ~0.50, REINFORCE got us to 0.55 but no further. PPO with proper
components (value function, GAE, clipped surrogate, vec-env, K epochs)
is the standard recipe BabyAI papers use to hit 0.95+.

Algorithm: vectorized PPO.
  for iteration in range(N):
    rollout T steps × B parallel envs:
      z = jepa.encode(obs_batch)            # frozen
      logits, value, h = policy.step_with_value(z, prev_a, mission, h)
      action ~ Categorical(masked_logits)   # disallowed actions → -inf
      env.step(action) → next_obs, reward, done
      store (obs, action, log_prob, reward, value, done)
      reset h on done (per-env)
    compute GAE advantages from stored (rewards, values, dones, last_value)
    returns = advantages + values
    for K epochs:
      shuffle into mini-batches, re-run policy with stored hidden states
      clipped surrogate loss + value loss + entropy bonus
      optimizer.step (only RecurrentPolicy params; JEPA frozen)

Vec-env: SyncVectorEnv with 16 parallel BabyAI envs. Each env runs its
own episode independently; resets and `agent.reset` are per-env.

Mission encoding matches collect_bc_data.py — one-hot of (type, color)
slot index. Action masking: per-env disallowed actions get -inf logits.

Usage:
    python -m scripts.ppo_train \
        --jepa-checkpoint runs/<...>/jepa_final.pt \
        --bc-checkpoint runs/bc_recurrent_v0.9b/policy_final.pt \
        --total-steps 500_000 --device cuda
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F

from prism.agents import goal_predicates_for_mission
from prism.agents.grounded_agent import allowed_actions_for_spec
from prism.agents.pose_tracker import MEM_FEAT_DIM, PoseTracker
from prism.envs.babyai import _encode_image, make_env_with_max_steps, set_max_steps  # noqa: F401
from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config
from prism.models.hybrid_policy import HybridPolicy
from prism.models.recurrent_policy import RecurrentPolicy
from prism.perception import compute_distances, extract_slots
from prism.perception.predicates import type_color_index
from prism.perception.slots import NUM_COLORS, OBJECT_TYPES
from prism.utils.seed import set_global_seed


def _goal_distance_from_raw_obs(raw_image_hwc, goal_pair) -> float:
    """Compute normalized manhattan distance to the closest goal slot in the
    current view. Returns 1.0 (max) when goal not in view. Used by reward
    shaping. raw_image_hwc is the un-normalized BabyAI image obs (H, W, 3)."""
    if goal_pair is None:
        return 1.0
    gt, gc = goal_pair
    slots = extract_slots(raw_image_hwc)
    cands = [s for s in slots if s.type_id == gt and s.color_id == gc]
    if not cands:
        return 1.0
    # Reuse the existing distance computation: returns (24,) where index i is
    # min normalized manhattan dist to (type, color)_i, 1.0 if absent.
    dists = compute_distances(slots)
    return float(dists[type_color_index(gt, gc)])


def latent_dim_for_cfg(cfg: JepaConfig) -> int:
    enc = getattr(cfg, "encoder_type", "flat")
    if enc == "categorical_spatial":
        C = getattr(cfg, "spatial_channels", 64)
        return C * cfg.obs_h * cfg.obs_w
    return cfg.embed_dim


# ----------------------------------------------------------------------
# Per-env state — wraps a BabyAI env and holds the per-env recurrent state
# the agent needs to maintain across timesteps.
# ----------------------------------------------------------------------
class EnvWorker:
    """Wraps one BabyAI env + per-env recurrent state. Sync vec-env steps
    each worker by calling .step(action) and reading .obs_encoded etc."""

    def __init__(self, env_id: str, base_seed: int, worker_id: int,
                 mission_dim: int, n_actions: int,
                 max_steps: int = 64, shaping_coef: float = 0.0,
                 use_pose_tracker: bool = False,
                 goal_provider=None,
                 held_out_combos: set[tuple[int, int]] | None = None):
        # Pass max_episode_steps to gym.make AND apply set_max_steps post-
        # construction for belt-and-suspenders coverage. Print the diagnostic
        # only on the first worker so we don't spam 16 lines.
        if worker_id == 0:
            self.env = make_env_with_max_steps(env_id, max_steps)
        else:
            self.env = gym.make(env_id, max_episode_steps=max_steps)
            set_max_steps(self.env, max_steps)
        self.base_seed = base_seed
        self.worker_id = worker_id
        self.episode_idx = 0
        self.n_actions = n_actions
        self.mission_dim = mission_dim
        self.shaping_coef = shaping_coef
        # Hold raw HWC image to compute goal-distance for reward shaping.
        # The encoded version stored in self.obs_encoded is for the policy.
        self.raw_image = None
        self.prev_goal_dist = 1.0
        # Path B — per-worker pose tracker. Disabled when use_pose_tracker=False
        # so legacy runs without --mem-feat-dim behave bit-for-bit identically.
        self.pose_tracker: PoseTracker | None = PoseTracker() if use_pose_tracker else None
        self.mem_feat = (
            np.zeros(MEM_FEAT_DIM, dtype=np.float32) if use_pose_tracker else None
        )
        # Optional callable mission_text -> (type_id, color_id) that
        # overrides the rule-parsed (type, color). The rule parser is
        # still used for `spec` (mission *type* — go-to vs pickup) and
        # the resulting `allowed_actions`. With goal_provider=None, this
        # is bit-identical to the original rule-only PPO.
        self.goal_provider = goal_provider
        # Stage 1.3 — when set, episodes whose mission target's
        # (color_id, type_id) is in this set are re-rolled (new seed)
        # until they fall outside. Used to train PPO with specific
        # compositional combos held out, then eval on those held-out
        # combos to measure compositional generalization at the policy
        # level.
        self.held_out_combos = held_out_combos or set()
        self._reset_episode()

    def _reset_episode(self):
        seed = self.base_seed + self.worker_id * 1_000_003 + self.episode_idx * 7919
        self.episode_idx += 1
        obs, _ = self.env.reset(seed=seed)
        # Re-seed loop. Two conditions to satisfy:
        # 1. Mission must be parseable.
        # 2. If --held-out-combos is set, mission target's (color, type)
        #    must NOT be in the held-out set.
        # Up to 50 attempts before giving up (effectively never for go-to
        # envs with 24 combos and ~4 held out → 5/6 chance per draw).
        for _ in range(50):
            parsed = goal_predicates_for_mission(obs["mission"])
            if parsed is not None:
                goal_preds, _spec = parsed
                if not goal_preds:
                    in_held_out = False
                else:
                    key = (
                        int(goal_preds[0].color_id),
                        int(goal_preds[0].type_id),
                    )
                    in_held_out = key in self.held_out_combos
                if not in_held_out:
                    break
            seed += 13
            obs, _ = self.env.reset(seed=seed)
            self.episode_idx += 1
        if parsed is None:
            self.allowed = (0, 1, 2)  # fallback for go-to-style
            self.mission_oh = np.zeros(self.mission_dim, dtype=np.float32)
            self.goal_pair = None
        else:
            goal_preds, spec = parsed
            self.allowed = allowed_actions_for_spec(spec, self.n_actions)
            # Default: rule-parsed (type, color).
            goal_type_id = goal_preds[0].type_id
            goal_color_id = goal_preds[0].color_id
            # If a language goal provider is attached, override the (type,
            # color) it predicts from the mission text. spec / allowed
            # actions are kept from the rule parse (mission-template-level).
            if self.goal_provider is not None:
                lang_type_id, lang_color_id = self.goal_provider(obs["mission"])
                if lang_type_id >= 0 and lang_color_id >= 0:
                    goal_type_id = lang_type_id
                    goal_color_id = lang_color_id
            tc_idx = type_color_index(goal_type_id, goal_color_id)
            self.mission_oh = np.zeros(self.mission_dim, dtype=np.float32)
            self.mission_oh[tc_idx] = 1.0
            self.goal_pair = (goal_type_id, goal_color_id)
        self.raw_image = obs["image"]
        self.obs_encoded = _encode_image(obs["image"])
        self.prev_goal_dist = _goal_distance_from_raw_obs(self.raw_image, self.goal_pair)
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.prev_action = -1
        # h_prev is owned by the trainer and reset externally on done; we
        # don't carry it on the worker.
        if self.pose_tracker is not None:
            gt = self.goal_pair[0] if self.goal_pair is not None else None
            gc = self.goal_pair[1] if self.goal_pair is not None else None
            self.pose_tracker.reset(gt, gc)
            self.pose_tracker.observe(self.obs_encoded)
            self.mem_feat = self.pose_tracker.features()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        # Force action into allowed set (defensive — the masked sample
        # should already respect this).
        if action not in self.allowed:
            action = self.allowed[0]
        # Commit the action to the tracker BEFORE env.step so a turn rotates
        # facing while observe() — fired below on next_obs — reconciles a
        # forward by comparing pre/post obs and updating pose if it moved.
        if self.pose_tracker is not None:
            self.pose_tracker.commit(action)
        next_obs, env_reward, term, trunc, info = self.env.step(action)
        done = bool(term or trunc)

        # Component 2 — potential-based shaping.
        # Compute goal-distance BEFORE potentially resetting, on the new
        # observation. Bonus = shaping_coef * (prev_dist - cur_dist):
        #   - positive when the agent moved closer (or brought goal into view)
        #   - negative when it moved further / lost sight of the goal
        # Per Ng et al. 1999, this is potential-based and preserves the
        # optimal policy under terminal reward, but provides dense gradient
        # signal that PPO needs to converge fast on this sparse env.
        shaping_bonus = 0.0
        if self.shaping_coef != 0.0:
            cur_goal_dist = _goal_distance_from_raw_obs(next_obs["image"], self.goal_pair)
            shaping_bonus = self.shaping_coef * (self.prev_goal_dist - cur_goal_dist)
            self.prev_goal_dist = cur_goal_dist

        # Total reward seen by PPO; episode-reward summary uses the env
        # reward only so logged window_R stays comparable across runs.
        total_reward = float(env_reward) + float(shaping_bonus)
        self.episode_reward += float(env_reward)  # log unshaped for honesty
        self.episode_steps += 1
        self.prev_action = action

        if done:
            ep_summary = {
                "ep_reward": self.episode_reward,
                "ep_steps": self.episode_steps,
            }
            self._reset_episode()
            return self.obs_encoded, total_reward, True, ep_summary
        else:
            self.raw_image = next_obs["image"]
            self.obs_encoded = _encode_image(next_obs["image"])
            if self.pose_tracker is not None:
                self.pose_tracker.observe(self.obs_encoded)
                self.mem_feat = self.pose_tracker.features()
            return self.obs_encoded, total_reward, False, {}


def make_action_mask(allowed_per_env, n_actions: int, device: torch.device):
    """Returns (B, n_actions) tensor of 0 for allowed, -inf for disallowed."""
    B = len(allowed_per_env)
    mask = torch.full((B, n_actions), float("-inf"), device=device)
    for i, allowed in enumerate(allowed_per_env):
        for a in allowed:
            mask[i, a] = 0.0
    return mask


def compute_gae(
    rewards: torch.Tensor,        # (T, B)
    values: torch.Tensor,         # (T, B)
    dones: torch.Tensor,          # (T, B)
    last_value: torch.Tensor,     # (B,)
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard GAE: A_t = delta_t + (gamma*lam)*(1-done_t) * A_{t+1}.
    Returns (advantages, returns) each of shape (T, B). returns = adv + values."""
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_adv = delta + gamma * lam * nonterminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def _find_substrate_banks(policy) -> dict | None:
    """Return {'concept': bank, 'operator': bank} if the policy is a
    transformer-trunk UniversalPolicy with retrieval; else None. Used
    by both curriculum mode and the --log-bank-stats path so they don't
    duplicate substrate-shape detection.
    """
    if getattr(policy, "state_kind", None) != "tuple":
        return None
    inner = getattr(policy, "inner", None)
    if inner is None or not hasattr(inner, "retrieval"):
        return None
    if not hasattr(inner.retrieval, "concept_bank"):
        return None
    return {
        "concept": inner.retrieval.concept_bank,
        "operator": inner.retrieval.operator_bank,
    }


def _log_bank_stats(banks: dict, it: int) -> None:
    """Print a one-line summary per managed bank. Quiet skip when the
    bank has no tracking data yet (activation_steps == 0)."""
    for name, bank in banks.items():
        if int(bank.activation_steps.item()) == 0:
            print(f"[bank-stats:iter {it+1}] {name}: no tracking yet "
                  f"(activation_steps=0)")
            continue
        frac = bank.slot_activation_fraction()
        n_active = int(bank.n_active)
        n_frozen = int(bank.frozen_mask.sum().item())
        active_frac = frac[bank.active_mask].clamp(min=1e-12)
        # Distribution stats over active slots.
        p = active_frac / active_frac.sum()
        entropy = float(-(p * p.log()).sum().item())
        # Max-entropy reference for active count.
        import math as _math
        ln_n = _math.log(max(n_active, 1))
        ratio = entropy / ln_n if ln_n > 0 else 0.0
        top = active_frac.topk(min(5, n_active))
        top_idx = top.indices.tolist()
        top_vals = [round(float(v), 4) for v in top.values.tolist()]
        print(
            f"[bank-stats:iter {it+1}] {name}: "
            f"active={n_active} frozen={n_frozen}/{bank.n_slots} "
            f"({100 * n_frozen / bank.n_slots:.1f}%) "
            f"max={float(active_frac.max()):.4f} "
            f"med={float(active_frac.median()):.4f} "
            f"entropy={entropy:.3f}/{ln_n:.3f} (ratio={ratio:.2f}) "
            f"top5_idx={top_idx} top5_frac={top_vals}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--bc-checkpoint", default=None,
                        help="path to RecurrentPolicy .pt to initialize from")
    parser.add_argument("--env-id", default="BabyAI-GoToLocal-v0")
    parser.add_argument("--total-steps", type=int, default=500_000,
                        help="total env steps across all workers")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=128,
                        help="T per iteration; total transitions = T * n_envs")
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--n-minibatches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-decay", action="store_true", default=True,
                        help="linearly decay lr from --lr to 0 over training")
    parser.add_argument("--ent-coef-start", type=float, default=0.01)
    parser.add_argument("--ent-coef-end", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=64,
                        help="env's truncation budget (Component 1). 64 = "
                             "BabyAI default, 128 = extended budget for harder spawns. "
                             "Affects per-episode reward via 1 - 0.9*(steps/max_steps).")
    parser.add_argument("--shaping-coef", type=float, default=0.0,
                        help="Component 2 reward shaping coefficient. 0.0 = "
                             "disabled (env reward only). 0.1 = recommended; "
                             "shaping_bonus = coef * (prev_dist - cur_dist) where "
                             "dist is normalized manhattan to closest goal slot, "
                             "1.0 if goal not in view. Potential-based per Ng 1999, "
                             "preserves optimal policy.")
    parser.add_argument("--mem-feat-dim", type=int, default=0,
                        help="Path B: 0 disables, 5 enables explicit memory "
                             "features (n_visited, n_blocked, goal_seen, "
                             "goal_fwd, goal_right) projected as a zero-init "
                             "residual into the policy/value head input. "
                             "Loading an old checkpoint with strict=False "
                             "leaves mem_proj at zero so initial behavior is "
                             "identical to the base policy.")
    parser.add_argument("--seed", type=int, default=2_000_000,
                        help="training seed; large to avoid eval-seed overlap")
    parser.add_argument("--run-name", default="ppo_v1")
    parser.add_argument("--save-every-iters", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # From-scratch policy initialization. Useful for ablations like Stage 1.2
    # where we don't have a BC checkpoint and want a clean apples-to-apples
    # comparison between two PPO runs. Slower convergence than warm-starting
    # from BC, but acceptable for short comparison runs.
    parser.add_argument("--no-bc", action="store_true",
                        help="skip loading --bc-checkpoint; initialize the "
                             "policy from random with the architecture params "
                             "given below.")
    parser.add_argument("--policy-hidden-dim", type=int, default=256)
    parser.add_argument("--policy-latent-proj-dim", type=int, default=128)
    # v5.0 — Hopfield-augmented hybrid policy. When --policy-type hybrid,
    # ConceptMemory + OperatorMemory replace the raw JEPA latent on the
    # GRU input. Same step_with_value contract as RecurrentPolicy.
    parser.add_argument(
        "--policy-type", choices=["recurrent", "hybrid", "universal"],
        default="recurrent",
        help="recurrent = original RecurrentPolicy (GRU on JEPA latent). "
             "hybrid = HybridPolicy (Hopfield ConceptMemory + OperatorMemory "
             "in front of the same GRU trunk). Use 'hybrid' for v5.0. "
             "universal = v6.0 UniversalPolicy with explicit DomainAdapter; "
             "in Phase A this delegates to HybridPolicy internally for "
             "behavior parity. Use 'universal' for v6.0 substrate refactor.",
    )
    # v6.0 PR-2: trunk selector for `--policy-type universal`. Phase A
    # uses 'gru' (delegates to existing HybridPolicy/RecurrentPolicy
    # internals). Phase B switches to 'transformer' (live UniversalTrunk
    # with two-tensor buffer). Ignored when --policy-type is not 'universal'.
    parser.add_argument(
        "--trunk", choices=["gru", "transformer"], default="gru",
        help="Trunk variant for --policy-type universal. 'gru' = Phase A "
             "behavior-parity wrapper; 'transformer' = Phase B live "
             "UniversalTrunk (PR-4+).",
    )
    # v6.0 PR-2: which inner policy class the universal trunk delegates to
    # in Phase A. Defaults to 'hybrid' (Hopfield-augmented), matching v5.
    parser.add_argument(
        "--universal-inner", choices=["hybrid", "recurrent"], default="hybrid",
        help="Inner v5 policy that --policy-type universal --trunk gru "
             "delegates to. PR-4 removes this flag when transformer trunk "
             "becomes the only path.",
    )
    # v6.0 PR-6: curriculum mode. When --n-stages > 1, total_steps is
    # divided evenly across N stages on the same env (smoke test for the
    # CurriculumEngine wiring). Per-stage env_factories arrive in PR-6b;
    # this gets us the freeze/expand/advance loop end-to-end on BabyAI.
    parser.add_argument(
        "--n-stages", type=int, default=1,
        help="Number of curriculum stages. Default 1 (no curriculum). "
             "When >1, total_steps is divided evenly; the engine runs "
             "stage transitions between segments. Requires --policy-type "
             "universal --trunk transformer. Ignored when --curriculum "
             "is set (the curriculum builder declares its own stage count).",
    )
    # PR-6b: named curricula with per-stage env factories.
    # When --curriculum is set, --n-stages is ignored and the stage
    # list comes from the registered builder. --curriculum-order
    # picks forward / reverse / shuffled for E1 ordering ablation.
    parser.add_argument(
        "--curriculum", default=None,
        help="Name of a registered curriculum. Currently: "
             "'babyai_developmental' (3-stage: GoToObj → GoToLocal → "
             "PickupLoc). When set, supersedes --n-stages and uses the "
             "named curriculum's per-stage env factories.",
    )
    parser.add_argument(
        "--curriculum-order", default="forward",
        choices=["forward", "reverse", "shuffled"],
        help="Stage ordering for --curriculum. E1 ordering ablation "
             "uses all three. shuffled is deterministic (seed=0).",
    )
    parser.add_argument(
        "--stage-expand-slots", type=int, default=0,
        help="Number of new slots each bank activates BEFORE each stage "
             "(except the first). Requires headroom: build the substrate "
             "with concept_n_slots >= initial_active + (n_stages-1)*this. "
             "0 means no growth.",
    )
    parser.add_argument(
        "--curriculum-warmup", type=int, default=0,
        help="Gradient steps each newly-expanded slot must run before "
             "the NEXT stage transition can freeze. Maps to "
             "CurriculumEngineConfig.warmup_steps. 0 disables the check.",
    )
    # AMP / mixed precision: forward + backward go through
    # torch.cuda.amp.autocast + GradScaler. Typically 1.5-2× speedup on
    # Ampere+ GPUs. Caveat: fp16 reductions in matmul change the log_prob
    # rollout/replay max_abs_diff from ~1e-7 to ~1e-3 — automatically
    # relaxes --check-replay-equality tolerance when --amp is set.
    # Subprocess-parallel env stepping. Forks n_envs subprocesses; each
    # owns one EnvWorker. Realistic 2-3× wall-clock speedup on BabyAI
    # at n_envs=32 — env stepping moves off the main process so the
    # GPU isn't gated on serial Python env.step calls. NOT compatible
    # with --goal-source lang (language head isn't shared across procs).
    parser.add_argument(
        "--async-envs", action="store_true",
        help="Run env workers in parallel subprocesses. 2-3× faster on "
             "BabyAI at n_envs=32. Disables --goal-source lang.",
    )
    parser.add_argument(
        "--amp", action="store_true",
        help="Enable mixed-precision training via torch.cuda.amp. "
             "1.5-2× faster on Ampere+ GPUs. Relaxes replay-equality "
             "tolerance from 1e-4 to 5e-3 to accommodate fp16 reduction "
             "noise.",
    )
    parser.add_argument(
        "--log-bank-stats", type=int, default=0,
        help="Print per-bank activation stats every N iterations. 0 "
             "disables. Requires --policy-type universal --trunk "
             "transformer. Reports active/frozen counts, max/median "
             "activation_fraction, attention entropy (over active "
             "slots), and top-5 slot indices. Useful for understanding "
             "how Hopfield retrieval evolves during long runs.",
    )
    parser.add_argument(
        "--curriculum-freeze-threshold", type=float, default=0.005,
        help="Default activation-fraction threshold for the freeze "
             "decision at each stage end. Used by any bank not overridden "
             "by the per-bank flags below. Provisional default; the v6 "
             "plan calls for an ablation sweep before Phase C.",
    )
    # Per-bank threshold overrides — PR-6 smoke showed Operator (β=4)
    # saturates at 0.005 while Concept (β=1) freezes ~1%/stage; the
    # same threshold doesn't fit both. None = use --curriculum-freeze-
    # threshold; a float overrides it for that bank.
    parser.add_argument(
        "--concept-freeze-threshold", type=float, default=None,
        help="Per-bank override for the concept bank. Default: use "
             "--curriculum-freeze-threshold.",
    )
    parser.add_argument(
        "--operator-freeze-threshold", type=float, default=None,
        help="Per-bank override for the operator bank. Suggested 0.05 "
             "given the β=4 sharp-retrieval saturation observed in the "
             "PR-6 smoke run. Default: use --curriculum-freeze-threshold.",
    )
    # v6.0 Phase B pre-condition (resolution-7 / audit pass-2 issue 4a):
    # one-shot check that the replay path produces bit-identical log_probs
    # to the rollout path on the first mini-batch of the first iteration.
    # Exits non-zero on mismatch; exits 0 with a PASS line otherwise. The
    # check uses the production rollout/replay code, so it cannot drift
    # from a parallel test rig.
    parser.add_argument(
        "--check-replay-equality", action="store_true",
        help="Assert log_prob(rollout) == log_prob(replay) bit-exactly on "
             "the first PPO mini-batch, then exit. Gates PR-4: the two-tensor "
             "buffer must preserve this invariant.",
    )
    parser.add_argument("--concept-n-slots", type=int, default=1024)
    parser.add_argument("--concept-slot-dim", type=int, default=64)
    parser.add_argument("--concept-scaling", type=float, default=1.0)
    parser.add_argument("--operator-n-slots", type=int, default=64)
    parser.add_argument("--operator-slot-dim", type=int, default=64)
    parser.add_argument("--operator-scaling", type=float, default=4.0)
    parser.add_argument(
        "--no-operator-memory", action="store_true",
        help="disable OperatorMemory for the hybrid policy. Only "
             "ConceptMemory is used. Lighter, faster.",
    )
    # Stage 1.2 — replace the rule-based mission parser's (type, color)
    # extraction with a trained text→(color, type) classifier. spec /
    # allowed_actions still come from the rule parser. With "rule" (default)
    # this script behaves identically to the original.
    parser.add_argument("--goal-source", choices=["rule", "lang"], default="rule",
                        help="source of the goal (type, color) signal that "
                             "drives the mission one-hot fed to the policy.")
    parser.add_argument("--lang-checkpoint", default=None,
                        help="text→(color, type) head; required when "
                             "--goal-source lang")
    parser.add_argument("--vocab-checkpoint", default=None,
                        help="WhitespaceVocab checkpoint; required when "
                             "--goal-source lang")
    # Stage 1.3 — hold out specific (color, type_idx) combos from training.
    # Each entry is "color_id,type_idx" where type_idx is the position in
    # OBJECT_TYPES (0=door, 1=key, 2=ball, 3=box). Episodes whose mission
    # target matches any held-out combo are re-rolled until they don't.
    parser.add_argument("--held-out-combos", nargs="*", default=[],
                        help="space-separated 'color_id,type_idx' pairs to "
                             "exclude from PPO training. Used for Stage 1.3 "
                             "compositional generalization. type_idx is the "
                             "position in OBJECT_TYPES (0=DOOR 1=KEY 2=BALL "
                             "3=BOX). Example: --held-out-combos 0,2 1,1")
    args = parser.parse_args()
    if args.goal_source == "lang":
        if args.lang_checkpoint is None or args.vocab_checkpoint is None:
            parser.error(
                "--goal-source lang requires both --lang-checkpoint and "
                "--vocab-checkpoint"
            )
    if not args.no_bc and args.bc_checkpoint is None:
        parser.error(
            "--bc-checkpoint is required unless --no-bc is set"
        )
    # Parse --held-out-combos "color,type_idx" strings into (color, type_id)
    # tuples (type_id = OBJECT_TYPES[type_idx]).
    held_out_combos: set[tuple[int, int]] = set()
    for s in args.held_out_combos:
        try:
            c_str, t_str = s.split(",")
            c_id = int(c_str)
            t_idx = int(t_str)
        except ValueError:
            parser.error(
                f"--held-out-combos entry {s!r} must be 'color_id,type_idx'"
            )
        if not (0 <= t_idx < len(OBJECT_TYPES)):
            parser.error(
                f"--held-out-combos type_idx {t_idx} out of range "
                f"[0, {len(OBJECT_TYPES)})"
            )
        held_out_combos.add((c_id, int(OBJECT_TYPES[t_idx])))

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # ---------- frozen JEPA ----------
    ckpt = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
    cfg: JepaConfig = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    n_actions = cfg.n_actions
    latent_dim = latent_dim_for_cfg(cfg)
    mission_dim = len(OBJECT_TYPES) * NUM_COLORS
    print(f"[ppo] frozen JEPA: encoder={cfg.encoder_type} latent_dim={latent_dim} n_actions={n_actions}")

    # ---------- recurrent / hybrid / universal policy ----------
    def _build_policy(**kwargs):
        """Construct one of: RecurrentPolicy, HybridPolicy, or
        UniversalPolicy (v6.0 PR-2).

        For --policy-type universal --trunk gru, the UniversalPolicy
        delegates to the inner v5 class selected by --universal-inner.
        This is the Phase A behavior-parity path: no math changes; we
        only add the substrate-side API surface (adapter ownership,
        action-mask routing, future buffer reshape).
        """
        if args.policy_type == "universal":
            # v6.0 substrate path. Encoder ownership lives in the adapter
            # (resolution 1). The adapter wraps the already-loaded
            # frozen JEPA without re-reading the checkpoint.
            from prism.adapters.babyai_adapter import BabyAIAdapter
            from prism.cognition.policy import UniversalPolicy
            adapter = BabyAIAdapter(jepa=jepa, cfg=cfg, device=device)
            return UniversalPolicy.from_adapter(
                adapter,
                trunk=args.trunk,
                hidden_dim=kwargs["hidden_dim"],
                latent_proj_dim=kwargs["latent_proj_dim"],
                action_emb_dim=kwargs.get("action_emb_dim", 16),
                mem_feat_dim=kwargs.get("mem_feat_dim", 0),
                policy_type=args.universal_inner,
                concept_n_slots=args.concept_n_slots,
                concept_slot_dim=args.concept_slot_dim,
                concept_scaling=args.concept_scaling,
                operator_n_slots=args.operator_n_slots,
                operator_slot_dim=args.operator_slot_dim,
                operator_scaling=args.operator_scaling,
                use_operator_memory=not args.no_operator_memory,
            ).to(device)
        if args.policy_type == "hybrid":
            return HybridPolicy(
                **kwargs,
                concept_n_slots=args.concept_n_slots,
                concept_slot_dim=args.concept_slot_dim,
                concept_scaling=args.concept_scaling,
                operator_n_slots=args.operator_n_slots,
                operator_slot_dim=args.operator_slot_dim,
                operator_scaling=args.operator_scaling,
                use_operator_memory=not args.no_operator_memory,
            ).to(device)
        return RecurrentPolicy(**kwargs).to(device)

    if args.no_bc:
        mem_feat_dim = int(args.mem_feat_dim)
        policy_latent_in_dim = latent_dim
        policy_n_actions = n_actions
        policy_mission_dim = mission_dim
        policy_hidden_dim = args.policy_hidden_dim
        policy_latent_proj_dim = args.policy_latent_proj_dim
        policy = _build_policy(
            latent_in_dim=policy_latent_in_dim,
            n_actions=policy_n_actions,
            mission_dim=policy_mission_dim,
            hidden_dim=policy_hidden_dim,
            latent_proj_dim=policy_latent_proj_dim,
            mem_feat_dim=mem_feat_dim,
        )
        print(f"[ppo] policy={args.policy_type} initialized from scratch (no BC): "
              f"hidden={policy_hidden_dim} "
              f"latent_proj={policy_latent_proj_dim} "
              f"mem={mem_feat_dim}")
    else:
        bc = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
        # mem_feat_dim is a CLI knob: when starting from an old checkpoint that
        # didn't have a residual, mem_proj is added and zero-init so the loaded
        # weights produce identical step-0 behavior. The checkpoint's own
        # mem_feat_dim (if present) only sets the floor — CLI can override to
        # match a new tracker layout.
        mem_feat_dim = max(int(args.mem_feat_dim), int(bc.get("mem_feat_dim", 0) or 0))
        policy_latent_in_dim = bc["latent_in_dim"]
        policy_n_actions = bc["n_actions"]
        policy_mission_dim = bc["mission_dim"]
        policy_hidden_dim = bc["hidden_dim"]
        policy_latent_proj_dim = bc["latent_proj_dim"]
        policy = _build_policy(
            latent_in_dim=policy_latent_in_dim,
            n_actions=policy_n_actions,
            mission_dim=policy_mission_dim,
            hidden_dim=policy_hidden_dim,
            latent_proj_dim=policy_latent_proj_dim,
            mem_feat_dim=mem_feat_dim,
        )
        # strict=False so the value_head (newly added) loads with random init.
        # When loading a RecurrentPolicy BC checkpoint into HybridPolicy,
        # the Hopfield memories are also random-init (no overlap in keys),
        # and the latent_proj / action_emb / mission_proj / gru / heads
        # load via name match.
        missing, unexpected = policy.load_state_dict(bc["policy_state_dict"], strict=False)
        print(f"[ppo] BC weights loaded: missing={missing} unexpected={unexpected}")
        if bc["latent_in_dim"] != latent_dim:
            raise SystemExit("BC policy / JEPA latent_dim mismatch")
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[ppo] policy params: {n_params:,}  (value head random-init)")
    print(f"[ppo] mem_feat_dim={mem_feat_dim}  "
          f"(mem_proj {'enabled (zero-init)' if mem_feat_dim > 0 else 'disabled'})")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    # AMP scaler — no-op when disabled. GradScaler manages fp16 gradient
    # underflow: scale loss before backward, unscale before clip + step.
    amp_scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    if args.amp:
        print(f"[ppo] AMP enabled: forward+backward in mixed precision; "
              f"replay-equality tolerance relaxed to 5e-3 (fp16 reduction noise).")

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ppo] writing to {out_dir}")

    # ---------- vectorized envs (sync) ----------
    use_pose_tracker = mem_feat_dim > 0
    if use_pose_tracker and mem_feat_dim != MEM_FEAT_DIM:
        raise SystemExit(
            f"--mem-feat-dim must be {MEM_FEAT_DIM} (PoseTracker layout) "
            f"or 0 (disabled); got {mem_feat_dim}"
        )
    # Stage 1.2 — optionally swap the rule-based mission parser's
    # (type, color) with a trained text→(color, type) classifier.
    goal_provider = None
    if args.goal_source == "lang":
        from prism.agents.lang_goal_provider import LangGoalProvider
        goal_provider = LangGoalProvider(
            lang_checkpoint=args.lang_checkpoint,
            vocab_checkpoint=args.vocab_checkpoint,
            device=device,
        )
        print(f"[ppo] goal source = lang  "
              f"(lang={args.lang_checkpoint}, vocab={args.vocab_checkpoint})")
    else:
        print(f"[ppo] goal source = rule (regex parser)")
    if held_out_combos:
        print(f"[ppo] held-out combos ({len(held_out_combos)}): "
              f"{sorted(held_out_combos)}  (rejected during training)")
    def _build_workers(env_id: str):
        """Construct n_envs fresh EnvWorkers on the given env_id. Used at
        startup and at curriculum-mode stage transitions to hot-swap envs
        between stages. When --async-envs is set, returns a
        ParallelEnvWorkers subprocess pool instead of a list; the rollout
        loop branches on type."""
        if args.async_envs:
            if goal_provider is not None:
                raise SystemExit(
                    "--async-envs is incompatible with --goal-source lang "
                    "(language head can't be shared across processes)."
                )
            from prism.envs.parallel_workers import ParallelEnvWorkers
            return ParallelEnvWorkers(
                env_id=env_id, n_envs=args.n_envs, base_seed=args.seed,
                mission_dim=mission_dim, n_actions=n_actions,
                max_steps=args.max_steps, shaping_coef=args.shaping_coef,
                use_pose_tracker=use_pose_tracker,
                held_out_combos=held_out_combos,
            )
        return [
            EnvWorker(
                env_id, args.seed, i, mission_dim, n_actions,
                max_steps=args.max_steps, shaping_coef=args.shaping_coef,
                use_pose_tracker=use_pose_tracker,
                goal_provider=goal_provider,
                held_out_combos=held_out_combos,
            )
            for i in range(args.n_envs)
        ]

    def _close_workers(w):
        """Cleanly close a ParallelEnvWorkers pool; no-op for list."""
        if hasattr(w, "close") and callable(w.close):
            try:
                w.close()
            except Exception:
                pass

    workers = _build_workers(args.env_id)
    print(f"[ppo] env: max_steps={args.max_steps} shaping_coef={args.shaping_coef} "
          f"pose_tracker={use_pose_tracker}")

    # ---------- training loop ----------
    n_iterations = args.total_steps // (args.rollout_steps * args.n_envs)
    print(f"[ppo] target {args.total_steps} env steps "
          f"= {n_iterations} iterations @ {args.rollout_steps}*{args.n_envs}")

    # v6.0 PR-6: optional curriculum mode. n_stages > 1 wires the
    # CurriculumEngine to drive activation tracking + freeze/expand
    # between iteration segments. Requires universal+transformer policy
    # so the banks are at policy.inner.retrieval.{concept,operator}_bank.
    curriculum_engine = None
    iters_per_stage = n_iterations
    # Banks dict for curriculum and/or --log-bank-stats. Constructed
    # once; None if the policy is not transformer+retrieval.
    substrate_banks = _find_substrate_banks(policy)
    if args.log_bank_stats > 0 and substrate_banks is None:
        raise SystemExit(
            "--log-bank-stats requires --policy-type universal --trunk "
            "transformer with use_retrieval=True."
        )
    # If logging is on but curriculum isn't, enable tracking manually so
    # activation_mass actually accumulates.
    if args.log_bank_stats > 0 and args.n_stages <= 1 and substrate_banks is not None:
        for _b in substrate_banks.values():
            _b.tracking = True
    # PR-6b: curriculum can be either named (--curriculum) with per-stage
    # env factories, OR synthesized (--n-stages > 1) on a single env.
    # Named curriculum takes precedence.
    use_curriculum = args.curriculum is not None or args.n_stages > 1
    if use_curriculum:
        if substrate_banks is None:
            raise SystemExit(
                "curriculum mode requires --policy-type universal --trunk "
                "transformer (the curriculum engine reads Concept/Operator "
                "banks off the transformer's RetrievalBlock)."
            )
        from prism.curriculum.engine import CurriculumEngine, CurriculumEngineConfig
        from prism.curriculum.stage import Stage as _Stage

        banks = substrate_banks

        if args.curriculum is not None:
            # PR-6b path: named curriculum with per-stage env factories.
            from prism.curriculum.babyai_curriculum import (
                BABYAI_PROBE_ENV_ID, get_curriculum, reorder_curriculum,
            )
            # Per-stage env_steps budget. The builder's default is 167k×3;
            # honor --total-steps by rescaling proportionally if user
            # overrides it.
            n_stages_from_curr = len(get_curriculum(args.curriculum))
            per_stage_env = args.total_steps // n_stages_from_curr
            stages = get_curriculum(
                args.curriculum,
                per_stage_env_steps=per_stage_env,
                expand_slots_per_transition=args.stage_expand_slots,
            )
            stages = reorder_curriculum(stages, args.curriculum_order)
            iters_per_stage = max(
                1, per_stage_env // (args.rollout_steps * args.n_envs)
            )
            n_iterations = iters_per_stage * len(stages)
            args.n_stages = len(stages)
            probe_env_id = BABYAI_PROBE_ENV_ID
            print(f"[ppo] named curriculum '{args.curriculum}' "
                  f"(order={args.curriculum_order}): "
                  f"{[s.name + '@' + s.extra.get('env_id', '?') for s in stages]}")
        else:
            # PR-6 path: synthesize n_stages on the same env.
            iters_per_stage = max(1, n_iterations // args.n_stages)
            n_iterations = iters_per_stage * args.n_stages
            env_factory_placeholder = lambda: None
            stages = [
                _Stage(
                    name=f"stage{k}",
                    env_factory=env_factory_placeholder,
                    max_env_steps=iters_per_stage * args.rollout_steps * args.n_envs,
                    expand_slots_before=(0 if k == 0 else args.stage_expand_slots),
                    freeze_after=(k < args.n_stages - 1),
                    extra={"env_id": args.env_id},
                )
                for k in range(args.n_stages)
            ]
            probe_env_id = args.env_id

        per_bank: dict[str, float] = {}
        if args.concept_freeze_threshold is not None:
            per_bank["concept"] = args.concept_freeze_threshold
        if args.operator_freeze_threshold is not None:
            per_bank["operator"] = args.operator_freeze_threshold
        config = CurriculumEngineConfig(
            activation_freeze_threshold=args.curriculum_freeze_threshold,
            per_bank_threshold=per_bank,
            warmup_steps=args.curriculum_warmup,
        )
        curriculum_engine = CurriculumEngine(
            stages=stages, banks=banks, config=config,
        )

        # Auto probe-set: collect (or load) at Stage 0 init per
        # resolution 6. Probe is always on the canonical probe_env_id
        # (BabyAI-GoToLocal for the babyai_developmental curriculum)
        # so E4 across curriculum-order arms uses the SAME probes.
        from prism.envs.babyai import _encode_image, make_env_with_max_steps

        def _probe_env_factory():
            return make_env_with_max_steps(probe_env_id, max_steps=64)

        def _probe_obs_fn(raw):
            return _encode_image(raw["image"])

        def _probe_mission_fn(raw):
            from prism.perception.slots import NUM_COLORS, OBJECT_TYPES
            v = np.zeros(len(OBJECT_TYPES) * NUM_COLORS, dtype=np.float32)
            try:
                preds = goal_predicates_for_mission(raw["mission"], None)
                if preds and preds[0][1] is not None and preds[0][2] is not None:
                    col, typ = preds[0][1], preds[0][2]
                    if 0 <= col < NUM_COLORS and 0 <= typ < len(OBJECT_TYPES):
                        v[typ * NUM_COLORS + col] = 1.0
            except Exception:
                pass
            return v

        curriculum_engine.init_probe_set(
            env_factory=_probe_env_factory,
            env_id=probe_env_id,
            n_actions=n_actions,
            obs_fn=_probe_obs_fn,
            mission_fn=_probe_mission_fn,
            save_dir=out_dir,
        )
        if curriculum_engine.probe_set is not None:
            print(f"[ppo] probe set ready: {curriculum_engine.probe_set.n_frames} frames "
                  f"from {probe_env_id} hash={curriculum_engine.probe_set.hash[:16]}…")

        threshold_repr = (
            f"default={args.curriculum_freeze_threshold}"
            + (f" concept={per_bank['concept']}" if "concept" in per_bank else "")
            + (f" operator={per_bank['operator']}" if "operator" in per_bank else "")
        )
        print(f"[ppo] curriculum mode: {len(stages)} stages × "
              f"{iters_per_stage} iters each (= {n_iterations} total). "
              f"expand={args.stage_expand_slots}/stage threshold=({threshold_repr}) "
              f"warmup={args.curriculum_warmup}")

        # If first stage's env_id differs from --env-id, hot-swap workers.
        first_stage_env = stages[0].extra.get("env_id")
        if first_stage_env is not None and first_stage_env != args.env_id:
            print(f"[ppo] swapping env for stage 0: {args.env_id} → {first_stage_env}")
            _close_workers(workers)
            workers = _build_workers(first_stage_env)

        # Begin tracking activations on stage 0 immediately.
        curriculum_engine.start_stage_tracking()

    h = policy.init_hidden(args.n_envs, device)  # (B, hidden_dim) or (tokens, valid_len)
    prev_actions = torch.full((args.n_envs,), -1, device=device, dtype=torch.long)

    # State kind: 'tensor' for v5 / universal+gru paths (single hidden tensor),
    # 'tuple' for universal+transformer paths ((buf_tokens, buf_valid_len)).
    # The trainer branches on this to size and record the rollout buffer.
    state_kind = getattr(policy, "state_kind", "tensor")
    # Pull L/D_tok off the policy when it's a tuple-state path. Defaults are
    # ignored on the tensor path.
    trunk_L = int(getattr(policy, "L", 1))
    trunk_D = int(getattr(policy, "hidden_dim", 0))

    ep_reward_window = deque(maxlen=200)
    ep_steps_window = deque(maxlen=200)
    total_env_steps = 0
    metrics_log: list[dict] = []

    for it in range(n_iterations):
        # ===== ROLLOUT PHASE =====
        T, B = args.rollout_steps, args.n_envs
        # buffers
        buf_z = torch.zeros(T, B, latent_dim, device=device)
        buf_actions = torch.zeros(T, B, dtype=torch.long, device=device)
        buf_log_probs = torch.zeros(T, B, device=device)
        buf_rewards = torch.zeros(T, B, device=device)
        buf_values = torch.zeros(T, B, device=device)
        buf_dones = torch.zeros(T, B, device=device)
        # Per-state-kind hidden buffer. tensor path: single (T, B, hidden_dim).
        # tuple path: two tensors stored independently (audit pass-2 issue 4a:
        # both must be restored at replay time or log_probs desync).
        if state_kind == "tuple":
            buf_h_init = None
            buf_h_tokens = torch.zeros(T, B, trunk_L, trunk_D, device=device)
            buf_h_valid = torch.zeros(T, B, dtype=torch.long, device=device)
        else:
            buf_h_init = torch.zeros(T, B, policy.hidden_dim, device=device)
            buf_h_tokens = None
            buf_h_valid = None
        buf_prev_actions = torch.zeros(T, B, dtype=torch.long, device=device)
        buf_missions = torch.zeros(T, B, mission_dim, device=device)
        buf_action_mask = torch.zeros(T, B, n_actions, device=device)
        buf_mem = (
            torch.zeros(T, B, mem_feat_dim, device=device)
            if use_pose_tracker else None
        )

        with torch.no_grad():
            for t in range(T):
                # gather obs / mission / mask from workers (list or pool)
                if args.async_envs:
                    state = workers.current_state()
                    obs_batch_np = state["obs_encoded"]
                    missions_np = state["mission_oh"]
                    allowed_per_env = state["allowed_lists"]
                else:
                    obs_batch_np = np.stack([w.obs_encoded for w in workers], axis=0)
                    missions_np = np.stack([w.mission_oh for w in workers])
                    allowed_per_env = [w.allowed for w in workers]
                obs_batch = torch.from_numpy(obs_batch_np).float().to(device)
                missions = torch.from_numpy(missions_np).float().to(device)
                mask = make_action_mask(allowed_per_env, n_actions, device)

                # encode + step policy (in autocast if --amp)
                with torch.amp.autocast("cuda", enabled=args.amp):
                    z = jepa.encode(obs_batch)
                z_flat = z.float().flatten(start_dim=1)  # (B, latent_dim) — store fp32
                if state_kind == "tuple":
                    buf_h_tokens[t] = h[0]
                    buf_h_valid[t] = h[1]
                else:
                    buf_h_init[t] = h
                buf_prev_actions[t] = prev_actions
                buf_missions[t] = missions
                buf_action_mask[t] = mask
                if use_pose_tracker:
                    if args.async_envs:
                        mem_np = state["mem_feat"]
                    else:
                        mem_np = np.stack([w.mem_feat for w in workers], axis=0)
                    mem_batch = torch.from_numpy(mem_np).float().to(device)
                    buf_mem[t] = mem_batch
                else:
                    mem_batch = None
                with torch.amp.autocast("cuda", enabled=args.amp):
                    logits, value, h_next = policy.step_with_value(
                        z, prev_actions, missions, h, mem_feat=mem_batch
                    )
                    # Categorical needs fp32 logits for stable log_prob.
                    logits = logits.float()
                    value = value.float()
                if args.policy_type == "universal":
                    dist = policy.action_dist(logits, mask)
                else:
                    dist = torch.distributions.Categorical(logits=logits + mask)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                buf_z[t] = z_flat
                buf_actions[t] = action
                buf_log_probs[t] = log_prob
                buf_values[t] = value

                # step envs (serial list or parallel pool)
                action_cpu = action.cpu().tolist()
                if args.async_envs:
                    state = workers.step_all(action_cpu)
                    rewards = state["rewards"].tolist()
                    dones = state["dones"].astype(np.float32).tolist()
                    for info, d in zip(state["infos"], state["dones"]):
                        if d and info:
                            ep_reward_window.append(info["ep_reward"])
                            ep_steps_window.append(info["ep_steps"])
                else:
                    rewards = []
                    dones = []
                    for i, w in enumerate(workers):
                        _obs, r, d, info = w.step(action_cpu[i])
                        rewards.append(r)
                        dones.append(1.0 if d else 0.0)
                        if d and info:
                            ep_reward_window.append(info["ep_reward"])
                            ep_steps_window.append(info["ep_steps"])
                buf_rewards[t] = torch.tensor(rewards, device=device)
                buf_dones[t] = torch.tensor(dones, device=device)
                # Reset h on done (per-env), set prev_action to action (or -1 if done).
                # For universal policies we go through `policy.reset_buffer(done, h)`,
                # the only API that resets both tensors of the tuple state atomically
                # (audit pass-2 issue 7g). For hybrid/recurrent paths we keep the v5
                # single-tensor torch.where to preserve bit-exactness.
                done_t = buf_dones[t].bool()
                if args.policy_type == "universal":
                    h = policy.reset_buffer(done_t, h_next)
                else:
                    h = torch.where(done_t.unsqueeze(1), policy.init_hidden(B, device), h_next)
                prev_actions = torch.where(done_t, torch.full_like(action, -1), action)

            # bootstrap value for the last state
            if args.async_envs:
                last_state = workers.current_state()
                obs_batch_np = last_state["obs_encoded"]
                missions_np = last_state["mission_oh"]
                mem_np_last = last_state["mem_feat"]
            else:
                obs_batch_np = np.stack([w.obs_encoded for w in workers], axis=0)
                missions_np = np.stack([w.mission_oh for w in workers])
                mem_np_last = (np.stack([w.mem_feat for w in workers], axis=0)
                               if use_pose_tracker else None)
            obs_batch = torch.from_numpy(obs_batch_np).float().to(device)
            missions = torch.from_numpy(missions_np).float().to(device)
            z = jepa.encode(obs_batch)
            if use_pose_tracker and mem_np_last is not None:
                mem_last = torch.from_numpy(mem_np_last).float().to(device)
            else:
                mem_last = None
            _, last_value, _ = policy.step_with_value(
                z, prev_actions, missions, h, mem_feat=mem_last
            )

        total_env_steps += T * B

        # GAE advantages and returns
        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones, last_value,
            gamma=args.gamma, lam=args.lam,
        )
        # Normalize advantages globally for this rollout.
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp(min=1e-8)
        advantages_norm = (advantages - adv_mean) / adv_std

        # ===== UPDATE PHASE =====
        # Linear schedules.
        progress = it / max(n_iterations - 1, 1)
        if args.lr_decay:
            for g in opt.param_groups:
                g["lr"] = args.lr * (1.0 - progress)
        ent_coef = args.ent_coef_start + (args.ent_coef_end - args.ent_coef_start) * progress

        # Flatten T*B → mini-batches (each mb is contiguous in env dimension
        # because we re-run the GRU per-env, but we shuffle the env-dim only).
        env_indices = np.arange(B)
        mb_size = max(B // args.n_minibatches, 1)

        last_pi_loss = last_v_loss = last_ent = last_kl = 0.0
        for epoch in range(args.ppo_epochs):
            np.random.shuffle(env_indices)
            for mb_start in range(0, B, mb_size):
                mb_envs = env_indices[mb_start:mb_start + mb_size]
                mb_envs_t = torch.from_numpy(mb_envs).to(device)

                # Re-run the policy across the rollout for this mini-batch
                # of envs. Use the recorded initial hidden state at t=0.
                mb_z = buf_z[:, mb_envs_t]                          # (T, mb, latent)
                mb_prev = buf_prev_actions[:, mb_envs_t]            # (T, mb)
                mb_missions = buf_missions[:, mb_envs_t]            # (T, mb, mdim)
                mb_mask = buf_action_mask[:, mb_envs_t]             # (T, mb, n_actions)
                mb_actions = buf_actions[:, mb_envs_t]              # (T, mb)
                mb_old_logp = buf_log_probs[:, mb_envs_t]           # (T, mb)
                mb_returns = returns[:, mb_envs_t]                  # (T, mb)
                mb_adv = advantages_norm[:, mb_envs_t]              # (T, mb)
                mb_dones = buf_dones[:, mb_envs_t]                  # (T, mb)
                mb_mem = buf_mem[:, mb_envs_t] if buf_mem is not None else None

                # We need to handle within-rollout episode boundaries: when
                # done at step t, reset hidden for env at step t+1. Use the
                # per-state-kind buffer at t=0 as the very-first hidden, then
                # re-derive. For tuple state this is BOTH tensors restored
                # together — audit pass-2 issue 4a: a missing restore here
                # silently desyncs log_probs in PPO replay.
                if state_kind == "tuple":
                    h_run = (
                        buf_h_tokens[0, mb_envs_t],
                        buf_h_valid[0, mb_envs_t],
                    )
                else:
                    h_run = buf_h_init[0, mb_envs_t]
                # latent passed to policy: we kept a flat (B, latent_dim)
                # version in buf_z; reshape into the encoder's natural
                # spatial form by reading cfg.
                # The policy's latent_proj already does Flatten internally,
                # so passing flat (mb, latent_dim) works as long as the
                # input shape's last-dim matches latent_in_dim.
                logits_seq = []
                values_seq = []
                for t in range(T):
                    mem_t = mb_mem[t] if mb_mem is not None else None
                    with torch.amp.autocast("cuda", enabled=args.amp):
                        logits_t, value_t, h_run = policy.step_with_value(
                            mb_z[t], mb_prev[t], mb_missions[t], h_run, mem_feat=mem_t
                        )
                        logits_t = logits_t.float()
                        value_t = value_t.float()
                    logits_seq.append(logits_t)
                    values_seq.append(value_t)
                    # reset h on done at step t (for step t+1 onward).
                    # Same dispatch as the rollout reset above: universal
                    # goes through policy.reset_buffer (single paired-reset
                    # API); hybrid/recurrent uses the v5 inline torch.where
                    # to preserve bit-exactness.
                    done_t = mb_dones[t].bool()
                    if args.policy_type == "universal":
                        h_run = policy.reset_buffer(done_t, h_run)
                    else:
                        h_run = torch.where(
                            done_t.unsqueeze(1),
                            policy.init_hidden(mb_z[t].shape[0], device),
                            h_run,
                        )
                logits_all = torch.stack(logits_seq, dim=0)  # (T, mb, n_actions)
                values_all = torch.stack(values_seq, dim=0)  # (T, mb)

                if args.policy_type == "universal":
                    dist = policy.action_dist(logits_all, mb_mask)
                else:
                    dist = torch.distributions.Categorical(logits=logits_all + mb_mask)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                # Resolution-7 / audit issue 4a: replay path must produce
                # log_probs that agree with rollout to within fp32 reduction
                # noise on the first replay mini-batch (no gradient steps
                # have happened yet, so the policy is provably identical to
                # its rollout state). The realistic failure mode is a
                # missing buffer-tensor restore — that produces O(1)
                # divergence in logits, not the ~1-ULP (≈1.2e-7) noise
                # that comes from log_softmax being run on differently
                # shaped batches in rollout (per-step, (B, n_actions)) vs
                # replay (stacked, (T*mb, n_actions)). The tolerance is
                # ~1000× ULP and ~10000× smaller than any real desync.
                if args.check_replay_equality and it == 0 and epoch == 0 and mb_start == 0:
                    max_abs_diff = float((new_logp - mb_old_logp).abs().max().item())
                    # fp16 matmul reductions in autocast push the diff
                    # from ~1e-7 to ~1e-3. Stay strict in fp32, relax in
                    # AMP — real desync is still O(1) and well above
                    # either threshold.
                    tol = 5e-3 if args.amp else 1e-4
                    bit_equal = torch.equal(new_logp, mb_old_logp)
                    print(
                        f"[E0b] replay log_prob max_abs_diff={max_abs_diff:.3e} "
                        f"tol={tol:.0e} bit_equal={bit_equal}"
                    )
                    if max_abs_diff > tol:
                        print(
                            f"[E0b] FAIL: max_abs_diff {max_abs_diff:.3e} > tol "
                            f"{tol:.0e}. The replay path is desynced from the "
                            f"rollout state. Suspect: missing buffer-tensor "
                            f"restore at ppo_train.py:740 (mini-batch slicing) "
                            f"or a non-deterministic op in step_with_value."
                        )
                        return 5
                    print(
                        f"[E0b] PASS: replay log_prob within fp32-reduction tolerance"
                        f"{' (bit-exact)' if bit_equal else ''}"
                    )
                    return 0

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_all, mb_returns)
                entropy_term = entropy.mean()
                loss = policy_loss + args.value_coef * value_loss - ent_coef * entropy_term

                opt.zero_grad(set_to_none=True)
                # AMP-aware backward: scaler.scale() handles fp16 underflow.
                # In fp32 (--amp not set) the scaler is a no-op pass-through.
                amp_scaler.scale(loss).backward()
                # Unscale BEFORE grad clip so the clip works on the
                # un-scaled gradients (otherwise the clip norm is
                # multiplied by scaler's scale factor).
                amp_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                amp_scaler.step(opt)
                amp_scaler.update()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - (new_logp - mb_old_logp)).mean()
                last_pi_loss = float(policy_loss.item())
                last_v_loss = float(value_loss.item())
                last_ent = float(entropy_term.item())
                last_kl = float(approx_kl.item())

        # Logging
        mean_R = float(np.mean(ep_reward_window)) if ep_reward_window else float("nan")
        mean_steps = float(np.mean(ep_steps_window)) if ep_steps_window else float("nan")
        metrics_log.append({
            "iter": it + 1,
            "env_steps": total_env_steps,
            "window_mean_R": mean_R,
            "ep_steps": mean_steps,
            "pi_loss": last_pi_loss,
            "v_loss": last_v_loss,
            "entropy": last_ent,
            "kl": last_kl,
        })
        if (it + 1) % 5 == 0 or it == 0 or it == n_iterations - 1:
            print(
                f"[iter {it+1:4d}/{n_iterations}] env_steps={total_env_steps:>7d} "
                f"window_R={mean_R:.3f} ep_steps={mean_steps:.1f} "
                f"pi={last_pi_loss:+.4f} v={last_v_loss:.4f} H={last_ent:.3f} KL={last_kl:.4f} "
                f"lr={opt.param_groups[0]['lr']:.2e} ent_coef={ent_coef:.4f}"
            )

        # v6.0 PR-6 follow-up: bank-stats logging cadence. Fires every
        # --log-bank-stats iterations. Works with or without curriculum;
        # when curriculum is off, tracking was enabled manually above so
        # activation_mass actually accumulates.
        if args.log_bank_stats > 0 and substrate_banks is not None \
                and (it + 1) % args.log_bank_stats == 0:
            _log_bank_stats(substrate_banks, it)

        # v6.0 PR-6: curriculum stage transition. Fires at the end of
        # every `iters_per_stage`-th iteration EXCEPT the final iteration
        # (final stage just ends without transitioning forward).
        if curriculum_engine is not None and (it + 1) % iters_per_stage == 0 \
                and (it + 1) < n_iterations:
            # Record gradient steps taken in this stage so the engine's
            # warmup check (audit 7a) has accurate counts. Each iter does
            # ppo_epochs * n_minibatches optimizer steps.
            grad_steps_this_stage = iters_per_stage * args.ppo_epochs * args.n_minibatches
            curriculum_engine.record_gradient_steps(grad_steps_this_stage)
            curriculum_engine.stop_stage_tracking()
            stage_idx_completed = curriculum_engine._current_stage_idx
            try:
                report = curriculum_engine.advance_stage(opt)
            except RuntimeError as e:
                print(f"[curriculum] advance_stage RAISED at stage {stage_idx_completed}: {e}")
                raise
            if report is not None:
                concept_r = report.bank_reports["concept"]
                operator_r = report.bank_reports["operator"]
                print(
                    f"[curriculum] stage {stage_idx_completed} → "
                    f"{stage_idx_completed + 1}: "
                    f"concept(frozen+={len(concept_r['frozen_idx'])}, "
                    f"expanded+={len(concept_r['expanded_idx'])}, "
                    f"active={concept_r['n_active_after']}, "
                    f"total_frozen={concept_r['n_frozen_after']}) | "
                    f"operator(frozen+={len(operator_r['frozen_idx'])}, "
                    f"expanded+={len(operator_r['expanded_idx'])}, "
                    f"active={operator_r['n_active_after']}, "
                    f"total_frozen={operator_r['n_frozen_after']})"
                )
            # PR-6b: hot-swap env workers if the next stage targets a
            # different env. The current `h` may carry buffer state from
            # the old env; reset it to a fresh per-env init so the new
            # stage starts with no stale token history.
            next_stage = curriculum_engine.current_stage
            next_env_id = next_stage.extra.get("env_id")
            if next_env_id is not None:
                prev_env_id = stages[stage_idx_completed].extra.get("env_id")
                if next_env_id != prev_env_id:
                    print(f"[curriculum] swapping env: {prev_env_id} → {next_env_id}")
                    _close_workers(workers)
                    workers = _build_workers(next_env_id)
                    # Reset rolling state — a buffer of stale tokens from
                    # the old env would pollute the new stage's first
                    # forward passes. Tuple state goes through reset_buffer.
                    full_done = torch.ones(B, dtype=torch.bool, device=device)
                    h = policy.reset_buffer(full_done, h)
                    prev_actions = torch.full(
                        (B,), -1, device=device, dtype=torch.long
                    )
            # Begin tracking for the next stage.
            curriculum_engine.start_stage_tracking()

        if (it + 1) % args.save_every_iters == 0 or it == n_iterations - 1:
            ckpt_path = out_dir / f"policy_iter{it+1}.pt"
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "policy_type": args.policy_type,
                "concept_n_slots": args.concept_n_slots,
                "concept_slot_dim": args.concept_slot_dim,
                "concept_scaling": args.concept_scaling,
                "operator_n_slots": args.operator_n_slots,
                "operator_slot_dim": args.operator_slot_dim,
                "operator_scaling": args.operator_scaling,
                "use_operator_memory": not args.no_operator_memory,
                "latent_in_dim": policy_latent_in_dim,
                "n_actions": policy_n_actions,
                "mission_dim": policy_mission_dim,
                "hidden_dim": policy_hidden_dim,
                "latent_proj_dim": policy_latent_proj_dim,
                "mem_feat_dim": mem_feat_dim,
                "jepa_checkpoint": args.jepa_checkpoint,
                "bc_checkpoint": args.bc_checkpoint,
                "goal_source": args.goal_source,
                "lang_checkpoint": args.lang_checkpoint,
                "held_out_combos": sorted(held_out_combos),
                "iteration": it + 1,
                "env_steps": total_env_steps,
                "window_mean_reward": float(np.mean(ep_reward_window)) if ep_reward_window else 0.0,
            }, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

    # v6.0 PR-6: tidy up after the final stage (no advance — the engine
    # ran its last `advance_stage` at the boundary between the
    # penultimate and final stages, and the final stage's `freeze_after`
    # was forced False in the synthesized curriculum).
    if curriculum_engine is not None:
        last_stage_grad_steps = iters_per_stage * args.ppo_epochs * args.n_minibatches
        curriculum_engine.record_gradient_steps(last_stage_grad_steps)
        curriculum_engine.stop_stage_tracking()
        print(f"[curriculum] final stage complete; "
              f"total gradient steps tracked = "
              f"{curriculum_engine._cumulative_gradient_steps}")

    # final
    final_path = out_dir / "policy_final.pt"
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "latent_in_dim": policy_latent_in_dim,
        "n_actions": policy_n_actions,
        "mission_dim": policy_mission_dim,
        "hidden_dim": policy_hidden_dim,
        "latent_proj_dim": policy_latent_proj_dim,
        "mem_feat_dim": mem_feat_dim,
        "jepa_checkpoint": args.jepa_checkpoint,
        "bc_checkpoint": args.bc_checkpoint,
        "goal_source": args.goal_source,
        "lang_checkpoint": args.lang_checkpoint,
        "held_out_combos": sorted(held_out_combos),
        "env_steps": total_env_steps,
        "window_mean_reward": float(np.mean(ep_reward_window)) if ep_reward_window else 0.0,
    }, final_path)
    print(f"[done] saved {final_path}")
    print(f"[done] final window_mean_R = {float(np.mean(ep_reward_window)):.3f}")

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump({"iterations": metrics_log}, f, indent=2)
    print(f"[done] saved {metrics_path}")
    _close_workers(workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
