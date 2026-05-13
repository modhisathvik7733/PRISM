"""PPO trainer for the synthetic Unity nav env.

Reward-only training — NO hand-coded expert. Substrate explores the env,
gets +1 for reaching the green target, small penalty for approaching the
red distractor, small per-step penalty. Learns from reward signal alone.

Phase 3 of the commercial pipeline:
  Phase 1: Concept pretrain (perception)   → runs/v6_concept_phaseAB_v2
  Phase 2: (skipped — BC has compounding-error ceiling on this task)
  Phase 3: PPO on env reward (policy)      → THIS SCRIPT

Frozen during training:
  * JEPA encoder weights (perception locked after concept-pretrain)
  * Concept memory K/V banks (anti-forgetting guarantee)
  * Operator memory K/V banks (anti-forgetting guarantee)
Trainable:
  * Trunk + action_head + value_head + projection layers
  → New policy is learned on top of the sharp concept representation.

Usage:
    python unity_demo/ppo_train_unity.py \\
        --jepa runs/v6_concept_phaseAB_v2/jepa.pt \\
        --base-policy runs/v6_concept_phaseAB_v2/policy.pt \\
        --out-path runs/v6_ppo_v1/policy.pt \\
        --total-steps 200000 --rollout-len 512 --eval-every-iters 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from prism.adapters.babyai_adapter import BabyAIAdapter
from prism.cognition.policy import UniversalPolicy
from prism.language.mission_parser import parse_mission
from prism.models.jepa import JepaWorldModel, upgrade_config
from prism.perception.predicates import type_color_index
from prism.perception.slots import NUM_COLORS, NUM_TYPES

from unity_demo.unity_nav_env import UnityNavEnv


# Allowed BabyAI actions for "go-to" missions (matches inference server mask).
_ALLOWED_ACTIONS = (0, 1, 2)
_ACTION_MASK_LOGIT = float("-inf")


# ===========================================================================
# Checkpoint loading (shared with other unity_demo scripts)
# ===========================================================================
def load_jepa(path: Path, device: torch.device, trainable: bool = False):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    if trainable:
        jepa.train()
        for p in jepa.parameters():
            p.requires_grad_(True)
    else:
        jepa.eval()
        for p in jepa.parameters():
            p.requires_grad_(False)
    return jepa, cfg


def build_policy(ckpt: dict, jepa, cfg, device: torch.device, trunk: str):
    adapter = BabyAIAdapter(jepa=jepa, cfg=cfg, device=device)
    policy = UniversalPolicy.from_adapter(
        adapter,
        trunk=trunk,
        hidden_dim=ckpt["hidden_dim"],
        latent_proj_dim=ckpt["latent_proj_dim"],
        mem_feat_dim=ckpt.get("mem_feat_dim", 0),
        concept_n_slots=ckpt.get("concept_n_slots", 1024),
        operator_n_slots=ckpt.get("operator_n_slots", 64),
        concept_scaling=ckpt.get("concept_scaling", 1.0),
        operator_scaling=ckpt.get("operator_scaling", 4.0),
        use_operator_memory=ckpt.get("use_operator_memory", True),
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    return policy


def freeze_memory_banks(policy: UniversalPolicy) -> dict[str, int]:
    """Freeze Concept/Operator K/V banks — preserves prior knowledge
    while PPO trains the trunk + heads on the new reward signal."""
    inner = policy._inner
    frozen = 0
    if hasattr(inner, "retrieval") and not isinstance(inner.retrieval, torch.nn.Identity):
        retrieval = inner.retrieval
        for bank_name in ("concept_bank", "operator_bank"):
            bank = getattr(retrieval, bank_name, None)
            if bank is None:
                continue
            for slot_name in ("keys", "values"):
                p = getattr(bank, slot_name, None)
                if p is None:
                    continue
                p.requires_grad_(False)
                frozen += p.numel()
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    return {"frozen_params": frozen, "trainable_params": trainable}


# ===========================================================================
# Mission encoding (shared with continual_finetune.py)
# ===========================================================================
def mission_to_onehot(mission_str: str, device: torch.device) -> torch.Tensor:
    spec = parse_mission(mission_str)
    if spec is None or spec.color_id is None:
        return torch.zeros(NUM_TYPES * NUM_COLORS, device=device)
    idx = type_color_index(spec.type_id, spec.color_id)
    v = torch.zeros(NUM_TYPES * NUM_COLORS, device=device)
    v[idx] = 1.0
    return v


# ===========================================================================
# Hidden-state helpers (universal policy uses a tuple for transformer trunk)
# ===========================================================================
def _detach_h(h):
    if isinstance(h, tuple):
        return tuple(t.detach() for t in h)
    return h.detach()


def _select_h(h, idx: int):
    """Return a batch-of-1 view of hidden state at batch index `idx`.
    Used to replay individual trajectory steps during PPO updates."""
    if isinstance(h, tuple):
        return tuple(t[idx:idx + 1] for t in h)
    return h[idx:idx + 1]


# ===========================================================================
# Rollout collection
# ===========================================================================
@torch.no_grad()
def collect_rollout(
    env: UnityNavEnv,
    policy: UniversalPolicy,
    jepa: JepaWorldModel,
    device: torch.device,
    rollout_len: int,
):
    """Run one env for `rollout_len` steps. Returns a dict of tensors plus
    the hidden state at each step (for replay during PPO updates).

    Reset on done; each transition records (obs, action, log_prob, value,
    reward, done, mission, prev_action, h_before_step).
    """
    state_kind = getattr(policy, "state_kind", "tensor")
    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    logp_buf: list[float] = []
    val_buf: list[float] = []
    rew_buf: list[float] = []
    done_buf: list[bool] = []
    mission_buf: list[torch.Tensor] = []
    prev_act_buf: list[int] = []
    # Hidden state list — element t is the h FED INTO step t (the state
    # before that transition). We need this for proper recurrent PPO replay.
    h_list: list = []
    # Per-episode tracking for SUCCESS / clean / reward stats.
    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    ep_success: list[bool] = []
    ep_clean: list[bool] = []
    cur_ep_reward = 0.0
    cur_ep_len = 0
    cur_ep_distracted = False

    obs, info = env.reset()
    h = policy.init_hidden(1, device)
    mission = mission_to_onehot(obs["mission"], device).unsqueeze(0)
    prev_action = -1

    for t in range(rollout_len):
        img = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(device)
        z = jepa.encode(img)
        prev_a_t = torch.tensor([prev_action], device=device, dtype=torch.long)
        logits, value, h_next = policy.step_with_value(z, prev_a_t, mission, h)
        # Action mask.
        mask = torch.full_like(logits, _ACTION_MASK_LOGIT)
        for a in _ALLOWED_ACTIONS:
            mask[..., a] = 0.0
        masked_logits = logits + mask
        dist = Categorical(logits=masked_logits)
        action = int(dist.sample().item())
        log_prob = float(dist.log_prob(torch.tensor([action], device=device)).item())

        obs_buf.append(obs["image"].copy())
        act_buf.append(action)
        logp_buf.append(log_prob)
        val_buf.append(float(value.item()))
        mission_buf.append(mission.squeeze(0).cpu())
        prev_act_buf.append(prev_action)
        h_list.append(_detach_h(h))

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rew_buf.append(float(reward))
        done_buf.append(bool(done))

        cur_ep_reward += float(reward)
        cur_ep_len += 1
        if info.get("approached_distractor", False):
            cur_ep_distracted = True

        if done:
            ep_rewards.append(cur_ep_reward)
            ep_lengths.append(cur_ep_len)
            ep_success.append(bool(info.get("reached_target", False) or reward >= 0.9))
            ep_clean.append(ep_success[-1] and not cur_ep_distracted)
            cur_ep_reward = 0.0
            cur_ep_len = 0
            cur_ep_distracted = False
            obs, info = env.reset()
            h = policy.init_hidden(1, device)
            mission = mission_to_onehot(obs["mission"], device).unsqueeze(0)
            prev_action = -1
        else:
            obs = next_obs
            h = h_next
            prev_action = action

    # Last value for bootstrap.
    img = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(device)
    z = jepa.encode(img)
    prev_a_t = torch.tensor([prev_action], device=device, dtype=torch.long)
    _, last_value, _ = policy.step_with_value(z, prev_a_t, mission, h)
    last_value = float(last_value.item())

    return {
        "obs": np.stack(obs_buf, axis=0),            # (T, 3, 7, 7)
        "actions": np.asarray(act_buf, dtype=np.int64),       # (T,)
        "log_probs": np.asarray(logp_buf, dtype=np.float32),  # (T,)
        "values": np.asarray(val_buf, dtype=np.float32),      # (T,)
        "rewards": np.asarray(rew_buf, dtype=np.float32),     # (T,)
        "dones": np.asarray(done_buf, dtype=np.bool_),        # (T,)
        "missions": torch.stack(mission_buf, dim=0),          # (T, 24)
        "prev_actions": np.asarray(prev_act_buf, dtype=np.int64),
        "hidden_states": h_list,                              # list len T
        "last_value": last_value,
        "ep_rewards": ep_rewards,
        "ep_lengths": ep_lengths,
        "ep_success": ep_success,
        "ep_clean": ep_clean,
    }


# ===========================================================================
# GAE
# ===========================================================================
def compute_gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
    last_value: float, gamma: float, lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation. Returns (advantages, returns)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - float(dones[t])
            next_value = last_value
        else:
            next_nonterminal = 1.0 - float(dones[t])
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


# ===========================================================================
# PPO update — per-sample replay through the policy (handles recurrent
# state by replaying each timestep with its stored h_before).
# ===========================================================================
def ppo_update(
    policy: UniversalPolicy,
    jepa: JepaWorldModel,
    optimizer: torch.optim.Optimizer,
    rollout: dict,
    advantages: np.ndarray,
    returns: np.ndarray,
    device: torch.device,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    n_epochs: int,
    minibatch_size: int,
    grad_clip: float,
):
    """Run K epochs of PPO updates on the rollout."""
    T = len(rollout["actions"])
    obs_t = torch.from_numpy(rollout["obs"]).float().to(device)
    actions_t = torch.from_numpy(rollout["actions"]).to(device)
    old_logp_t = torch.from_numpy(rollout["log_probs"]).to(device)
    missions_t = rollout["missions"].to(device)
    prev_actions_t = torch.from_numpy(rollout["prev_actions"]).to(device)
    adv_t = torch.from_numpy(advantages).to(device)
    ret_t = torch.from_numpy(returns).to(device)
    # Normalize advantages per rollout.
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    h_list = rollout["hidden_states"]

    total_loss = 0.0
    total_pi_loss = 0.0
    total_v_loss = 0.0
    total_ent = 0.0
    total_kl = 0.0
    n_updates = 0

    for _epoch in range(n_epochs):
        perm = torch.randperm(T, device=device)
        for start in range(0, T, minibatch_size):
            mb_idx = perm[start:start + minibatch_size]
            B = mb_idx.size(0)

            # Replay each sample with ITS stored hidden state to preserve
            # recurrent semantics. We loop B samples — minibatch_size of
            # ~64 keeps this manageable on GPU.
            new_logp_list = []
            new_value_list = []
            entropies = []
            for j in range(B):
                t_idx = int(mb_idx[j].item())
                h_j = _select_h(h_list[t_idx], 0)  # already batch-of-1
                img = obs_t[t_idx:t_idx + 1]
                z = jepa.encode(img)
                logits, value, _h_next = policy.step_with_value(
                    z,
                    prev_actions_t[t_idx:t_idx + 1],
                    missions_t[t_idx:t_idx + 1],
                    h_j,
                )
                mask = torch.full_like(logits, _ACTION_MASK_LOGIT)
                for a in _ALLOWED_ACTIONS:
                    mask[..., a] = 0.0
                masked_logits = logits + mask
                dist = Categorical(logits=masked_logits)
                new_logp = dist.log_prob(actions_t[t_idx:t_idx + 1])
                new_logp_list.append(new_logp.squeeze(0))
                new_value_list.append(value.squeeze(0))
                entropies.append(dist.entropy().squeeze(0))

            new_logp = torch.stack(new_logp_list)
            new_value = torch.stack(new_value_list)
            entropy = torch.stack(entropies).mean()

            old_logp = old_logp_t[mb_idx]
            mb_adv = adv_t[mb_idx]
            mb_ret = ret_t[mb_idx]

            # PPO clipped surrogate.
            ratio = (new_logp - old_logp).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            pi_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(new_value, mb_ret)
            loss = pi_loss + vf_coef * v_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad],
                grad_clip,
            )
            optimizer.step()

            total_loss += float(loss.item())
            total_pi_loss += float(pi_loss.item())
            total_v_loss += float(v_loss.item())
            total_ent += float(entropy.item())
            total_kl += float((old_logp - new_logp).mean().item())
            n_updates += 1

    return {
        "loss": total_loss / max(1, n_updates),
        "pi_loss": total_pi_loss / max(1, n_updates),
        "v_loss": total_v_loss / max(1, n_updates),
        "entropy": total_ent / max(1, n_updates),
        "approx_kl": total_kl / max(1, n_updates),
    }


# ===========================================================================
# Eval (deterministic argmax, separate env)
# ===========================================================================
@torch.no_grad()
def evaluate_policy(
    policy: UniversalPolicy,
    jepa: JepaWorldModel,
    env: UnityNavEnv,
    n_episodes: int,
    device: torch.device,
) -> dict:
    n_success = 0
    n_clean = 0
    n_distracted = 0
    n_ep_steps = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        h = policy.init_hidden(1, device)
        prev_action = torch.tensor([-1], device=device, dtype=torch.long)
        mission = mission_to_onehot(obs["mission"], device).unsqueeze(0)
        terminated = truncated = False
        steps = 0
        ever_approached_distractor = False
        while not (terminated or truncated):
            img = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(device)
            z = jepa.encode(img)
            logits, h_next = policy.step(z, prev_action, mission, h)
            mask = torch.full_like(logits, _ACTION_MASK_LOGIT)
            for a in _ALLOWED_ACTIONS:
                mask[..., a] = 0.0
            action = int((logits + mask).argmax(dim=-1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            ever_approached_distractor = (
                ever_approached_distractor or bool(info.get("approached_distractor", False))
            )
            prev_action = torch.tensor([action], device=device, dtype=torch.long)
            h = h_next
            steps += 1
        n_ep_steps.append(steps)
        if info.get("reached_target", False) or reward >= 0.9:
            n_success += 1
            if not ever_approached_distractor:
                n_clean += 1
        if ever_approached_distractor:
            n_distracted += 1
    return {
        "success_rate": n_success / n_episodes,
        "clean_rate": n_clean / n_episodes,
        "distractor_visit_rate": n_distracted / n_episodes,
        "mean_steps": float(np.mean(n_ep_steps)),
    }


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    p = argparse.ArgumentParser(description="PPO trainer for synthetic Unity nav env.")
    p.add_argument("--jepa", required=True)
    p.add_argument("--base-policy", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--trunk", default="transformer", choices=["transformer", "gru"])
    # Env / dynamics (match Unity by default).
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--reach-threshold", type=float, default=1.6)
    p.add_argument("--forward-step", type=float, default=0.07)
    p.add_argument("--obs-scale", type=float, default=2.0)
    p.add_argument("--randomize-target-color", action="store_true")
    # Dense distance-progress reward shaping (Ng et al. 1999 potential-based).
    # 0.0 keeps the sparse +1-on-reach baseline; >0 adds γ·Φ(s')−Φ(s) where
    # Φ = -dist_to_target. Provably preserves the optimal policy but
    # dramatically accelerates credit assignment.
    p.add_argument("--shaping-weight", type=float, default=0.5)
    p.add_argument("--shaping-gamma", type=float, default=0.99)
    # PPO hyperparameters.
    p.add_argument("--total-steps", type=int, default=200000)
    p.add_argument("--rollout-len", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every-iters", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ppo] device={device}")
    print(f"[ppo] loading JEPA from {args.jepa}")
    jepa, cfg = load_jepa(Path(args.jepa), device)

    print(f"[ppo] loading base policy from {args.base_policy}")
    base_ckpt = torch.load(args.base_policy, map_location=device, weights_only=False)
    policy = build_policy(base_ckpt, jepa, cfg, device, trunk=args.trunk)

    print("[ppo] freezing Concept/Operator memory banks")
    freeze_info = freeze_memory_banks(policy)
    print(f"[ppo]   frozen params:    {freeze_info['frozen_params']:,}")
    print(f"[ppo]   trainable params: {freeze_info['trainable_params']:,}")

    env_kwargs = dict(
        max_steps=args.max_steps,
        reach_threshold=args.reach_threshold,
        forward_step=args.forward_step,
        obs_scale=args.obs_scale,
        randomize_target_color=args.randomize_target_color,
        shaping_weight=args.shaping_weight,
        shaping_gamma=args.shaping_gamma,
    )
    train_env = UnityNavEnv(**env_kwargs, seed=0)
    # Eval env uses sparse reward (shaping_weight=0) so the metric we report
    # is the true sparse success rate, not the shaped surrogate.
    eval_kwargs = dict(env_kwargs)
    eval_kwargs["shaping_weight"] = 0.0
    eval_env = UnityNavEnv(**eval_kwargs, seed=999)

    print(f"[ppo] BASE eval (pre-PPO):")
    base_eval = evaluate_policy(policy, jepa, eval_env, args.eval_episodes, device)
    print(f"[ppo]   success={base_eval['success_rate']:.1%} "
          f"clean={base_eval['clean_rate']:.1%} "
          f"mean_steps={base_eval['mean_steps']:.0f}")

    trainable = [p_ for p_ in policy.parameters() if p_.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    n_iters = max(1, args.total_steps // args.rollout_len)
    print(f"[ppo] training: {n_iters} iters x {args.rollout_len} steps = {n_iters * args.rollout_len} env steps")

    t_start = time.time()
    best_success = base_eval["success_rate"]
    for it in range(n_iters):
        rollout = collect_rollout(train_env, policy, jepa, device, args.rollout_len)
        advantages, returns = compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"],
            rollout["last_value"], args.gamma, args.lam,
        )
        update_stats = ppo_update(
            policy, jepa, optimizer, rollout, advantages, returns, device,
            clip_eps=args.clip_eps, vf_coef=args.vf_coef, ent_coef=args.ent_coef,
            n_epochs=args.n_epochs, minibatch_size=args.minibatch_size,
            grad_clip=args.grad_clip,
        )

        n_eps_in_rollout = len(rollout["ep_success"])
        mean_ep_R = float(np.mean(rollout["ep_rewards"])) if rollout["ep_rewards"] else 0.0
        mean_ep_len = float(np.mean(rollout["ep_lengths"])) if rollout["ep_lengths"] else 0.0
        train_success = (
            sum(rollout["ep_success"]) / n_eps_in_rollout if n_eps_in_rollout else 0.0
        )

        elapsed = time.time() - t_start
        print(
            f"[ppo] iter {it+1}/{n_iters} "
            f"eps={n_eps_in_rollout} train_R={mean_ep_R:+.2f} "
            f"train_succ={train_success:.1%} ep_len={mean_ep_len:.0f} "
            f"pi_loss={update_stats['pi_loss']:+.3f} "
            f"v_loss={update_stats['v_loss']:.3f} "
            f"ent={update_stats['entropy']:.2f} "
            f"kl={update_stats['approx_kl']:+.3f} "
            f"t={elapsed:.0f}s"
        )

        if (it + 1) % args.eval_every_iters == 0 or it == n_iters - 1:
            eval_stats = evaluate_policy(
                policy, jepa, eval_env, args.eval_episodes, device,
            )
            print(
                f"[ppo] EVAL  iter {it+1}: "
                f"success={eval_stats['success_rate']:.1%} "
                f"clean={eval_stats['clean_rate']:.1%} "
                f"mean_steps={eval_stats['mean_steps']:.0f}"
            )
            if eval_stats["success_rate"] > best_success:
                best_success = eval_stats["success_rate"]
                # Save current-best checkpoint inline.
                torch.save({
                    **base_ckpt,
                    "policy_state_dict": policy.state_dict(),
                    "ppo_train": {
                        "iter": it + 1,
                        "eval_success": eval_stats["success_rate"],
                        "eval_clean": eval_stats["clean_rate"],
                    },
                }, out_path)
                print(f"[ppo]   saved best checkpoint to {out_path}")

    # Final eval + save.
    print("[ppo] FINAL eval:")
    final_eval = evaluate_policy(policy, jepa, eval_env, args.eval_episodes, device)
    print(f"[ppo]   success={final_eval['success_rate']:.1%} "
          f"clean={final_eval['clean_rate']:.1%} "
          f"mean_steps={final_eval['mean_steps']:.0f}")
    print(f"\n[ppo] BASE:  success={base_eval['success_rate']:.1%} clean={base_eval['clean_rate']:.1%}")
    print(f"[ppo] FINAL: success={final_eval['success_rate']:.1%} clean={final_eval['clean_rate']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
