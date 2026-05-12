"""Continual fine-tune: behavior-clone the substrate on Unity nav tasks
while keeping Concept/Operator memory FROZEN.

The thesis under test: PRISM's domain-general substrate can learn a new
task (Unity 2D nav with distractors) from a small batch of expert
trajectories WITHOUT forgetting its previous BabyAI training. The
mechanism is PRISM's own frozen Concept/Operator K/V banks — new
knowledge lands in the trunk + action head; old concept slots stay
locked.

Pipeline:
  1. Load existing v6 checkpoint (e.g., v6_phaseB_GoToLocal_500k).
  2. Freeze JEPA + ConceptMemory K/V + OperatorMemory K/V.
  3. Collect N trajectories with a greedy scripted expert on UnityNavEnv.
  4. Fine-tune trunk + action head with cross-entropy on (obs, action).
  5. Save new checkpoint.
  6. Print before/after eval accuracy on a fresh batch of episodes.

Usage (Vast.ai):
    python unity_demo/continual_finetune.py \\
        --jepa  runs/jepa_dev_v1_factored/jepa_final.pt \\
        --base-policy runs/v6_phaseB_GoToLocal_500k/policy_iter244.pt \\
        --out-path runs/v6_unity_continual_v1/policy.pt \\
        --n-train-episodes 2000 \\
        --n-eval-episodes 200 \\
        --epochs 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.adapters.babyai_adapter import BabyAIAdapter
from prism.adapters.unity_2d import _FORWARD_VEC, _RIGHT_VEC
from prism.cognition.policy import UniversalPolicy
from prism.language.mission_parser import parse_mission
from prism.models.jepa import JepaWorldModel, upgrade_config
from prism.perception.predicates import type_color_index
from prism.perception.slots import NUM_COLORS, NUM_TYPES

from unity_demo.unity_nav_env import UnityNavEnv


# Substrate action ids (matches BabyAI conventions).
_ACT_LEFT, _ACT_RIGHT, _ACT_FORWARD = 0, 1, 2


# ===========================================================================
# Checkpoint loading
# ===========================================================================
def load_jepa(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
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
    """Freeze Concept/Operator memory key/value banks (Hopfield slots).

    PRISM's anti-forgetting primitive: the keys/values in
    RetrievalBlock.{concept_bank, operator_bank} encode what was learned
    during BabyAI training. Locking them prevents the new task from
    overwriting old knowledge — new learning lands in trunk + heads +
    conditioning MLPs instead.
    """
    inner = policy._inner
    frozen = 0
    if hasattr(inner, "retrieval") and not isinstance(inner.retrieval, nn.Identity):
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
                print(f"[finetune]   FROZEN retrieval.{bank_name}.{slot_name} {tuple(p.shape)}")
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    return {"frozen_params": frozen, "trainable_params": trainable}


# ===========================================================================
# Greedy expert: knows ground-truth target position and emits the
# substrate action that reduces egocentric distance.
# ===========================================================================
def greedy_action(env: UnityNavEnv) -> int:
    """Emit one of {turn_left, turn_right, forward} that points the agent
    at the target. Uses env internals (cheat) — that's fine because this
    is a synthetic expert for BC.

    Strategy: commit to one axis at a time. Drive forward whenever the
    forward distance is meaningfully positive; only turn when the target
    is behind us, or when we've drawn level on the forward axis and the
    lateral offset still needs closing. This avoids the diagonal-target
    spin trap (alternating turn_right/turn_left every tick).
    """
    agent = env._agent_pos
    target = env._target_pos
    heading = env._adapter.heading

    delta = target - agent
    fwd = _FORWARD_VEC[heading]
    right = _RIGHT_VEC[heading]
    forward_dist = float(delta @ fwd)
    right_dist = float(delta @ right)

    # 1. Already nearly inside the touch zone — just push forward.
    if float(np.linalg.norm(delta)) < 0.8:
        return _ACT_FORWARD

    # 2. Target is significantly behind: about-face (any single turn).
    if forward_dist < -0.5:
        return _ACT_RIGHT

    # 3. Drawn level on the forward axis AND there's lateral offset:
    #    commit to turning toward the offset side. This is the ONLY
    #    place we turn; the "abs(forward_dist) < some_threshold" guard
    #    prevents the diagonal-spin oscillation.
    if forward_dist < 0.6 and abs(right_dist) > 0.4:
        return _ACT_RIGHT if right_dist > 0 else _ACT_LEFT

    # 4. Otherwise drive forward.
    return _ACT_FORWARD


# ===========================================================================
# Mission encoding (one-hot 24)
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
# Trajectory collection
# ===========================================================================
def collect_episode(env: UnityNavEnv, expert_fn) -> list[dict]:
    """Roll out one episode with the expert. Returns list of step records."""
    obs, info = env.reset()
    records = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = expert_fn(env)
        records.append({
            "image": obs["image"].copy(),
            "mission": obs["mission"],
            "action": int(action),
            "heading": int(obs["direction"]),
        })
        obs, reward, terminated, truncated, info = env.step(action)
    return records


def collect_dataset(env: UnityNavEnv, n_episodes: int, expert_fn) -> list[dict]:
    records = []
    n_solved = 0
    for ep in range(n_episodes):
        ep_records = collect_episode(env, expert_fn)
        # Only keep episodes where the expert actually solved it
        # (final action led to terminate=True). For robustness we accept
        # all but track success.
        records.extend(ep_records)
        # Heuristic: solved if last step's record action led to terminal
        # (we check by re-running last few steps' positions, but here we
        # just count short episodes < max_steps as solved).
        if len(ep_records) < env._max_steps:
            n_solved += 1
        if (ep + 1) % 200 == 0:
            print(f"[collect] {ep+1}/{n_episodes} eps, "
                  f"{n_solved} solved, {len(records)} records")
    print(f"[collect] DONE: {n_solved}/{n_episodes} expert solved, "
          f"{len(records)} total records")
    return records


# ===========================================================================
# Forward pass through substrate (for BC training)
# ===========================================================================
def forward_logits(
    policy: UniversalPolicy,
    jepa: JepaWorldModel,
    images: torch.Tensor,
    missions: torch.Tensor,
    prev_actions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Per-sample forward (no recurrent state — fine for BC since the
    expert acts greedily per-state, not relying on memory of past steps).

    Args:
        images: (B, 3, 7, 7)
        missions: (B, 24)
        prev_actions: (B,) int64
    Returns:
        logits: (B, n_actions)
    """
    B = images.size(0)
    h_init = policy.init_hidden(B, device)
    with torch.no_grad():
        z = jepa.encode(images)
    logits, _h = policy.step(z, prev_actions, missions, h_init)
    # Mask to allowed actions {0, 1, 2}.
    mask = torch.full_like(logits, float("-inf"))
    mask[..., :3] = 0.0
    return logits + mask


# ===========================================================================
# Evaluation: run policy in env, measure success + discrimination
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
    n_clean = 0  # success without approaching any distractor
    n_ep_steps = []
    n_distractor_visits = 0
    for _ in range(n_episodes):
        obs, info = env.reset()
        h = policy.init_hidden(1, device)
        prev_action = torch.tensor([-1], device=device, dtype=torch.long)
        mission = mission_to_onehot(obs["mission"], device).unsqueeze(0)
        ever_approached_distractor = False
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            img = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(device)
            z = jepa.encode(img)
            logits, h_next = policy.step(z, prev_action, mission, h)
            mask = torch.full_like(logits, float("-inf"))
            mask[..., :3] = 0.0
            action = int((logits + mask).argmax(dim=-1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            ever_approached_distractor = (
                ever_approached_distractor or bool(info.get("approached_distractor", False))
            )
            if reward < -0.1:  # distractor proximity penalty fired
                pass  # already tracked above
            prev_action = torch.tensor([action], device=device, dtype=torch.long)
            h = h_next
            steps += 1
        n_ep_steps.append(steps)
        if info.get("reached_target", False) or reward >= 0.9:
            n_success += 1
            if not ever_approached_distractor:
                n_clean += 1
        if ever_approached_distractor:
            n_distractor_visits += 1
    return {
        "n_episodes": n_episodes,
        "success_rate": n_success / n_episodes,
        "discrimination_rate_of_successes": n_clean / max(1, n_success),
        "clean_rate_overall": n_clean / n_episodes,
        "distractor_visit_rate": n_distractor_visits / n_episodes,
        "mean_steps": float(np.mean(n_ep_steps)),
    }


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    p = argparse.ArgumentParser(description="Continual BC fine-tune for Unity nav.")
    p.add_argument("--jepa", required=True, help="JEPA checkpoint")
    p.add_argument("--base-policy", required=True, help="Base policy checkpoint to fine-tune")
    p.add_argument("--out-path", required=True, help="Where to save the fine-tuned policy .pt")
    p.add_argument("--trunk", default="transformer", choices=["transformer", "gru"])
    p.add_argument("--n-train-episodes", type=int, default=2000)
    p.add_argument("--n-eval-episodes", type=int, default=200)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--obs-scale", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--reach-threshold", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--randomize-target-color", action="store_true",
        help="Vary target color per episode (red, green, blue, ...). "
             "Forces the policy to actually use the mission one-hot.",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[finetune] device={device}")
    print(f"[finetune] loading JEPA from {args.jepa}")
    jepa, cfg = load_jepa(Path(args.jepa), device)

    print(f"[finetune] loading base policy from {args.base_policy}")
    base_ckpt = torch.load(args.base_policy, map_location=device, weights_only=False)
    policy = build_policy(base_ckpt, jepa, cfg, device, trunk=args.trunk)

    print("[finetune] freezing Concept/Operator memory banks")
    freeze_info = freeze_memory_banks(policy)
    print(f"[finetune]   total frozen params: {freeze_info['frozen_params']:,}")
    print(f"[finetune]   total trainable params: {freeze_info['trainable_params']:,}")

    # Set up envs.
    env_kwargs = dict(
        max_steps=args.max_steps,
        reach_threshold=args.reach_threshold,
        obs_scale=args.obs_scale,
        randomize_target_color=args.randomize_target_color,
    )
    train_env = UnityNavEnv(**env_kwargs, seed=0)
    eval_env = UnityNavEnv(**env_kwargs, seed=42)

    # Pre-finetune eval.
    print("[finetune] evaluating BASE policy (pre-finetune)...")
    base_eval = evaluate_policy(policy, jepa, eval_env, args.n_eval_episodes, device)
    print(f"[finetune] BASE eval: {base_eval}")

    # Collect expert dataset.
    print(f"[finetune] collecting {args.n_train_episodes} expert episodes...")
    t0 = time.time()
    records = collect_dataset(train_env, args.n_train_episodes, greedy_action)
    print(f"[finetune] collection took {time.time()-t0:.1f}s")

    # Build tensors.
    images = torch.from_numpy(np.stack([r["image"] for r in records])).float().to(device)
    actions = torch.tensor([r["action"] for r in records], dtype=torch.long, device=device)
    missions = torch.stack([mission_to_onehot(r["mission"], device) for r in records])
    # prev_actions: use -1 sentinel for the first step of each ep, otherwise the previous action.
    # For BC simplicity we use -1 for everything (matches inference cold-start).
    prev_actions = torch.full((len(records),), -1, dtype=torch.long, device=device)
    print(f"[finetune] dataset tensors: images={tuple(images.shape)} actions={actions.shape}")

    # BC training loop.
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable_params, lr=args.lr)
    N = len(records)
    for epoch in range(args.epochs):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        n_batches = 0
        n_correct = 0
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            b_imgs = images[idx]
            b_acts = actions[idx]
            b_miss = missions[idx]
            b_prev = prev_actions[idx]
            logits = forward_logits(policy, jepa, b_imgs, b_miss, b_prev, device)
            loss = F.cross_entropy(logits, b_acts)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
            n_correct += int((logits.argmax(-1) == b_acts).sum().item())
        print(f"[finetune] epoch {epoch+1}/{args.epochs} "
              f"loss={total_loss/n_batches:.4f} "
              f"acc={n_correct/N:.3f}")

    # Post-finetune eval.
    print("[finetune] evaluating FINETUNED policy...")
    new_eval = evaluate_policy(policy, jepa, eval_env, args.n_eval_episodes, device)
    print(f"[finetune] FINETUNED eval: {new_eval}")

    print(f"\n[finetune] BEFORE: success={base_eval['success_rate']:.1%} "
          f"clean={base_eval['clean_rate_overall']:.1%} "
          f"mean_steps={base_eval['mean_steps']:.1f}")
    print(f"[finetune] AFTER:  success={new_eval['success_rate']:.1%} "
          f"clean={new_eval['clean_rate_overall']:.1%} "
          f"mean_steps={new_eval['mean_steps']:.1f}")

    # Save checkpoint (matching ppo_train.py's format so the inference
    # server's loader works unchanged).
    new_ckpt = {
        "policy_state_dict": policy.state_dict(),
        "policy_type": "universal",
        "latent_in_dim": base_ckpt["latent_in_dim"],
        "n_actions": base_ckpt["n_actions"],
        "mission_dim": base_ckpt["mission_dim"],
        "hidden_dim": base_ckpt["hidden_dim"],
        "latent_proj_dim": base_ckpt["latent_proj_dim"],
        "mem_feat_dim": base_ckpt.get("mem_feat_dim", 0),
        "concept_n_slots": base_ckpt.get("concept_n_slots", 1024),
        "concept_slot_dim": base_ckpt.get("concept_slot_dim", 64),
        "concept_scaling": base_ckpt.get("concept_scaling", 1.0),
        "operator_n_slots": base_ckpt.get("operator_n_slots", 64),
        "operator_slot_dim": base_ckpt.get("operator_slot_dim", 64),
        "operator_scaling": base_ckpt.get("operator_scaling", 4.0),
        "use_operator_memory": base_ckpt.get("use_operator_memory", True),
        "jepa_checkpoint": args.jepa,
        "base_policy_checkpoint": args.base_policy,
        "continual_finetune": {
            "n_train_episodes": args.n_train_episodes,
            "epochs": args.epochs,
            "base_eval": base_eval,
            "new_eval": new_eval,
            "frozen_params": freeze_info["frozen_params"],
            "trainable_params": freeze_info["trainable_params"],
        },
    }
    torch.save(new_ckpt, out_path)
    print(f"[finetune] saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
