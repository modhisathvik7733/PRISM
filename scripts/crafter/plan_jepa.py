"""Reward-free latent planning with the JEPA world model on Crafter.

Three experiments:
  goal      — encode a goal observation, beam-search toward it in psi-space
               via receding-horizon replanning (no reward, no PPO gradient)
  curiosity — no goal; score each beam by novelty from visited psi-states,
               pure latent-space exploration
  chain     — two-phase: (1) curiosity exploration builds z_memory,
               (2) find_subgoal_chain picks waypoints along the start→goal
               axis, beam_search chained through each waypoint in order

Usage:
    python -m scripts.crafter.plan_jepa \\
        --jepa-checkpoint runs/crafter_jepa/jepa_final.pt \\
        --ppo-checkpoint  runs/crafter_ppo_jepa/policy_final.pt \\
        --experiment chain \\
        --n-subgoals 3 --explore-steps 300 --plan-steps 300 \\
        --horizon 15 --beam-k 10 --exec-steps 5 \\
        --switch-threshold 0.20 --trials 3 --seed 42 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from prism.crafter.cnn_encoder import CrafterCNN
from prism.crafter.env_worker import CrafterEnvWorker
from prism.crafter.jepa_crafter import CrafterJepaConfig, _LatentDynamics
from prism.crafter.latent_planner import LatentPlanner
from prism.crafter.policy_jepa import CrafterPolicyJepa
from prism.utils.seed import set_global_seed


# Action index that fires each achievement (action 5 = "do").
ACHIEVEMENT_ACTIONS: dict[str, int] = {
    "collect_wood":        5,
    "collect_stone":       5,
    "collect_coal":        5,
    "collect_iron":        5,
    "collect_diamond":     5,
    "collect_drink":       5,
    "eat_cow":             5,
    "eat_plant":           5,
    "defeat_zombie":       5,
    "defeat_skeleton":     5,
    "place_table":         8,
    "place_stone":         7,
    "place_furnace":       9,
    "place_plant":        10,
    "make_wood_pickaxe":  11,
    "make_stone_pickaxe": 12,
    "make_iron_pickaxe":  13,
    "make_wood_sword":    14,
    "make_stone_sword":   15,
    "make_iron_sword":    16,
}


# ── checkpoint helpers ────────────────────────────────────────────────────────

def load_memory_from_npz(
    npz_path: str,
    encoder: torch.nn.Module,
    n_samples: int,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    """Encode N randomly-sampled observations from crafter_rollouts.npz → (N, E).

    The rollout file contains obs_t: (Total, 3, 64, 64) uint8.
    We sample n_samples indices and run them through the frozen encoder in one
    batched forward pass (no gradient). Returns z_memory: (N, E) float32.
    """
    d       = np.load(npz_path)
    obs_all = d["obs_t"]                                  # (Total, 3, 64, 64) uint8
    Total   = len(obs_all)
    rng     = np.random.default_rng(seed)
    idx     = rng.choice(Total, size=min(n_samples, Total), replace=False)
    obs_sub = obs_all[idx].astype(np.float32) / 255.0    # (N, 3, 64, 64) float32

    batch_size = 512
    zs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(obs_sub), batch_size):
            batch = torch.from_numpy(obs_sub[start:start + batch_size]).to(device)
            zs.append(encoder(batch))                     # (B, E)
    z_memory = torch.cat(zs, dim=0)                      # (N, E)
    print(f"[memory] encoded {len(z_memory):,} observations from {npz_path}")
    return z_memory


def load_jepa(path: str, device: torch.device):
    """Load frozen encoder + dynamics from jepa_final.pt."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg: CrafterJepaConfig = ckpt["cfg"]

    state_dim = getattr(cfg, "state_dim", 0)
    encoder = CrafterCNN(embed_dim=cfg.embed_dim, state_dim=state_dim).to(device)
    encoder.load_state_dict(ckpt["online_encoder_state"])
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    dynamics = _LatentDynamics(cfg.embed_dim, cfg.n_actions, cfg.dynamics_hidden).to(device)
    dynamics.load_state_dict(ckpt["dynamics_state"])
    for p in dynamics.parameters():
        p.requires_grad_(False)
    dynamics.eval()

    print(f"[jepa] loaded from {path}  embed_dim={cfg.embed_dim}")
    return encoder, dynamics, cfg


def collect_goal_obs(
    jepa_checkpoint: str,
    ppo_checkpoint: Optional[str],
    trigger_achievement: str = "place_table",
    max_episodes: int = 100,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Roll out a policy until trigger_achievement fires; return (pre_obs, pre_state).

    Returns the PRE-achievement observation (player facing the target object,
    before the achievement action is executed) so the planner can navigate
    toward this common precondition state rather than the rare post-obs.

    If ppo_checkpoint exists, uses the trained PPO policy.
    Otherwise falls back to a random policy (more episodes needed).
    """
    use_ppo = ppo_checkpoint is not None and Path(ppo_checkpoint).exists()
    if use_ppo:
        policy = CrafterPolicyJepa(
            jepa_checkpoint=jepa_checkpoint,
            n_actions=17,
            device=device,
        ).to(device)
        ckpt = torch.load(ppo_checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.eval()
        h           = policy.init_hidden(1, device)
        prev_action = torch.full((1,), -1, dtype=torch.long, device=device)
        print(f"[goal] collecting with PPO policy from {ppo_checkpoint}")
    else:
        policy = None
        print(f"[goal] PPO checkpoint not found — using random policy "
              f"(max_episodes raised to {max_episodes})")

    worker   = CrafterEnvWorker(seed, worker_id=0, reward_mode="reward")
    episodes = 0

    while episodes < max_episodes:
        # Capture pre-step obs and game state BEFORE the action fires.
        obs_np     = worker.obs.copy()
        pre_state  = worker.env.get_game_state()
        ach_before = set(worker.achievements)

        if use_ppo:
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _, h_next = policy.step_with_value(obs_t, prev_action, h)
            action_int = int(torch.distributions.Categorical(logits=logits).sample().item())
        else:
            action_int = int(np.random.randint(17))
            h_next     = None

        _, _, done, info = worker.step(action_int)

        if not done:
            new_ach = worker.achievements - ach_before
            if trigger_achievement in new_ach:
                print(f"[goal] '{trigger_achievement}' fired (ep {episodes}, non-terminal) "
                      f"— returning PRE-achievement obs")
                return obs_np, pre_state
        else:
            if trigger_achievement in info.get("achievements", set()):
                print(f"[goal] '{trigger_achievement}' fired (ep {episodes}, terminal) "
                      f"— returning PRE-achievement obs")
                return obs_np, pre_state
            episodes += 1
            if use_ppo:
                h           = policy.init_hidden(1, device)
                prev_action = torch.full((1,), -1, dtype=torch.long, device=device)
            if episodes % 20 == 0:
                print(f"[goal] {episodes}/{max_episodes} episodes — trigger not yet fired")
            continue

        if use_ppo:
            h           = h_next
            prev_action = torch.tensor([action_int], dtype=torch.long, device=device)

    print(f"[goal] WARNING: '{trigger_achievement}' not fired in {max_episodes} episodes")
    return None


# ── experiments ───────────────────────────────────────────────────────────────

def run_goal_experiment(
    planner: LatentPlanner,
    goal_obs: np.ndarray,
    n_trials: int = 5,
    horizon: int = 15,
    exec_steps: int = 5,
    max_total_steps: int = 200,
    seed: int = 42,
) -> None:
    """Receding-horizon goal-directed planning. No reward signal."""
    z_goal   = planner.encode_obs(goal_obs)                    # (E,)
    z_goal_n = F.normalize(z_goal, dim=-1)

    print("\n" + "=" * 60)
    print("EXPERIMENT A: Goal-directed latent planning")
    print(f"  horizon={horizon}  exec_steps={exec_steps}  max_steps={max_total_steps}")
    print("=" * 60)

    all_final_dists: list[float] = []
    all_achievements: list[set[str]] = []

    for trial in range(n_trials):
        worker = CrafterEnvWorker(seed + trial * 1_000_003, worker_id=0, reward_mode="reward")
        step = 0
        dist_trace: list[float] = []
        trial_achievements: set[str] = set()

        while step < max_total_steps:
            z_cur   = planner.encode_obs(worker.obs)            # (E,)
            z_cur_n = F.normalize(z_cur, dim=-1)
            cos_dist = 1.0 - float((z_cur_n * z_goal_n).sum())
            dist_trace.append(cos_dist)

            actions = planner.beam_search(z_cur, z_goal, horizon=horizon)

            for a in actions[:exec_steps]:
                if step >= max_total_steps:
                    break
                _, _, done, info = worker.step(a)
                step += 1
                if info:
                    trial_achievements |= info.get("achievements", set())
                if done:
                    break  # replan from new episode's first obs

        z_final_n  = F.normalize(planner.encode_obs(worker.obs), dim=-1)
        final_dist = 1.0 - float((z_final_n * z_goal_n).sum())
        all_final_dists.append(final_dist)
        all_achievements.append(trial_achievements)

        trace_str = "  ".join(f"{d:.3f}" for d in dist_trace[:8])
        print(f"[trial {trial+1}/{n_trials}]  replans={len(dist_trace)}"
              f"  dist=[{trace_str}{'...' if len(dist_trace)>8 else ''}]"
              f"  final={final_dist:.3f}"
              f"  ach={sorted(trial_achievements)}")

    print(f"\n[goal] mean final cos_dist = {np.mean(all_final_dists):.3f}  "
          f"(lower = closer to goal in psi-space; random baseline ~1.0)")
    all_ach = set().union(*all_achievements)
    print(f"[goal] achievements across all trials: {sorted(all_ach)}")


def run_curiosity_experiment(
    planner: LatentPlanner,
    n_trials: int = 3,
    horizon: int = 15,
    exec_steps: int = 5,
    max_total_steps: int = 300,
    max_memory: int = 500,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Curiosity-driven exploration. No goal, no reward."""
    E = 256  # embed_dim

    print("\n" + "=" * 60)
    print("EXPERIMENT B: Curiosity-driven latent exploration")
    print(f"  horizon={horizon}  exec_steps={exec_steps}  max_steps={max_total_steps}"
          f"  max_memory={max_memory}")
    print("=" * 60)

    all_achievements: list[set[str]] = []

    for trial in range(n_trials):
        worker   = CrafterEnvWorker(seed + trial * 1_000_003, worker_id=0, reward_mode="reward")
        z_memory = torch.zeros(0, E, device=device)
        step     = 0
        trial_achievements: set[str] = set()

        while step < max_total_steps:
            z_cur   = planner.encode_obs(worker.obs)
            actions = planner.novelty_plan(z_cur, z_memory, horizon=horizon)

            for a in actions[:exec_steps]:
                if step >= max_total_steps:
                    break
                obs_visited = worker.obs.copy()
                _, _, done, info = worker.step(a)
                step += 1
                if info:
                    trial_achievements |= info.get("achievements", set())

                if len(z_memory) < max_memory:
                    z_vis = planner.encode_obs(obs_visited)
                    z_memory = torch.cat([z_memory, z_vis.unsqueeze(0)], dim=0)

                if done:
                    break

        # Memory diversity: mean pairwise cosine distance (sampled 100 pairs)
        if len(z_memory) >= 2:
            idx    = torch.randperm(len(z_memory), device=device)[:min(100, len(z_memory))]
            sample = F.normalize(z_memory[idx], dim=-1)
            sim    = sample @ sample.T
            n      = len(sample)
            mask   = ~torch.eye(n, dtype=torch.bool, device=device)
            mean_cos_dist = float(1.0 - sim[mask].mean())
        else:
            mean_cos_dist = 0.0

        all_achievements.append(trial_achievements)
        print(f"[trial {trial+1}/{n_trials}]  steps={step}"
              f"  memory={len(z_memory)}"
              f"  diversity={mean_cos_dist:.3f}"
              f"  ach={sorted(trial_achievements)}")

    all_ach = set().union(*all_achievements)
    print(f"\n[curiosity] achievements across all trials: {sorted(all_ach)}")


def run_chain_experiment(
    planner: LatentPlanner,
    goal_obs: np.ndarray,
    n_trials: int = 3,
    n_subgoals: int = 3,
    explore_steps: int = 300,
    plan_steps: int = 300,
    max_memory: int = 500,
    horizon: int = 15,
    beam_k: int = 10,
    exec_steps: int = 5,
    switch_threshold: float = 0.20,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    z_memory_seed: Optional[torch.Tensor] = None,
) -> None:
    """Two-phase subgoal-chaining experiment.

    Phase 1 (explore_steps): curiosity beam search builds z_memory.
    Phase 2 (plan_steps):    find_subgoal_chain selects waypoints from
        z_memory along the start→goal axis; beam_search chains through
        each waypoint, switching when cos_dist < switch_threshold.

    Key diagnostics:
      chain_dists  — cos_dist of each waypoint to z_goal (quality of chain)
      dist_to_sg   — distance to current subgoal target
      dist_to_goal — distance to final z_goal (should trend down vs. direct)
      sg_idx       — how far along the chain we got
    """
    z_goal   = planner.encode_obs(goal_obs)          # (E,)
    z_goal_n = F.normalize(z_goal, dim=-1)
    E        = int(z_goal.shape[0])

    print("\n" + "=" * 60)
    print("EXPERIMENT C: Subgoal chaining (explore → chain plan)")
    print(f"  n_subgoals={n_subgoals}  explore={explore_steps}  plan={plan_steps}")
    print(f"  horizon={horizon}  beam_k={beam_k}  exec_steps={exec_steps}"
          f"  switch_threshold={switch_threshold}")
    print("=" * 60)

    all_final_dists:   list[float]    = []
    all_achievements:  list[set[str]] = []
    all_sg_reached:    list[int]      = []

    for trial in range(n_trials):
        base_seed = seed + trial * 1_000_003
        worker   = CrafterEnvWorker(base_seed, worker_id=0, reward_mode="reward")
        # Pre-seed memory from rollout data if provided; curiosity exploration adds more.
        z_memory = z_memory_seed.clone() if z_memory_seed is not None else torch.zeros(0, E, device=device)
        trial_achievements: set[str] = set()

        # ── Phase 1: curiosity exploration ───────────────────────────────────
        step = 0
        while step < explore_steps:
            z_cur   = planner.encode_obs(worker.obs)
            actions = planner.novelty_plan(z_cur, z_memory, horizon=horizon, beam_k=beam_k)
            for a in actions[:exec_steps]:
                if step >= explore_steps:
                    break
                obs_visited = worker.obs.copy()
                _, _, done, info = worker.step(a)
                step += 1
                if info:
                    trial_achievements |= info.get("achievements", set())
                if len(z_memory) < max_memory:
                    z_memory = torch.cat(
                        [z_memory, planner.encode_obs(obs_visited).unsqueeze(0)], dim=0
                    )
                if done:
                    break

        # ── Build subgoal chain from z_memory ────────────────────────────────
        z_cur   = planner.encode_obs(worker.obs)
        chain   = planner.find_subgoal_chain(z_cur, z_goal, z_memory, n_subgoals)
        n_chain = len(chain)  # n_subgoals + 1 (last entry is z_goal)

        chain_dists = []
        for z_sg in chain:
            d = 1.0 - float(F.normalize(z_sg, dim=-1) @ z_goal_n)
            chain_dists.append(d)
        print(f"\n[trial {trial+1}/{n_trials}] memory={len(z_memory)}"
              f"  chain_len={n_chain}  chain_dists={[f'{d:.3f}' for d in chain_dists]}")

        # ── Phase 2: chain-guided planning ───────────────────────────────────
        sg_idx      = 0
        step        = 0
        sg_patience = 0          # replans on current subgoal without progress
        dist_trace: list[tuple[int, float, float]] = []  # (sg_idx, d_sg, d_goal)
        max_sg_idx  = 0
        patience_limit = 10      # skip unreachable subgoal after this many replans

        def _rebuild_chain() -> list:
            z_c = planner.encode_obs(worker.obs)
            return planner.find_subgoal_chain(z_c, z_goal, z_memory, n_subgoals)

        while step < plan_steps:
            z_cur    = planner.encode_obs(worker.obs)
            z_cur_n  = F.normalize(z_cur, dim=-1)
            z_target = chain[sg_idx]
            z_tgt_n  = F.normalize(z_target, dim=-1)

            dist_to_sg   = float(1.0 - z_cur_n @ z_tgt_n)
            dist_to_goal = float(1.0 - z_cur_n @ z_goal_n)
            dist_trace.append((sg_idx, dist_to_sg, dist_to_goal))

            # Advance subgoal when close enough
            if dist_to_sg < switch_threshold and sg_idx < n_chain - 1:
                sg_idx     += 1
                sg_patience = 0
                max_sg_idx  = max(max_sg_idx, sg_idx)
                z_target    = chain[sg_idx]
            else:
                sg_patience += 1
                # Skip a stuck subgoal and move to the next one
                if sg_patience >= patience_limit and sg_idx < n_chain - 1:
                    sg_idx     += 1
                    sg_patience = 0
                    z_target    = chain[sg_idx]

            actions = planner.beam_search(z_cur, z_target, horizon=horizon, beam_k=beam_k)

            episode_done = False
            for a in actions[:exec_steps]:
                if step >= plan_steps:
                    break
                _, _, done, info = worker.step(a)
                step += 1
                if info:
                    trial_achievements |= info.get("achievements", set())
                if done:
                    episode_done = True
                    break

            # After episode reset, rebuild the chain from the new starting state
            if episode_done:
                chain       = _rebuild_chain()
                n_chain     = len(chain)
                sg_idx      = 0
                sg_patience = 0

        z_final_n  = F.normalize(planner.encode_obs(worker.obs), dim=-1)
        final_dist = float(1.0 - z_final_n @ z_goal_n)
        all_final_dists.append(final_dist)
        all_achievements.append(trial_achievements)
        all_sg_reached.append(max_sg_idx)

        # Print first 8 plan steps of the trace
        trace_str = "  ".join(
            f"[sg{r}|sg={ds:.2f}|g={dg:.2f}]"
            for r, ds, dg in dist_trace[:8]
        )
        print(f"         trace=[{trace_str}{'...' if len(dist_trace) > 8 else ''}]")
        print(f"         final_dist={final_dist:.3f}  max_sg_reached={max_sg_idx}/{n_chain-1}"
              f"  ach={sorted(trial_achievements)}")

    print(f"\n[chain] mean final_dist={np.mean(all_final_dists):.3f}  "
          f"mean_sg_reached={np.mean(all_sg_reached):.1f}/{len(chain)-1}")
    all_ach = set().union(*all_achievements)
    print(f"[chain] achievements across all trials: {sorted(all_ach)}")


def run_precondition_experiment(
    planner: LatentPlanner,
    goal_obs: np.ndarray,           # pre-achievement obs (player facing tree, etc.)
    trigger_achievement: str,
    n_trials: int = 5,
    n_subgoals: int = 3,
    explore_steps: int = 100,
    plan_steps: int = 400,
    max_memory: int = 2000,
    horizon: int = 15,
    beam_k: int = 10,
    exec_steps: int = 5,
    switch_threshold: float = 0.20,
    execute_threshold: float = 0.15,  # cos_dist below which to try the achievement action
    execute_burst: int = 20,          # how many times to try the action per close encounter
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    z_memory_seed: Optional[torch.Tensor] = None,
) -> None:
    """Two-phase precondition planning + execution burst experiment.

    Phase 1: curiosity exploration builds z_memory.
    Phase 2: subgoal chain navigates toward the PRE-achievement obs (player
             facing the target). When cos_dist < execute_threshold, fire the
             achievement action in a burst and check if the achievement fires.
    """
    z_goal   = planner.encode_obs(goal_obs)      # (E,)
    z_goal_n = F.normalize(z_goal, dim=-1)
    E        = int(z_goal.shape[0])
    ach_action = ACHIEVEMENT_ACTIONS.get(trigger_achievement, 5)

    print("\n" + "=" * 60)
    print("EXPERIMENT D: Precondition planning + execution burst")
    print(f"  trigger={trigger_achievement}  ach_action={ach_action}")
    print(f"  n_subgoals={n_subgoals}  explore={explore_steps}  plan={plan_steps}")
    print(f"  execute_threshold={execute_threshold}  execute_burst={execute_burst}")
    print("=" * 60)

    all_final_dists:  list[float]    = []
    all_achievements: list[set[str]] = []
    all_fired:        list[bool]     = []

    for trial in range(n_trials):
        base_seed = seed + trial * 1_000_003
        worker    = CrafterEnvWorker(base_seed, worker_id=0, reward_mode="reward")
        z_memory  = z_memory_seed.clone() if z_memory_seed is not None else torch.zeros(0, E, device=device)
        trial_achievements: set[str] = set()
        achievement_fired = False

        # ── Phase 1: curiosity exploration ───────────────────────────────────
        step = 0
        while step < explore_steps:
            z_cur   = planner.encode_obs(worker.obs)
            actions = planner.novelty_plan(z_cur, z_memory, horizon=horizon, beam_k=beam_k)
            for a in actions[:exec_steps]:
                if step >= explore_steps:
                    break
                obs_visited = worker.obs.copy()
                _, _, done, info = worker.step(a)
                step += 1
                if info:
                    trial_achievements |= info.get("achievements", set())
                    if trigger_achievement in trial_achievements:
                        achievement_fired = True
                if len(z_memory) < max_memory:
                    z_memory = torch.cat(
                        [z_memory, planner.encode_obs(obs_visited).unsqueeze(0)], dim=0
                    )
                if done:
                    break

        # ── Build subgoal chain ───────────────────────────────────────────────
        z_cur  = planner.encode_obs(worker.obs)
        chain  = planner.find_subgoal_chain(z_cur, z_goal, z_memory, n_subgoals)
        n_chain = len(chain)

        chain_dists = [
            1.0 - float(F.normalize(z_sg, dim=-1) @ z_goal_n) for z_sg in chain
        ]
        print(f"\n[trial {trial+1}/{n_trials}] memory={len(z_memory)}"
              f"  chain_len={n_chain}  chain_dists={[f'{d:.3f}' for d in chain_dists]}")

        # ── Phase 2: chain-guided planning with execution bursts ──────────────
        sg_idx      = 0
        step        = 0
        sg_patience = 0
        patience_limit = 10
        dist_trace: list[float] = []

        def _rebuild_chain() -> list:
            z_c = planner.encode_obs(worker.obs)
            return planner.find_subgoal_chain(z_c, z_goal, z_memory, n_subgoals)

        while step < plan_steps and not achievement_fired:
            z_cur    = planner.encode_obs(worker.obs)
            z_cur_n  = F.normalize(z_cur, dim=-1)
            dist_to_goal = float(1.0 - z_cur_n @ z_goal_n)
            dist_trace.append(dist_to_goal)

            # Check if we're close enough to attempt the achievement action.
            if dist_to_goal < execute_threshold:
                for _ in range(execute_burst):
                    if step >= plan_steps:
                        break
                    _, _, done, info = worker.step(ach_action)
                    step += 1
                    if info:
                        trial_achievements |= info.get("achievements", set())
                        if trigger_achievement in trial_achievements:
                            achievement_fired = True
                    if done or achievement_fired:
                        break
                if achievement_fired:
                    break
                if done:
                    chain       = _rebuild_chain()
                    n_chain     = len(chain)
                    sg_idx      = 0
                    sg_patience = 0
                continue

            z_target = chain[sg_idx]
            z_tgt_n  = F.normalize(z_target, dim=-1)
            dist_to_sg = float(1.0 - z_cur_n @ z_tgt_n)

            if dist_to_sg < switch_threshold and sg_idx < n_chain - 1:
                sg_idx     += 1
                sg_patience = 0
                z_target    = chain[sg_idx]
            else:
                sg_patience += 1
                if sg_patience >= patience_limit and sg_idx < n_chain - 1:
                    sg_idx     += 1
                    sg_patience = 0
                    z_target    = chain[sg_idx]

            actions = planner.beam_search(z_cur, z_target, horizon=horizon, beam_k=beam_k)

            episode_done = False
            for a in actions[:exec_steps]:
                if step >= plan_steps:
                    break
                _, _, done, info = worker.step(a)
                step += 1
                if info:
                    trial_achievements |= info.get("achievements", set())
                    if trigger_achievement in trial_achievements:
                        achievement_fired = True
                if done:
                    episode_done = True
                    break

            if episode_done:
                chain       = _rebuild_chain()
                n_chain     = len(chain)
                sg_idx      = 0
                sg_patience = 0

        z_final_n  = F.normalize(planner.encode_obs(worker.obs), dim=-1)
        final_dist = float(1.0 - z_final_n @ z_goal_n)
        all_final_dists.append(final_dist)
        all_achievements.append(trial_achievements)
        all_fired.append(achievement_fired)

        trace_str = "  ".join(f"{d:.3f}" for d in dist_trace[:8])
        print(f"[trial {trial+1}/{n_trials}]"
              f"  chain={[f'{d:.3f}' for d in chain_dists]}"
              f"  dist_trace=[{trace_str}{'...' if len(dist_trace) > 8 else ''}]"
              f"  achievement_fired={achievement_fired}"
              f"  final_dist={final_dist:.3f}"
              f"  ach={sorted(trial_achievements)}")

    fired_count = sum(all_fired)
    print(f"\n[precondition] achievement_fired={fired_count}/{n_trials} trials")
    print(f"[precondition] mean final_dist={np.mean(all_final_dists):.3f}")
    all_ach = set().union(*all_achievements)
    print(f"[precondition] achievements across all trials: {sorted(all_ach)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jepa-checkpoint",     default="runs/crafter_jepa/jepa_final.pt")
    p.add_argument("--ppo-checkpoint",      default="runs/crafter_ppo_jepa/policy_final.pt")
    p.add_argument("--experiment",          default="both",
                   choices=["goal", "curiosity", "both", "chain", "precondition"])
    p.add_argument("--trigger-achievement", default="place_table")
    p.add_argument("--horizon",             type=int,   default=15)
    p.add_argument("--beam-k",              type=int,   default=10)
    p.add_argument("--exec-steps",          type=int,   default=5)
    p.add_argument("--trials",              type=int,   default=5)
    p.add_argument("--max-steps",           type=int,   default=200)
    p.add_argument("--max-memory",          type=int,   default=500)
    p.add_argument("--seed",                type=int,   default=42)
    # chain / precondition shared args
    p.add_argument("--n-subgoals",          type=int,   default=3)
    p.add_argument("--explore-steps",       type=int,   default=300)
    p.add_argument("--plan-steps",          type=int,   default=300)
    p.add_argument("--switch-threshold",    type=float, default=0.20)
    p.add_argument("--memory-npz",          default=None,
                   help="path to crafter_rollouts.npz; samples --memory-npz-n obs "
                        "to pre-populate z_memory before curiosity exploration")
    p.add_argument("--memory-npz-n",        type=int,   default=2000,
                   help="number of observations to sample from --memory-npz")
    # precondition-specific args
    p.add_argument("--execute-threshold",   type=float, default=0.15,
                   help="cos_dist below which to fire the achievement action burst")
    p.add_argument("--execute-burst",       type=int,   default=20,
                   help="how many times to try the achievement action per close encounter")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device(args.device)

    encoder, dynamics, cfg = load_jepa(args.jepa_checkpoint, device)
    planner = LatentPlanner(encoder, dynamics, n_actions=cfg.n_actions, device=device)

    if args.experiment in ("goal", "both"):
        result = collect_goal_obs(
            args.jepa_checkpoint,
            args.ppo_checkpoint,
            trigger_achievement=args.trigger_achievement,
            seed=args.seed,
            device=device,
        )
        if result is None:
            print("[goal] could not collect goal obs — skipping")
        else:
            goal_obs, _goal_state = result
            run_goal_experiment(
                planner, goal_obs,
                n_trials=args.trials,
                horizon=args.horizon,
                exec_steps=args.exec_steps,
                max_total_steps=args.max_steps,
                seed=args.seed + 1,
            )

    if args.experiment in ("curiosity", "both"):
        n_curiosity_trials = max(1, args.trials // 2) if args.experiment == "both" else args.trials
        run_curiosity_experiment(
            planner,
            n_trials=n_curiosity_trials,
            horizon=args.horizon,
            exec_steps=args.exec_steps,
            max_total_steps=args.max_steps + 100,
            max_memory=args.max_memory,
            seed=args.seed + 2,
            device=device,
        )

    if args.experiment == "chain":
        result = collect_goal_obs(
            args.jepa_checkpoint,
            args.ppo_checkpoint,
            trigger_achievement=args.trigger_achievement,
            seed=args.seed,
            device=device,
        )
        if result is None:
            print("[chain] could not collect goal obs — aborting")
        else:
            goal_obs, _goal_state = result
            z_mem_seed: Optional[torch.Tensor] = None
            if args.memory_npz is not None and Path(args.memory_npz).exists():
                z_mem_seed = load_memory_from_npz(
                    args.memory_npz, encoder,
                    n_samples=args.memory_npz_n,
                    device=device, seed=args.seed,
                )
            elif args.memory_npz is not None:
                print(f"[chain] WARNING: --memory-npz {args.memory_npz} not found, "
                      f"starting with empty memory")
            run_chain_experiment(
                planner, goal_obs,
                n_trials=args.trials,
                n_subgoals=args.n_subgoals,
                explore_steps=args.explore_steps,
                plan_steps=args.plan_steps,
                max_memory=args.max_memory,
                horizon=args.horizon,
                beam_k=args.beam_k,
                exec_steps=args.exec_steps,
                switch_threshold=args.switch_threshold,
                seed=args.seed + 3,
                device=device,
                z_memory_seed=z_mem_seed,
            )

    if args.experiment == "precondition":
        result = collect_goal_obs(
            args.jepa_checkpoint,
            args.ppo_checkpoint,
            trigger_achievement=args.trigger_achievement,
            seed=args.seed,
            device=device,
        )
        if result is None:
            print("[precondition] could not collect goal obs — aborting")
        else:
            goal_obs, _goal_state = result
            z_mem_seed2: Optional[torch.Tensor] = None
            if args.memory_npz is not None and Path(args.memory_npz).exists():
                z_mem_seed2 = load_memory_from_npz(
                    args.memory_npz, encoder,
                    n_samples=args.memory_npz_n,
                    device=device, seed=args.seed,
                )
            elif args.memory_npz is not None:
                print(f"[precondition] WARNING: --memory-npz {args.memory_npz} not found, "
                      f"starting with empty memory")
            run_precondition_experiment(
                planner, goal_obs,
                trigger_achievement=args.trigger_achievement,
                n_trials=args.trials,
                n_subgoals=args.n_subgoals,
                explore_steps=args.explore_steps,
                plan_steps=args.plan_steps,
                max_memory=args.max_memory,
                horizon=args.horizon,
                beam_k=args.beam_k,
                exec_steps=args.exec_steps,
                switch_threshold=args.switch_threshold,
                execute_threshold=args.execute_threshold,
                execute_burst=args.execute_burst,
                seed=args.seed + 4,
                device=device,
                z_memory_seed=z_mem_seed2,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
