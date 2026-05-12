"""PRISM ↔ Unity WebSocket bridge — Day 2 inference server.

Replaces Day 1's random-action server with the trained v6 substrate.
The substrate was trained on BabyAI (egocentric 7x7x3 obs + 24-d
mission one-hot, 7-action discrete output). Unity sends 2D (x, z)
positions, so Unity2DAdapter renders fake BabyAI obs each step and
remaps the substrate's actions back to Unity's 5-action space.

This is the "domain-general substrate" thesis under test: trained
weights are unchanged from BabyAI; only the adapter layer changes.

Protocol (identical to Day 1):
  Unity → Python: {"agent_pos":[x,z], "target_pos":[x,z],
                   "delta":[dx,dz], "step":int, "episode_done":bool}
  Python → Unity: {"action": 0..4}   # stay, N, S, E, W

Usage on Vast.ai:
    python unity_demo/prism_inference_server.py \\
        --jepa  /workspace/runs/jepa_dev_v1_factored/jepa_final.pt \\
        --policy /workspace/runs/v6_pr6_curriculum/policy_final.pt \\
        --trunk transformer     # PR-4+; use 'gru' for older checkpoints
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import torch

HOST = "0.0.0.0"  # bind on all interfaces (SSH tunnel friendly)
PORT = 8765


def _build_policy(ckpt: dict, jepa, cfg, device: torch.device, trunk: str, universal_inner: str):
    """Reconstruct policy from a v6 ppo_train.py checkpoint."""
    from prism.adapters.babyai_adapter import BabyAIAdapter
    from prism.cognition.policy import UniversalPolicy
    from prism.models.hybrid_policy import HybridPolicy
    from prism.models.recurrent_policy import RecurrentPolicy

    policy_type = ckpt.get("policy_type", "recurrent")
    shared = dict(
        latent_in_dim=ckpt["latent_in_dim"],
        n_actions=ckpt["n_actions"],
        mission_dim=ckpt["mission_dim"],
        hidden_dim=ckpt["hidden_dim"],
        latent_proj_dim=ckpt["latent_proj_dim"],
        mem_feat_dim=ckpt.get("mem_feat_dim", 0),
    )

    if policy_type == "universal":
        adapter = BabyAIAdapter(jepa=jepa, cfg=cfg, device=device)
        policy = UniversalPolicy.from_adapter(
            adapter,
            trunk=trunk,
            hidden_dim=shared["hidden_dim"],
            latent_proj_dim=shared["latent_proj_dim"],
            mem_feat_dim=shared["mem_feat_dim"],
            policy_type=universal_inner,
            concept_n_slots=ckpt.get("concept_n_slots", 1024),
            concept_slot_dim=ckpt.get("concept_slot_dim", 64),
            concept_scaling=ckpt.get("concept_scaling", 1.0),
            operator_n_slots=ckpt.get("operator_n_slots", 64),
            operator_slot_dim=ckpt.get("operator_slot_dim", 64),
            operator_scaling=ckpt.get("operator_scaling", 4.0),
            use_operator_memory=ckpt.get("use_operator_memory", True),
        ).to(device)
    elif policy_type == "hybrid":
        policy = HybridPolicy(
            **shared,
            concept_n_slots=ckpt.get("concept_n_slots", 1024),
            concept_slot_dim=ckpt.get("concept_slot_dim", 64),
            concept_scaling=ckpt.get("concept_scaling", 1.0),
            operator_n_slots=ckpt.get("operator_n_slots", 64),
            operator_slot_dim=ckpt.get("operator_slot_dim", 64),
            operator_scaling=ckpt.get("operator_scaling", 4.0),
            use_operator_memory=ckpt.get("use_operator_memory", True),
        ).to(device)
    else:
        policy = RecurrentPolicy(**shared).to(device)

    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy, policy_type


def _load_jepa(path: Path, device: torch.device):
    """Mirrors prism/adapters/babyai_adapter.py:from_jepa_checkpoint."""
    from prism.models.jepa import JepaConfig, JepaWorldModel, upgrade_config

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg: JepaConfig = upgrade_config(ckpt["cfg"])
    jepa = JepaWorldModel(cfg).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    return jepa, cfg


async def _handle_connection(websocket, *, policy, jepa, device, args):
    """One Unity client → one async handler.

    Maintains per-connection: virtual heading, recurrent hidden state,
    previous action. Reset on `episode_done=true`.
    """
    import numpy as np

    from prism.adapters.unity_2d import Unity2DAdapter

    print(f"[infer] connected: {websocket.remote_address}")
    adapter = Unity2DAdapter(
        target_color=args.target_color,
        target_type=args.target_type,
        n_actions=policy.n_actions if hasattr(policy, "n_actions") else 7,
    )
    h = policy.init_hidden(1, device)
    prev_action = torch.tensor([-1], device=device, dtype=torch.long)
    mission = adapter.mission_onehot(device).unsqueeze(0)  # (1, 24)
    step_count = 0
    state_kind = getattr(policy, "state_kind", "tensor")

    # Episode telemetry for path-efficiency metric.
    episode_idx = 0
    episode_start: tuple[float, float] | None = None
    episode_target: tuple[float, float] | None = None
    episode_path_len = 0.0
    episode_start_step = 0
    prev_pos: tuple[float, float] | None = None

    try:
        async for message in websocket:
            try:
                state = json.loads(message)
            except json.JSONDecodeError:
                print(f"[infer] invalid JSON: {message[:80]}")
                continue

            # ---- handle episode boundary ----
            if state.get("episode_done", False):
                # Compute and print path-efficiency before resetting.
                if episode_start is not None and episode_target is not None:
                    optimal = abs(episode_target[0] - episode_start[0]) + abs(
                        episode_target[1] - episode_start[1]
                    )
                    actual = episode_path_len
                    eff = (optimal / actual) if actual > 1e-6 else 0.0
                    eps_steps = step_count - episode_start_step
                    print(
                        f"[infer] *** TOUCH ep#{episode_idx} step={step_count} "
                        f"steps_in_ep={eps_steps} "
                        f"start=({episode_start[0]:+.2f},{episode_start[1]:+.2f}) "
                        f"target=({episode_target[0]:+.2f},{episode_target[1]:+.2f}) "
                        f"optimal_manhattan={optimal:.2f} actual_path={actual:.2f} "
                        f"efficiency={eff:.3f}"
                    )
                episode_idx += 1
                episode_start = None
                episode_target = None
                episode_path_len = 0.0
                episode_start_step = step_count
                prev_pos = None
                adapter.reset()
                if state_kind == "tuple":
                    done_t = torch.ones(1, dtype=torch.bool, device=device)
                    h = policy.reset_buffer(done_t, h)
                else:
                    h = policy.init_hidden(1, device)
                prev_action = torch.tensor([-1], device=device, dtype=torch.long)

            agent_pos = tuple(state.get("agent_pos", [0.0, 0.0]))
            target_pos = tuple(state.get("target_pos", [0.0, 0.0]))

            # Episode start: first state of a new episode.
            if episode_start is None:
                episode_start = agent_pos
                episode_target = target_pos
                prev_pos = agent_pos

            # Accumulate path length.
            if prev_pos is not None:
                episode_path_len += (
                    (agent_pos[0] - prev_pos[0]) ** 2
                    + (agent_pos[1] - prev_pos[1]) ** 2
                ) ** 0.5
            prev_pos = agent_pos

            # ---- adapter: render fake BabyAI obs ----
            obs_np = adapter.render_obs(agent_pos, target_pos)  # (3, 7, 7)
            obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)  # (1, 3, 7, 7)

            # ---- substrate forward ----
            with torch.no_grad():
                z = jepa.encode(obs_t)
                logits, h = policy.step(z, prev_action, mission, h)
                logits = adapter.mask_logits(logits)
                substrate_action = int(logits.argmax(dim=-1).item())

            prev_action = torch.tensor([substrate_action], device=device, dtype=torch.long)
            unity_action = adapter.map_action(substrate_action)

            await websocket.send(json.dumps({"action": unity_action}))

            step_count += 1
            # Verbose for the first 10 steps so we can see initial behavior;
            # then drop to every-100 to keep logs readable.
            if step_count <= 10 or step_count % 100 == 1:
                logits_list = logits.squeeze(0).tolist()
                logits_str = ",".join(f"{x:+.2f}" for x in logits_list)
                print(
                    f"[infer] step={step_count} "
                    f"agent=({agent_pos[0]:+.2f},{agent_pos[1]:+.2f}) "
                    f"target=({target_pos[0]:+.2f},{target_pos[1]:+.2f}) "
                    f"heading={adapter.heading} "
                    f"sub_act={substrate_action} unity_act={unity_action} "
                    f"logits=[{logits_str}]"
                )
    except Exception as e:
        print(f"[infer] connection error: {type(e).__name__}: {e}")
    finally:
        print(f"[infer] disconnected after {step_count} steps")


async def _main(args):
    try:
        import websockets
    except ImportError:
        print("[infer] missing dependency: pip install websockets")
        sys.exit(1)

    device = torch.device(args.device)
    print(f"[infer] device={device}")

    print(f"[infer] loading JEPA from {args.jepa}")
    jepa, cfg = _load_jepa(Path(args.jepa), device)
    print(f"[infer] JEPA encoder={cfg.encoder_type} embed_dim={cfg.embed_dim}")

    print(f"[infer] loading policy from {args.policy}")
    ckpt = torch.load(args.policy, map_location=device, weights_only=False)
    policy, policy_type = _build_policy(
        ckpt, jepa, cfg, device, trunk=args.trunk, universal_inner=args.universal_inner,
    )
    print(
        f"[infer] policy={policy_type} "
        f"trunk={args.trunk if policy_type == 'universal' else 'n/a'} "
        f"n_actions={ckpt['n_actions']} mission_dim={ckpt['mission_dim']} "
        f"hidden_dim={ckpt['hidden_dim']}"
    )

    print(f"[infer] PRISM Day 2 server listening on ws://{HOST}:{PORT}")
    print(f"[infer] target = {args.target_color} {args.target_type}")

    async def handler(ws):
        await _handle_connection(ws, policy=policy, jepa=jepa, device=device, args=args)

    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()


def main() -> int:
    p = argparse.ArgumentParser(description="PRISM Day 2 inference server.")
    p.add_argument("--jepa", required=True, help="path to JEPA checkpoint (.pt)")
    p.add_argument("--policy", required=True, help="path to policy checkpoint (.pt)")
    p.add_argument(
        "--trunk", default="transformer", choices=["transformer", "gru"],
        help="UniversalPolicy trunk type (universal checkpoints only)",
    )
    p.add_argument(
        "--universal-inner", default="hybrid", choices=["hybrid", "recurrent"],
        help="Inner policy class for universal+gru trunk",
    )
    p.add_argument("--target-color", default="green")
    p.add_argument("--target-type", default="ball")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("\n[infer] shut down by user")
    return 0


if __name__ == "__main__":
    sys.exit(main())
