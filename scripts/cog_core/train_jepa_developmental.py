"""Train a JEPA from scratch with strict developmental curriculum
(Path B). The JEPA passes through stages 0a → 0b → 0c → 0d in order;
each stage gate is competence-based (1-step cosine sim threshold).

This is the foundational PATH B implementation: instead of training
the JEPA on random rollouts of one env (the v1.3 approach), we train
it like a child — easiest concept first, advance only when prior
concept is mastered, build on transferred weights.

Differences from scripts/train_jepa.py (the v1.x trainer):
  - Multi-stage env progression (DEFAULT_STAGES from dev_curriculum)
  - Per-stage held-out validation set for transition gating
  - Stage transition logged in checkpoint
  - Same JEPA architecture and loss as v1.3 — only the data ordering changes

Usage:
    python -m scripts.cog_core.train_jepa_developmental \
        --total-steps 80000 \
        --batch-size 128 \
        --encoder-type categorical_spatial --spatial-channels 64 \
        --dynamics-type spatial_film --dynamics-hidden 256 --dynamics-layers 3 \
        --aux-predicate-weight 3.0 --aux-distance-dim 24 --aux-distance-weight 0.5 \
        --run-name jepa_dev_v0 --device cuda

Then re-run all Phase 1 emergence tests pointing at this checkpoint
instead of v1.3 — the comparison IS the test of the developmental
principle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  — registers BabyAI envs
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from prism.cog_core.dev_curriculum import (
    DEFAULT_STAGES, DevStage, DevelopmentalCurriculum,
)
from prism.envs.babyai import _encode_image
from prism.models.jepa import JepaConfig, JepaWorldModel
from prism.perception import (
    compute_augmented_predicates, compute_predicates, extract_slots,
)
from prism.perception.slots import AGENT_POS, OBJECT_TYPES
from prism.utils.seed import set_global_seed


# ---- primary-object (color, type) label for factored aux supervision ----
_TYPE_TO_IDX = {t: i for i, t in enumerate(OBJECT_TYPES)}


def primary_object_label(slots) -> tuple[int, int]:
    """Return (color_id, type_idx) of the most-prominent visible object,
    or (-1, -1) if no recognized object is in view.

    Heuristic: prefer slots in the agent's facing column (x=AGENT_POS[0])
    with smallest y (closest in front); fall back to the slot with smallest
    Manhattan distance to AGENT_POS."""
    if not slots:
        return -1, -1
    ax, ay = AGENT_POS
    in_col = [s for s in slots if int(s.x) == ax]
    if in_col:
        s = min(in_col, key=lambda s: int(s.y))
    else:
        s = min(
            slots,
            key=lambda s: abs(int(s.x) - ax) + abs(int(s.y) - ay),
        )
    t_id = int(s.type_id)
    if t_id not in _TYPE_TO_IDX:
        return -1, -1
    return int(s.color_id), _TYPE_TO_IDX[t_id]


# ---------------------------------------------------------------- data
def _process_frame_pair(
    raw_t: np.ndarray,
    raw_tp1: np.ndarray,
    action_id: int,
    obs_t_list: list,
    act_list: list,
    obs_tp1_list: list,
    pred_t_list: list,
    pred_tp1_list: list,
    col_t_list: list,
    typ_t_list: list,
    col_tp1_list: list,
    typ_tp1_list: list,
    *,
    with_predicates: bool,
    augmented: bool,
) -> None:
    """Shared frame-pair processing — encode, slot extraction, predicate
    computation, factored primary-object label. Used by both the single-env
    and vectorized collection paths so they produce identical data."""
    obs_t_list.append(_encode_image(raw_t))
    act_list.append(action_id)
    obs_tp1_list.append(_encode_image(raw_tp1))
    slots_t = extract_slots(raw_t)
    slots_tp1 = extract_slots(raw_tp1)
    if with_predicates:
        pred_fn = compute_augmented_predicates if augmented else compute_predicates
        pred_t_list.append(pred_fn(slots_t))
        pred_tp1_list.append(pred_fn(slots_tp1))
    c_t, ty_t = primary_object_label(slots_t)
    c_tp1, ty_tp1 = primary_object_label(slots_tp1)
    col_t_list.append(c_t)
    typ_t_list.append(ty_t)
    col_tp1_list.append(c_tp1)
    typ_tp1_list.append(ty_tp1)


def collect_random_transitions(
    env_id: str,
    n: int,
    rng: np.random.Generator,
    *,
    with_predicates: bool = False,
    augmented: bool = False,
    n_envs: int = 1,
):
    """Collect n one-step transitions under random policy.

    When `n_envs == 1`, behaves identically to the original single-env
    implementation. When `n_envs > 1`, runs N parallel BabyAI envs in the
    same process (manual list of envs, matching the `EnvWorker` pattern
    used elsewhere in PRISM, e.g. `scripts/ppo_train.py:90`). Each outer
    iteration steps N envs and produces up to N transitions, amortizing
    Python/loop overhead and (more importantly) removing the long single-
    threaded blocking call that starves the GPU during rollout refresh.

    Transitions crossing an episode boundary are valid: env.step's
    returned `next_obs` is the terminal frame, which is the correct
    `obs_tp1`; we then reset that env for the next iteration. This
    matches the single-env path exactly so the science is unchanged.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")

    # Build N envs (lazily, so a bad env_id raises a clear single error).
    try:
        envs = [gym.make(env_id) for _ in range(n_envs)]
    except Exception as e:
        raise SystemExit(
            f"\n[dev-jepa] env {env_id} unavailable in this minigrid build: {e}\n"
            "Edit prism/cog_core/dev_curriculum.DEFAULT_STAGES if needed."
        )
    n_actions = envs[0].action_space.n

    obs_t_list, act_list, obs_tp1_list = [], [], []
    pred_t_list, pred_tp1_list = [], []
    col_t_list, typ_t_list, col_tp1_list, typ_tp1_list = [], [], [], []

    # Per-env "current obs" — initialized via reset with distinct seeds.
    cur_obs: list = [None] * n_envs
    for i in range(n_envs):
        seed = int(rng.integers(0, 1_000_000)) + i * 7919
        obs_i, _ = envs[i].reset(seed=seed)
        cur_obs[i] = obs_i

    while len(obs_t_list) < n:
        # Sample N actions in one numpy call (cheaper than per-env sampling).
        actions = rng.integers(n_actions, size=n_envs)
        for i in range(n_envs):
            if len(obs_t_list) >= n:
                break
            raw_t = cur_obs[i]["image"]
            a_i = int(actions[i])
            next_obs, _r, term, trunc, _ = envs[i].step(a_i)
            raw_tp1 = next_obs["image"]
            _process_frame_pair(
                raw_t, raw_tp1, a_i,
                obs_t_list, act_list, obs_tp1_list,
                pred_t_list, pred_tp1_list,
                col_t_list, typ_t_list, col_tp1_list, typ_tp1_list,
                with_predicates=with_predicates, augmented=augmented,
            )
            if term or trunc:
                new_seed = int(rng.integers(0, 1_000_000))
                cur_obs[i], _ = envs[i].reset(seed=new_seed)
            else:
                cur_obs[i] = next_obs

    for env in envs:
        env.close()
    return (
        np.stack(obs_t_list).astype(np.float32),
        np.array(act_list, dtype=np.int64),
        np.stack(obs_tp1_list).astype(np.float32),
        np.stack(pred_t_list).astype(np.float32) if with_predicates else None,
        np.stack(pred_tp1_list).astype(np.float32) if with_predicates else None,
        np.array(col_t_list, dtype=np.int64),
        np.array(typ_t_list, dtype=np.int64),
        np.array(col_tp1_list, dtype=np.int64),
        np.array(typ_tp1_list, dtype=np.int64),
    )


# ---------------------------------------------------- transition-gate eval
@torch.no_grad()
def measure_stage_competence(
    model: JepaWorldModel,
    env_id: str,
    rng: np.random.Generator,
    device: torch.device,
    n_transitions: int = 1000,
) -> float:
    """Held-out 1-step latent cosine similarity for the current stage's
    env. This is the metric the curriculum gate reads."""
    obs_t, actions, obs_tp1, _, _, _, _, _, _ = collect_random_transitions(
        env_id, n_transitions, rng, with_predicates=False, n_envs=1,
    )
    obs_t_t = torch.from_numpy(obs_t).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    obs_tp1_t = torch.from_numpy(obs_tp1).to(device)
    z_t = model.encode(obs_t_t)
    z_pred = model.predict(z_t, actions_t)
    z_actual = model.encode(obs_tp1_t)
    cos = torch.nn.functional.cosine_similarity(
        z_pred.flatten(1), z_actual.flatten(1), dim=-1,
    )
    return float(cos.median().item())


# --------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser()
    # JEPA architecture (matches v1.3 by default)
    parser.add_argument("--encoder-type", default="categorical_spatial",
                        choices=["flat", "categorical", "categorical_spatial"])
    parser.add_argument("--aux-predicate-weight", type=float, default=3.0)
    parser.add_argument("--aux-distance-dim", type=int, default=24)
    parser.add_argument("--aux-distance-weight", type=float, default=0.5)
    parser.add_argument("--aux-factored-weight", type=float, default=0.0,
                        help="factored (color, type) softmax CE on the "
                             "primary visible object. Forces shared-weight "
                             "color and type axes in the latent — required "
                             "for compositional predicate readout. Try 1.0.")
    parser.add_argument("--bf16", action="store_true",
                        help="bf16 autocast on the forward/loss pass")
    parser.add_argument("--compile", action="store_true",
                        help="wrap model.loss with torch.compile. The model "
                             "is dispatch-bound (30-40 small ops per step at "
                             "~1-2ms Python overhead each = ~35ms forward + "
                             "~50ms backward) so compilation typically gives "
                             "2-3x speedup by fusing ops into fewer kernels. "
                             "First step is slow (~10-30s compile time).")
    parser.add_argument("--min-steps-scale", type=float, default=1.0,
                        help="multiply every stage's min_steps and max_steps "
                             "by this factor. Use < 1.0 when running with a "
                             "much larger --batch-size so the curriculum "
                             "doesn't over-train (e.g. with batch=4096 vs "
                             "the default 128, try --min-steps-scale 0.0625 "
                             "for an equivalent total data exposure).")
    parser.add_argument("--dynamics-hidden", type=int, default=256)
    parser.add_argument("--dynamics-layers", type=int, default=3)
    parser.add_argument("--dynamics-type", default="spatial_film",
                        choices=["mlp", "film", "spatial_film"])
    parser.add_argument("--spatial-channels", type=int, default=64)
    # Training
    parser.add_argument("--total-steps", type=int, default=80_000,
                        help="Hard cap. Curriculum may finish earlier if all "
                             "stages converge under their max_steps budgets.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--collect-every", type=int, default=500,
                        help="Refresh stage rollout buffer every N steps.")
    parser.add_argument("--rollout-size", type=int, default=5000)
    parser.add_argument("--n-collect-envs", type=int, default=8,
                        help="number of parallel BabyAI envs used during "
                             "`collect_random_transitions`. 1 = the original "
                             "single-env path; 8 = ~6-8× lower per-collect "
                             "blocking time (the real GPU-idle bottleneck).")
    parser.add_argument("--gate-eval-every", type=int, default=500,
                        help="Measure competence (and consider stage-advance) "
                             "every N optimizer steps.")
    parser.add_argument("--gate-eval-n-transitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    # Probe env to grab n_actions (all BabyAI envs use the same 7-action set).
    sample_env = gym.make(DEFAULT_STAGES[0].env_id)
    n_actions = sample_env.action_space.n
    sample_env.close()

    cfg = JepaConfig(
        n_actions=n_actions,
        encoder_type=args.encoder_type,
        aux_predicate_weight=args.aux_predicate_weight,
        aux_distance_dim=args.aux_distance_dim,
        aux_distance_weight=args.aux_distance_weight,
        aux_factored_weight=args.aux_factored_weight,
        dynamics_hidden_dim=args.dynamics_hidden,
        dynamics_layers=args.dynamics_layers,
        dynamics_type=args.dynamics_type,
        spatial_channels=args.spatial_channels,
    )
    model = JepaWorldModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[dev-jepa] model: encoder={cfg.encoder_type} dyn={cfg.dynamics_type} "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # Apply --min-steps-scale so the curriculum doesn't over-train when the
    # user passes a much larger --batch-size. Each stage's min_steps/max_steps
    # is scaled by the factor; transition_cos and env_id are unchanged.
    if abs(args.min_steps_scale - 1.0) > 1e-9:
        scale = args.min_steps_scale
        scaled_stages = [
            DevStage(
                name=s.name,
                env_id=s.env_id,
                description=s.description,
                min_steps=max(1, int(round(s.min_steps * scale))),
                max_steps=max(1, int(round(s.max_steps * scale))),
                transition_cos=s.transition_cos,
            )
            for s in DEFAULT_STAGES
        ]
        print(f"[dev-jepa] curriculum min_steps scaled by {scale} "
              f"(was sum={sum(s.min_steps for s in DEFAULT_STAGES)}, "
              f"now sum={sum(s.min_steps for s in scaled_stages)})")
        curr = DevelopmentalCurriculum(stages=scaled_stages)
    else:
        curr = DevelopmentalCurriculum(stages=list(DEFAULT_STAGES))
    print("[dev-jepa] curriculum stages:")
    for s in curr.stages:
        print(f"  {s}")

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    print(f"[dev-jepa] writing to {out_dir}")

    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 7)
    use_aux = args.aux_predicate_weight > 0.0
    use_distance = args.aux_distance_dim > 0
    use_factored = args.aux_factored_weight > 0.0
    use_amp = args.bf16 and device.type == "cuda"
    if use_amp:
        print("[dev-jepa] BF16 autocast enabled")
    print(f"[dev-jepa] collection: n_collect_envs={args.n_collect_envs}  "
          f"rollout_size={args.rollout_size}  collect_every={args.collect_every}")
    # `loss_fn` is what the training loop actually calls. With --compile,
    # torch.compile fuses the 30-40 small ops in `model.loss` into a much
    # smaller set of generated kernels. mode="reduce-overhead" enables CUDA
    # graphs, the largest win for dispatch-bound small models. fullgraph=False
    # allows graph breaks at the dict-return / optional-label branches.
    loss_fn = model.loss
    if args.compile:
        print("[dev-jepa] torch.compile(model.loss, mode='reduce-overhead') — "
              "first step will be slow (~10-30s) while compiling")
        loss_fn = torch.compile(
            model.loss, mode="reduce-overhead", fullgraph=False,
        )

    # Per-stage rollout buffer (refreshed when stage changes OR
    # every collect_every steps within a stage). Kept on GPU after
    # collection so the train loop never blocks on CPU→GPU copies.
    obs_t_gpu = act_gpu = obs_tp1_gpu = None
    preds_t_gpu = preds_tp1_gpu = None
    col_t_gpu = typ_t_gpu = col_tp1_gpu = typ_tp1_gpu = None
    last_collected_for: str | None = None
    # GPU-side loss accumulator — synced to CPU only at log time.
    loss_accum = torch.zeros((), device=device)
    loss_count = 0

    for step in range(args.total_steps):
        if curr.is_done():
            print(f"[dev-jepa] all stages complete at step {step}; stopping early")
            break

        stage = curr.current_stage()

        # Refresh rollout buffer if stage changed or interval hit.
        need_refresh = (
            obs_t_gpu is None
            or last_collected_for != stage.env_id
            or step % args.collect_every == 0
        )
        if need_refresh:
            (obs_t_buf, act_buf, obs_tp1_buf,
             pred_t_buf, pred_tp1_buf,
             col_t_buf, typ_t_buf, col_tp1_buf, typ_tp1_buf
             ) = collect_random_transitions(
                stage.env_id, args.rollout_size, rng,
                with_predicates=use_aux, augmented=use_distance,
                n_envs=args.n_collect_envs,
            )
            # Move the entire buffer to GPU once; per-step sampling is then
            # pure GPU indexing — no CPU→GPU copies in the hot path.
            obs_t_gpu = torch.from_numpy(obs_t_buf).to(device)
            act_gpu = torch.from_numpy(act_buf).to(device)
            obs_tp1_gpu = torch.from_numpy(obs_tp1_buf).to(device)
            preds_t_gpu = torch.from_numpy(pred_t_buf).to(device) if use_aux else None
            preds_tp1_gpu = torch.from_numpy(pred_tp1_buf).to(device) if use_aux else None
            col_t_gpu = torch.from_numpy(col_t_buf).to(device)
            typ_t_gpu = torch.from_numpy(typ_t_buf).to(device)
            col_tp1_gpu = torch.from_numpy(col_tp1_buf).to(device)
            typ_tp1_gpu = torch.from_numpy(typ_tp1_buf).to(device)
            last_collected_for = stage.env_id

        # GPU-side batch sampling — no CPU work, no PCIe transfer.
        idx = torch.randint(
            0, args.rollout_size, (args.batch_size,), device=device,
        )
        obs_t = obs_t_gpu[idx]
        a_t = act_gpu[idx]
        obs_tp1 = obs_tp1_gpu[idx]
        preds_t = preds_t_gpu[idx] if use_aux else None
        preds_tp1 = preds_tp1_gpu[idx] if use_aux else None
        if use_factored:
            col_t = col_t_gpu[idx]
            typ_t = typ_t_gpu[idx]
            col_tp1 = col_tp1_gpu[idx]
            typ_tp1 = typ_tp1_gpu[idx]
        else:
            col_t = typ_t = col_tp1 = typ_tp1 = None

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = loss_fn(
                    obs_t, a_t, obs_tp1,
                    predicates_t=preds_t, predicates_tp1=preds_tp1,
                    color_label_t=col_t, type_label_t=typ_t,
                    color_label_tp1=col_tp1, type_label_tp1=typ_tp1,
                )
        else:
            out = loss_fn(
                obs_t, a_t, obs_tp1,
                predicates_t=preds_t, predicates_tp1=preds_tp1,
                color_label_t=col_t, type_label_t=typ_t,
                color_label_tp1=col_tp1, type_label_tp1=typ_tp1,
            )
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.update_target()
        curr.increment_step()

        # Accumulate loss on GPU — sync only every 100 steps for the log.
        loss_accum = loss_accum + loss.detach()
        loss_count += 1

        if step % 100 == 0:
            loss_val = float(loss_accum.item() / max(loss_count, 1))
            loss_accum = torch.zeros((), device=device)
            loss_count = 0
            writer.add_scalar(f"train/{stage.name}/loss", loss_val, step)
            writer.add_scalar("train/loss_total", loss_val, step)
            writer.add_scalar("train/mean200", loss_val, step)
            # Factored CE values from the latest batch (cheap; out is fresh).
            fac_c = (
                float(out["loss_fac_color_t"].item())
                if "loss_fac_color_t" in out else None
            )
            fac_t = (
                float(out["loss_fac_type_t"].item())
                if "loss_fac_type_t" in out else None
            )
            fac_str = ""
            if fac_c is not None and fac_t is not None:
                writer.add_scalar("train/fac_color", fac_c, step)
                writer.add_scalar("train/fac_type", fac_t, step)
                fac_str = f" fac_c={fac_c:.3f} fac_t={fac_t:.3f}"
            print(f"[step {step:6d}] stage={stage.name:3s} "
                  f"({stage.env_id.replace('BabyAI-', '').replace('-v0', ''):20s}) "
                  f"loss={loss_val:.4f}{fac_str}")

        # Stage-transition gate
        if (step + 1) % args.gate_eval_every == 0 or step == args.total_steps - 1:
            cos = measure_stage_competence(
                model, stage.env_id, eval_rng, device,
                n_transitions=args.gate_eval_n_transitions,
            )
            writer.add_scalar(f"gate/{stage.name}/cosine", cos, step)
            steps_in_stage = curr.stage_step_counts.get(stage.name, 0)
            print(f"  [gate @ step {step+1}] stage={stage.name} "
                  f"steps_in_stage={steps_in_stage} "
                  f"cosine={cos:.4f} target={stage.transition_cos:.2f} "
                  f"min_steps={stage.min_steps} max_steps={stage.max_steps}")
            transition = curr.maybe_advance(global_step=step + 1, recent_cosine=cos)
            if transition is not None:
                print(f"  >>> ADVANCING: {transition.from_stage} → {transition.to_stage} "
                      f"({transition.reason}, cos={transition.cosine_at_transition:.3f})")
                writer.add_scalar(
                    f"transitions/{transition.from_stage}_to_{transition.to_stage}",
                    cos, step,
                )
                # Save a stage-boundary checkpoint.
                ck = out_dir / f"jepa_after_{transition.from_stage}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "step": step + 1,
                    "stage_name": transition.from_stage,
                    "transition_cosine": transition.cosine_at_transition,
                    "transition_reason": transition.reason,
                    "curriculum_summary": curr.summary(),
                }, ck)
                print(f"  >>> saved {ck}")

    # Final
    final = out_dir / "jepa_final.pt"
    torch.save({
        "model": model.state_dict(),
        "cfg": cfg,
        "step": args.total_steps,
        "curriculum_summary": curr.summary(),
    }, final)
    summary_path = out_dir / "curriculum_summary.json"
    with open(summary_path, "w") as f:
        json.dump(curr.summary(), f, indent=2)
    print(f"\n[done] saved {final}")
    print(f"[done] curriculum summary → {summary_path}")
    print(f"[done] stages completed: {len(curr.transitions)}/{len(curr.stages)}")
    for t in curr.transitions:
        print(f"  {t.from_stage} → {t.to_stage}: at step {t.at_step} "
              f"(cosine={t.cosine_at_transition:.3f}, reason={t.reason})")
    print()
    print("Next: re-run Phase 1 emergence tests pointing at this checkpoint:")
    print(f"  --jepa-checkpoint {final}")
    print("Then compare to the same tests on v1.3 JEPA — that comparison")
    print("IS the test of the developmental-curriculum principle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
