"""Diagnostics for the dual-depth ACT policy. Reads only; changes nothing.

Subcommands
  data       phase balance, per-phase MAE, per-horizon MAE, zero-delta baseline
  blind      does the policy actually use depth, or only proprio?
  onpolicy   open-loop error on the teacher trajectory vs closed-loop rollout
  qvel       is the rollout proprio input the same quantity the dataset stored?

Run every subcommand from the project root on the machine that holds the
dataset. Only `data` and `blind` need a checkpoint plus the dataset; `qvel`
needs neither.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from collect_fixed_delta_depth_dataset import place_nearby_scene
from depth_act_dataset import (
    DepthACTEpisodeDataset,
    EpisodeBatchSampler,
    delta_episode_records,
    split_delta_records,
)
from delta_depth_dataset import CONTROL_HZ, PHASE_NAMES
from play_waypoint_teacher import execute_waypoint_episode
from so101_ball_bins_env import SO101BallBinsEnv
from so101_depth import DepthConfig, TopDepthRenderer
from so101_depth_act import (
    JointVelocityEstimator,
    TemporalActionEnsembler,
    depth_act_loss,
    load_depth_act_checkpoint,
    predict_delta_chunk,
)
from train_depth_act import resolve_device

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
HORIZON_BUCKETS = ((0, 5), (5, 15), (15, 30))


# ----------------------------------------------------------------- helpers
def _load_policy(path: Path, device: torch.device):
    policy, checkpoint = load_depth_act_checkpoint(str(path), map_location="cpu")
    policy = policy.to(device).eval()
    allowed = {field.name for field in fields(DepthConfig)}
    config = DepthConfig(**{k: v for k, v in checkpoint["depth_config"].items() if k in allowed})
    scale = torch.tensor(checkpoint["delta_scale"], dtype=torch.float32, device=device)
    return policy, checkpoint, config, scale


def _validation_loader(dataset_root: Path, config: DepthConfig, chunk: int, seed: int, batch: int):
    records = delta_episode_records(dataset_root)
    _, validation = split_delta_records(records, seed=seed)
    data = DepthACTEpisodeDataset(validation, chunk_size=chunk, depth_config=config, training=False)
    sampler = EpisodeBatchSampler(
        data.episode_offsets, batch_size=batch,
        shuffle_episodes=False, shuffle_frames=False, seed=seed,
    )
    return DataLoader(data, batch_sampler=sampler, num_workers=0), len(validation)


def _degrees(value: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """value is a per-joint error already in radians."""
    del scale
    return np.rad2deg(value)


# --------------------------------------------------------------- data mode
def run_data(args) -> None:
    device = resolve_device(args.device)
    policy, checkpoint, config, scale = _load_policy(args.model, device)
    loader, episodes = _validation_loader(
        args.dataset, config, policy.config.chunk_size, args.seed, args.batch_size
    )
    chunk = policy.config.chunk_size
    scale_np = scale.detach().cpu().numpy()

    abs_error = np.zeros((chunk, 6))          # summed |pred - target| in rad
    zero_error = np.zeros((chunk, 6))         # summed |0 - target|   in rad
    counts = np.zeros(chunk)
    phase_error = np.zeros((len(PHASE_NAMES), 6))
    phase_zero = np.zeros((len(PHASE_NAMES), 6))
    phase_frames = np.zeros(len(PHASE_NAMES))
    phase_elements = np.zeros(len(PHASE_NAMES))
    loss_total = zero_total = 0.0
    samples = 0

    with torch.inference_mode():
        for batch in loader:
            top = batch["top_depth"].to(device)
            side = batch["side_depth"].to(device)
            proprio = batch["proprio"].to(device)
            target = batch["delta_chunk"].to(device)
            mask = batch["chunk_mask"].to(device)
            phase = batch["phase_id"].numpy()
            prediction = policy(top, side, proprio)
            _, metrics = depth_act_loss(prediction, target, mask, delta_scale=scale)
            loss_total += metrics["loss"] * top.shape[0]
            zero_total += metrics["zero_delta_baseline"] * top.shape[0]
            samples += top.shape[0]

            valid = mask.float().unsqueeze(2).cpu().numpy()
            err = (prediction - target).abs().cpu().numpy() * valid
            zer = target.abs().cpu().numpy() * valid
            abs_error += err.sum(axis=0)
            zero_error += zer.sum(axis=0)
            counts += valid[:, :, 0].sum(axis=0)
            for row in range(top.shape[0]):
                pid = int(phase[row])
                phase_error[pid] += err[row].sum(axis=0)
                phase_zero[pid] += zer[row].sum(axis=0)
                phase_frames[pid] += 1
                phase_elements[pid] += valid[row, :, 0].sum()

    print(f"validation episodes={episodes} frames={samples} chunk_size={chunk}")
    print(f"delta_scale = {np.array2string(scale_np, precision=4)}\n")

    print("=" * 78)
    print("Q14  MODEL vs ZERO-DELTA BASELINE  (identical weighting, so directly comparable)")
    print(f"  loss                {loss_total / samples:.6e}")
    print(f"  zero_delta_baseline {zero_total / samples:.6e}")
    print(f"  zero_ratio          {loss_total / max(zero_total, 1e-12):.4f}   (<1 means better than standing still)")
    overall = abs_error.sum(0) / np.maximum(counts.sum(), 1)
    overall_zero = zero_error.sum(0) / np.maximum(counts.sum(), 1)
    print(f"\n  {'joint':<15}{'model MAE(deg)':>16}{'zero MAE(deg)':>15}{'ratio':>8}")
    for j, name in enumerate(JOINTS):
        print(f"  {name:<15}{np.rad2deg(overall[j]):>16.4f}{np.rad2deg(overall_zero[j]):>15.4f}"
              f"{overall[j] / max(overall_zero[j], 1e-12):>8.3f}")

    print("\n" + "=" * 78)
    print("Q13  MAE BY HORIZON BUCKET")
    print(f"  {'bucket':<10}{'steps':>7}" + "".join(f"{n[:8]:>10}" for n in JOINTS) + f"{'mean':>10}")
    print(f"  {'':<10}{'':>7}" + "".join(f"{'(deg)':>10}" for _ in JOINTS) + f"{'(deg)':>10}")
    for lo, hi in HORIZON_BUCKETS:
        hi = min(hi, chunk)
        if lo >= hi:
            continue
        mae = abs_error[lo:hi].sum(0) / np.maximum(counts[lo:hi].sum(), 1)
        row = "".join(f"{np.rad2deg(mae[j]):>10.4f}" for j in range(6))
        print(f"  {f'{lo}-{hi-1}':<10}{hi-lo:>7}{row}{np.rad2deg(mae.mean()):>10.4f}")
    print("\n  per-step MAE (rad) for the first 5 horizons:")
    for k in range(min(5, chunk)):
        mae = abs_error[k] / max(counts[k], 1)
        print(f"    k={k}  " + "  ".join(f"{name[:6]}={mae[j]:.5f}" for j, name in enumerate(JOINTS)))

    print("\n" + "=" * 78)
    print("Q12  PHASE BALANCE AND PER-PHASE MAE  (phase_id is the phase at frame t)")
    total_frames = max(phase_frames.sum(), 1)
    print(f"  {'phase':<12}{'frames':>9}{'share':>8}{'model MAE':>11}{'zero MAE':>10}{'ratio':>8}")
    print(f"  {'':<12}{'':>9}{'':>8}{'(deg)':>11}{'(deg)':>10}{'':>8}")
    for pid, name in enumerate(PHASE_NAMES):
        if phase_frames[pid] == 0:
            print(f"  {name:<12}{0:>9}{'-':>8}{'-':>11}{'-':>10}{'-':>8}")
            continue
        m = phase_error[pid].sum() / max(phase_elements[pid] * 6, 1)
        z = phase_zero[pid].sum() / max(phase_elements[pid] * 6, 1)
        print(f"  {name:<12}{int(phase_frames[pid]):>9}{100*phase_frames[pid]/total_frames:>7.1f}%"
              f"{np.rad2deg(m):>11.4f}{np.rad2deg(z):>10.4f}{m/max(z,1e-12):>8.3f}")
    print("\n  ratio near 1.0 means the policy is no better than standing still in that phase.")


# -------------------------------------------------------------- blind mode
def run_blind(args) -> None:
    """Q15: replace depth with a constant and see whether the loss moves."""
    device = resolve_device(args.device)
    policy, _, config, scale = _load_policy(args.model, device)
    loader, _ = _validation_loader(
        args.dataset, config, policy.config.chunk_size, args.seed, args.batch_size
    )
    modes = ("normal", "zero_depth", "shuffled_depth", "zero_proprio")
    totals = {name: 0.0 for name in modes}
    samples = 0
    with torch.inference_mode():
        for batch in loader:
            top = batch["top_depth"].to(device)
            side = batch["side_depth"].to(device)
            proprio = batch["proprio"].to(device)
            target = batch["delta_chunk"].to(device)
            mask = batch["chunk_mask"].to(device)
            if top.shape[0] < 2:
                continue
            roll = torch.roll(torch.arange(top.shape[0], device=device), 1)
            variants = {
                "normal": (top, side, proprio),
                "zero_depth": (torch.zeros_like(top), torch.zeros_like(side), proprio),
                "shuffled_depth": (top[roll], side[roll], proprio),
                "zero_proprio": (top, side, torch.zeros_like(proprio)),
            }
            for name, (t, s, p) in variants.items():
                _, metrics = depth_act_loss(policy(t, s, p), target, mask, delta_scale=scale)
                totals[name] += metrics["loss"] * top.shape[0]
            samples += top.shape[0]

    print("Q15  IS DEPTH ACTUALLY USED?   (validation loss under input corruption)")
    base = totals["normal"] / max(samples, 1)
    print(f"  {'input':<18}{'loss':>14}{'vs normal':>12}")
    for name in modes:
        value = totals[name] / max(samples, 1)
        print(f"  {name:<18}{value:>14.6e}{value / max(base, 1e-12):>11.2f}x")
    print("\n  shuffled_depth close to 1.0x means the policy is ignoring depth and")
    print("  replaying an average trajectory from proprio alone.")


# ----------------------------------------------------------- onpolicy mode
def run_onpolicy(args) -> None:
    """Q16: open-loop error ON the teacher trajectory vs closed-loop rollout."""
    device = resolve_device(args.device)
    policy, _, config, _ = _load_policy(args.model, device)
    environment = SO101BallBinsEnv(spawn_stage="stage1", max_steps=args.max_steps)
    top_renderer = TopDepthRenderer(environment.model, camera_name="top")
    side_renderer = TopDepthRenderer(environment.model, camera_name="side_depth")
    period = 1.0 / CONTROL_HZ
    open_loop, closed = [], []
    try:
        for episode in range(args.episodes):
            seed = args.seed_start + episode

            # --- open loop: the scripted teacher drives, the policy only watches
            environment.reset(seed=seed)
            place_nearby_scene(environment, seed=seed,
                               cube_jitter=args.cube_jitter, bin_jitter=args.bin_jitter)
            previous = {"pos": None}
            errors = []

            def watch(clean_action, executed_action, phase):
                qpos = environment.joint_positions()
                qvel = (np.zeros(6, dtype=np.float32) if previous["pos"] is None
                        else ((qpos - previous["pos"]) / np.float32(period)).astype(np.float32))
                previous["pos"] = qpos.copy()
                delta = predict_delta_chunk(
                    policy, top_renderer.render(environment.data),
                    side_renderer.render(environment.data), qpos, qvel,
                    depth_config=config, device=device,
                )
                goal = environment.control_target(clean_action)
                errors.append(np.abs((qpos + delta[0]) - goal))

            info = execute_waypoint_episode(environment, on_control_step=watch)
            open_loop.append((np.mean(errors, axis=0), bool(info["is_success"])))

            # --- closed loop: the policy drives, exactly as train/play do
            environment.reset(seed=seed)
            place_nearby_scene(environment, seed=seed,
                               cube_jitter=args.cube_jitter, bin_jitter=args.bin_jitter)
            ensemble = TemporalActionEnsembler(chunk_size=policy.config.chunk_size)
            velocity_estimator = JointVelocityEstimator(
                control_period_s=environment.model.opt.timestep * environment.frame_skip
            )
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                qpos = environment.joint_positions()
                qvel = velocity_estimator.update(qpos)
                delta = predict_delta_chunk(
                    policy, top_renderer.render(environment.data),
                    side_renderer.render(environment.data), qpos, qvel,
                    depth_config=config, device=device,
                )
                absolute = np.clip(qpos[None, :] + delta,
                                   environment.task_ctrl_low, environment.task_ctrl_high)
                target = ensemble.add_and_get(absolute)
                action = np.clip((target - environment.ctrl) / environment.action_scale, -1.0, 1.0)
                _, _, terminated, truncated, info = environment.step(action.astype(np.float32))
            closed.append({
                "success": bool(info.get("is_success")),
                "grasp": bool(info.get("has_grasped")),
                "lift": bool(info.get("has_lifted")),
                "phase": info.get("phase"),
            })
    finally:
        top_renderer.close()
        side_renderer.close()
        environment.close()

    errors = np.stack([row[0] for row in open_loop])
    print("Q16  TEACHER-FORCED (ON-DISTRIBUTION) vs CLOSED LOOP")
    print(f"\n  open-loop |predicted next goal - teacher goal| on the teacher trajectory")
    print(f"  {'joint':<15}{'MAE (rad)':>12}{'MAE (deg)':>12}")
    mean = errors.mean(axis=0)
    for j, name in enumerate(JOINTS):
        print(f"  {name:<15}{mean[j]:>12.5f}{np.rad2deg(mean[j]):>12.4f}")
    print(f"\n  closed-loop over {len(closed)} episodes:")
    print(f"    success {sum(c['success'] for c in closed)}/{len(closed)}"
          f"   grasp {sum(c['grasp'] for c in closed)}/{len(closed)}"
          f"   lift {sum(c['lift'] for c in closed)}/{len(closed)}")
    print(f"    phases reached: {[c['phase'] for c in closed]}")
    print("\n  small open-loop error together with zero closed-loop success is")
    print("  compounding error / covariate shift, not a loss-weighting problem.")


# --------------------------------------------------------------- qvel mode
def run_qvel(args) -> None:
    """Does the rollout feed the same velocity quantity the dataset stored?"""
    period = 1.0 / CONTROL_HZ
    finite, instant = [], []
    collected = 0
    seed = args.seed_start - 1
    while collected < args.episodes:
        seed += 1
        environment = SO101BallBinsEnv(spawn_stage="stage1", max_steps=args.max_steps)
        environment.reset(seed=seed)
        place_nearby_scene(environment, seed=seed,
                           cube_jitter=args.cube_jitter, bin_jitter=args.bin_jitter)
        positions, fd, iv = [], [], []

        def capture(clean_action, executed_action, phase):
            p = environment.joint_positions().astype(np.float64)
            fd.append(np.zeros(6) if not positions else (p - positions[-1]) / period)
            iv.append(environment.data.qvel[environment.qvel_ids].astype(np.float64).copy())
            positions.append(p)

        try:
            info = execute_waypoint_episode(environment, on_control_step=capture)
        except Exception:
            environment.close()
            continue
        environment.close()
        if not info["is_success"]:
            continue
        finite.append(np.array(fd))
        instant.append(np.array(iv))
        collected += 1

    fd = np.concatenate(finite)
    iv = np.concatenate(instant)
    print("PROPRIO INPUT CONSISTENCY")
    print("  dataset stores : (qpos[t] - qpos[t-1]) / control_period   "
          "collect_fixed_delta_depth_dataset.py:210")
    print("  rollout feeds  : data.qvel[qvel_ids]                      "
          "train_depth_act.py:229 / play_depth_act.py:73")
    print(f"\n  {len(fd)} control steps over {collected} successful episodes")
    print(f"\n  {'joint':<15}{'fd std':>10}{'qvel std':>10}{'ratio':>8}{'corr':>9}")
    for j, name in enumerate(JOINTS):
        a, b = fd[:, j], iv[:, j]
        print(f"  {name:<15}{a.std():>10.4f}{b.std():>10.4f}"
              f"{b.std()/max(a.std(),1e-12):>8.2f}{np.corrcoef(a,b)[0,1]:>9.3f}")
    print("\n  a correlation far below 1.0 means the policy is fed a different")
    print("  quantity at rollout than the one it was trained on.")


# ------------------------------------------------------------------- entry
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub, need_model=True, need_dataset=True):
        if need_model:
            sub.add_argument("--model", type=Path, required=True)
        if need_dataset:
            sub.add_argument("--dataset", type=Path, required=True)
        sub.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        sub.add_argument("--seed", type=int, default=31)
        sub.add_argument("--batch-size", type=int, default=16)

    add_common(subparsers.add_parser("data"))
    add_common(subparsers.add_parser("blind"))

    onpolicy = subparsers.add_parser("onpolicy")
    add_common(onpolicy, need_dataset=False)
    onpolicy.add_argument("--episodes", type=int, default=5)
    onpolicy.add_argument("--seed-start", type=int, default=30000)
    onpolicy.add_argument("--max-steps", type=int, default=1100)
    onpolicy.add_argument("--cube-jitter", type=float, default=0.020)
    onpolicy.add_argument("--bin-jitter", type=float, default=0.010)

    qvel = subparsers.add_parser("qvel")
    qvel.add_argument("--episodes", type=int, default=3)
    qvel.add_argument("--seed-start", type=int, default=10000)
    qvel.add_argument("--max-steps", type=int, default=1100)
    qvel.add_argument("--cube-jitter", type=float, default=0.020)
    qvel.add_argument("--bin-jitter", type=float, default=0.010)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    {"data": run_data, "blind": run_blind, "onpolicy": run_onpolicy, "qvel": run_qvel}[args.command](args)


if __name__ == "__main__":
    main()
