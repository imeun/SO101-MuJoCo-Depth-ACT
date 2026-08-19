from __future__ import annotations

import argparse
from dataclasses import fields
import math
from pathlib import Path
import time

import numpy as np

from collect_fixed_delta_depth_dataset import candidate_seeds, place_nearby_scene, scene_provenance
from delta_depth_dataset import PHASE_TO_ID, DeltaDepthEpisodeWriter, atomic_json, load_delta_manifest
from so101_ball_bins_env import SO101BallBinsEnv
from so101_depth import DepthConfig, TopDepthRenderer, depth_to_millimetres
from so101_depth_act import (
    JointVelocityEstimator,
    TemporalActionEnsembler,
    load_depth_act_checkpoint,
    predict_delta_chunk,
)
from so101_depth_act_dagger import InterventionController, StateAwareWaypointTeacher
from train_depth_act import resolve_device


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative value")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive value")
    return parsed


def _checkpoint(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"ACT checkpoint does not exist: {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect MuJoCo DAgger corrections for the dual-depth SO-101 ACT policy."
    )
    parser.add_argument("--policy", type=_checkpoint)
    parser.add_argument("--teacher-only", action="store_true")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--episodes", required=True, type=_positive_int)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--max-attempts", type=_positive_int)
    parser.add_argument("--max-steps", type=_positive_int, default=1600)
    parser.add_argument("--cube-jitter", type=_nonnegative_float, default=0.020)
    parser.add_argument("--bin-jitter", type=_nonnegative_float, default=0.010)
    parser.add_argument("--trigger-threshold", type=_nonnegative_float, default=0.35)
    parser.add_argument("--release-threshold", type=_nonnegative_float, default=0.12)
    parser.add_argument("--minimum-teacher-steps", type=_positive_int, default=20)
    parser.add_argument("--ensemble-decay", type=_nonnegative_float, default=0.08)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--playback-speed", type=_positive_float, default=1.0)
    args = parser.parse_args(argv)
    if not args.teacher_only and args.policy is None:
        parser.error("--policy is required unless --teacher-only is set")
    if args.release_threshold >= args.trigger_threshold:
        parser.error("--release-threshold must be smaller than --trigger-threshold")
    if args.max_attempts is None:
        args.max_attempts = args.episodes * 5
    return args


def _depth_config(payload: dict) -> DepthConfig:
    allowed = {field.name for field in fields(DepthConfig)}
    return DepthConfig(**{key: value for key, value in payload["depth_config"].items() if key in allowed})


def _stack(values: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    return np.stack(values).astype(dtype, copy=False)


def collect_episodes(args: argparse.Namespace) -> dict:
    output_root = Path(args.out_dir).expanduser().resolve()
    device = resolve_device(args.device)
    policy = None
    depth_config = DepthConfig()
    if not args.teacher_only:
        policy, payload = load_depth_act_checkpoint(str(args.policy), map_location="cpu")
        policy = policy.to(device).eval()
        depth_config = _depth_config(payload)

    environment = SO101BallBinsEnv(spawn_stage="stage1", max_steps=args.max_steps)
    initial_conditions = "fixed" if args.cube_jitter == 0.0 and args.bin_jitter == 0.0 else "near_position"
    writer = DeltaDepthEpisodeWriter(
        output_root,
        provenance=scene_provenance(environment),
        initial_conditions=initial_conditions,
    )
    existing = load_delta_manifest(output_root)
    if len(existing["episodes"]) > args.episodes:
        environment.close()
        raise ValueError("output dataset already contains more than --episodes")

    top_renderer = TopDepthRenderer(environment.model, camera_name="top")
    side_renderer = TopDepthRenderer(environment.model, camera_name="side_depth")
    seeds = candidate_seeds(args.seed, {entry["seed"] for entry in existing["episodes"]})
    attempts = 0
    failures: list[dict] = []
    duplicate_count = 0
    current_teacher_steps = 0
    current_total_steps = 0
    current_interventions = 0
    control_period_s = environment.model.opt.timestep * environment.frame_skip
    viewer_context = None
    viewer = None
    if args.visualize:
        import mujoco.viewer

        viewer_context = mujoco.viewer.launch_passive(environment.model, environment.data)
        viewer = viewer_context.__enter__()

    try:
        while len(writer.manifest["episodes"]) < args.episodes and attempts < args.max_attempts:
            seed = next(seeds)
            attempts += 1
            environment.reset(seed=seed)
            initial_joint_pos = environment.joint_positions().copy()
            cube_position, bin_position = place_nearby_scene(
                environment,
                seed=seed,
                cube_jitter=args.cube_jitter,
                bin_jitter=args.bin_jitter,
            )
            teacher = StateAwareWaypointTeacher(environment)
            intervention = InterventionController(
                joint_scale=np.full(6, environment.action_scale, dtype=np.float64),
                trigger_threshold=args.trigger_threshold,
                release_threshold=args.release_threshold,
                minimum_teacher_steps=args.minimum_teacher_steps,
            )
            ensemble = None
            if policy is not None:
                ensemble = TemporalActionEnsembler(
                    chunk_size=policy.config.chunk_size,
                    decay=args.ensemble_decay,
                )
            velocity = JointVelocityEstimator(control_period_s=control_period_s)

            top_frames: list[np.ndarray] = []
            side_frames: list[np.ndarray] = []
            joint_positions: list[np.ndarray] = []
            joint_velocities: list[np.ndarray] = []
            teacher_goals: list[np.ndarray] = []
            delta_targets: list[np.ndarray] = []
            teacher_actions: list[np.ndarray] = []
            executed_actions: list[np.ndarray] = []
            phases: list[int] = []
            timestamps: list[float] = []
            ever_success = False
            failure_reason: str | None = None

            try:
                for step in range(args.max_steps):
                    qpos = environment.joint_positions().astype(np.float32, copy=True)
                    qvel = velocity.update(qpos)
                    top_depth = top_renderer.render(environment.data)
                    side_depth = side_renderer.render(environment.data)
                    teacher_action, teacher_goal, phase = teacher.command()

                    previous_intervention = intervention.active
                    if policy is None:
                        use_teacher = True
                    else:
                        delta_chunk = predict_delta_chunk(
                            policy,
                            top_depth,
                            side_depth,
                            qpos,
                            qvel,
                            depth_config=depth_config,
                            device=device,
                        )
                        policy_target = np.clip(
                            qpos + delta_chunk[0],
                            environment.task_ctrl_low,
                            environment.task_ctrl_high,
                        )
                        use_teacher = intervention.update(policy_target, teacher_goal)
                        if use_teacher != previous_intervention:
                            ensemble.reset()

                    if use_teacher:
                        executed_action = teacher_action.copy()
                    else:
                        absolute_chunk = np.clip(
                            qpos[None, :] + delta_chunk,
                            environment.task_ctrl_low,
                            environment.task_ctrl_high,
                        )
                        target = ensemble.add_and_get(absolute_chunk)
                        executed_action = np.clip(
                            (target - environment.ctrl) / environment.action_scale,
                            -1.0,
                            1.0,
                        ).astype(np.float32)

                    top_frames.append(depth_to_millimetres(top_depth))
                    side_frames.append(depth_to_millimetres(side_depth))
                    joint_positions.append(qpos)
                    joint_velocities.append(qvel.copy())
                    teacher_goals.append(teacher_goal.copy())
                    delta_targets.append((teacher_goal - qpos).astype(np.float32))
                    teacher_actions.append(teacher_action.copy())
                    executed_actions.append(executed_action.copy())
                    phases.append(PHASE_TO_ID[phase])
                    timestamps.append(step * control_period_s)

                    _, _, _, truncated, info = environment.step(executed_action)
                    ever_success |= bool(info["is_success"])
                    teacher.observe(info)
                    if viewer is not None:
                        if not viewer.is_running():
                            raise KeyboardInterrupt
                        viewer.sync()
                        if step % 30 == 0:
                            error = 0.0 if policy is None else intervention.last_error
                            print(
                                f"seed={seed} step={step} phase={phase} "
                                f"teacher={use_teacher} action_error={error:.3f}",
                                flush=True,
                            )
                        time.sleep(control_period_s / args.playback_speed)
                    if info["is_failure"]:
                        failure_reason = "task_failed"
                        break
                    if teacher.complete:
                        break
                    if truncated:
                        failure_reason = "max_steps"
                        break
            except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                failure_reason = f"{type(error).__name__}: {error}"

            success = ever_success and teacher.complete and failure_reason is None
            if not success:
                failures.append({"seed": seed, "reason": failure_reason or "teacher_incomplete"})
                print(
                    f"saved={len(writer.manifest['episodes'])}/{args.episodes} "
                    f"attempt={attempts}/{args.max_attempts} seed={seed} success=False "
                    f"reason={failures[-1]['reason']}",
                    flush=True,
                )
                continue

            arrays = (
                _stack(top_frames, np.uint16),
                _stack(side_frames, np.uint16),
                _stack(joint_positions, np.float32),
                _stack(joint_velocities, np.float32),
                _stack(teacher_goals, np.float32),
                _stack(delta_targets, np.float32),
                _stack(teacher_actions, np.float32),
                _stack(executed_actions, np.float32),
                np.asarray(phases, dtype=np.uint8),
                np.asarray(timestamps, dtype=np.float64),
            )
            try:
                writer.save_episode(
                    *arrays,
                    seed=seed,
                    initial_joint_pos=initial_joint_pos,
                    cube_position=cube_position,
                    bin_position=bin_position,
                )
            except ValueError as error:
                if "duplicate trajectory" not in str(error):
                    raise
                duplicate_count += 1
                continue

            episode_teacher_steps = len(timestamps) if policy is None else intervention.teacher_steps
            current_teacher_steps += episode_teacher_steps
            current_total_steps += len(timestamps)
            current_interventions += 1 if policy is None else intervention.intervention_count
            print(
                f"saved={len(writer.manifest['episodes'])}/{args.episodes} "
                f"attempt={attempts}/{args.max_attempts} seed={seed} frames={len(timestamps)} "
                f"teacher_fraction={episode_teacher_steps / len(timestamps):.3f} "
                f"interventions={1 if policy is None else intervention.intervention_count}",
                flush=True,
            )
    finally:
        if viewer_context is not None:
            viewer_context.__exit__(None, None, None)
        top_renderer.close()
        side_renderer.close()
        environment.close()

    summary = {
        "mode": "teacher_only" if args.teacher_only else "dagger",
        "policy": None if args.policy is None else str(args.policy),
        "output_dataset": str(output_root),
        "requested_episodes": args.episodes,
        "saved_episodes": len(writer.manifest["episodes"]),
        "collected_now": len(writer.manifest["episodes"]) - len(existing["episodes"]),
        "attempts": attempts,
        "failed_attempts": len(failures),
        "duplicate_trajectories": duplicate_count,
        "teacher_fraction": current_teacher_steps / max(current_total_steps, 1),
        "interventions": current_interventions,
        "trigger_threshold_action_steps": args.trigger_threshold,
        "release_threshold_action_steps": args.release_threshold,
        "failures": failures,
    }
    atomic_json(output_root / "dagger_summary.json", summary)
    print(summary, flush=True)
    if len(writer.manifest["episodes"]) < args.episodes:
        raise RuntimeError(
            f"collected {len(writer.manifest['episodes'])}/{args.episodes} episodes "
            f"after {attempts} attempts"
        )
    return summary


def main(argv: list[str] | None = None) -> None:
    collect_episodes(parse_args(argv))


if __name__ == "__main__":
    main()
