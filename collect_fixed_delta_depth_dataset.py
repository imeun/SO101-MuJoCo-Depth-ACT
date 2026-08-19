from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from delta_depth_dataset import CONTROL_HZ, PHASE_TO_ID, DeltaDepthEpisodeWriter, atomic_json, load_delta_manifest
from play_waypoint_teacher import (
    BASE_BIN_POSITION,
    BASE_CUBE_POSITION,
    execute_waypoint_episode,
    randomize_bin_position,
    randomize_cube_pose,
)
from so101_ball_bins_env import SO101BallBinsEnv
from so101_depth import TopDepthRenderer, depth_to_millimetres


STATIC_PHASES = {"hold", "settle", "home_hold"}


def scene_provenance(environment: SO101BallBinsEnv) -> dict:
    xml_path = environment.xml_path.resolve()
    cameras: dict[str, dict] = {}
    for name in ("top", "side_depth"):
        camera_id = mujoco.mj_name2id(environment.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera_id < 0:
            raise ValueError(f"camera {name!r} is missing")
        intrinsic = environment.model.cam_intrinsic[camera_id]
        cameras[name] = {
            "resolution": environment.model.cam_resolution[camera_id].astype(int).tolist(),
            "focalpixel": intrinsic[:2].astype(float).tolist(),
            "principalpixel_offset": intrinsic[2:].astype(float).tolist(),
            "fovy_degrees": float(environment.model.cam_fovy[camera_id]),
        }
    return {
        "scene_xml_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
        "cameras": cameras,
    }


def candidate_seeds(start: int, used: set[int]):
    seed = int(start)
    while True:
        if seed not in used:
            yield seed
        seed += 1


def place_nearby_scene(
    environment: SO101BallBinsEnv,
    *,
    seed: int,
    cube_jitter: float,
    bin_jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place a deterministic nearby cube/bin pair before waypoint IK is built."""
    if cube_jitter < 0.0 or bin_jitter < 0.0:
        raise ValueError("scene jitter must be non-negative")
    bin_body_id = mujoco.mj_name2id(environment.model, mujoco.mjtObj.mjOBJ_BODY, "square_bin")
    environment.model.body_pos[bin_body_id, :2] = BASE_BIN_POSITION
    randomize_cube_pose(
        environment,
        np.random.default_rng(seed),
        xy_jitter=cube_jitter,
        yaw_range_degrees=0.0,
    )
    if bin_jitter > 0.0:
        randomize_bin_position(
            environment,
            np.random.default_rng(seed ^ 0x5F3759DF),
            jitter=bin_jitter,
        )
    else:
        mujoco.mj_forward(environment.model, environment.data)
    return environment._cube_position().copy(), environment._target_position().copy()


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


def _correlation(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("must be within [0, 1)")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect clean nearby-scene SO101 episodes with cumulative-delta-ready labels."
    )
    parser.add_argument("--episodes", type=_positive_int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-attempts", type=_positive_int)
    parser.add_argument("--cube-jitter", type=_nonnegative_float, default=0.020)
    parser.add_argument("--bin-jitter", type=_nonnegative_float, default=0.010)
    parser.add_argument("--perturbation-std", type=_nonnegative_float, default=0.0)
    parser.add_argument("--perturbation-limit", type=_nonnegative_float, default=0.0)
    parser.add_argument("--perturbation-correlation", type=_correlation, default=0.90)
    args = parser.parse_args(argv)
    if args.perturbation_std > args.perturbation_limit:
        parser.error("--perturbation-std must not exceed --perturbation-limit")
    if args.max_attempts is None:
        args.max_attempts = args.episodes * 5
    return args


class SmoothActionPerturbation:
    def __init__(self, *, seed: int, std: float, limit: float, correlation: float):
        if std < 0.0 or limit < 0.0 or std > limit or not 0.0 <= correlation < 1.0:
            raise ValueError("invalid perturbation configuration")
        self.rng = np.random.default_rng(seed)
        self.std = float(std)
        self.limit = float(limit)
        self.correlation = float(correlation)
        self.state = np.zeros(6, dtype=np.float64)

    def __call__(self, clean_action: np.ndarray, phase: str, step: int) -> np.ndarray:
        del step
        clean = np.asarray(clean_action, dtype=np.float64)
        if clean.shape != (6,) or not np.all(np.isfinite(clean)):
            raise ValueError("clean_action must be finite with shape (6,)")
        if phase in STATIC_PHASES or self.std == 0.0:
            self.state.fill(0.0)
            return clean.astype(np.float32, copy=True)
        innovation_scale = self.std * math.sqrt(1.0 - self.correlation**2)
        self.state = self.correlation * self.state + self.rng.normal(0.0, innovation_scale, size=6)
        self.state = np.clip(self.state, -self.limit, self.limit)
        self.state[5] = 0.0
        return np.clip(clean + self.state, -1.0, 1.0).astype(np.float32)


def collect_episodes(args: argparse.Namespace) -> dict:
    output_root = Path(args.out_dir).resolve()
    environment = SO101BallBinsEnv(spawn_stage="stage1", max_steps=1100)
    initial_conditions = (
        "fixed" if args.cube_jitter == 0.0 and args.bin_jitter == 0.0 else "near_position"
    )
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
    failures: list[dict] = []
    duplicate_count = 0
    attempts = 0
    seeds = candidate_seeds(args.seed, {entry["seed"] for entry in existing["episodes"]})
    try:
        bin_body_id = mujoco.mj_name2id(environment.model, mujoco.mjtObj.mjOBJ_BODY, "square_bin")
        while len(writer.manifest["episodes"]) < args.episodes and attempts < args.max_attempts:
            seed = next(seeds)
            attempts += 1
            environment.model.body_pos[bin_body_id, :2] = BASE_BIN_POSITION
            environment.reset(seed=seed)
            initial_joint_pos = environment.joint_positions().copy()
            cube_position, bin_position = place_nearby_scene(
                environment,
                seed=seed,
                cube_jitter=args.cube_jitter,
                bin_jitter=args.bin_jitter,
            )
            perturb = SmoothActionPerturbation(
                seed=seed,
                std=args.perturbation_std,
                limit=args.perturbation_limit,
                correlation=args.perturbation_correlation,
            )

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
            control_period_s = 1.0 / CONTROL_HZ

            def capture(clean_action: np.ndarray, executed_action: np.ndarray, phase: str) -> None:
                if phase not in PHASE_TO_ID:
                    raise ValueError(f"unknown scripted phase: {phase}")
                joint_pos = environment.joint_positions().astype(np.float32, copy=True)
                joint_velocity = (
                    np.zeros(6, dtype=np.float32)
                    if not joint_positions
                    else ((joint_pos - joint_positions[-1]) / np.float32(control_period_s)).astype(np.float32)
                )
                teacher_goal = environment.control_target(clean_action).astype(np.float32, copy=True)
                top_frames.append(depth_to_millimetres(top_renderer.render(environment.data)))
                side_frames.append(depth_to_millimetres(side_renderer.render(environment.data)))
                joint_positions.append(joint_pos)
                joint_velocities.append(joint_velocity)
                teacher_goals.append(teacher_goal)
                delta_targets.append((teacher_goal - joint_pos).astype(np.float32))
                teacher_actions.append(clean_action.astype(np.float32, copy=True))
                executed_actions.append(executed_action.astype(np.float32, copy=True))
                phases.append(PHASE_TO_ID[phase])
                timestamps.append(len(timestamps) * control_period_s)

            try:
                info = execute_waypoint_episode(
                    environment,
                    action_transform=perturb,
                    on_control_step=capture,
                )
                success = bool(info["is_success"])
                reason = "task_failed"
            except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                success = False
                reason = f"{type(error).__name__}: {error}"
            if not success:
                failures.append({"seed": seed, "reason": reason})
                print(
                    f"saved={len(writer.manifest['episodes'])}/{args.episodes} "
                    f"attempt={attempts}/{args.max_attempts} seed={seed} success=False",
                    flush=True,
                )
                continue

            arrays = (
                np.stack(top_frames).astype(np.uint16, copy=False),
                np.stack(side_frames).astype(np.uint16, copy=False),
                np.stack(joint_positions).astype(np.float32, copy=False),
                np.stack(joint_velocities).astype(np.float32, copy=False),
                np.stack(teacher_goals).astype(np.float32, copy=False),
                np.stack(delta_targets).astype(np.float32, copy=False),
                np.stack(teacher_actions).astype(np.float32, copy=False),
                np.stack(executed_actions).astype(np.float32, copy=False),
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
                print(
                    f"saved={len(writer.manifest['episodes'])}/{args.episodes} "
                    f"attempt={attempts}/{args.max_attempts} seed={seed} duplicate=True",
                    flush=True,
                )
                continue
            print(
                f"saved={len(writer.manifest['episodes'])}/{args.episodes} "
                f"attempt={attempts}/{args.max_attempts} seed={seed} "
                f"frames={len(timestamps)} success=True",
                flush=True,
            )
    finally:
        top_renderer.close()
        side_renderer.close()
        environment.close()

    summary = {
        "dataset_root": str(output_root),
        "requested_episodes": args.episodes,
        "saved_episodes": len(writer.manifest["episodes"]),
        "attempts": attempts,
        "failed_attempts": len(failures),
        "duplicate_trajectories": duplicate_count,
        "failures": failures,
        "control_hz": CONTROL_HZ,
        "target_type": "immediate_joint_delta",
        "initial_conditions": initial_conditions,
        "cube_jitter": args.cube_jitter,
        "bin_jitter": args.bin_jitter,
        "perturbation_std": args.perturbation_std,
        "perturbation_limit": args.perturbation_limit,
        "perturbation_correlation": args.perturbation_correlation,
        "provenance": writer.manifest["provenance"],
    }
    atomic_json(output_root / "collection_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if summary["saved_episodes"] < args.episodes:
        raise RuntimeError(
            f"collected {summary['saved_episodes']}/{args.episodes} unique successful episodes "
            f"after {attempts} attempts"
        )
    return summary


def main(argv: list[str] | None = None) -> None:
    collect_episodes(parse_args(argv))


if __name__ == "__main__":
    main()
