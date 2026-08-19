from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from delta_depth_dataset import load_delta_manifest
from delta_depth_dataset import PHASE_NAMES
from play_waypoint_teacher import configure_viewer_camera, playback_delay_seconds
from so101_ball_bins_env import SO101BallBinsEnv


REQUIRED_ARRAYS = {
    "top_depth_mm",
    "side_depth_mm",
    "joint_pos",
    "joint_velocity",
    "teacher_goal_pos",
    "delta_target_rad",
    "teacher_action",
    "executed_action",
    "phase_id",
    "timestamp_s",
}


def load_delta_episode(
    dataset: str | Path,
    episode_index: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    root = Path(dataset).resolve()
    manifest = load_delta_manifest(root)
    entry = next(
        (item for item in manifest["episodes"] if item["index"] == episode_index),
        None,
    )
    if entry is None:
        raise ValueError(f"episode index {episode_index} does not exist in {root}")

    with np.load(root / entry["file"], allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS.difference(archive.files)
        if missing:
            raise ValueError(f"episode is missing arrays: {sorted(missing)}")
        arrays = {name: archive[name].copy() for name in REQUIRED_ARRAYS}
    if arrays["executed_action"].shape != (entry["frames"], 6):
        raise ValueError("episode executed_action has an incompatible shape")
    return dict(entry), arrays


def replay_delta_episode(
    dataset: str | Path,
    *,
    episode_index: int = 0,
    render: bool = True,
    view_camera: str = "free",
    playback_speed: float = 1.0,
    hold_final_seconds: float = 3.0,
) -> dict:
    if view_camera not in {"free", "top", "side_depth"}:
        raise ValueError("view_camera must be one of: free, top, side_depth")
    if not math.isfinite(hold_final_seconds) or hold_final_seconds < 0.0:
        raise ValueError("hold_final_seconds must be a finite non-negative value")

    entry, arrays = load_delta_episode(dataset, episode_index)
    playback_delay_seconds(0.002, 17, playback_speed)
    env = SO101BallBinsEnv(
        spawn_stage="stage1",
        max_steps=max(int(entry["frames"]) + 50, 1_100),
    )
    viewer_context = nullcontext(None)
    frames_replayed = 0
    maximum_joint_error = 0.0
    final_info: dict = {}
    previous_phase_id: int | None = None
    try:
        bin_body_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "square_bin",
        )
        env.model.body_pos[bin_body_id, :2] = np.asarray(entry["bin_position"][:2])
        env.reset(seed=int(entry["seed"]))

        cube_position = np.asarray(entry["cube_position"], dtype=np.float64)
        env.data.qpos[env.cube_qpos_id : env.cube_qpos_id + 7] = np.array(
            [*cube_position, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        env.data.qvel[env.cube_qvel_id : env.cube_qvel_id + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)

        initial_error = float(
            np.max(
                np.abs(
                    env.joint_positions()
                    - np.asarray(entry["initial_joint_pos"], dtype=np.float32)
                )
            )
        )
        if initial_error > 1e-5:
            raise ValueError(
                "current MuJoCo home pose does not match the dataset "
                f"(max error={initial_error:.6g} rad)"
            )

        if render:
            viewer_context = mujoco.viewer.launch_passive(env.model, env.data)
        with viewer_context as viewer:
            if viewer is not None:
                configure_viewer_camera(viewer, env.model, view_camera)

            for frame_index, action in enumerate(arrays["executed_action"]):
                if viewer is not None and not viewer.is_running():
                    break
                joint_error = float(
                    np.max(
                        np.abs(
                            env.joint_positions()
                            - arrays["joint_pos"][frame_index]
                        )
                    )
                )
                maximum_joint_error = max(maximum_joint_error, joint_error)

                phase_id = int(arrays["phase_id"][frame_index])
                if phase_id != previous_phase_id:
                    print(
                        f"frame={frame_index}/{entry['frames']} "
                        f"phase={PHASE_NAMES[phase_id]}",
                        flush=True,
                    )
                    previous_phase_id = phase_id

                _, _, _, _, final_info = env.step(action)
                frames_replayed += 1
                if viewer is not None:
                    viewer.sync()
                    time.sleep(
                        playback_delay_seconds(
                            env.model.opt.timestep,
                            env.frame_skip,
                            playback_speed,
                        )
                    )

            if viewer is not None and viewer.is_running() and hold_final_seconds > 0.0:
                deadline = time.monotonic() + hold_final_seconds
                while viewer.is_running() and time.monotonic() < deadline:
                    viewer.sync()
                    time.sleep(0.02)
    finally:
        env.close()

    result = {
        "dataset": str(Path(dataset).resolve()),
        "episode_index": int(episode_index),
        "seed": int(entry["seed"]),
        "frames": int(entry["frames"]),
        "frames_replayed": frames_replayed,
        "view_camera": view_camera,
        "max_joint_replay_error_rad": maximum_joint_error,
        "has_grasped": bool(final_info.get("has_grasped", False)),
        "has_lifted": bool(final_info.get("has_lifted", False)),
        "is_success": bool(final_info.get("is_success", False)),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive value")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative value")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one saved fixed-delta episode in MuJoCo."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--view-camera",
        choices=("free", "top", "side_depth"),
        default="free",
    )
    parser.add_argument("--playback-speed", type=_positive_float, default=1.0)
    parser.add_argument("--hold-final-seconds", type=_nonnegative_float, default=3.0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    replay_delta_episode(
        args.dataset,
        episode_index=args.episode_index,
        render=not args.headless,
        view_camera=args.view_camera,
        playback_speed=args.playback_speed,
        hold_final_seconds=args.hold_final_seconds,
    )


if __name__ == "__main__":
    main()
