from __future__ import annotations

import argparse
from contextlib import nullcontext
import time
from typing import Callable

import mujoco
import mujoco.viewer
import numpy as np

from so101_ball_bins_env import SO101BallBinsEnv


BASE_BIN_POSITION = np.array([0.310, -0.050], dtype=np.float64)
BASE_CUBE_POSITION = np.array([0.320, 0.155], dtype=np.float64)
StepCallback = Callable[[np.ndarray, str], None]
ActionTransform = Callable[[np.ndarray, str, int], np.ndarray]
ControlStepCallback = Callable[[np.ndarray, np.ndarray, str], None]


def configure_overview_camera(viewer) -> None:
    """Frame both physical camera mounts and the complete robot workspace."""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0.22, 0.0, 0.30]
    viewer.cam.distance = 1.25
    viewer.cam.azimuth = -40.0
    viewer.cam.elevation = -18.0


def configure_viewer_camera(viewer, model: mujoco.MjModel, view_camera: str) -> None:
    if view_camera == "free":
        configure_overview_camera(viewer)
        return
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, view_camera)
    if camera_id < 0:
        raise ValueError(f"camera {view_camera!r} does not exist in the MuJoCo model")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id


def playback_delay_seconds(timestep: float, frame_skip: int, playback_speed: float) -> float:
    if not np.isfinite(playback_speed) or playback_speed <= 0.0:
        raise ValueError("playback_speed must be a finite positive value")
    return float(timestep * frame_skip / playback_speed)


def _solve_position_ik(env: SO101BallBinsEnv, start: np.ndarray, target: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(env.model)
    mujoco.mj_resetData(env.model, data)
    data.qpos[env.cube_qpos_id : env.cube_qpos_id + 7] = env.data.qpos[
        env.cube_qpos_id : env.cube_qpos_id + 7
    ]
    result = start.copy()
    for _ in range(2_000):
        data.qpos[env.qpos_ids] = result
        mujoco.mj_forward(env.model, data)
        error = target - data.site_xpos[env.gripper_site_id]
        if np.linalg.norm(error) < 1e-5:
            break
        jacobian_position = np.zeros((3, env.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, env.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            env.model,
            data,
            jacobian_position,
            jacobian_rotation,
            env.gripper_site_id,
        )
        jacobian = jacobian_position[:, env.qvel_ids[:5]]
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1e-4 * np.eye(3),
            error,
        )
        result[:5] = np.clip(
            result[:5] + 0.1 * delta,
            env.task_ctrl_low[:5],
            env.task_ctrl_high[:5],
        )
    else:
        raise RuntimeError("inverse kinematics did not converge")
    return result


def build_waypoints(env: SO101BallBinsEnv) -> dict[str, np.ndarray]:
    cube = env._cube_position()
    goal = env._target_position()

    opened = env.home_qpos.copy()
    opened[5] = env.GRIPPER_OPEN_POSITION
    pregrasp = _solve_position_ik(env, opened, env._pregrasp_position())
    pregrasp[5] = env.GRIPPER_OPEN_POSITION
    grasp = _solve_position_ik(env, pregrasp, cube)
    grasp[5] = env.GRIPPER_OPEN_POSITION
    close = grasp.copy()
    close[5] = 0.10
    hold = close.copy()
    lift = _solve_position_ik(env, grasp, np.array([cube[0], cube[1], 0.145]))
    lift[5] = 0.10
    above_bin = _solve_position_ik(env, lift, np.array([goal[0], goal[1], 0.155]))
    above_bin[5] = 0.10
    release = above_bin.copy()
    release[5] = env.GRIPPER_OPEN_POSITION
    settle = release.copy()
    retreat = _solve_position_ik(env, release, np.array([goal[0], goal[1], 0.230]))
    retreat[5] = env.GRIPPER_OPEN_POSITION
    home = env.home_qpos.copy()
    home_hold = home.copy()
    return {
        "open": opened,
        "pregrasp": pregrasp,
        "grasp": grasp,
        "close": close,
        "hold": hold,
        "lift": lift,
        "bin": above_bin,
        "release": release,
        "settle": settle,
        "retreat": retreat,
        "home": home,
        "home_hold": home_hold,
    }


def randomize_bin_position(
    env: SO101BallBinsEnv,
    rng: np.random.Generator,
    *,
    jitter: float = 0.03,
) -> np.ndarray:
    if jitter < 0.0 or not np.isfinite(jitter):
        raise ValueError("jitter must be a finite non-negative value")
    bin_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "square_bin")
    cube_xy = env._cube_position()[:2]
    for _ in range(1_000):
        candidate = BASE_BIN_POSITION + rng.uniform(-jitter, jitter, size=2)
        if np.linalg.norm(candidate - cube_xy) <= 0.10:
            continue
        env.model.body_pos[bin_body_id, :2] = candidate
        mujoco.mj_forward(env.model, env.data)
        return candidate.astype(np.float64, copy=True)
    raise RuntimeError("could not sample a collision-free bin position")


def randomize_cube_pose(
    env: SO101BallBinsEnv,
    rng: np.random.Generator,
    *,
    xy_jitter: float = 0.02,
    yaw_range_degrees: float = 0.0,
) -> np.ndarray:
    if xy_jitter < 0.0 or not np.isfinite(xy_jitter):
        raise ValueError("xy_jitter must be a finite non-negative value")
    if yaw_range_degrees < 0.0 or not np.isfinite(yaw_range_degrees):
        raise ValueError("yaw_range_degrees must be a finite non-negative value")
    xy = BASE_CUBE_POSITION + rng.uniform(-xy_jitter, xy_jitter, size=2)
    yaw = np.deg2rad(rng.uniform(-yaw_range_degrees, yaw_range_degrees))
    pose = np.array(
        [xy[0], xy[1], 0.0336, np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
        dtype=np.float64,
    )
    env.data.qpos[env.cube_qpos_id : env.cube_qpos_id + 7] = pose
    env.data.qvel[env.cube_qvel_id : env.cube_qvel_id + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    return pose.copy()


def execute_waypoint_episode(
    env: SO101BallBinsEnv,
    *,
    on_step: StepCallback | None = None,
    action_transform: ActionTransform | None = None,
    on_control_step: ControlStepCallback | None = None,
    viewer=None,
    playback_speed: float = 1.0,
) -> dict:
    waypoints = build_waypoints(env)
    _, info = env._get_obs(), env._info()
    schedule = (
        ("open", 40),
        ("pregrasp", 80),
        ("grasp", 80),
        ("close", 45),
        ("hold", 5),
        ("lift", 120),
        ("bin", 160),
        ("release", 240),
        ("settle", 20),
        ("retreat", 60),
        ("home", 90),
        ("home_hold", 30),
    )
    control_step = 0
    for name, duration in schedule:
        start = env.ctrl.copy()
        target = waypoints[name]
        for index in range(duration):
            ratio = (index + 1) / duration
            desired = start + ratio * (target - start)
            action = np.clip((desired - env.ctrl) / env.action_scale, -1.0, 1.0).astype(np.float32)
            if on_step is not None:
                on_step(action.copy(), name)
            executed_action = (
                action.copy()
                if action_transform is None
                else np.asarray(action_transform(action.copy(), name, control_step), dtype=np.float32)
            )
            if (
                executed_action.shape != (6,)
                or not np.all(np.isfinite(executed_action))
                or np.any(executed_action < -1.0)
                or np.any(executed_action > 1.0)
            ):
                raise ValueError("action_transform must return a finite action in [-1, 1] with shape (6,)")
            if on_control_step is not None:
                on_control_step(action.copy(), executed_action.copy(), name)
            _, _, terminated, truncated, info = env.step(executed_action)
            control_step += 1
            if viewer is not None:
                viewer.sync()
                time.sleep(
                    playback_delay_seconds(env.model.opt.timestep, env.frame_skip, playback_speed)
                )
            if truncated or info["is_failure"]:
                return info
    return info


def run_waypoint_episode(
    *,
    seed: int = 7,
    render: bool = True,
    spawn_stage: str = "stage1",
    randomize_bin: bool = False,
    bin_jitter: float = 0.03,
    cube_jitter: float = 0.0,
    yaw_range_degrees: float = 0.0,
    view_camera: str = "free",
    playback_speed: float = 1.0,
    hold_final_seconds: float = 3.0,
) -> dict:
    if view_camera not in {"free", "top", "side_depth"}:
        raise ValueError("view_camera must be one of: free, top, side_depth")
    if not np.isfinite(hold_final_seconds) or hold_final_seconds < 0.0:
        raise ValueError("hold_final_seconds must be a finite non-negative value")
    playback_delay_seconds(0.002, 17, playback_speed)
    env = SO101BallBinsEnv(spawn_stage=spawn_stage, max_steps=1100)
    viewer_context = nullcontext(None)
    try:
        env.reset(seed=seed)
        if cube_jitter > 0.0 or yaw_range_degrees > 0.0:
            randomize_cube_pose(
                env,
                np.random.default_rng(seed),
                xy_jitter=cube_jitter,
                yaw_range_degrees=yaw_range_degrees,
            )
        if randomize_bin:
            randomize_bin_position(env, np.random.default_rng(seed ^ 0x5F3759DF), jitter=bin_jitter)
        if render:
            viewer_context = mujoco.viewer.launch_passive(env.model, env.data)
        with viewer_context as viewer:
            if viewer is not None:
                configure_viewer_camera(viewer, env.model, view_camera)
            result = execute_waypoint_episode(
                env,
                viewer=viewer,
                playback_speed=playback_speed,
            )
            if viewer is not None and hold_final_seconds > 0.0:
                deadline = time.monotonic() + hold_final_seconds
                while viewer.is_running() and time.monotonic() < deadline:
                    viewer.sync()
                    time.sleep(0.02)
            return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--spawn-stage", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument("--randomize-bin", action="store_true")
    parser.add_argument("--bin-jitter", type=float, default=0.03)
    parser.add_argument("--cube-jitter", type=float, default=0.0)
    parser.add_argument("--yaw-range-deg", type=float, default=0.0)
    parser.add_argument("--view-camera", choices=["free", "top", "side_depth"], default="free")
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--hold-final-seconds", type=float, default=3.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    result = run_waypoint_episode(
        seed=args.seed,
        render=not args.headless,
        spawn_stage=args.spawn_stage,
        randomize_bin=args.randomize_bin,
        bin_jitter=args.bin_jitter,
        cube_jitter=args.cube_jitter,
        yaw_range_degrees=args.yaw_range_deg,
        view_camera=args.view_camera,
        playback_speed=args.playback_speed,
        hold_final_seconds=args.hold_final_seconds,
    )
    print(
        f"success={result['is_success']} grasped={result['has_grasped']} "
        f"lifted={result['has_lifted']} phase={result['phase']}"
    )
    if not result["is_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
