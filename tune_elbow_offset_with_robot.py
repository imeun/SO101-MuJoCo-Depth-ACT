"""Interactively tune one SO-101 joint offset against a fixed MuJoCo pose."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from mirror_sim_real import load_travel_bounds, plan_command, send_real_degrees, follower_writer
from play_waypoint_teacher import build_waypoints, configure_viewer_camera
from run_waypoint_on_real_robot import EmergencyStopPanel
from sim2real_joint_mapping import JOINT_NAMES, observation_to_real_degrees
from so101_ball_bins_env import SO101BallBinsEnv
from sweep_joint_calibration import AffineJointMapping, follower_reader


DEGREE_JOINT_NAMES = tuple(JOINT_NAMES[:5])
KEY_NAMES = {
    glfw.KEY_1: "select_0",
    glfw.KEY_2: "select_1",
    glfw.KEY_3: "select_2",
    glfw.KEY_4: "select_3",
    glfw.KEY_5: "select_4",
    glfw.KEY_UP: "up",
    glfw.KEY_DOWN: "down",
    glfw.KEY_S: "save",
    glfw.KEY_R: "reset",
    glfw.KEY_SPACE: "stop",
    glfw.KEY_ESCAPE: "stop",
}


def adjusted_mapping_many(
    mapping: AffineJointMapping,
    joint_trims_degrees: dict[str, float],
) -> AffineJointMapping:
    offset = mapping.offset_rad.copy()
    for joint_name, trim_degrees in joint_trims_degrees.items():
        if joint_name not in DEGREE_JOINT_NAMES:
            raise ValueError(f"joint must be one of {DEGREE_JOINT_NAMES}")
        if not np.isfinite(trim_degrees):
            raise ValueError("joint trim must be finite")
        offset[JOINT_NAMES.index(joint_name)] += np.deg2rad(trim_degrees)
    return AffineJointMapping(
        scale_rad_per_real_unit=mapping.scale_rad_per_real_unit.copy(),
        offset_rad=offset,
        real_period=mapping.real_period.copy(),
        real_reference=mapping.real_reference.copy(),
    )


def adjusted_mapping(
    mapping: AffineJointMapping,
    *,
    joint_name: str,
    trim_degrees: float,
) -> AffineJointMapping:
    return adjusted_mapping_many(mapping, {joint_name: trim_degrees})


def save_adjusted_mapping_many(
    source: str | Path,
    output: str | Path,
    *,
    joint_trims_degrees: dict[str, float],
) -> Path:
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if source_path == output_path:
        raise ValueError("output must differ from the source calibration")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    offset = np.asarray(payload["offset_rad"], dtype=np.float64)
    if offset.shape != (6,) or not np.all(np.isfinite(offset)):
        raise ValueError("source calibration offset_rad is invalid")
    normalized_trims: dict[str, float] = {}
    for joint_name, trim_degrees in joint_trims_degrees.items():
        if joint_name not in DEGREE_JOINT_NAMES:
            raise ValueError(f"joint must be one of {DEGREE_JOINT_NAMES}")
        if not np.isfinite(trim_degrees):
            raise ValueError("joint trim must be finite")
        offset[JOINT_NAMES.index(joint_name)] += np.deg2rad(trim_degrees)
        normalized_trims[joint_name] = float(trim_degrees)
    payload["offset_rad"] = offset.tolist()
    payload["manual_joint_trims_degrees"] = normalized_trims
    payload["manual_joint_trim_source"] = str(source_path)
    payload["manual_joint_trim_rule"] = (
        "positive trim increases the selected simulation offset and decreases "
        "the corresponding real joint command for a fixed MuJoCo pose"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_adjusted_mapping(
    source: str | Path,
    output: str | Path,
    *,
    joint_name: str,
    trim_degrees: float,
) -> Path:
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if source_path == output_path:
        raise ValueError("output must differ from the source calibration")
    if joint_name not in DEGREE_JOINT_NAMES:
        raise ValueError(f"joint must be one of {DEGREE_JOINT_NAMES}")
    if not np.isfinite(trim_degrees):
        raise ValueError("joint trim must be finite")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    offset = np.asarray(payload["offset_rad"], dtype=np.float64)
    if offset.shape != (6,) or not np.all(np.isfinite(offset)):
        raise ValueError("source calibration offset_rad is invalid")
    joint_index = JOINT_NAMES.index(joint_name)
    offset[joint_index] += np.deg2rad(trim_degrees)
    payload["offset_rad"] = offset.tolist()
    payload["manual_joint_name"] = joint_name
    payload["manual_joint_trim_degrees"] = float(trim_degrees)
    payload["manual_joint_trim_source"] = str(source_path)
    payload["manual_joint_trim_rule"] = (
        "positive trim increases the selected simulation offset and decreases "
        "the corresponding real joint command for a fixed MuJoCo pose"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a fixed MuJoCo pose and tune one real joint offset in degree steps."
    )
    parser.add_argument("--calibration", type=Path, default=Path("sim2real_joint_map_swept.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sim2real_joint_map_swept_joint_tuned.json"),
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument(
        "--phase",
        choices=("home", "pregrasp", "grasp", "close", "lift"),
        default="grasp",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--joint",
        choices=DEGREE_JOINT_NAMES,
        default="wrist_flex",
        help="joint to tune; motor 4 is wrist_flex",
    )
    parser.add_argument("--step-degrees", type=float, default=1.0)
    parser.add_argument("--max-absolute-trim", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--margin", type=float, default=3.0)
    parser.add_argument("--gripper-margin", type=float, default=0.0)
    parser.add_argument("--slew", type=float, default=0.5)
    parser.add_argument("--gripper-slew", type=float, default=5.0)
    parser.add_argument("--send", action="store_true")
    return parser.parse_args(argv)


def _fixed_pose(environment: SO101BallBinsEnv, phase: str) -> np.ndarray:
    if phase == "home":
        return environment.home_qpos.copy()
    return np.asarray(build_waypoints(environment)[phase], dtype=np.float64)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if (
        not np.isfinite(args.step_degrees)
        or args.step_degrees <= 0.0
        or not np.isfinite(args.max_absolute_trim)
        or args.max_absolute_trim <= 0.0
        or not np.isfinite(args.rate)
        or args.rate <= 0.0
    ):
        raise ValueError("step, trim limit, and rate must be positive and finite")

    source = args.calibration.expanduser().resolve()
    output = args.output.expanduser().resolve()
    mapping = AffineJointMapping.load(source)
    travel_low, travel_high = load_travel_bounds(source)
    environment = SO101BallBinsEnv(spawn_stage="stage1", max_steps=1000)
    environment.reset(seed=args.seed)
    sim_pose = _fixed_pose(environment, args.phase)
    environment.data.qpos[environment.qpos_ids] = sim_pose
    environment.data.qvel[:] = 0.0
    environment.ctrl = sim_pose.copy()
    environment.data.ctrl[:] = sim_pose
    mujoco.mj_forward(environment.model, environment.data)

    margin = np.array([args.margin] * 5 + [args.gripper_margin], dtype=np.float64)
    slew = np.array([args.slew] * 5 + [args.gripper_slew], dtype=np.float64)
    period = 1.0 / args.rate
    events: deque[str] = deque()
    panel = robot = None
    selected_joint = args.joint
    trims = {joint_name: 0.0 for joint_name in DEGREE_JOINT_NAMES}
    saved = False

    def on_key(keycode: int) -> None:
        event = KEY_NAMES.get(keycode)
        if event is not None:
            events.append(event)

    try:
        print("Fixed MuJoCo phase:", args.phase)
        print("Joint order:", ", ".join(JOINT_NAMES))
        print(f"Selected: motor {JOINT_NAMES.index(selected_joint) + 1} ({selected_joint})")
        print(f"Trim limit: +/-{args.max_absolute_trim:g} degrees")
        print("MuJoCo pose radians:", np.round(sim_pose, 5))
        print("\nKeys in the MuJoCo window:")
        print("  1..5       : select shoulder_pan .. wrist_roll")
        print("  Up arrow   : offset +1 step; real motor command moves in - direction")
        print("  Down arrow : offset -1 step; real motor command moves in + direction")
        print("  R          : reset the selected joint trim to 0 degrees")
        print("  S          : save all joint trims to a NEW calibration file")
        print("  Space/Esc  : stop and exit; torque is released on exit")
        print("\nRemove the real block before tuning and support the arm when exiting.")

        with follower_reader(args.port, args.robot_id) as read_values:
            start_real = mapping.canonicalize_real(read_values())
        print("Current real joints:", np.round(start_real, 3))
        initial_target = mapping.sim_to_real(sim_pose)
        print("Fixed-pose real target:", np.round(mapping.to_reported_domain(initial_target), 3))

        if not args.send:
            print("\nDRY RUN: MuJoCo opens, but no motor command is sent. Add --send to tune the real arm.")
        else:
            print(
                "\nThis energises the arm and moves it to the displayed pose. "
                "Keep a hand near the power switch."
            )
            if input("Type ENGAGE to continue: ").strip() != "ENGAGE":
                print("Aborted; nothing was sent.")
                return
            panel = EmergencyStopPanel()
            robot = follower_writer(args.port, args.robot_id)

        previous_real = start_real.copy()
        with mujoco.viewer.launch_passive(
            environment.model,
            environment.data,
            key_callback=on_key,
        ) as viewer:
            configure_viewer_camera(viewer, environment.model, "free")
            viewer.sync()
            print("\nMuJoCo is the fixed reference. Click its window before using the keys.")
            while viewer.is_running():
                started = time.perf_counter()
                if panel is not None:
                    panel.pump()
                    if panel.stopped:
                        break

                while events:
                    event = events.popleft()
                    if event == "stop":
                        if panel is not None:
                            panel.request_stop()
                        else:
                            return
                        break
                    if event.startswith("select_"):
                        selected_joint = DEGREE_JOINT_NAMES[int(event.removeprefix("select_"))]
                        selected_index = JOINT_NAMES.index(selected_joint)
                        print(
                            f"SELECTED motor {selected_index + 1} ({selected_joint}) | "
                            f"trim={trims[selected_joint]:+.1f} deg",
                            flush=True,
                        )
                        continue
                    if event == "up":
                        next_trim = min(
                            args.max_absolute_trim,
                            trims[selected_joint] + args.step_degrees,
                        )
                    elif event == "down":
                        next_trim = max(
                            -args.max_absolute_trim,
                            trims[selected_joint] - args.step_degrees,
                        )
                    elif event == "reset":
                        next_trim = 0.0
                    elif event == "save":
                        active_trims = {
                            joint_name: value
                            for joint_name, value in trims.items()
                            if not np.isclose(value, 0.0)
                        }
                        saved_path = save_adjusted_mapping_many(
                            source,
                            output,
                            joint_trims_degrees=active_trims,
                        )
                        saved = True
                        print(f"SAVED trims={active_trims} -> {saved_path}", flush=True)
                        continue
                    else:
                        continue
                    hit_limit = (
                        next_trim == trims[selected_joint]
                        and event in {"up", "down"}
                    )
                    trims[selected_joint] = next_trim
                    selected_index = JOINT_NAMES.index(selected_joint)
                    adjusted = adjusted_mapping_many(mapping, trims)
                    target_real = adjusted.sim_to_real(sim_pose)
                    print(
                        f"trim={next_trim:+.1f} deg | motor {selected_index + 1} "
                        f"({selected_joint}) command="
                        f"{adjusted.to_reported_domain(target_real)[selected_index]:.2f}"
                        + (" | LIMIT REACHED" if hit_limit else ""),
                        flush=True,
                    )

                if panel is not None and panel.stopped:
                    break

                if robot is not None:
                    adjusted = adjusted_mapping_many(mapping, trims)
                    previous_real, reported = plan_command(
                        sim_pose,
                        previous_real,
                        adjusted,
                        travel_low,
                        travel_high,
                        margin=margin,
                        slew=slew,
                    )
                    send_real_degrees(robot, reported)

                environment.data.qpos[environment.qpos_ids] = sim_pose
                environment.data.qvel[:] = 0.0
                environment.data.ctrl[:] = sim_pose
                viewer.sync()
                time.sleep(max(0.0, period - (time.perf_counter() - started)))

        if not saved:
            print("No calibration was saved. Press S before exit to keep the trim.")
    finally:
        if robot is not None:
            robot.disconnect()
            print("Robot disconnected; torque released. Support the arm as it relaxes.")
        if panel is not None:
            panel.close()
        environment.close()


if __name__ == "__main__":
    main()
