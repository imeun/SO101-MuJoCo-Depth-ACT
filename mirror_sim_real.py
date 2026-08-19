"""Verify the swept calibration by mirroring poses between MuJoCo and the arm.

Two directions, and the safe one is the default:

  real-to-sim  Torque stays OFF. Move the arm by hand and watch MuJoCo follow.
               Sends nothing to the motors, so a bad calibration only looks
               wrong instead of driving the arm into something.

  sim-to-real  Drives the motors. Drag the sliders in the MuJoCo Control panel
               and the arm follows. Requires --send plus a typed confirmation,
               starts from the arm's current pose so nothing jumps, clamps to
               the measured travel, and rate-limits every step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from sim2real_joint_mapping import JOINT_NAMES
from sweep_joint_calibration import AffineJointMapping, follower_reader

MARGIN_DEGREES = 3.0
SLEW_DEGREES_PER_TICK = 1.5
DEFAULT_RATE_HZ = 30.0


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return parsed


def load_travel_bounds(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the swept travel per joint, on the calibrated branch."""
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    for key in ("real_at_sim_low", "real_at_sim_high"):
        if key not in payload:
            raise ValueError(f"calibration file has no {key}; re-run sweep_fit.py")
    low = np.asarray(payload["real_at_sim_low"], dtype=np.float64)
    high = np.asarray(payload["real_at_sim_high"], dtype=np.float64)
    return np.minimum(low, high), np.maximum(low, high)


def limit_step(
    target: np.ndarray, previous: np.ndarray, *, slew: float
) -> np.ndarray:
    """Move at most `slew` units per tick so the servos never slam."""
    return previous + np.clip(target - previous, -slew, slew)


def plan_command(
    sim_radians: Sequence[float],
    previous_real: np.ndarray,
    mapping: AffineJointMapping,
    travel_low: np.ndarray,
    travel_high: np.ndarray,
    *,
    margin: float = MARGIN_DEGREES,
    slew: float = SLEW_DEGREES_PER_TICK,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (unwrapped command, value to report to the robot).

    Everything is computed on the calibrated branch, where the travel bounds are
    monotonic, and only folded into the robot's reported domain at the very end.
    """
    desired = mapping.sim_to_real(sim_radians)
    safe_low = travel_low + margin
    safe_high = np.maximum(travel_high - margin, safe_low)
    clamped = np.clip(desired, safe_low, safe_high)
    stepped = limit_step(clamped, previous_real, slew=slew)
    return stepped, mapping.to_reported_domain(stepped)


def send_real_degrees(robot, values: Sequence[float]) -> None:
    action = {f"{joint}.pos": float(value) for joint, value in zip(JOINT_NAMES, values)}
    robot.send_action(action)


def follower_writer(port: str, robot_id: str):
    """Full connect, which enables torque. Caller must close it."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id))
    robot.connect()
    return robot


def _print_row(label: str, values: Sequence[float]) -> None:
    print(f"  {label:<14}" + "".join(f"{float(v):>9.2f}" for v in values))


def _print_header() -> None:
    print("  " + " " * 14 + "".join(f"{name[:8]:>9}" for name in JOINT_NAMES))


def run_real_to_sim(mapping: AffineJointMapping, args) -> None:
    import mujoco

    from so101_ball_bins_env import SO101BallBinsEnv

    environment = SO101BallBinsEnv(spawn_stage="stage1", xml_path=args.xml)
    viewer_handle = None
    try:
        with follower_reader(args.port, args.robot_id) as read_values:
            read_values()
            print(f"Connected to {args.robot_id} on {args.port}. Torque stays OFF.")
            import mujoco.viewer

            viewer_handle = mujoco.viewer.launch_passive(environment.model, environment.data)
            print(
                "\nMove the arm BY HAND. MuJoCo should copy it.\n"
                "Close the MuJoCo window to stop.\n"
            )
            _print_header()
            period = 1.0 / args.rate
            tick = 0
            while viewer_handle.is_running():
                real = read_values()
                sim = mapping.real_to_sim(real)
                environment.data.qpos[environment.qpos_ids] = sim
                environment.data.qvel[:] = 0.0
                mujoco.mj_forward(environment.model, environment.data)
                viewer_handle.sync()
                if tick % int(max(1, args.rate)) == 0:
                    _print_row("real", real)
                    _print_row("sim deg", np.rad2deg(sim))
                tick += 1
                time.sleep(period)
    finally:
        if viewer_handle is not None:
            try:
                viewer_handle.close()
            except Exception:  # noqa: BLE001
                pass
        environment.close()


def run_sim_to_real(mapping: AffineJointMapping, args) -> None:
    import mujoco

    from so101_ball_bins_env import SO101BallBinsEnv

    travel_low, travel_high = load_travel_bounds(args.calibration)
    environment = SO101BallBinsEnv(spawn_stage="stage1", xml_path=args.xml)
    robot = None
    viewer_handle = None
    try:
        with follower_reader(args.port, args.robot_id) as read_values:
            start_real = mapping.canonicalize_real(read_values())
        start_sim = mapping.real_to_sim(start_real)
        print("Starting from the arm's current pose so nothing jumps:")
        _print_header()
        _print_row("real", start_real)
        _print_row("sim deg", np.rad2deg(start_sim))

        clamped_start = np.clip(start_sim, environment.ctrl_low, environment.ctrl_high)
        if not np.allclose(clamped_start, start_sim, atol=1e-6):
            outside = [
                JOINT_NAMES[i]
                for i in range(len(JOINT_NAMES))
                if abs(clamped_start[i] - start_sim[i]) > 1e-6
            ]
            print("NOTE: the current pose maps outside the MuJoCo limits for: " + ", ".join(outside))

        if not args.send:
            print(
                "\nDry run: nothing will be sent to the motors. Re-run with --send "
                "once the real-to-sim direction looks correct."
            )
        else:
            print(
                f"\nThis ENERGISES the arm and drives it. Limits in effect:\n"
                f"  travel margin {args.margin:.1f} deg, max {args.slew:.1f} deg per tick "
                f"at {args.rate:.0f} Hz.\n"
                "  Keep a hand near the power switch."
            )
            if input('Type ENGAGE to continue: ').strip() != "ENGAGE":
                print("Aborted; nothing was sent.")
                return
            robot = follower_writer(args.port, args.robot_id)

        environment.data.qpos[environment.qpos_ids] = clamped_start
        environment.data.qvel[:] = 0.0
        environment.ctrl = clamped_start.copy()
        environment.data.ctrl[:] = clamped_start
        mujoco.mj_forward(environment.model, environment.data)

        import mujoco.viewer

        viewer_handle = mujoco.viewer.launch_passive(environment.model, environment.data)
        print(
            "\nOpen the Control panel and drag the six actuator sliders.\n"
            "Close the MuJoCo window to stop.\n"
        )
        _print_header()

        previous_real = start_real.copy()
        period = 1.0 / args.rate
        steps = max(1, round(period / environment.model.opt.timestep))
        tick = 0
        while viewer_handle.is_running():
            for _ in range(steps):
                mujoco.mj_step(environment.model, environment.data)
            viewer_handle.sync()
            sim_ctrl = np.asarray(environment.data.ctrl[:len(JOINT_NAMES)], dtype=np.float64)
            previous_real, reported = plan_command(
                sim_ctrl, previous_real, mapping, travel_low, travel_high,
                margin=args.margin, slew=args.slew,
            )
            if robot is not None:
                send_real_degrees(robot, reported)
            if tick % int(max(1, args.rate)) == 0:
                _print_row("sim deg", np.rad2deg(sim_ctrl))
                _print_row("send" if robot is not None else "would send", reported)
            tick += 1
            time.sleep(period)
    finally:
        if viewer_handle is not None:
            try:
                viewer_handle.close()
            except Exception:  # noqa: BLE001
                pass
        if robot is not None:
            robot.disconnect()
            print("Arm disconnected.")
        environment.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mirror poses between MuJoCo and the SO101 to check the calibration."
    )
    parser.add_argument(
        "--direction",
        choices=("real-to-sim", "sim-to-real"),
        default="real-to-sim",
        help="real-to-sim (default) never touches the motors.",
    )
    parser.add_argument("--calibration", type=Path, default=Path("sim2real_joint_map_swept.json"))
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--xml", default=None)
    parser.add_argument("--rate", type=_positive_float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--margin", type=float, default=MARGIN_DEGREES)
    parser.add_argument("--slew", type=_positive_float, default=SLEW_DEGREES_PER_TICK)
    parser.add_argument(
        "--send",
        action="store_true",
        help="sim-to-real only: actually drive the motors instead of dry-running.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.margin < 0.0 or not np.isfinite(args.margin):
        raise ValueError("--margin must be finite and non-negative")
    mapping = AffineJointMapping.load(args.calibration)
    print("Joint order:", ", ".join(JOINT_NAMES))
    if args.direction == "real-to-sim":
        run_real_to_sim(mapping, args)
    else:
        run_sim_to_real(mapping, args)


if __name__ == "__main__":
    main()
