"""Run the scripted waypoint success episode on the real SO101 arm.

MuJoCo drives the trajectory exactly as in play_waypoint_teacher.py, and every
control step is converted through the swept calibration and sent to the arm,
reusing the clamp and slew limiter that mirror_sim_real.py already validated.

A separate always-on-top window carries the emergency stop. Pressing it (or
Space, or Escape, or closing either window) reads where the arm actually is and
commands exactly that, so the arm freezes in place with torque still holding.

    python run_waypoint_on_real_robot.py --send
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import tkinter as tk
from typing import Sequence

import mujoco
import mujoco.viewer
import numpy as np

from mirror_sim_real import (
    load_travel_bounds,
    plan_command,
    send_real_degrees,
    follower_writer,
    _print_header,
    _print_row,
)
from play_waypoint_teacher import (
    BASE_BIN_POSITION,
    configure_viewer_camera,
    execute_waypoint_episode,
    randomize_cube_pose,
)
from sim2real_joint_mapping import JOINT_NAMES, observation_to_real_degrees
from so101_ball_bins_env import SO101BallBinsEnv
from sweep_joint_calibration import AffineJointMapping, follower_reader


ALIGN_SLEW_DEGREES_PER_TICK = 0.5
ALIGN_TOLERANCE = 0.15
ALIGN_TIMEOUT_SECONDS = 60.0


class EmergencyStop(Exception):
    """Raised inside the control loop to abort the episode immediately."""


class EmergencyStopPanel:
    """Always-on-top stop button, pumped from the control loop itself.

    No threads: the loop calls pump() every tick, so the button is serviced at
    the control rate and the stop flag is read at a known point each step. If
    the loop ever stalls, no new targets go out and the servos keep holding the
    last one, which is the same thing the stop does.
    """

    def __init__(self) -> None:
        self.stopped = False
        self._alive = True
        self._root = tk.Tk()
        self._root.title("SO101 EMERGENCY STOP")
        self._root.geometry("360x220")
        self._root.attributes("-topmost", True)
        self._root.configure(bg="#1a1a1a")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._status = tk.Label(
            self._root,
            text="RUNNING",
            font=("TkDefaultFont", 16, "bold"),
            fg="#39d353",
            bg="#1a1a1a",
        )
        self._status.pack(pady=(14, 6))

        self._button = tk.Button(
            self._root,
            text="STOP",
            command=self.request_stop,
            font=("TkDefaultFont", 34, "bold"),
            fg="white",
            bg="#c0392b",
            activebackground="#e74c3c",
            activeforeground="white",
            relief="raised",
            borderwidth=6,
        )
        self._button.pack(expand=True, fill="both", padx=18, pady=(0, 10))

        tk.Label(
            self._root,
            text="Space / Esc also stops",
            font=("TkDefaultFont", 9),
            fg="#888888",
            bg="#1a1a1a",
        ).pack(pady=(0, 10))

        self._root.bind("<space>", lambda _event: self.request_stop())
        self._root.bind("<Escape>", lambda _event: self.request_stop())
        self._button.focus_set()
        self.pump()

    @property
    def is_open(self) -> bool:
        return self._alive

    def request_stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        if self._alive:
            try:
                self._status.configure(text="STOPPED", fg="#e74c3c")
                self._button.configure(state="disabled", text="STOPPED", bg="#555555")
                self._root.update()
            except tk.TclError:
                self._alive = False

    def _on_close(self) -> None:
        """First X press stops the arm; a second one dismisses the window."""
        if self.stopped:
            self.close()
        else:
            self.request_stop()

    def pump(self) -> None:
        """Service the window. A window that is gone counts as a stop request.

        update() does not raise on a destroyed root, so losing the window some
        other way would otherwise leave the episode running with no stop button.
        """
        if not self._alive:
            return
        try:
            if not self._root.winfo_exists():
                raise tk.TclError("stop window no longer exists")
            self._root.update()
        except tk.TclError:
            self._alive = False
            self.stopped = True

    def close(self) -> None:
        if not self._alive:
            return
        self._alive = False
        try:
            self._root.destroy()
        except tk.TclError:
            pass


def read_real_values(robot) -> np.ndarray:
    """Read the arm through an already-connected follower."""
    return observation_to_real_degrees(robot.get_observation())


def hold_where_it_is(robot, mapping: AffineJointMapping) -> np.ndarray:
    """Command the arm's own measured pose so it freezes without going limp."""
    measured = mapping.canonicalize_real(read_real_values(robot))
    send_real_degrees(robot, mapping.to_reported_domain(measured))
    return measured


def align_to_start_pose(
    robot,
    mapping: AffineJointMapping,
    sim_target: np.ndarray,
    previous_real: np.ndarray,
    travel_low: np.ndarray,
    travel_high: np.ndarray,
    *,
    margin: float,
    slew: float,
    period: float,
    panel: EmergencyStopPanel,
    viewer,
) -> np.ndarray:
    """Walk the arm to the episode start pose at a deliberately slow slew.

    The waypoints are solved from the MuJoCo home pose, so the arm has to be
    there before the episode starts or the two immediately diverge.
    """
    settled, _ = plan_command(
        sim_target, previous_real, mapping, travel_low, travel_high,
        margin=margin, slew=np.inf,
    )
    gap = float(np.max(np.abs(settled - previous_real)))
    print(f"\nAligning to the episode start pose (largest gap {gap:.1f}).")
    _print_header()
    _print_row("from", previous_real)
    _print_row("to", settled)

    deadline = time.monotonic() + ALIGN_TIMEOUT_SECONDS
    while True:
        panel.pump()
        if panel.stopped:
            raise EmergencyStop("stop pressed during alignment")
        if viewer is not None and not viewer.is_running():
            raise EmergencyStop("MuJoCo window closed during alignment")
        previous_real, reported = plan_command(
            sim_target, previous_real, mapping, travel_low, travel_high,
            margin=margin, slew=slew,
        )
        send_real_degrees(robot, reported)
        if viewer is not None:
            viewer.sync()
        if float(np.max(np.abs(settled - previous_real))) <= ALIGN_TOLERANCE:
            print("Aligned.\n")
            return previous_real
        if time.monotonic() > deadline:
            raise EmergencyStop(
                f"alignment did not converge within {ALIGN_TIMEOUT_SECONDS:.0f} s"
            )
        time.sleep(period)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay the scripted waypoint success episode on the real SO101."
    )
    parser.add_argument("--calibration", type=Path, default=Path("sim2real_joint_map_swept.json"))
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--xml", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--spawn-stage", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument(
        "--cube-jitter",
        type=float,
        default=0.0,
        help="Sim block jitter in metres. Keep 0 so the real block sits at a known spot.",
    )
    parser.add_argument("--view-camera", choices=["free", "top", "side_depth"], default="free")
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument(
        "--margin",
        type=float,
        default=3.0,
        help="Degrees held back from each end of the swept travel, rotary joints only. "
             "The episode leaves 12.2 deg of slack at the tightest joint (shoulder_lift).",
    )
    parser.add_argument(
        "--gripper-margin",
        type=float,
        default=0.0,
        help="Percent held back on the gripper. Its closed end IS the home pose, so any "
             "margin here stops it closing fully at rest; the swept travel is the guard.",
    )
    parser.add_argument(
        "--slew",
        type=float,
        default=2.5,
        help="Max degrees per control tick during the episode, rotary joints only. "
             "The trajectory peaks at 1.03 deg, so this only catches blowups.",
    )
    parser.add_argument(
        "--gripper-slew",
        type=float,
        default=10.0,
        help="Max percent per tick on the gripper. The close phase needs 6.05 percent "
             "per tick, so the rotary slew would make it lag and drop the block.",
    )
    parser.add_argument(
        "--align-slew",
        type=float,
        default=ALIGN_SLEW_DEGREES_PER_TICK,
        help="Max change per tick while moving to the start pose.",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Actually drive the motors. Without it nothing is sent.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    mapping = AffineJointMapping.load(args.calibration)
    travel_low, travel_high = load_travel_bounds(args.calibration)

    # The gripper is the odd one out: it is a 0-100 percent axis while the rest
    # are degrees, so a single margin and slew cannot serve both. plan_command
    # broadcasts, so per-joint vectors drop straight in.
    margin = np.array([args.margin] * 5 + [args.gripper_margin], dtype=np.float64)
    slew = np.array([args.slew] * 5 + [args.gripper_slew], dtype=np.float64)
    align_slew = np.array([args.align_slew] * 5 + [args.gripper_slew], dtype=np.float64)

    # The schedule runs 970 steps and only returns home in the last 120, so a
    # 900-step cap would strand the arm over the bin.
    env = SO101BallBinsEnv(spawn_stage=args.spawn_stage, xml_path=args.xml, max_steps=1000)
    panel = None
    robot = None
    keep_torque = False
    try:
        env.reset(seed=args.seed)
        if args.cube_jitter > 0.0:
            randomize_cube_pose(
                env, np.random.default_rng(args.seed), xy_jitter=args.cube_jitter
            )
        cube = env._cube_position()
        period = float(env.model.opt.timestep * env.frame_skip / args.playback_speed)

        print("Joint order:", ", ".join(JOINT_NAMES))
        print(
            f"\nPlace the real block at x={cube[0] * 1000:.0f} mm, y={cube[1] * 1000:.0f} mm "
            f"and the bin at x={BASE_BIN_POSITION[0] * 1000:.0f} mm, "
            f"y={BASE_BIN_POSITION[1] * 1000:.0f} mm in the arm base frame."
        )

        with follower_reader(args.port, args.robot_id) as read_values:
            start_real = mapping.canonicalize_real(read_values())
        print("\nArm is currently at:")
        _print_header()
        _print_row("real", start_real)

        if not args.send:
            print(
                "\nDry run: --send was not given, so nothing will be sent to the motors "
                "and the episode is not started."
            )
            return

        print(
            f"\nThis ENERGISES the arm and runs the full pick-and-place episode.\n"
            f"  rotary: margin {args.margin:.1f} deg, max {args.slew:.1f} deg per tick "
            f"({args.align_slew:.1f} while aligning)\n"
            f"  gripper: margin {args.gripper_margin:.1f} pct, max "
            f"{args.gripper_slew:.1f} pct per tick\n"
            "  The STOP window freezes the arm where it is. Keep a hand near the power switch."
        )
        if input("Type ENGAGE to continue: ").strip() != "ENGAGE":
            print("Aborted; nothing was sent.")
            return

        panel = EmergencyStopPanel()
        robot = follower_writer(args.port, args.robot_id)
        previous_real = start_real.copy()
        clamped = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        info = None

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            configure_viewer_camera(viewer, env.model, args.view_camera)
            viewer.sync()

            def on_step(action: np.ndarray, phase: str) -> None:
                nonlocal previous_real
                panel.pump()
                if panel.stopped:
                    raise EmergencyStop(f"stop pressed during {phase}")
                if not viewer.is_running():
                    raise EmergencyStop(f"MuJoCo window closed during {phase}")
                sim_ctrl = env.control_target(action)
                desired = mapping.sim_to_real(sim_ctrl)
                previous_real, reported = plan_command(
                    sim_ctrl, previous_real, mapping, travel_low, travel_high,
                    margin=margin, slew=slew,
                )
                np.maximum(clamped, np.abs(desired - previous_real), out=clamped)
                send_real_degrees(robot, reported)

            try:
                previous_real = align_to_start_pose(
                    robot, mapping, env.ctrl.copy(), previous_real, travel_low, travel_high,
                    margin=margin, slew=align_slew, period=period,
                    panel=panel, viewer=viewer,
                )
                print("Running the episode. Watch the STOP window.\n")
                _print_header()
                info = execute_waypoint_episode(
                    env,
                    on_step=on_step,
                    viewer=viewer,
                    playback_speed=args.playback_speed,
                )
            except EmergencyStop as stop:
                held = hold_where_it_is(robot, mapping)
                keep_torque = True
                print(f"\nEMERGENCY STOP: {stop}")
                _print_row("holding at", held)
                print("The arm is holding position with torque on.")
                print("Close the STOP window, then power down or move it by hand.")
                while panel.is_open:
                    panel.pump()
                    time.sleep(0.05)

        if info is None:
            raise SystemExit(1)

        _print_row("send", previous_real)
        print(
            f"\nsuccess={info['is_success']} grasped={info['has_grasped']} "
            f"lifted={info['has_lifted']} phase={info['phase']}"
        )
        worst = int(np.argmax(clamped))
        if float(clamped[worst]) > 0.5:
            print(
                f"NOTE: commands were held back by up to {clamped[worst]:.1f} on "
                f"{JOINT_NAMES[worst]}. Lower --margin or raise --slew if the arm "
                "did not follow the sim."
            )
    finally:
        if robot is not None:
            if keep_torque:
                # disconnect() writes Torque_Enable=0 and the arm would drop out
                # of the pose the stop just froze it in. False closes the port
                # and leaves the servos holding.
                robot.bus.disconnect(False)
                print("Port closed. The arm is still energised and holding.")
            else:
                robot.disconnect()
                print("Arm disconnected.")
        if panel is not None:
            panel.close()
        env.close()


if __name__ == "__main__":
    main()
