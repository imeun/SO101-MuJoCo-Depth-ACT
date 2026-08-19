"""Safely deploy a trained dual-depth ACT checkpoint on the real SO-101."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from collect_fixed_delta_depth_dataset import scene_provenance
from mirror_sim_real import load_travel_bounds, plan_command, send_real_degrees, follower_writer
from play_waypoint_teacher import BASE_BIN_POSITION, BASE_CUBE_POSITION, configure_viewer_camera
from run_waypoint_on_real_robot import EmergencyStop, EmergencyStopPanel, hold_where_it_is
from sim2real_joint_mapping import observation_to_real_degrees
from so101_ball_bins_env import SO101BallBinsEnv
from so101_depth import DepthConfig
from so101_depth_act import (
    JointVelocityEstimator,
    TemporalActionEnsembler,
    load_depth_act_checkpoint,
    predict_delta_chunk,
)
from sweep_joint_calibration import AffineJointMapping, follower_reader
from train_depth_act import resolve_device


CONTROL_HZ = 1.0 / (0.002 * 17)


def depth_units_to_metres(raw: np.ndarray, scale: float) -> np.ndarray:
    values = np.asarray(raw)
    if values.ndim != 2 or values.dtype != np.uint16:
        raise ValueError("raw RealSense depth must be uint16 HxW")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("depth scale must be positive and finite")
    return values.astype(np.float32) * np.float32(scale)


def depth_to_bgr(depth_m: np.ndarray, *, near_m: float, far_m: float) -> np.ndarray:
    """Colorize metric depth for display without changing policy input data."""
    depth = np.asarray(depth_m)
    if depth.ndim != 2 or depth.size == 0 or not np.issubdtype(depth.dtype, np.number):
        raise ValueError("depth must be a nonempty numeric HxW array")
    if not np.isfinite(near_m) or not np.isfinite(far_m) or near_m >= far_m:
        raise ValueError("display depth range must be finite and ordered")
    valid = np.isfinite(depth) & (depth > 0.0)
    clipped = np.clip(depth.astype(np.float32, copy=True), near_m, far_m)
    proximity = (far_m - clipped) / (far_m - near_m)
    red = np.clip(1.5 - np.abs(4.0 * proximity - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * proximity - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * proximity - 1.0), 0.0, 1.0)
    bgr = np.stack((blue, green, red), axis=-1)
    bgr[~valid] = 0.0
    return np.rint(bgr * 255.0).astype(np.uint8)


class DeploymentVisualizer:
    def __init__(
        self,
        environment: SO101BallBinsEnv,
        depth_config: DepthConfig,
        *,
        refresh_rate: float,
    ):
        self.environment = environment
        self.depth_config = depth_config
        self.refresh_period = 1.0 / refresh_rate
        self.last_refresh = float("-inf")
        self.depth_window = None
        self.viewer = None
        try:
            self.depth_window = MatplotlibDepthWindow(depth_config)
            self.viewer = mujoco.viewer.launch_passive(environment.model, environment.data)
            configure_viewer_camera(self.viewer, environment.model, "free")
        except BaseException:
            self.close()
            raise

    def update(
        self,
        sim_joint_pos: np.ndarray,
        top_depth_m: np.ndarray,
        side_depth_m: np.ndarray,
        *,
        force: bool = False,
    ) -> None:
        pose = np.asarray(sim_joint_pos, dtype=np.float64)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise RuntimeError("visualizer received an invalid MuJoCo joint pose")
        self.environment.data.qpos[self.environment.qpos_ids] = pose
        self.environment.data.qvel[:] = 0.0
        self.environment.data.ctrl[:] = pose
        mujoco.mj_forward(self.environment.model, self.environment.data)

        now = time.perf_counter()
        if force or now - self.last_refresh >= self.refresh_period:
            if not self.viewer.is_running():
                raise EmergencyStop("MuJoCo viewer was closed")
            self.viewer.sync()
            self.depth_window.update(top_depth_m, side_depth_m)
            self.last_refresh = now
        if self.depth_window.stop_requested or not self.depth_window.is_running():
            raise EmergencyStop("Depth viewer was closed or its stop key was pressed")

    def close(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
        if self.depth_window is not None:
            self.depth_window.close()


class MatplotlibDepthWindow:
    WINDOW_NAME = "SO101 Live Depth - Top | Side"

    def __init__(
        self,
        depth_config: DepthConfig,
    ):
        try:
            import matplotlib
            import matplotlib.pyplot as pyplot
        except ImportError as error:
            raise RuntimeError(
                "matplotlib with a GUI backend is required for --display; "
                "install it with: python -m pip install matplotlib"
            ) from error
        if str(matplotlib.get_backend()).lower() in {
            "agg", "cairo", "pdf", "pgf", "ps", "svg", "template"
        }:
            raise RuntimeError(
                f"Matplotlib backend '{matplotlib.get_backend()}' cannot open a window; "
                "on Ubuntu install python3-tk and retry"
            )
        self.pyplot = pyplot
        self.depth_config = depth_config
        self.stop_requested = False
        pyplot.ion()
        self.figure, axes = pyplot.subplots(1, 2, num=self.WINDOW_NAME, figsize=(12, 5))
        blank = np.zeros((depth_config.height, depth_config.width, 3), dtype=np.uint8)
        self.top_image = axes[0].imshow(blank)
        self.side_image = axes[1].imshow(blank)
        axes[0].set_title("TOP DEPTH")
        axes[1].set_title("SIDE DEPTH")
        for axis in axes:
            axis.axis("off")
        self.figure.tight_layout()
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        pyplot.show(block=False)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def _on_key(self, event) -> None:
        if event.key in {"q", "escape"}:
            self.stop_requested = True

    def update(
        self,
        top_depth_m: np.ndarray,
        side_depth_m: np.ndarray,
    ) -> None:
        top_rgb = depth_to_bgr(
            top_depth_m,
            near_m=self.depth_config.near_m,
            far_m=self.depth_config.far_m,
        )[..., ::-1]
        side_rgb = depth_to_bgr(
            side_depth_m,
            near_m=self.depth_config.near_m,
            far_m=self.depth_config.far_m,
        )[..., ::-1]
        self.top_image.set_data(top_rgb)
        self.side_image.set_data(side_rgb)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def is_running(self) -> bool:
        return self.pyplot.fignum_exists(self.figure.number)

    def close(self) -> None:
        try:
            self.pyplot.close(self.figure)
        except Exception:
            pass


def validate_camera_intrinsics(
    expected_provenance: dict,
    actual_intrinsics: dict,
    *,
    tolerance_px: float = 3.0,
) -> dict[str, dict[str, float]]:
    if not np.isfinite(tolerance_px) or tolerance_px < 0.0:
        raise ValueError("camera intrinsic tolerance must be finite and non-negative")
    expected_cameras = expected_provenance.get("cameras")
    if not isinstance(expected_cameras, dict):
        raise ValueError("checkpoint has no camera provenance")
    diagnostics = {}
    for name in ("top", "side_depth"):
        if name not in expected_cameras or name not in actual_intrinsics:
            raise ValueError(f"camera intrinsics are missing for {name}")
        expected = expected_cameras[name]
        actual = actual_intrinsics[name]
        resolution = np.asarray(actual["resolution"], dtype=np.float64)
        if resolution.shape != (2,) or actual["resolution"] != expected["resolution"]:
            raise ValueError(
                f"{name} resolution mismatch: expected {expected['resolution']}, "
                f"got {actual['resolution']}"
            )
        expected_focal = np.asarray(expected["focalpixel"], dtype=np.float64)
        actual_focal = np.asarray(actual["focalpixel"], dtype=np.float64)
        principal = np.asarray(actual["principalpixel"], dtype=np.float64)
        expected_offset = np.asarray(expected["principalpixel_offset"], dtype=np.float64)
        actual_offset = np.array(
            [principal[0] - resolution[0] / 2.0, resolution[1] / 2.0 - principal[1]],
            dtype=np.float64,
        )
        focal_error = float(np.max(np.abs(expected_focal - actual_focal)))
        principal_error = float(np.max(np.abs(expected_offset - actual_offset)))
        if focal_error > tolerance_px or principal_error > tolerance_px:
            raise ValueError(
                f"{name} camera intrinsics mismatch: focal_error={focal_error:.2f}px, "
                f"principal_error={principal_error:.2f}px"
            )
        diagnostics[name] = {
            "focal_error_px": focal_error,
            "principal_error_px": principal_error,
        }
    return diagnostics


class DeploymentSafetyMonitor:
    def __init__(
        self,
        *,
        min_valid_depth_fraction: float = 0.20,
        max_immediate_delta_rad: float = 0.12,
        max_overrun_factor: float = 2.0,
        max_consecutive_overruns: int = 5,
        max_tracking_error: np.ndarray = np.array([15.0] * 5 + [25.0]),
        max_tracking_failures: int = 8,
    ):
        tracking = np.asarray(max_tracking_error, dtype=np.float64)
        if (
            not 0.0 <= min_valid_depth_fraction <= 1.0
            or not np.isfinite(max_immediate_delta_rad)
            or max_immediate_delta_rad <= 0.0
            or not np.isfinite(max_overrun_factor)
            or max_overrun_factor <= 1.0
            or max_consecutive_overruns <= 0
            or tracking.shape != (6,)
            or not np.all(np.isfinite(tracking))
            or np.any(tracking <= 0.0)
            or max_tracking_failures <= 0
        ):
            raise ValueError("invalid deployment safety configuration")
        self.min_valid_depth_fraction = float(min_valid_depth_fraction)
        self.max_immediate_delta_rad = float(max_immediate_delta_rad)
        self.max_overrun_factor = float(max_overrun_factor)
        self.max_consecutive_overruns = int(max_consecutive_overruns)
        self.max_tracking_error = tracking
        self.max_tracking_failures = int(max_tracking_failures)
        self.consecutive_overruns = 0
        self.consecutive_tracking_failures = 0

    def check_depth_pair(self, top: np.ndarray, side: np.ndarray) -> dict[str, float]:
        fractions = {}
        for name, depth in (("top", top), ("side_depth", side)):
            values = np.asarray(depth)
            if values.ndim != 2 or values.size == 0 or not np.issubdtype(values.dtype, np.number):
                raise RuntimeError(f"{name} depth frame is invalid")
            valid_fraction = float(np.mean(np.isfinite(values) & (values > 0.0)))
            if valid_fraction < self.min_valid_depth_fraction:
                raise RuntimeError(
                    f"{name} valid depth fraction {valid_fraction:.3f} is below "
                    f"{self.min_valid_depth_fraction:.3f}"
                )
            fractions[name] = valid_fraction
        return fractions

    def check_prediction(self, delta_chunk: np.ndarray) -> float:
        delta = np.asarray(delta_chunk)
        if delta.ndim != 2 or delta.shape[1] != 6 or delta.shape[0] == 0 or not np.all(np.isfinite(delta)):
            raise RuntimeError("policy produced an invalid delta chunk")
        immediate = float(np.max(np.abs(delta[0])))
        if immediate > self.max_immediate_delta_rad:
            raise RuntimeError(
                f"immediate policy delta {immediate:.3f} rad exceeds "
                f"{self.max_immediate_delta_rad:.3f} rad"
            )
        return immediate

    def check_cycle(self, elapsed_s: float, period_s: float) -> None:
        if not np.isfinite(elapsed_s) or not np.isfinite(period_s) or elapsed_s < 0.0 or period_s <= 0.0:
            raise RuntimeError("control loop timing is invalid")
        if elapsed_s > period_s * self.max_overrun_factor:
            self.consecutive_overruns += 1
        else:
            self.consecutive_overruns = 0
        if self.consecutive_overruns >= self.max_consecutive_overruns:
            raise RuntimeError(
                f"control loop exceeded {self.max_overrun_factor:.1f}x target period for "
                f"{self.consecutive_overruns} consecutive steps"
            )

    def check_tracking(self, commanded_real: np.ndarray, measured_real: np.ndarray) -> float:
        commanded = np.asarray(commanded_real, dtype=np.float64)
        measured = np.asarray(measured_real, dtype=np.float64)
        if commanded.shape != (6,) or measured.shape != (6,) or not np.all(np.isfinite(commanded)) or not np.all(np.isfinite(measured)):
            raise RuntimeError("real joint tracking vectors are invalid")
        error = np.abs(commanded - measured)
        if np.any(error > self.max_tracking_error):
            self.consecutive_tracking_failures += 1
        else:
            self.consecutive_tracking_failures = 0
        if self.consecutive_tracking_failures >= self.max_tracking_failures:
            joint = int(np.argmax(error / self.max_tracking_error))
            raise RuntimeError(
                f"joint tracking error persisted: joint={joint}, error={error[joint]:.2f}, "
                f"limit={self.max_tracking_error[joint]:.2f}"
            )
        return float(np.max(error / self.max_tracking_error))


class DualRealSenseDepth:
    def __init__(self, top_serial: str, side_serial: str, *, width: int = 640, height: int = 480, fps: int = 30):
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError("pyrealsense2 is required for real deployment") from error
        self.rs = rs
        self.pipelines = []
        self.scales = []
        self.intrinsics = {}
        try:
            for name, serial in (("top", top_serial), ("side_depth", side_serial)):
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(str(serial))
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                profile = pipeline.start(config)
                sensor = profile.get_device().first_depth_sensor()
                stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
                intrinsic = stream.get_intrinsics()
                self.pipelines.append(pipeline)
                self.scales.append(float(sensor.get_depth_scale()))
                self.intrinsics[name] = {
                    "resolution": [int(intrinsic.width), int(intrinsic.height)],
                    "focalpixel": [float(intrinsic.fx), float(intrinsic.fy)],
                    "principalpixel": [float(intrinsic.ppx), float(intrinsic.ppy)],
                }
            for _ in range(30):
                self.read(timeout_ms=5000)
        except BaseException:
            self.close()
            raise

    def read(self, *, timeout_ms: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        output = []
        for pipeline, scale in zip(self.pipelines, self.scales):
            frames = pipeline.wait_for_frames(timeout_ms)
            frame = frames.get_depth_frame()
            if not frame:
                raise RuntimeError("RealSense depth frame is missing")
            output.append(depth_units_to_metres(np.asanyarray(frame.get_data()), scale))
        return output[0], output[1]

    def close(self) -> None:
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except Exception:
                pass
        self.pipelines.clear()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dual-depth ACT checkpoint on the real SO101.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--top-serial", default="138422072965")
    parser.add_argument("--side-serial", default="047322070492")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int, default=1100)
    parser.add_argument("--rate", type=float, default=CONTROL_HZ)
    parser.add_argument("--margin", type=float, default=3.0)
    parser.add_argument("--gripper-margin", type=float, default=0.0)
    parser.add_argument("--slew", type=float, default=1.5)
    parser.add_argument("--gripper-slew", type=float, default=10.0)
    parser.add_argument("--align-slew", type=float, default=0.5)
    parser.add_argument("--ensemble-decay", type=float, default=0.08)
    parser.add_argument("--allow-provenance-mismatch", action="store_true")
    parser.add_argument("--allow-camera-intrinsics-mismatch", action="store_true")
    parser.add_argument("--intrinsics-tolerance-px", type=float, default=3.0)
    parser.add_argument("--preflight-frames", type=int, default=15)
    parser.add_argument("--min-valid-depth-fraction", type=float, default=0.20)
    parser.add_argument("--max-immediate-delta-rad", type=float, default=0.12)
    parser.add_argument("--max-overrun-factor", type=float, default=2.0)
    parser.add_argument("--max-consecutive-overruns", type=int, default=5)
    parser.add_argument("--tracking-error-degrees", type=float, default=15.0)
    parser.add_argument("--gripper-tracking-error", type=float, default=25.0)
    parser.add_argument("--max-tracking-failures", type=int, default=8)
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show live Top/Side depth and a MuJoCo digital twin.",
    )
    parser.add_argument("--display-rate", type=float, default=10.0)
    parser.add_argument("--send", action="store_true", help="Actually energise and command the motors.")
    return parser.parse_args(argv)


def _check_provenance(payload: dict, environment: SO101BallBinsEnv, allow_mismatch: bool) -> None:
    expected = payload.get("dataset_provenance")
    actual = scene_provenance(environment)
    if expected == actual:
        print("MuJoCo scene/camera provenance: MATCH", flush=True)
        return
    message = "checkpoint dataset provenance does not match the current scene_ball_bins.xml"
    if not allow_mismatch:
        raise ValueError(message + "; use --allow-provenance-mismatch only after manual verification")
    print("WARNING:", message, flush=True)


def _align(
    robot,
    mapping: AffineJointMapping,
    sim_target: np.ndarray,
    previous_real: np.ndarray,
    travel_low: np.ndarray,
    travel_high: np.ndarray,
    panel: EmergencyStopPanel,
    *,
    margin: np.ndarray,
    slew: np.ndarray,
    period: float,
) -> np.ndarray:
    desired, _ = plan_command(
        sim_target, previous_real, mapping, travel_low, travel_high,
        margin=margin, slew=np.full(6, np.inf),
    )
    deadline = time.monotonic() + 60.0
    while float(np.max(np.abs(desired - previous_real))) > 0.15:
        panel.pump()
        if panel.stopped:
            raise EmergencyStop("stop pressed during home alignment")
        previous_real, reported = plan_command(
            sim_target, previous_real, mapping, travel_low, travel_high,
            margin=margin, slew=slew,
        )
        send_real_degrees(robot, reported)
        if time.monotonic() > deadline:
            raise EmergencyStop("home alignment timed out")
        time.sleep(period)
    return previous_real


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    positive = {
        "max_steps": args.max_steps,
        "rate": args.rate,
        "intrinsics_tolerance_px": args.intrinsics_tolerance_px,
        "preflight_frames": args.preflight_frames,
        "max_immediate_delta_rad": args.max_immediate_delta_rad,
        "max_overrun_factor": args.max_overrun_factor,
        "max_consecutive_overruns": args.max_consecutive_overruns,
        "tracking_error_degrees": args.tracking_error_degrees,
        "gripper_tracking_error": args.gripper_tracking_error,
        "max_tracking_failures": args.max_tracking_failures,
        "display_rate": args.display_rate,
    }
    if any(not np.isfinite(value) or value <= 0 for value in positive.values()):
        raise ValueError("deployment limits and timing arguments must be positive and finite")
    if not 0.0 <= args.min_valid_depth_fraction <= 1.0:
        raise ValueError("--min-valid-depth-fraction must be within [0, 1]")
    device = resolve_device(args.device)
    policy, payload = load_depth_act_checkpoint(str(args.model), map_location="cpu")
    policy = policy.to(device).eval()
    allowed = {field.name for field in fields(DepthConfig)}
    depth_config = DepthConfig(**{key: value for key, value in payload["depth_config"].items() if key in allowed})
    mapping = AffineJointMapping.load(args.calibration)
    travel_low, travel_high = load_travel_bounds(args.calibration)
    margin = np.array([args.margin] * 5 + [args.gripper_margin], dtype=np.float64)
    slew = np.array([args.slew] * 5 + [args.gripper_slew], dtype=np.float64)
    align_slew = np.array([args.align_slew] * 5 + [args.gripper_slew], dtype=np.float64)
    environment = SO101BallBinsEnv(spawn_stage="stage1", max_steps=args.max_steps)
    _check_provenance(payload, environment, args.allow_provenance_mismatch)
    home = np.asarray(payload.get("home_joint_pos", environment.home_qpos), dtype=np.float64)
    if home.shape != (6,) or not np.all(np.isfinite(home)):
        raise ValueError("checkpoint home_joint_pos is invalid")
    cameras = robot = panel = visualizer = None
    keep_torque = False
    period = 1.0 / args.rate
    monitor = DeploymentSafetyMonitor(
        min_valid_depth_fraction=args.min_valid_depth_fraction,
        max_immediate_delta_rad=args.max_immediate_delta_rad,
        max_overrun_factor=args.max_overrun_factor,
        max_consecutive_overruns=args.max_consecutive_overruns,
        max_tracking_error=np.array(
            [args.tracking_error_degrees] * 5 + [args.gripper_tracking_error],
            dtype=np.float64,
        ),
        max_tracking_failures=args.max_tracking_failures,
    )
    try:
        cameras = DualRealSenseDepth(args.top_serial, args.side_serial)
        print("RealSense cameras connected:", cameras.intrinsics, flush=True)
        try:
            intrinsic_diagnostics = validate_camera_intrinsics(
                payload.get("dataset_provenance", {}),
                cameras.intrinsics,
                tolerance_px=args.intrinsics_tolerance_px,
            )
            print("RealSense/MuJoCo intrinsics: MATCH", intrinsic_diagnostics, flush=True)
        except ValueError as error:
            if not args.allow_camera_intrinsics_mismatch:
                raise ValueError(
                    f"{error}; use --allow-camera-intrinsics-mismatch only after checking "
                    "the camera serials and mounting"
                ) from error
            print("WARNING: camera intrinsic check bypassed:", error, flush=True)
        metrics = payload.get("metrics", {})
        print(
            f"checkpoint={args.model.resolve()} epoch={payload.get('epoch')} "
            f"rollout={metrics.get('rollout')}",
            flush=True,
        )
        print(
            f"Real scene: cube x={BASE_CUBE_POSITION[0] * 1000:.0f} mm, "
            f"y={BASE_CUBE_POSITION[1] * 1000:.0f} mm; "
            f"bin x={BASE_BIN_POSITION[0] * 1000:.0f} mm, "
            f"y={BASE_BIN_POSITION[1] * 1000:.0f} mm.",
            flush=True,
        )
        if args.display:
            visualizer = DeploymentVisualizer(
                environment,
                depth_config,
                refresh_rate=args.display_rate,
            )
            print(
                "DISPLAY: MuJoCo shows the encoder-mapped real arm; "
                "the Depth window shows live Top and Side frames. Press Q/Esc to stop.",
                flush=True,
            )
        if not args.send:
            print("DRY RUN: torque remains off and no motor command will be sent.", flush=True)
            with follower_reader(args.port, args.robot_id) as read_values:
                velocity_estimator = JointVelocityEstimator(control_period_s=period)
                for step in range(args.max_steps):
                    started = time.perf_counter()
                    real = mapping.canonicalize_real(read_values())
                    sim = mapping.real_to_sim(real)
                    qvel = velocity_estimator.update(sim)
                    top, side = cameras.read()
                    monitor.check_depth_pair(top, side)
                    if visualizer is not None:
                        visualizer.update(sim, top, side)
                    delta = predict_delta_chunk(
                        policy, top, side, sim.astype(np.float32), qvel.astype(np.float32),
                        depth_config=depth_config, device=device,
                    )
                    immediate_delta = monitor.check_prediction(delta)
                    monitor.check_cycle(time.perf_counter() - started, period)
                    if step % 30 == 0:
                        target = np.clip(sim + delta[0], environment.task_ctrl_low, environment.task_ctrl_high)
                        _, reported = plan_command(
                            target, real, mapping, travel_low, travel_high,
                            margin=margin, slew=slew,
                        )
                        print(
                            f"step={step} would_send={np.round(reported, 2)} "
                            f"max_delta={immediate_delta:.3f}rad "
                            f"inference_ms={(time.perf_counter() - started) * 1000:.1f}",
                            flush=True,
                        )
                    time.sleep(max(0.0, period - (time.perf_counter() - started)))
            return

        print(
            "This will energise and move the SO101 using live Top+Side depth. "
            "The arm first aligns to the training home pose.",
            flush=True,
        )
        if input("Type ENGAGE to continue: ").strip() != "ENGAGE":
            print("Aborted; nothing was sent.")
            return
        with follower_reader(args.port, args.robot_id) as read_values:
            previous_real = mapping.canonicalize_real(read_values())
        panel = EmergencyStopPanel()
        robot = follower_writer(args.port, args.robot_id)
        previous_real = _align(
            robot, mapping, home, previous_real, travel_low, travel_high, panel,
            margin=margin, slew=align_slew, period=period,
        )
        print(
            f"Running {args.preflight_frames} preflight frames at the aligned home pose; "
            "motors hold position and policy commands are not sent.",
            flush=True,
        )
        velocity_estimator = JointVelocityEstimator(control_period_s=period)
        for preflight_step in range(args.preflight_frames):
            started = time.perf_counter()
            panel.pump()
            if panel.stopped:
                raise EmergencyStop("stop pressed during policy preflight")
            observation = robot.get_observation()
            measured_real = mapping.canonicalize_real(observation_to_real_degrees(observation))
            tracking_ratio = monitor.check_tracking(previous_real, measured_real)
            sim = mapping.real_to_sim(measured_real)
            qvel = velocity_estimator.update(sim)
            top, side = cameras.read()
            valid = monitor.check_depth_pair(top, side)
            if visualizer is not None:
                visualizer.update(sim, top, side)
            delta = predict_delta_chunk(
                policy,
                top,
                side,
                sim.astype(np.float32),
                qvel.astype(np.float32),
                depth_config=depth_config,
                device=device,
            )
            immediate_delta = monitor.check_prediction(delta)
            elapsed = time.perf_counter() - started
            monitor.check_cycle(elapsed, period)
            if preflight_step == 0 or preflight_step == args.preflight_frames - 1:
                print(
                    f"preflight={preflight_step + 1}/{args.preflight_frames} "
                    f"valid_top={valid['top']:.3f} valid_side={valid['side_depth']:.3f} "
                    f"max_delta={immediate_delta:.3f}rad tracking_ratio={tracking_ratio:.2f} "
                    f"loop_ms={elapsed * 1000:.1f}",
                    flush=True,
                )
            time.sleep(max(0.0, period - elapsed))
        print("Preflight passed. Starting closed-loop policy control.", flush=True)

        ensemble = TemporalActionEnsembler(chunk_size=policy.config.chunk_size, decay=args.ensemble_decay)
        velocity_estimator = JointVelocityEstimator(control_period_s=period)
        for step in range(args.max_steps):
            started = time.perf_counter()
            panel.pump()
            if panel.stopped:
                raise EmergencyStop("stop pressed during policy execution")
            observation = robot.get_observation()
            measured_real = mapping.canonicalize_real(observation_to_real_degrees(observation))
            tracking_ratio = monitor.check_tracking(previous_real, measured_real)
            sim = mapping.real_to_sim(measured_real)
            qvel = velocity_estimator.update(sim)
            top, side = cameras.read()
            valid = monitor.check_depth_pair(top, side)
            if visualizer is not None:
                visualizer.update(sim, top, side)
            delta = predict_delta_chunk(
                policy, top, side, sim.astype(np.float32), qvel.astype(np.float32),
                depth_config=depth_config, device=device,
            )
            immediate_delta = monitor.check_prediction(delta)
            absolute_chunk = np.clip(
                sim[None, :] + delta,
                environment.task_ctrl_low,
                environment.task_ctrl_high,
            )
            sim_target = ensemble.add_and_get(absolute_chunk)
            previous_real, reported = plan_command(
                sim_target, previous_real, mapping, travel_low, travel_high,
                margin=margin, slew=slew,
            )
            elapsed = time.perf_counter() - started
            monitor.check_cycle(elapsed, period)
            send_real_degrees(robot, reported)
            if step % 30 == 0:
                print(
                    f"step={step}/{args.max_steps} send={np.round(reported, 2)} "
                    f"max_delta={immediate_delta:.3f}rad tracking_ratio={tracking_ratio:.2f} "
                    f"valid=({valid['top']:.2f},{valid['side_depth']:.2f}) "
                    f"loop_ms={elapsed * 1000:.1f}",
                    flush=True,
                )
            time.sleep(max(0.0, period - elapsed))
        previous_real = _align(
            robot, mapping, home, previous_real, travel_low, travel_high, panel,
            margin=margin, slew=align_slew, period=period,
        )
        print("Policy cycle completed and arm returned to home.", flush=True)
    except EmergencyStop as error:
        if robot is not None:
            held = hold_where_it_is(robot, mapping)
            keep_torque = True
            print(f"EMERGENCY STOP: {error}; holding at {np.round(held, 2)}", flush=True)
    except KeyboardInterrupt:
        if robot is not None:
            held = hold_where_it_is(robot, mapping)
            keep_torque = True
            print(f"KEYBOARD STOP: holding at {np.round(held, 2)}", flush=True)
    except Exception as error:
        if robot is not None:
            try:
                held = hold_where_it_is(robot, mapping)
                keep_torque = True
                print(
                    f"RUNTIME SAFETY STOP: {type(error).__name__}: {error}; "
                    f"holding at {np.round(held, 2)}",
                    flush=True,
                )
            except Exception as hold_error:
                print(
                    f"FAILED TO HOLD after {type(error).__name__}: {error}; "
                    f"hold error: {hold_error}",
                    flush=True,
                )
        raise
    finally:
        if visualizer is not None:
            visualizer.close()
        if cameras is not None:
            cameras.close()
        if robot is not None:
            if keep_torque:
                robot.bus.disconnect(False)
                print("Port closed; servos remain energised and holding.")
            else:
                robot.disconnect()
        if panel is not None:
            panel.close()
        environment.close()


if __name__ == "__main__":
    main()
