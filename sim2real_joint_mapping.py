from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

SIM_JOINT_LOW_RAD = np.array(
    [-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453],
    dtype=np.float64,
)
SIM_JOINT_HIGH_RAD = np.array(
    [1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533],
    dtype=np.float64,
)


def _joint_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(JOINT_NAMES),):
        raise ValueError(f"{name} must contain exactly {len(JOINT_NAMES)} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class JointMapping:
    """Convert LeRobot calibrated degrees to and from MuJoCo joint radians."""

    signs: np.ndarray
    offset_rad: np.ndarray

    def __post_init__(self) -> None:
        signs = _joint_vector(self.signs, "signs")
        offset_rad = _joint_vector(self.offset_rad, "offset_rad")
        if not np.all(np.isin(signs, (-1.0, 1.0))):
            raise ValueError("signs must contain only -1 or 1")
        object.__setattr__(self, "signs", signs)
        object.__setattr__(self, "offset_rad", offset_rad)

    @classmethod
    def identity(cls) -> "JointMapping":
        return cls(np.ones(len(JOINT_NAMES)), np.zeros(len(JOINT_NAMES)))

    def real_degrees_to_sim_radians(self, real_degrees: Sequence[float]) -> np.ndarray:
        real = _joint_vector(real_degrees, "real_degrees")
        return self.signs * np.deg2rad(real) + self.offset_rad

    def sim_radians_to_real_degrees(self, sim_radians: Sequence[float]) -> np.ndarray:
        sim = _joint_vector(sim_radians, "sim_radians")
        return np.rad2deg((sim - self.offset_rad) / self.signs)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "joint_names": list(JOINT_NAMES),
            "real_unit": "degree",
            "sim_unit": "radian",
            "formula_real_to_sim": "sim_rad = sign * deg2rad(real_deg) + offset_rad",
            "signs": self.signs.tolist(),
            "offset_rad": self.offset_rad.tolist(),
            "sim_joint_low_rad": SIM_JOINT_LOW_RAD.tolist(),
            "sim_joint_high_rad": SIM_JOINT_HIGH_RAD.tolist(),
        }

    def save(self, path: str | Path, metadata: Mapping[str, object] | None = None) -> None:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(metadata or {})
        payload.update(self.to_dict())
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "JointMapping":
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if payload.get("joint_names") != list(JOINT_NAMES):
            raise ValueError("mapping joint_names do not match the SO101 joint order")
        return cls(signs=payload["signs"], offset_rad=payload["offset_rad"])


def observation_to_real_degrees(observation: Mapping[str, object]) -> np.ndarray:
    missing = [f"{joint}.pos" for joint in JOINT_NAMES if f"{joint}.pos" not in observation]
    if missing:
        raise KeyError(f"missing joint observations: {', '.join(missing)}")
    return np.array([float(observation[f"{joint}.pos"]) for joint in JOINT_NAMES], dtype=np.float64)


def _parse_six_csv(text: str, name: str) -> np.ndarray:
    try:
        values = [float(value.strip()) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be six comma-separated numbers") from error
    try:
        return _joint_vector(values, name)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def read_degrees_from_robot_bus(robot: object) -> np.ndarray:
    """Read calibrated positions without running follower motor configuration."""
    connected = False
    try:
        robot.bus.connect()
        connected = True
        return observation_to_real_degrees(robot.get_observation())
    finally:
        if connected:
            # False closes the serial port without writing Torque_Enable.
            robot.bus.disconnect(False)


def read_real_joint_degrees(port: str, robot_id: str) -> np.ndarray:
    # Import lazily so conversion and tests work without a LeRobot installation.
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id))
    return read_degrees_from_robot_bus(robot)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read an SO101 follower and save the calibrated degree/MuJoCo radian mapping."
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--output", default="sim2real_joint_map.json")
    parser.add_argument(
        "--signs",
        default="1,1,1,1,1,1",
        help="Six axis signs in SO101 joint order; each value must be -1 or 1.",
    )
    parser.add_argument(
        "--offset-deg",
        default="0,0,0,0,0,0",
        help="Six MuJoCo zero offsets in degrees, in SO101 joint order.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    signs = _parse_six_csv(args.signs, "signs")
    offset_deg = _parse_six_csv(args.offset_deg, "offset_deg")
    mapping = JointMapping(signs=signs, offset_rad=np.deg2rad(offset_deg))

    real_degrees = read_real_joint_degrees(args.port, args.robot_id)
    sim_radians = mapping.real_degrees_to_sim_radians(real_degrees)
    within_limits = (sim_radians >= SIM_JOINT_LOW_RAD) & (sim_radians <= SIM_JOINT_HIGH_RAD)

    output = Path(args.output).expanduser().resolve()
    mapping.save(
        output,
        metadata={
            "port": args.port,
            "robot_id": args.robot_id,
            "captured_real_deg": real_degrees.tolist(),
            "captured_sim_rad": sim_radians.tolist(),
            "captured_within_sim_limits": within_limits.tolist(),
        },
    )

    print("SO101 joint order:", ", ".join(JOINT_NAMES))
    print("captured real degrees:", np.array2string(real_degrees, precision=4))
    print("mapped MuJoCo radians:", np.array2string(sim_radians, precision=4))
    print("within MuJoCo limits:", within_limits.tolist())
    print("saved mapping:", output)
    if not np.all(within_limits):
        outside = [JOINT_NAMES[index] for index, valid in enumerate(within_limits) if not valid]
        print("WARNING: mapped value is outside the training limits for:", ", ".join(outside))
    print("No motor action was sent.")


if __name__ == "__main__":
    main()
