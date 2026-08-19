from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np

from sim2real_joint_mapping import (
    JOINT_NAMES,
    SIM_JOINT_LOW_RAD,
    SIM_JOINT_HIGH_RAD,
    observation_to_real_degrees,
)


CALIBRATION_VERSION = 4
FORMULA_REAL_TO_SIM = "sim_rad = scale_rad_per_real_unit * real_value + offset_rad"
FORMULA_SIM_TO_REAL = "real_value = (sim_rad - offset_rad) / scale_rad_per_real_unit"

# STS3215 magnetic encoder resolution. A measured sweep showed the five arm
# joints reporting exactly 360/4095 per tick, i.e. LeRobot reports physical
# degrees; readings land on half-integer multiples because the normalization is
# centred on tick 2047.5, which is easy to mistake for a 180/4095 quantum if
# only a couple of poses are inspected.
PHYSICAL_DEGREES_PER_TICK = 360.0 / 4096.0
OBSERVED_ARM_UNIT_PER_TICK = 360.0 / 4095.0
DEGREES_PER_TURN = 360.0
# A phase whose reported value swept nearly a full turn crossed the wrap point.
_WRAP_EXCURSION_DEGREES = 300.0

# The gripper is range-normalized to percent, so degree-based span diagnostics
# do not apply to it.
GRIPPER_INDEX = JOINT_NAMES.index("gripper")

_MINIMUM_REAL_SPAN = 1e-6
_TAIL_SAMPLES = 12
# Guards for the encoder-quantum estimate; see PhaseRecorder.estimated_tick.
_MIN_TICK_LEVELS = 20
_MAX_TICK_FRACTION_OF_SPAN = 0.02


def _joint_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(JOINT_NAMES),):
        raise ValueError(f"{name} must contain exactly {len(JOINT_NAMES)} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _real_period() -> np.ndarray:
    """A full turn for the arm joints; the percent gripper never wraps."""
    period = np.full(len(JOINT_NAMES), DEGREES_PER_TURN)
    period[GRIPPER_INDEX] = 0.0
    return period


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False, encoding="utf-8",
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    """Atomically write a JSON document, creating parent directories."""
    _atomic_write_json(Path(path).expanduser(), payload)


# --------------------------------------------------------------------------
# Sweep recording (pure, no I/O)
# --------------------------------------------------------------------------


class PhaseRecorder:
    """Accumulate joint samples for one sweep phase without touching hardware."""

    def __init__(self) -> None:
        self._samples: list[np.ndarray] = []
        self._lock = threading.Lock()

    def update(self, values: Sequence[float]) -> None:
        sample = _joint_vector(values, "sweep sample")
        with self._lock:
            self._samples.append(sample)

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._samples)

    def _stack(self) -> np.ndarray:
        with self._lock:
            if not self._samples:
                raise ValueError("sweep phase recorded no samples")
            return np.stack(self._samples)

    def settled(self, tail: int = _TAIL_SAMPLES) -> np.ndarray:
        """Return the median of the trailing samples, i.e. the held pose.

        A median over the tail is used rather than the extreme reading so an
        operator who overshoots and settles back still yields the held value,
        and so a single noisy tick does not define the calibration endpoint.
        """
        if tail <= 0:
            raise ValueError("tail must be positive")
        stacked = self._stack()
        return np.median(stacked[-tail:], axis=0)

    def excursion(self) -> np.ndarray:
        """Return per-joint (max - min) over the phase, i.e. operator wander."""
        stacked = self._stack()
        return stacked.max(axis=0) - stacked.min(axis=0)

    def estimated_tick(self, *, minimum_levels: int = _MIN_TICK_LEVELS) -> np.ndarray:
        """Return the encoder quantum per joint, or NaN where it is not trustworthy.

        Recovering the quantum needs a hand sweep that actually crawls through
        adjacent encoder counts. Three checks guard against reporting a bogus
        value, because a wrong quantum produces confident but false warnings
        downstream: enough distinct levels, a quantum far smaller than the
        travel, and every observed gap being a near-integer multiple of it.
        """
        stacked = self._stack()
        ticks = np.full(len(JOINT_NAMES), np.nan, dtype=np.float64)
        for index in range(len(JOINT_NAMES)):
            unique = np.unique(stacked[:, index])
            if unique.size < minimum_levels:
                continue
            span = float(unique[-1] - unique[0])
            gaps = np.diff(unique)
            gaps = gaps[gaps > 1e-12]
            if gaps.size == 0 or span <= 0.0:
                continue
            candidate = float(gaps.min())
            if candidate > _MAX_TICK_FRACTION_OF_SPAN * span:
                continue
            ratios = gaps / candidate
            if float(np.max(np.abs(ratios - np.round(ratios)))) > 0.05:
                continue
            ticks[index] = candidate
        return ticks


# --------------------------------------------------------------------------
# Affine mapping with a real inverse
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AffineJointMapping:
    """Per-joint affine map between LeRobot real units and MuJoCo radians.

    Unlike the v1 `JointMapping`, the scale is a free parameter per joint
    rather than a fixed `pi/180`, so the arm joints and the range-normalized
    gripper share one representation and the inverse is exact.

    The arm joints report an angle on a circle, so a joint whose travel crosses
    the seam is calibrated on an unwrapped branch that the robot never reports
    directly. `real_period` and `real_reference` record that branch, so
    `real_to_sim` accepts whatever the robot reports and `sim_to_reported_real`
    returns something the robot will accept. Without this a joint calibrated
    across the seam is commanded a full turn away from where it should go.
    """

    scale_rad_per_real_unit: np.ndarray
    offset_rad: np.ndarray
    real_period: np.ndarray | None = None
    real_reference: np.ndarray | None = None

    def __post_init__(self) -> None:
        scale = _joint_vector(self.scale_rad_per_real_unit, "scale_rad_per_real_unit")
        offset = _joint_vector(self.offset_rad, "offset_rad")
        if np.any(np.abs(scale) < 1e-12):
            raise ValueError("scale_rad_per_real_unit must be nonzero for every joint")
        period = (
            np.zeros(len(JOINT_NAMES))
            if self.real_period is None
            else _joint_vector(self.real_period, "real_period")
        )
        if np.any(period < 0.0):
            raise ValueError("real_period must be non-negative; use 0 for a joint that never wraps")
        reference = (
            np.zeros(len(JOINT_NAMES))
            if self.real_reference is None
            else _joint_vector(self.real_reference, "real_reference")
        )
        object.__setattr__(self, "scale_rad_per_real_unit", scale)
        object.__setattr__(self, "offset_rad", offset)
        object.__setattr__(self, "real_period", period)
        object.__setattr__(self, "real_reference", reference)

    def _wrapping(self) -> np.ndarray:
        return self.real_period > 0.0

    def canonicalize_real(self, real_values: Sequence[float]) -> np.ndarray:
        """Move each reported value onto the revolution this fit was made on."""
        values = _joint_vector(real_values, "real_values")
        wrapping = self._wrapping()
        if not np.any(wrapping):
            return values
        safe_period = np.where(wrapping, self.real_period, 1.0)
        turns = np.round((self.real_reference - values) / safe_period)
        return np.where(wrapping, values + turns * safe_period, values)

    def to_reported_domain(self, real_values: Sequence[float]) -> np.ndarray:
        """Fold unwrapped values back into the +/- half-turn the robot reports."""
        values = _joint_vector(real_values, "real_values")
        wrapping = self._wrapping()
        if not np.any(wrapping):
            return values
        safe_period = np.where(wrapping, self.real_period, 1.0)
        folded = np.mod(values + safe_period / 2.0, safe_period) - safe_period / 2.0
        return np.where(wrapping, folded, values)

    def real_to_sim(self, real_values: Sequence[float]) -> np.ndarray:
        return self.scale_rad_per_real_unit * self.canonicalize_real(real_values) + self.offset_rad

    def sim_to_real(self, sim_radians: Sequence[float]) -> np.ndarray:
        """Return the value on the calibrated branch, which may exceed one turn."""
        sim = _joint_vector(sim_radians, "sim_radians")
        return (sim - self.offset_rad) / self.scale_rad_per_real_unit

    def sim_to_reported_real(self, sim_radians: Sequence[float]) -> np.ndarray:
        """Return a value in the domain the robot reports and accepts."""
        return self.to_reported_domain(self.sim_to_real(sim_radians))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": CALIBRATION_VERSION,
            "joint_names": list(JOINT_NAMES),
            "real_units": ["lerobot_degree"] * 5 + ["percent_0_100"],
            "sim_unit": "radian",
            "formula_real_to_sim": FORMULA_REAL_TO_SIM,
            "formula_sim_to_real": FORMULA_SIM_TO_REAL,
            "scale_rad_per_real_unit": self.scale_rad_per_real_unit.tolist(),
            "offset_rad": self.offset_rad.tolist(),
            "real_period": self.real_period.tolist(),
            "real_reference": self.real_reference.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AffineJointMapping":
        version = payload.get("version")
        if version != CALIBRATION_VERSION:
            raise ValueError(
                f"calibration version {version!r} is not supported; expected {CALIBRATION_VERSION}. "
                "v1/v2 mapping files use a different formula and must be re-swept."
            )
        if payload.get("joint_names") != list(JOINT_NAMES):
            raise ValueError("mapping joint_names do not match the SO101 joint order")
        if payload.get("formula_real_to_sim") != FORMULA_REAL_TO_SIM:
            raise ValueError("mapping formula_real_to_sim does not match this loader")
        if "scale_rad_per_real_unit" not in payload:
            raise ValueError("mapping has no scale_rad_per_real_unit")
        return cls(
            scale_rad_per_real_unit=payload["scale_rad_per_real_unit"],
            offset_rad=payload["offset_rad"],
            real_period=payload.get("real_period"),
            real_reference=payload.get("real_reference"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "AffineJointMapping":
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("calibration file must contain a JSON object")
        return cls.from_dict(payload)


def fit_endpoint_mapping(
    real_at_sim_low: Sequence[float],
    real_at_sim_high: Sequence[float],
    sim_low: Sequence[float] = SIM_JOINT_LOW_RAD,
    sim_high: Sequence[float] = SIM_JOINT_HIGH_RAD,
) -> AffineJointMapping:
    """Map each joint's measured travel endpoints onto the MuJoCo joint range."""
    low_real = _joint_vector(real_at_sim_low, "real_at_sim_low")
    high_real = _joint_vector(real_at_sim_high, "real_at_sim_high")
    low_sim = _joint_vector(sim_low, "sim_low")
    high_sim = _joint_vector(sim_high, "sim_high")
    if np.any(high_sim <= low_sim):
        raise ValueError("sim_high must exceed sim_low for every joint")
    span = high_real - low_real
    degenerate = np.abs(span) < _MINIMUM_REAL_SPAN
    if np.any(degenerate):
        names = [JOINT_NAMES[index] for index in np.flatnonzero(degenerate)]
        raise ValueError(
            "measured real travel is zero for: " + ", ".join(names) +
            " -- the joint was not moved between the two sweep phases"
        )
    scale = (high_sim - low_sim) / span
    offset = low_sim - scale * low_real
    return AffineJointMapping(
        scale_rad_per_real_unit=scale,
        offset_rad=offset,
        real_period=_real_period(),
        real_reference=0.5 * (low_real + high_real),
    )


# --------------------------------------------------------------------------
# Wrap-around correction
# --------------------------------------------------------------------------


ARM_INDICES = tuple(index for index in range(len(JOINT_NAMES)) if index != GRIPPER_INDEX)



def unwrap_endpoints(
    real_at_sim_low: Sequence[float],
    real_at_sim_high: Sequence[float],
    excursion_at_sim_low: Sequence[float],
    excursion_at_sim_high: Sequence[float],
    *,
    period: float = DEGREES_PER_TURN,
    threshold: float = _WRAP_EXCURSION_DEGREES,
) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """Put both endpoints of each arm joint on the same revolution.

    The servo reports an angle on a circle, so a joint whose travel crosses the
    wrap point yields endpoints that look far closer together than the arm
    actually moved. The tell is the excursion: a phase that reported nearly a
    full turn of motion crossed the seam, because no joint here travels that
    far. Deciding from the excursion keeps this independent of the MuJoCo
    ranges, which is what the span diagnostic is supposed to be testing.
    """
    low = _joint_vector(real_at_sim_low, "real_at_sim_low").copy()
    high = _joint_vector(real_at_sim_high, "real_at_sim_high").copy()
    excursion_low = _joint_vector(excursion_at_sim_low, "excursion_at_sim_low")
    excursion_high = _joint_vector(excursion_at_sim_high, "excursion_at_sim_high")
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be positive and finite")

    corrected = [False] * len(JOINT_NAMES)
    for index in ARM_INDICES:
        if max(float(excursion_low[index]), float(excursion_high[index])) <= threshold:
            continue
        high[index] += period if high[index] < low[index] else -period
        corrected[index] = True
    return low, high, corrected


def fit_fixed_unit_mapping(
    real_at_sim_low: Sequence[float],
    real_at_sim_high: Sequence[float],
    sim_low: Sequence[float] = SIM_JOINT_LOW_RAD,
    sim_high: Sequence[float] = SIM_JOINT_HIGH_RAD,
) -> tuple[AffineJointMapping, np.ndarray]:
    """Lock each arm joint's scale to the known unit and fit only its offset.

    Endpoint fitting silently rescales a joint whenever the MuJoCo range and the
    physical travel disagree. Once the reported unit is known to be degrees, the
    scale is not something to fit: only the sign and the zero are unknown. The
    gripper keeps its endpoint fit because its percent unit is defined by the
    travel itself. Returns the mapping and the per-endpoint residual in degrees,
    which is where any range mismatch becomes visible.
    """
    low_real = _joint_vector(real_at_sim_low, "real_at_sim_low")
    high_real = _joint_vector(real_at_sim_high, "real_at_sim_high")
    low_sim = _joint_vector(sim_low, "sim_low")
    high_sim = _joint_vector(sim_high, "sim_high")

    endpoint = fit_endpoint_mapping(low_real, high_real, low_sim, high_sim)
    scale = endpoint.scale_rad_per_real_unit.copy()
    offset = endpoint.offset_rad.copy()
    for index in ARM_INDICES:
        sign = 1.0 if high_real[index] >= low_real[index] else -1.0
        scale[index] = sign * np.pi / 180.0
        # Two endpoints, one unknown: the mean centres any range mismatch
        # instead of letting one end absorb all of it.
        offset[index] = 0.5 * (
            (low_sim[index] - scale[index] * low_real[index])
            + (high_sim[index] - scale[index] * high_real[index])
        )
    mapping = AffineJointMapping(
        scale_rad_per_real_unit=scale,
        offset_rad=offset,
        real_period=_real_period(),
        real_reference=0.5 * (low_real + high_real),
    )
    residual_deg = np.rad2deg(
        np.stack((
            mapping.real_to_sim(low_real) - low_sim,
            mapping.real_to_sim(high_real) - high_sim,
        ))
    )
    return mapping, residual_deg


# --------------------------------------------------------------------------
# Diagnostics: does endpoint mapping actually hold up?
# --------------------------------------------------------------------------


def sweep_diagnostics(
    real_at_sim_low: Sequence[float],
    real_at_sim_high: Sequence[float],
    *,
    sim_low: Sequence[float] = SIM_JOINT_LOW_RAD,
    sim_high: Sequence[float] = SIM_JOINT_HIGH_RAD,
    estimated_tick: Sequence[float] | None = None,
) -> list[dict[str, object]]:
    """Report, per joint, whether the sim range plausibly matches real travel.

    Endpoint calibration silently absorbs any mismatch between the modelled
    MuJoCo limits and the physical hard stops into the scale. This surfaces
    that mismatch instead: for the arm joints the implied degrees-per-real-unit
    should land near 1.0 (reported values are physical degrees) or near 2.0
    (reported values are half-degrees). Anything else means the two ranges
    disagree and the scale is compensating for it.
    """
    low_real = _joint_vector(real_at_sim_low, "real_at_sim_low")
    high_real = _joint_vector(real_at_sim_high, "real_at_sim_high")
    low_sim = _joint_vector(sim_low, "sim_low")
    high_sim = _joint_vector(sim_high, "sim_high")
    ticks = (
        np.full(len(JOINT_NAMES), np.nan)
        if estimated_tick is None
        else _joint_vector(estimated_tick, "estimated_tick")
    )

    report: list[dict[str, object]] = []
    for index, name in enumerate(JOINT_NAMES):
        real_span = float(abs(high_real[index] - low_real[index]))
        sim_span_deg = float(np.rad2deg(high_sim[index] - low_sim[index]))
        entry: dict[str, object] = {
            "joint": name,
            "real_at_sim_low": float(low_real[index]),
            "real_at_sim_high": float(high_real[index]),
            "real_span": real_span,
            "sim_span_deg": sim_span_deg,
            "reversed_axis": bool(high_real[index] < low_real[index]),
            "estimated_real_tick": None if np.isnan(ticks[index]) else float(ticks[index]),
            "warnings": [],
        }
        warnings: list[str] = entry["warnings"]  # type: ignore[assignment]

        if real_span < _MINIMUM_REAL_SPAN:
            warnings.append("joint did not move between sweep phases")
            report.append(entry)
            continue

        if not np.isnan(ticks[index]):
            entry["measured_ticks_in_span"] = round(real_span / float(ticks[index]))

        if index == GRIPPER_INDEX:
            entry["unit_note"] = "percent_0_100; degree-span diagnostics do not apply"
            if real_span < 50.0:
                warnings.append(
                    f"gripper travelled only {real_span:.1f}% of its normalized range; "
                    "expected close to 100"
                )
            warnings.append(
                "validate the gripper by jaw gap in mm against the sim jaw sites, not by angle"
            )
            report.append(entry)
            continue

        implied = sim_span_deg / real_span
        entry["implied_deg_per_real_unit"] = implied
        if abs(implied - 1.0) <= 0.05:
            entry["unit_interpretation"] = "reported values are physical degrees (scale ~1.0)"
        elif abs(implied - 2.0) <= 0.10:
            entry["unit_interpretation"] = "reported values are half-degrees (scale ~2.0)"
        else:
            entry["unit_interpretation"] = "unexpected"
            warnings.append(
                f"implied scale {implied:.3f} deg per real unit is neither ~1.0 nor ~2.0; "
                "the MuJoCo range and the physical travel disagree, and the fitted scale "
                "is absorbing that error across the whole joint"
            )

        if not np.isnan(ticks[index]):
            entry["tick_matches_observed_arm_unit"] = bool(
                abs(float(ticks[index]) - OBSERVED_ARM_UNIT_PER_TICK) < 1e-4
            )
            physical_span_deg = real_span * (PHYSICAL_DEGREES_PER_TICK / float(ticks[index]))
            entry["physical_span_deg_from_ticks"] = physical_span_deg
            discrepancy = abs(physical_span_deg - sim_span_deg)
            entry["sim_vs_physical_span_deg"] = discrepancy
            if discrepancy > 5.0:
                warnings.append(
                    f"physical travel {physical_span_deg:.1f} deg differs from the MuJoCo range "
                    f"{sim_span_deg:.1f} deg by {discrepancy:.1f} deg"
                )

        report.append(entry)
    return report


def envelope_check(
    mapping: AffineJointMapping,
    real_at_sim_low: Sequence[float],
    real_at_sim_high: Sequence[float],
    task_low_rad: Sequence[float],
    task_high_rad: Sequence[float],
) -> list[dict[str, object]]:
    """Verify the policy's control envelope maps inside the measured real travel."""
    low_real = _joint_vector(real_at_sim_low, "real_at_sim_low")
    high_real = _joint_vector(real_at_sim_high, "real_at_sim_high")
    measured_low = np.minimum(low_real, high_real)
    measured_high = np.maximum(low_real, high_real)
    required_a = mapping.sim_to_real(task_low_rad)
    required_b = mapping.sim_to_real(task_high_rad)
    required_low = np.minimum(required_a, required_b)
    required_high = np.maximum(required_a, required_b)

    results = []
    for index, name in enumerate(JOINT_NAMES):
        reachable = bool(
            required_low[index] >= measured_low[index] - 1e-9
            and required_high[index] <= measured_high[index] + 1e-9
        )
        results.append({
            "joint": name,
            "required_real_low": float(required_low[index]),
            "required_real_high": float(required_high[index]),
            "measured_real_low": float(measured_low[index]),
            "measured_real_high": float(measured_high[index]),
            "reachable": reachable,
        })
    return results


def robot_calibration_fingerprint(path: str | Path) -> dict[str, object]:
    """Record the follower calibration identity so re-homing is detectable."""
    calibration_path = Path(path).expanduser()
    raw = calibration_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    return {
        "path": str(calibration_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "homing_offset": {
            name: payload.get(name, {}).get("homing_offset") for name in JOINT_NAMES
        },
    }


# --------------------------------------------------------------------------
# Hardware polling
# --------------------------------------------------------------------------


@contextmanager
def follower_reader(port: str, robot_id: str) -> Iterator[Callable[[], np.ndarray]]:
    """Yield a torque-free position reader over one persistent bus connection."""
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError as error:
        raise SystemExit(
            "lerobot is not installed in this Python environment.\n"
            "  Activate the environment that talks to the arm, or install lerobot here.\n"
            "  Check with:  python -c \"import lerobot; print(lerobot.__file__)\"\n"
            "  This script needs lerobot; MuJoCo is optional -- add --no-viewer to run\n"
            "  the sweep in an environment that only has lerobot."
        ) from error

    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id))
    robot.bus.connect()
    try:
        yield lambda: observation_to_real_degrees(robot.get_observation())
    finally:
        # False closes the port without writing Torque_Enable, so the arm stays
        # back-drivable and nothing lurches when the script exits.
        robot.bus.disconnect(False)


def poll_until(
    read_values: Callable[[], np.ndarray],
    stop: threading.Event,
    recorder: PhaseRecorder,
    *,
    period_s: float = 0.05,
) -> None:
    while not stop.is_set():
        try:
            recorder.update(read_values())
        except Exception:  # noqa: BLE001 - a dropped frame must not kill the sweep
            pass
        stop.wait(period_s)


def record_phase(
    read_values: Callable[[], np.ndarray],
    wait_for_operator: Callable[[], None],
    *,
    period_s: float = 0.05,
) -> PhaseRecorder:
    """Poll the arm in the background while the operator holds a pose."""
    recorder = PhaseRecorder()
    stop = threading.Event()
    worker = threading.Thread(
        target=poll_until, args=(read_values, stop, recorder), kwargs={"period_s": period_s},
        daemon=True,
    )
    worker.start()
    try:
        wait_for_operator()
    finally:
        stop.set()
        worker.join(timeout=2.0)
    if recorder.sample_count == 0:
        raise RuntimeError("no joint samples were read during the sweep phase")
    return recorder


# --------------------------------------------------------------------------
# Reference images and the measurement handoff file
# --------------------------------------------------------------------------


MEASUREMENTS_VERSION = 1
TARGETS_VERSION = 1


def reference_image_name(joint_index: int, joint: str, phase: str) -> str:
    """Name shared by the pose renderer and the collector so they agree."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    order = 2 * joint_index + (0 if phase == "high" else 1) + 1
    return f"{order:02d}_{joint}_{phase.upper()}.png"


def save_targets(path: str | Path, sim_low: Sequence[float], sim_high: Sequence[float]) -> None:
    low = _joint_vector(sim_low, "sim_low")
    high = _joint_vector(sim_high, "sim_high")
    _atomic_write_json(Path(path).expanduser(), {
        "version": TARGETS_VERSION,
        "kind": "sweep_targets",
        "joint_names": list(JOINT_NAMES),
        "sim_joint_low_rad": low.tolist(),
        "sim_joint_high_rad": high.tolist(),
        "images": {
            f"{joint}:{phase}": reference_image_name(index, joint, phase)
            for index, joint in enumerate(JOINT_NAMES)
            for phase in PHASES
        },
    })


def load_targets(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if payload.get("kind") != "sweep_targets" or payload.get("joint_names") != list(JOINT_NAMES):
        raise ValueError("targets file is not a sweep_targets file for this joint order")
    return (
        _joint_vector(payload["sim_joint_low_rad"], "sim_joint_low_rad"),
        _joint_vector(payload["sim_joint_high_rad"], "sim_joint_high_rad"),
    )


def save_measurements(
    path: str | Path,
    measurements: Mapping[str, object],
    *,
    sim_low: Sequence[float],
    sim_high: Sequence[float],
    metadata: Mapping[str, object] | None = None,
) -> None:
    payload = {
        "version": MEASUREMENTS_VERSION,
        "kind": "sweep_measurements",
        "joint_names": list(JOINT_NAMES),
        "sim_joint_low_rad": _joint_vector(sim_low, "sim_low").tolist(),
        "sim_joint_high_rad": _joint_vector(sim_high, "sim_high").tolist(),
        "real_at_sim_low": np.asarray(measurements["real_at_sim_low"]).tolist(),
        "real_at_sim_high": np.asarray(measurements["real_at_sim_high"]).tolist(),
        "excursion_at_sim_low": np.asarray(measurements["excursion_at_sim_low"]).tolist(),
        "excursion_at_sim_high": np.asarray(measurements["excursion_at_sim_high"]).tolist(),
        "estimated_real_tick": [
            None if np.isnan(value) else float(value)
            for value in np.asarray(measurements["estimated_tick"])
        ],
        "sample_counts": measurements["sample_counts"],
    }
    payload.update(dict(metadata or {}))
    _atomic_write_json(Path(path).expanduser(), payload)


def load_measurements(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("measurement file must contain a JSON object")
    if payload.get("kind") != "sweep_measurements":
        raise ValueError("file is not a sweep_measurements file")
    if payload.get("joint_names") != list(JOINT_NAMES):
        raise ValueError("measurement joint_names do not match the SO101 joint order")
    for key in ("real_at_sim_low", "real_at_sim_high", "sim_joint_low_rad", "sim_joint_high_rad"):
        if key not in payload:
            raise ValueError(f"measurement file has no {key}")
        payload[key] = _joint_vector(payload[key], key)
    ticks = payload.get("estimated_real_tick")
    payload["estimated_real_tick"] = (
        np.full(len(JOINT_NAMES), np.nan)
        if ticks is None
        else np.array([np.nan if value is None else float(value) for value in ticks])
    )
    return payload


def print_report(diagnostics: Sequence[Mapping[str, object]], envelope: Sequence[Mapping[str, object]]) -> None:
    print("\n" + "=" * 78)
    print(f"{'joint':<15}{'real low':>11}{'real high':>11}{'span':>10}{'deg/unit':>11}  interpretation")
    print("-" * 78)
    for entry in diagnostics:
        implied = entry.get("implied_deg_per_real_unit")
        shown = f"{implied:.3f}" if implied is not None else "-"
        note = entry.get("unit_interpretation") or entry.get("unit_note") or ""
        print(
            f"{entry['joint']:<15}"
            f"{entry['real_at_sim_low']:>11.3f}"
            f"{entry['real_at_sim_high']:>11.3f}"
            f"{entry['real_span']:>10.3f}"
            f"{shown:>11}  {note}"
        )
    print("=" * 78)
    for entry in diagnostics:
        for warning in entry["warnings"]:
            print(f"WARNING [{entry['joint']}] {warning}")
    if not envelope:
        print("NOTE: ran without MuJoCo, so the policy control envelope was not checked.")
    unreachable = [item["joint"] for item in envelope if not item["reachable"]]
    if unreachable:
        print(
            "WARNING: the policy control envelope maps outside the measured travel for: "
            + ", ".join(unreachable)
        )



# --------------------------------------------------------------------------
# Interactive sweep driver
# --------------------------------------------------------------------------


PHASES = ("high", "low")


def _phase_instruction(
    joint: str,
    phase: str,
    target_deg: float,
    viewer_open: bool,
    reference_image: str | None = None,
) -> str:
    if viewer_open:
        where = "The MuJoCo window now shows this joint at that limit."
    elif reference_image is not None:
        where = f"Match the reference picture: {reference_image}"
    else:
        where = f"(MuJoCo target: {target_deg:+.1f} deg)"
    return (
        f"\n[{joint}] move ONLY this joint to its {phase.upper()} mechanical limit.\n"
        f"  {where}\n"
        f"  Hold it there, then press Enter: "
    )


def pose_sim_joint(env, joint_index: int, sim_radians: float) -> None:
    """Show one joint at a limit using kinematics only -- no servo droop."""
    import mujoco

    pose = env.home_qpos.copy()
    pose[joint_index] = sim_radians
    env.data.qpos[env.qpos_ids] = pose
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = np.clip(pose, env.ctrl_low, env.ctrl_high)
    mujoco.mj_forward(env.model, env.data)


def run_interactive_sweep(
    read_values: Callable[[], np.ndarray],
    *,
    env=None,
    viewer=None,
    prompt: Callable[[str], str] = input,
    tail: int = _TAIL_SAMPLES,
    period_s: float = 0.05,
    reference_dir: str | Path | None = None,
    sim_low: Sequence[float] = SIM_JOINT_LOW_RAD,
    sim_high: Sequence[float] = SIM_JOINT_HIGH_RAD,
) -> dict[str, object]:
    """Walk the operator through 12 held poses and return the raw measurements."""
    low_sim = _joint_vector(sim_low, "sim_low")
    high_sim = _joint_vector(sim_high, "sim_high")
    settled = {phase: np.zeros(len(JOINT_NAMES)) for phase in PHASES}
    excursion = {phase: np.zeros(len(JOINT_NAMES)) for phase in PHASES}
    counts = {phase: [0] * len(JOINT_NAMES) for phase in PHASES}
    ticks = np.full(len(JOINT_NAMES), np.nan)

    for joint_index, joint in enumerate(JOINT_NAMES):
        for phase in PHASES:
            target = float(high_sim[joint_index] if phase == "high" else low_sim[joint_index])
            if env is not None:
                pose_sim_joint(env, joint_index, target)
                if viewer is not None:
                    viewer.sync()
            reference = (
                str(Path(reference_dir) / reference_image_name(joint_index, joint, phase))
                if reference_dir is not None
                else None
            )
            message = _phase_instruction(
                joint, phase, float(np.rad2deg(target)), viewer is not None, reference
            )
            recorder = record_phase(read_values, lambda: prompt(message), period_s=period_s)
            settled[phase][joint_index] = float(recorder.settled(tail)[joint_index])
            excursion[phase][joint_index] = float(recorder.excursion()[joint_index])
            counts[phase][joint_index] = recorder.sample_count
            phase_tick = recorder.estimated_tick()[joint_index]
            if not np.isnan(phase_tick):
                ticks[joint_index] = (
                    phase_tick if np.isnan(ticks[joint_index]) else min(ticks[joint_index], phase_tick)
                )
            print(
                f"    recorded {settled[phase][joint_index]:+10.4f}"
                f"  (wander {excursion[phase][joint_index]:.4f},"
                f" {recorder.sample_count} samples)"
            )

    return {
        "real_at_sim_low": settled["low"],
        "real_at_sim_high": settled["high"],
        "excursion_at_sim_low": excursion["low"],
        "excursion_at_sim_high": excursion["high"],
        "sample_counts": counts,
        "estimated_tick": ticks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep each SO101 joint to both mechanical limits with torque disabled "
            "and map the measured travel onto the MuJoCo joint ranges."
        )
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--output", type=Path, default=Path("sim2real_joint_map_swept.json"))
    parser.add_argument(
        "--robot-calibration",
        type=Path,
        default=Path("my_awesome_follower_arm.json"),
        help="LeRobot follower calibration file, fingerprinted into the output.",
    )
    parser.add_argument("--xml", default=None, help="Optional scene XML path.")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--tail",
        type=int,
        default=_TAIL_SAMPLES,
        help="Samples averaged at the end of each phase to define the held pose.",
    )
    parser.add_argument(
        "--gripper-sim-high",
        type=float,
        default=None,
        help=(
            "Override the sim radian value paired with the fully open gripper. "
            "Defaults to the joint limit 1.74533; pass 1.0 to pair full opening "
            "with the env's operational open position instead."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    if args.tail <= 0:
        raise ValueError("--tail must be positive")

    sim_low = SIM_JOINT_LOW_RAD.copy()
    sim_high = SIM_JOINT_HIGH_RAD.copy()
    if args.gripper_sim_high is not None:
        sim_high[GRIPPER_INDEX] = float(args.gripper_sim_high)

    print("SO101 joint order:", ", ".join(JOINT_NAMES))
    env = None
    viewer_handle = None
    env_task_low = None
    env_task_high = None
    try:
        # Reach the arm before creating any OpenGL context. A failure here used
        # to unwind through an already-open viewer and segfault on teardown.
        with follower_reader(args.port, args.robot_id) as read_values:
            read_values()
            print(f"Connected to {args.robot_id} on {args.port}.")
            if not args.no_viewer:
                import mujoco.viewer

                from so101_ball_bins_env import SO101BallBinsEnv

                env = SO101BallBinsEnv(spawn_stage="stage1", xml_path=args.xml)
                env_task_low = env.task_ctrl_low.copy()
                env_task_high = env.task_ctrl_high.copy()
                viewer_handle = mujoco.viewer.launch_passive(env.model, env.data)
            else:
                print("Running without MuJoCo; each prompt prints its target angle instead.")
            print(
                "\nTorque stays disabled for the whole session -- move each joint BY HAND.\n"
                "Do not drive the servos into their hard stops.\n"
            )
            measurements = run_interactive_sweep(
                read_values,
                env=env,
                viewer=viewer_handle,
                tail=args.tail,
                sim_low=sim_low,
                sim_high=sim_high,
            )
    finally:
        if viewer_handle is not None:
            try:
                viewer_handle.close()
            except Exception:  # noqa: BLE001 - teardown must not mask a real error
                pass
        if env is not None:
            env.close()

    real_low = measurements["real_at_sim_low"]
    real_high = measurements["real_at_sim_high"]
    mapping = fit_endpoint_mapping(real_low, real_high, sim_low, sim_high)
    diagnostics = sweep_diagnostics(
        real_low, real_high,
        sim_low=sim_low, sim_high=sim_high,
        estimated_tick=measurements["estimated_tick"],
    )
    envelope = (
        envelope_check(mapping, real_low, real_high, env_task_low, env_task_high)
        if env_task_low is not None
        else []
    )

    fingerprint = None
    if args.robot_calibration is not None and Path(args.robot_calibration).expanduser().is_file():
        fingerprint = robot_calibration_fingerprint(args.robot_calibration)

    payload = dict(mapping.to_dict())
    payload.update({
        "method": "endpoint_sweep",
        "sim_joint_low_rad": sim_low.tolist(),
        "sim_joint_high_rad": sim_high.tolist(),
        "measurements": {
            "real_at_sim_low": np.asarray(real_low).tolist(),
            "real_at_sim_high": np.asarray(real_high).tolist(),
            "excursion_at_sim_low": np.asarray(measurements["excursion_at_sim_low"]).tolist(),
            "excursion_at_sim_high": np.asarray(measurements["excursion_at_sim_high"]).tolist(),
            "estimated_real_tick": [
                None if np.isnan(value) else float(value)
                for value in np.asarray(measurements["estimated_tick"])
            ],
            "sample_counts": measurements["sample_counts"],
        },
        "diagnostics": diagnostics,
        "task_envelope_check": envelope,
        "robot_calibration": fingerprint,
        "port": args.port,
        "robot_id": args.robot_id,
    })
    _atomic_write_json(Path(args.output).expanduser(), payload)

    print("\n" + "=" * 78)
    print(f"{'joint':<15}{'real low':>11}{'real high':>11}{'span':>10}{'deg/unit':>11}  interpretation")
    print("-" * 78)
    for entry in diagnostics:
        implied = entry.get("implied_deg_per_real_unit")
        print(
            f"{entry['joint']:<15}"
            f"{entry['real_at_sim_low']:>11.3f}"
            f"{entry['real_at_sim_high']:>11.3f}"
            f"{entry['real_span']:>10.3f}"
            f"{(f'{implied:.3f}' if implied is not None else '-'):>11}"
            f"  {entry.get('unit_interpretation', entry.get('unit_note', ''))}"
        )
    print("=" * 78)

    problems = [(entry["joint"], warning) for entry in diagnostics for warning in entry["warnings"]]
    for joint, warning in problems:
        print(f"WARNING [{joint}] {warning}")
    if not envelope:
        print("NOTE: ran without MuJoCo, so the policy control envelope was not checked.")
    unreachable = [item["joint"] for item in envelope if not item["reachable"]]
    if unreachable:
        print(
            "WARNING: the policy control envelope maps outside the measured travel for: "
            + ", ".join(unreachable)
        )
    print(f"\nSaved calibration: {Path(args.output).expanduser().resolve()}")
    print("No motor action was sent.")
    return payload


if __name__ == "__main__":
    main()
