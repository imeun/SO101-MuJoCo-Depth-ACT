from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

import mujoco
import numpy as np


@dataclass(frozen=True)
class DepthConfig:
    width: int = 320
    height: int = 240
    near_m: float = 0.20
    far_m: float = 1.00
    invalid_fill_m: float = 1.00
    noise_std_range_m: tuple[float, float] = (0.001, 0.006)
    invalid_pixel_probability: float = 0.01
    frame_bias_range_m: tuple[float, float] = (-0.003, 0.003)
    depth_scale_range: tuple[float, float] = (0.995, 1.005)
    invalid_pixel_probability_range: tuple[float, float] = (0.005, 0.02)
    edge_dropout_probability_range: tuple[float, float] = (0.0, 0.20)
    edge_threshold_m: float = 0.015
    hole_count_range: tuple[int, int] = (0, 3)
    hole_size_range_px: tuple[int, int] = (3, 20)

    def __post_init__(self) -> None:
        _validate_size(self.width, self.height)
        near_m = _validate_finite_real(self.near_m, "near_m")
        far_m = _validate_finite_real(self.far_m, "far_m")
        invalid_fill_m = _validate_finite_real(self.invalid_fill_m, "invalid_fill_m")
        if near_m >= far_m:
            raise ValueError("near_m must be less than far_m")
        if not near_m <= invalid_fill_m <= far_m:
            raise ValueError("invalid_fill_m must be within [near_m, far_m]")
        try:
            noise_min, noise_max = self.noise_std_range_m
        except (TypeError, ValueError) as error:
            raise ValueError("noise_std_range_m must contain two values") from error
        noise_min = _validate_finite_real(noise_min, "noise_std_range_m minimum")
        noise_max = _validate_finite_real(noise_max, "noise_std_range_m maximum")
        if noise_min < 0.0 or noise_min > noise_max:
            raise ValueError("noise_std_range_m must be non-negative and ordered")
        probability = _validate_finite_real(self.invalid_pixel_probability, "invalid_pixel_probability")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("invalid_pixel_probability must be within [0, 1]")
        _validate_real_pair(self.frame_bias_range_m, "frame_bias_range_m")
        scale_min, scale_max = _validate_real_pair(self.depth_scale_range, "depth_scale_range")
        if scale_min <= 0.0:
            raise ValueError("depth_scale_range must be positive")
        invalid_min, invalid_max = _validate_real_pair(
            self.invalid_pixel_probability_range, "invalid_pixel_probability_range"
        )
        edge_min, edge_max = _validate_real_pair(
            self.edge_dropout_probability_range, "edge_dropout_probability_range"
        )
        if not 0.0 <= invalid_min <= invalid_max <= 1.0:
            raise ValueError("invalid_pixel_probability_range must be within [0, 1]")
        if not 0.0 <= edge_min <= edge_max <= 1.0:
            raise ValueError("edge_dropout_probability_range must be within [0, 1]")
        edge_threshold = _validate_finite_real(self.edge_threshold_m, "edge_threshold_m")
        if edge_threshold <= 0.0:
            raise ValueError("edge_threshold_m must be positive")
        _validate_integer_pair(self.hole_count_range, "hole_count_range", minimum=0)
        _validate_integer_pair(self.hole_size_range_px, "hole_size_range_px", minimum=1)


def _validate_size(width: int, height: int) -> None:
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def _validate_finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real value")
    return float(value)


def _validate_real_pair(value: object, name: str) -> tuple[float, float]:
    try:
        minimum, maximum = value
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain two values") from error
    minimum = _validate_finite_real(minimum, f"{name} minimum")
    maximum = _validate_finite_real(maximum, f"{name} maximum")
    if minimum > maximum:
        raise ValueError(f"{name} must be ordered")
    return minimum, maximum


def _validate_integer_pair(value: object, name: str, *, minimum: int) -> tuple[int, int]:
    try:
        lower, upper = value
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain two values") from error
    if any(isinstance(item, bool) or not isinstance(item, numbers.Integral) for item in (lower, upper)):
        raise ValueError(f"{name} must contain integers")
    lower, upper = int(lower), int(upper)
    if lower < minimum or lower > upper:
        raise ValueError(f"{name} is invalid")
    return lower, upper


def _validate_depth_array(depth_m: np.ndarray) -> np.ndarray:
    if not isinstance(depth_m, np.ndarray):
        raise ValueError("depth must be a NumPy array")
    if depth_m.ndim != 2 or 0 in depth_m.shape:
        raise ValueError("depth must be a nonempty 2D array")
    if not np.issubdtype(depth_m.dtype, np.number) or np.issubdtype(depth_m.dtype, np.complexfloating):
        raise ValueError("depth must have a real numeric dtype")
    return depth_m


def resize_depth_nearest(depth_m: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return a float32 HxW nearest-neighbor resize using NumPy only."""
    depth_m = _validate_depth_array(depth_m)
    _validate_size(width, height)
    source_height, source_width = depth_m.shape
    y_indices = (np.arange(height) * source_height // height).astype(np.intp)
    x_indices = (np.arange(width) * source_width // width).astype(np.intp)
    return depth_m[np.ix_(y_indices, x_indices)].astype(np.float32, copy=True)


def preprocess_depth(
    depth_m: np.ndarray,
    config: DepthConfig = DepthConfig(),
    *,
    rng: np.random.Generator | None = None,
    augment: bool = False,
) -> np.ndarray:
    """Return normalized float32 depth with shape (1, H, W)."""
    if not isinstance(config, DepthConfig):
        raise ValueError("config must be a DepthConfig")
    if augment and rng is None:
        raise ValueError("rng is required when augment=True")

    resized = resize_depth_nearest(depth_m, config.width, config.height)
    valid = np.isfinite(resized) & (resized > 0.0)
    prepared = np.full(resized.shape, config.invalid_fill_m, dtype=np.float32)
    prepared[valid] = resized[valid]

    if augment:
        noise_std_m = rng.uniform(*config.noise_std_range_m)
        noise = rng.normal(0.0, noise_std_m, size=prepared.shape).astype(np.float32)
        prepared[valid] += noise[valid]
        prepared[rng.random(prepared.shape) < config.invalid_pixel_probability] = config.invalid_fill_m

    prepared = np.clip(prepared, config.near_m, config.far_m)
    normalized = (prepared - config.near_m) / (config.far_m - config.near_m)
    return normalized.astype(np.float32, copy=False)[np.newaxis, :, :]


def augment_depth_sequence(
    depth_sequence_m: np.ndarray,
    config: DepthConfig = DepthConfig(),
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply D435-like metric noise to a T×H×W sequence and return T×1×H×W."""
    if not isinstance(depth_sequence_m, np.ndarray) or depth_sequence_m.ndim != 3:
        raise ValueError("depth_sequence_m must have shape (T, H, W)")
    if depth_sequence_m.shape[0] <= 0 or 0 in depth_sequence_m.shape[1:]:
        raise ValueError("depth_sequence_m must be nonempty")
    if not np.issubdtype(depth_sequence_m.dtype, np.number):
        raise ValueError("depth_sequence_m must contain numeric depth")
    if not isinstance(config, DepthConfig) or not isinstance(rng, np.random.Generator):
        raise ValueError("config and rng must be DepthConfig and numpy Generator")

    resized = np.stack(
        [resize_depth_nearest(frame, config.width, config.height) for frame in depth_sequence_m]
    )
    frame_count, height, width = resized.shape
    persistent_holes = np.zeros((height, width), dtype=bool)
    hole_min, hole_max = config.hole_count_range
    size_min, size_max = config.hole_size_range_px
    hole_count = int(rng.integers(hole_min, hole_max + 1))
    for _ in range(hole_count):
        hole_height = min(height, int(rng.integers(size_min, size_max + 1)))
        hole_width = min(width, int(rng.integers(size_min, size_max + 1)))
        y = int(rng.integers(0, height - hole_height + 1))
        x = int(rng.integers(0, width - hole_width + 1))
        persistent_holes[y : y + hole_height, x : x + hole_width] = True

    base_bias = rng.uniform(*config.frame_bias_range_m)
    base_scale = rng.uniform(*config.depth_scale_range)
    output = np.empty((frame_count, 1, height, width), dtype=np.float32)
    for frame_index, frame in enumerate(resized):
        valid = np.isfinite(frame) & (frame > 0.0)
        metric = np.full(frame.shape, config.invalid_fill_m, dtype=np.float32)
        frame_bias = base_bias + float(rng.normal(0.0, 0.0005))
        frame_scale = base_scale + float(rng.normal(0.0, 0.0005))
        metric[valid] = frame[valid] * frame_scale + frame_bias
        noise_std = rng.uniform(*config.noise_std_range_m)
        metric[valid] += rng.normal(0.0, noise_std, size=int(valid.sum())).astype(np.float32)

        edge = np.zeros(frame.shape, dtype=bool)
        horizontal = np.abs(np.diff(frame, axis=1)) > config.edge_threshold_m
        vertical = np.abs(np.diff(frame, axis=0)) > config.edge_threshold_m
        edge[:, 1:] |= horizontal
        edge[:, :-1] |= horizontal
        edge[1:, :] |= vertical
        edge[:-1, :] |= vertical
        edge_probability = rng.uniform(*config.edge_dropout_probability_range)
        invalid_probability = rng.uniform(*config.invalid_pixel_probability_range)
        dropout = rng.random(frame.shape) < invalid_probability
        dropout |= edge & (rng.random(frame.shape) < edge_probability)
        dropout |= persistent_holes
        metric[~valid | dropout] = config.invalid_fill_m

        metric = np.round(metric * 1000.0) / 1000.0
        metric = np.clip(metric, config.near_m, config.far_m)
        output[frame_index, 0] = (metric - config.near_m) / (config.far_m - config.near_m)
    return output


def depth_to_millimetres(depth_m: np.ndarray) -> np.ndarray:
    """Return uint16 millimetres; invalid/non-positive/non-finite pixels become zero."""
    depth_m = _validate_depth_array(depth_m)
    result = np.zeros(depth_m.shape, dtype=np.uint16)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    millimetres = np.floor(depth_m[valid].astype(np.float64) * 1000.0 + 0.5)
    result[valid] = np.clip(millimetres, 0.0, np.iinfo(np.uint16).max).astype(np.uint16)
    return result


def millimetres_to_depth(depth_mm: np.ndarray) -> np.ndarray:
    """Return float32 metres; zero remains zero for shared invalid handling."""
    if not isinstance(depth_mm, np.ndarray) or depth_mm.ndim != 2 or 0 in depth_mm.shape:
        raise ValueError("depth_mm must be a nonempty 2D array")
    if depth_mm.dtype != np.uint16:
        raise ValueError("depth_mm must have dtype uint16")
    return depth_mm.astype(np.float32) / np.float32(1000.0)


class TopDepthRenderer:
    def __init__(
        self,
        model: mujoco.MjModel,
        config: DepthConfig = DepthConfig(),
        camera_name: str = "top",
    ):
        if not isinstance(config, DepthConfig):
            raise ValueError("config must be a DepthConfig")
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id == -1:
            raise ValueError(f"camera {camera_name!r} does not exist in the MuJoCo model")

        self._camera_name = camera_name
        self._renderer = mujoco.Renderer(model, height=config.height, width=config.width)
        self._renderer.enable_depth_rendering()
        self._closed = False

    def render(self, data: mujoco.MjData) -> np.ndarray:
        """Return a copied float32 metric-depth array with shape (H, W)."""
        if self._closed:
            raise RuntimeError("TopDepthRenderer is closed")
        self._renderer.update_scene(data, camera=self._camera_name)
        return self._renderer.render().astype(np.float32, copy=True)

    def close(self) -> None:
        if not self._closed:
            self._renderer.close()
            self._closed = True
