from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from so101_depth import DepthConfig


_AUGMENTATION_PROFILE_KEYS = (
    "temporal_std_mm_median",
    "temporal_std_mm_p95",
    "invalid_pixel_ratio_mean",
    "invalid_pixel_ratio_p95",
    "edge_invalid_ratio_mean",
    "frame_median_depth_mm_mean",
    "frame_median_depth_mm_std",
)


def depth_config_from_profiles(
    profiles: Sequence[Mapping[str, Any]], *, base: DepthConfig = DepthConfig()
) -> DepthConfig:
    """Build conservative online augmentation ranges from static RealSense profiles."""
    if not profiles:
        raise ValueError("at least one RealSense noise profile is required")
    values: list[dict[str, float]] = []
    for profile in profiles:
        try:
            converted = {key: float(profile[key]) for key in _AUGMENTATION_PROFILE_KEYS}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("RealSense noise profile is missing valid metrics") from error
        if not np.all(np.isfinite(list(converted.values()))) or any(value < 0.0 for value in converted.values()):
            raise ValueError("RealSense noise profile metrics must be finite and non-negative")
        if converted["invalid_pixel_ratio_mean"] > 1.0 or converted["invalid_pixel_ratio_p95"] > 1.0:
            raise ValueError("RealSense invalid-pixel ratios must be within [0, 1]")
        values.append(converted)

    noise_high = min(
        0.030,
        max(base.noise_std_range_m[1], 1.5 * max(item["temporal_std_mm_p95"] for item in values) / 1000.0),
    )
    noise_low = min(noise_high, max(0.0005, 0.5 * max(item["temporal_std_mm_median"] for item in values) / 1000.0))
    invalid_high = min(
        0.50,
        max(base.invalid_pixel_probability_range[1], 1.5 * max(item["invalid_pixel_ratio_p95"] for item in values)),
    )
    invalid_low = min(
        invalid_high,
        max(0.001, 0.5 * min(item["invalid_pixel_ratio_mean"] for item in values)),
    )
    edge_high = min(
        0.75,
        max(base.edge_dropout_probability_range[1], 1.5 * max(item["edge_invalid_ratio_mean"] for item in values)),
    )
    bias_span = min(
        0.030,
        max(abs(base.frame_bias_range_m[0]), base.frame_bias_range_m[1], 3.0 * max(item["frame_median_depth_mm_std"] for item in values) / 1000.0),
    )
    relative_jitter = max(
        item["frame_median_depth_mm_std"] / max(item["frame_median_depth_mm_mean"], 1.0)
        for item in values
    )
    scale_span = min(
        0.05,
        max(1.0 - base.depth_scale_range[0], base.depth_scale_range[1] - 1.0, 3.0 * relative_jitter),
    )
    return replace(
        base,
        noise_std_range_m=(noise_low, noise_high),
        frame_bias_range_m=(-bias_span, bias_span),
        depth_scale_range=(1.0 - scale_span, 1.0 + scale_span),
        invalid_pixel_probability_range=(invalid_low, invalid_high),
        edge_dropout_probability_range=(base.edge_dropout_probability_range[0], edge_high),
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def analyze_depth_stack(
    depth_mm: np.ndarray,
    *,
    serial: str,
    fps: int,
    intrinsics: dict[str, float],
) -> dict[str, Any]:
    if (
        not isinstance(depth_mm, np.ndarray)
        or depth_mm.dtype != np.uint16
        or depth_mm.ndim != 3
        or depth_mm.shape[0] <= 0
        or 0 in depth_mm.shape[1:]
    ):
        raise ValueError("depth_mm must be nonempty uint16 with shape (T, H, W)")
    if not isinstance(serial, str) or not serial or fps <= 0:
        raise ValueError("serial and fps are invalid")
    if set(intrinsics) != {"fx", "fy", "ppx", "ppy"}:
        raise ValueError("intrinsics must contain fx, fy, ppx, and ppy")
    values = np.asarray(list(intrinsics.values()), dtype=np.float64)
    if not np.all(np.isfinite(values)) or intrinsics["fx"] <= 0.0 or intrinsics["fy"] <= 0.0:
        raise ValueError("intrinsics must be finite with positive focal lengths")

    valid = depth_mm > 0
    valid_count = valid.sum(axis=0, dtype=np.int32)
    depth_float = depth_mm.astype(np.float32)
    sums = np.where(valid, depth_float, 0.0).sum(axis=0, dtype=np.float64)
    squares = np.where(valid, depth_float * depth_float, 0.0).sum(axis=0, dtype=np.float64)
    mean_depth = np.divide(sums, valid_count, out=np.zeros_like(sums), where=valid_count > 0)
    variance = np.divide(squares, valid_count, out=np.zeros_like(squares), where=valid_count > 0) - mean_depth**2
    temporal_std = np.sqrt(np.maximum(variance, 0.0))
    temporal_values = temporal_std[valid_count >= 2]
    frame_medians = np.asarray(
        [np.median(frame[frame > 0]) if np.any(frame > 0) else 0.0 for frame in depth_mm],
        dtype=np.float64,
    )

    edge = np.zeros(mean_depth.shape, dtype=bool)
    horizontal = np.abs(np.diff(mean_depth, axis=1)) > 15.0
    vertical = np.abs(np.diff(mean_depth, axis=0)) > 15.0
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    invalid_frequency = 1.0 - valid.mean(axis=0)
    edge_invalid_ratio = float(invalid_frequency[edge].mean()) if np.any(edge) else 0.0

    height, width = depth_mm.shape[1:]
    vertical_fov = math.degrees(2.0 * math.atan(height / (2.0 * intrinsics["fy"])))
    horizontal_fov = math.degrees(2.0 * math.atan(width / (2.0 * intrinsics["fx"])))
    return {
        "serial": serial,
        "frames": int(depth_mm.shape[0]),
        "fps": int(fps),
        "resolution": [int(width), int(height)],
        "intrinsics": {key: float(value) for key, value in intrinsics.items()},
        "horizontal_fov_degrees": float(horizontal_fov),
        "vertical_fov_degrees": float(vertical_fov),
        "invalid_pixel_ratio_mean": float((~valid).mean()),
        "invalid_pixel_ratio_p95": float(np.percentile((~valid).mean(axis=(1, 2)), 95)),
        "edge_invalid_ratio_mean": edge_invalid_ratio,
        "temporal_std_mm_median": float(np.median(temporal_values)) if temporal_values.size else 0.0,
        "temporal_std_mm_p95": float(np.percentile(temporal_values, 95)) if temporal_values.size else 0.0,
        "frame_median_depth_mm_mean": float(frame_medians.mean()),
        "frame_median_depth_mm_std": float(frame_medians.std()),
    }


def capture_depth_stack(
    *, serial: str, frames: int, width: int, height: int, fps: int, warmup_frames: int
) -> tuple[np.ndarray, dict[str, float]]:
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError("pyrealsense2 is required to measure a RealSense camera") from error
    pipeline = rs.pipeline()
    configuration = rs.config()
    configuration.enable_device(serial)
    configuration.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(configuration)
    try:
        for _ in range(warmup_frames):
            pipeline.wait_for_frames()
        captured: list[np.ndarray] = []
        intrinsics = None
        for _ in range(frames):
            depth_frame = pipeline.wait_for_frames().get_depth_frame()
            if not depth_frame:
                raise RuntimeError("RealSense returned an empty depth frame")
            captured.append(np.asanyarray(depth_frame.get_data()).astype(np.uint16, copy=True))
            if intrinsics is None:
                values = depth_frame.profile.as_video_stream_profile().intrinsics
                intrinsics = {
                    "fx": float(values.fx),
                    "fy": float(values.fy),
                    "ppx": float(values.ppx),
                    "ppy": float(values.ppy),
                }
    finally:
        pipeline.stop()
    if intrinsics is None:
        raise RuntimeError("RealSense did not provide intrinsics")
    return np.stack(captured), intrinsics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure raw D435 depth noise and camera intrinsics.")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=_positive_int, default=300)
    parser.add_argument("--warmup-frames", type=_positive_int, default=30)
    parser.add_argument("--width", type=_positive_int, default=640)
    parser.add_argument("--height", type=_positive_int, default=480)
    parser.add_argument("--fps", type=_positive_int, default=30)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    depth, intrinsics = capture_depth_stack(
        serial=args.serial,
        frames=args.frames,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
    )
    profile = analyze_depth_stack(depth, serial=args.serial, fps=args.fps, intrinsics=intrinsics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2, sort_keys=True))
    return profile


if __name__ == "__main__":
    main()
