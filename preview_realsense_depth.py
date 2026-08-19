from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np


def depth_to_display_u8(depth_mm: np.ndarray, *, max_depth_mm: int) -> np.ndarray:
    if not isinstance(depth_mm, np.ndarray) or depth_mm.dtype != np.uint16 or depth_mm.ndim != 2:
        raise ValueError("depth_mm must be a uint16 array with shape (H, W)")
    if isinstance(max_depth_mm, bool) or not isinstance(max_depth_mm, (int, np.integer)) or max_depth_mm <= 0:
        raise ValueError("max_depth_mm must be a positive integer")
    scaled = np.rint(np.clip(depth_mm.astype(np.float32), 0.0, float(max_depth_mm)) * (255.0 / max_depth_mm))
    scaled[depth_mm == 0] = 0.0
    return scaled.astype(np.uint8)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive value")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a selected RealSense RGB and raw depth stream.")
    parser.add_argument("--serial")
    parser.add_argument("--list", action="store_true", dest="list_devices")
    parser.add_argument("--width", type=_positive_int, default=640)
    parser.add_argument("--height", type=_positive_int, default=480)
    parser.add_argument("--fps", type=_positive_int, default=30)
    parser.add_argument("--warmup-frames", type=_positive_int, default=30)
    parser.add_argument("--max-depth-m", type=_positive_float, default=1.5)
    parser.add_argument("--snapshot-dir", type=Path, default=Path("realsense_previews"))
    return parser.parse_args(argv)


def _device_descriptions(rs) -> list[tuple[str, str]]:
    descriptions = []
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        descriptions.append((serial, name))
    return descriptions


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import cv2
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError("opencv-python and pyrealsense2 are required for RealSense preview") from error

    devices = _device_descriptions(rs)
    if not devices:
        raise RuntimeError("no RealSense devices were detected")
    print("Detected RealSense devices:", flush=True)
    for serial, name in devices:
        print(f"  serial={serial} name={name}", flush=True)
    if args.list_devices:
        return 0
    if args.serial is None:
        raise ValueError("--serial is required when opening a preview")
    matching = [name for serial, name in devices if serial == args.serial]
    if not matching:
        raise ValueError(f"RealSense serial {args.serial!r} is not connected")

    pipeline = rs.pipeline()
    configuration = rs.config()
    configuration.enable_device(args.serial)
    configuration.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    configuration.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    window_name = f"RealSense {args.serial} - RGB | Depth"
    max_depth_mm = int(round(args.max_depth_m * 1000.0))
    started = False
    try:
        pipeline.start(configuration)
        started = True
        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames(5_000)

        print(f"Previewing serial={args.serial} name={matching[0]}", flush=True)
        print("Press S to save a snapshot, Q or ESC to quit.", flush=True)
        while True:
            frames = pipeline.wait_for_frames(5_000)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_mm = np.asanyarray(depth_frame.get_data()).astype(np.uint16, copy=True)
            color = np.asanyarray(color_frame.get_data()).copy()
            depth_u8 = depth_to_display_u8(depth_mm, max_depth_mm=max_depth_mm)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
            depth_color[depth_mm == 0] = 0

            center_depth = int(depth_mm[depth_mm.shape[0] // 2, depth_mm.shape[1] // 2])
            invalid_ratio = float(np.mean(depth_mm == 0))
            label = (
                f"serial={args.serial} center={center_depth}mm "
                f"invalid={invalid_ratio * 100.0:.1f}%"
            )
            for image in (color, depth_color):
                cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
                cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1)
            canvas = np.hstack((color, depth_color))
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                destination = args.snapshot_dir / f"realsense_{args.serial}_{stamp}.png"
                if not cv2.imwrite(str(destination), canvas):
                    raise RuntimeError(f"failed to save snapshot: {destination}")
                print(f"Saved {destination.resolve()}", flush=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"RealSense {args.serial} stream failed. Close other camera programs and check USB 3 connectivity."
        ) from error
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
