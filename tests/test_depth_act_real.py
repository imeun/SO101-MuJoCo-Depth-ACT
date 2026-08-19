import unittest

import numpy as np

from run_depth_act_model_on_real_robot import (
    DeploymentVisualizer,
    DeploymentSafetyMonitor,
    MatplotlibDepthWindow,
    depth_to_bgr,
    depth_units_to_metres,
    parse_args,
    validate_camera_intrinsics,
)


class DepthACTRealTest(unittest.TestCase):
    def test_matplotlib_depth_window_close_ignores_gui_shutdown_errors(self):
        class BrokenPyplot:
            def close(self, _figure):
                raise RuntimeError("figure already closed")

        window = object.__new__(MatplotlibDepthWindow)
        window.pyplot = BrokenPyplot()
        window.figure = object()

        window.close()

    def test_dry_run_is_default(self):
        args = parse_args(["--model", "model.pt", "--calibration", "mapping.json"])
        self.assertFalse(args.send)
        self.assertEqual(args.gripper_slew, 10.0)

    def test_display_can_be_enabled_without_enabling_motor_commands(self):
        args = parse_args([
            "--model", "model.pt",
            "--calibration", "mapping.json",
            "--display",
            "--display-rate", "12",
        ])
        self.assertTrue(args.display)
        self.assertEqual(args.display_rate, 12.0)
        self.assertFalse(args.send)

    def test_depth_display_is_bgr_and_does_not_modify_metric_input(self):
        depth = np.array([[0.0, 0.2, 0.6, 1.0]], dtype=np.float32)
        original = depth.copy()

        display = depth_to_bgr(depth, near_m=0.2, far_m=1.0)

        self.assertEqual(display.shape, (1, 4, 3))
        self.assertEqual(display.dtype, np.uint8)
        np.testing.assert_array_equal(display[0, 0], [0, 0, 0])
        self.assertFalse(np.array_equal(display[0, 1], display[0, 3]))
        np.testing.assert_array_equal(depth, original)

    def test_realsense_units_convert_to_metric_depth(self):
        raw = np.array([[0, 500, 1000]], dtype=np.uint16)
        depth = depth_units_to_metres(raw, 0.001)
        np.testing.assert_allclose(depth, [[0.0, 0.5, 1.0]])
        self.assertEqual(depth.dtype, np.float32)

    def test_invalid_depth_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scale"):
            depth_units_to_metres(np.ones((2, 2), dtype=np.uint16), 0.0)

    def test_safety_monitor_rejects_sparse_depth(self):
        monitor = DeploymentSafetyMonitor(min_valid_depth_fraction=0.50)
        sparse = np.zeros((10, 10), dtype=np.float32)
        sparse[:4] = 0.5
        with self.assertRaisesRegex(RuntimeError, "valid depth"):
            monitor.check_depth_pair(sparse, np.full((10, 10), 0.6, dtype=np.float32))

    def test_safety_monitor_rejects_large_immediate_delta(self):
        monitor = DeploymentSafetyMonitor(max_immediate_delta_rad=0.12)
        chunk = np.zeros((30, 6), dtype=np.float32)
        chunk[0, 2] = 0.13
        with self.assertRaisesRegex(RuntimeError, "immediate policy delta"):
            monitor.check_prediction(chunk)

    def test_safety_monitor_stops_after_repeated_loop_overruns(self):
        monitor = DeploymentSafetyMonitor(max_overrun_factor=1.5, max_consecutive_overruns=3)
        monitor.check_cycle(0.06, 0.034)
        monitor.check_cycle(0.06, 0.034)
        with self.assertRaisesRegex(RuntimeError, "control loop"):
            monitor.check_cycle(0.06, 0.034)

    def test_safety_monitor_stops_after_repeated_tracking_error(self):
        monitor = DeploymentSafetyMonitor(max_tracking_error=np.array([10] * 5 + [20]), max_tracking_failures=2)
        commanded = np.zeros(6)
        measured = np.array([11, 0, 0, 0, 0, 0], dtype=np.float64)
        monitor.check_tracking(commanded, measured)
        with self.assertRaisesRegex(RuntimeError, "tracking error"):
            monitor.check_tracking(commanded, measured)

    def test_camera_intrinsics_are_compared_in_mujoco_offset_convention(self):
        expected = {
            "cameras": {
                "top": {
                    "resolution": [640, 480],
                    "focalpixel": [382.0, 383.0],
                    "principalpixel_offset": [1.0, 2.0],
                },
                "side_depth": {
                    "resolution": [640, 480],
                    "focalpixel": [385.0, 385.0],
                    "principalpixel_offset": [8.0, 1.0],
                },
            }
        }
        actual = {
            "top": {
                "resolution": [640, 480],
                "focalpixel": [382.5, 382.5],
                "principalpixel": [321.2, 238.2],
            },
            "side_depth": {
                "resolution": [640, 480],
                "focalpixel": [384.8, 385.1],
                "principalpixel": [327.9, 238.9],
            },
        }
        diagnostics = validate_camera_intrinsics(expected, actual, tolerance_px=2.0)
        self.assertEqual(set(diagnostics), {"top", "side_depth"})

        actual["top"]["principalpixel"] = [330.0, 238.2]
        with self.assertRaisesRegex(ValueError, "top camera intrinsics mismatch"):
            validate_camera_intrinsics(expected, actual, tolerance_px=2.0)


if __name__ == "__main__":
    unittest.main()
