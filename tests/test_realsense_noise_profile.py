import unittest

import numpy as np

from measure_realsense_depth_noise import analyze_depth_stack, depth_config_from_profiles


class RealSenseNoiseProfileTest(unittest.TestCase):
    def test_profile_reports_metric_noise_holes_and_intrinsics(self):
        rng = np.random.default_rng(7)
        depth = np.full((20, 12, 16), 650, dtype=np.int32)
        depth += rng.integers(-4, 5, size=depth.shape)
        depth[:, 2:4, 3:6] = 0
        depth[::2, 7, 8] = 0
        depth = depth.astype(np.uint16)

        profile = analyze_depth_stack(
            depth,
            serial="test-camera",
            fps=30,
            intrinsics={"fx": 12.0, "fy": 10.0, "ppx": 8.0, "ppy": 6.0},
        )

        self.assertEqual(profile["serial"], "test-camera")
        self.assertEqual(profile["frames"], 20)
        self.assertEqual(profile["resolution"], [16, 12])
        self.assertGreater(profile["invalid_pixel_ratio_mean"], 0.0)
        self.assertGreater(profile["temporal_std_mm_median"], 0.0)
        self.assertGreater(profile["vertical_fov_degrees"], 0.0)
        self.assertTrue(np.isfinite(list(profile["intrinsics"].values())).all())

    def test_profile_rejects_invalid_or_empty_depth_stacks(self):
        with self.assertRaises(ValueError):
            analyze_depth_stack(
                np.zeros((0, 12, 16), dtype=np.uint16),
                serial="test-camera",
                fps=30,
                intrinsics={"fx": 12.0, "fy": 10.0, "ppx": 8.0, "ppy": 6.0},
            )

    def test_profiles_configure_training_augmentation_from_worst_camera(self):
        profiles = [
            {
                "temporal_std_mm_median": 1.5,
                "temporal_std_mm_p95": 4.0,
                "invalid_pixel_ratio_mean": 0.02,
                "invalid_pixel_ratio_p95": 0.05,
                "edge_invalid_ratio_mean": 0.12,
                "frame_median_depth_mm_mean": 650.0,
                "frame_median_depth_mm_std": 1.0,
            },
            {
                "temporal_std_mm_median": 2.0,
                "temporal_std_mm_p95": 6.0,
                "invalid_pixel_ratio_mean": 0.03,
                "invalid_pixel_ratio_p95": 0.08,
                "edge_invalid_ratio_mean": 0.20,
                "frame_median_depth_mm_mean": 700.0,
                "frame_median_depth_mm_std": 2.0,
            },
        ]

        config = depth_config_from_profiles(profiles)

        self.assertGreaterEqual(config.noise_std_range_m[1], 0.009)
        self.assertGreaterEqual(config.invalid_pixel_probability_range[1], 0.12)
        self.assertGreaterEqual(config.edge_dropout_probability_range[1], 0.30)
        self.assertGreaterEqual(config.frame_bias_range_m[1], 0.006)
        with self.assertRaises(ValueError):
            analyze_depth_stack(
                np.zeros((2, 12, 16), dtype=np.float32),
                serial="test-camera",
                fps=30,
                intrinsics={"fx": 12.0, "fy": 10.0, "ppx": 8.0, "ppy": 6.0},
            )


if __name__ == "__main__":
    unittest.main()
