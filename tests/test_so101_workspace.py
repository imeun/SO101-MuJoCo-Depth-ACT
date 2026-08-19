import math
import unittest

import numpy as np

from so101_workspace import WorkspaceConfig, sample_block_pose


class SampleBlockPoseTest(unittest.TestCase):
    def test_stage1_uses_one_fixed_curriculum_pose(self):
        first = sample_block_pose(np.random.default_rng(1), "stage1")
        second = sample_block_pose(np.random.default_rng(999), "stage1")
        np.testing.assert_allclose(first, np.array([0.320, 0.155, 0.0336, 1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(first, second)

    def test_block_reset_height_uses_zero_table_surface(self):
        """Catches moving the table away from the measured SO101 base bottom."""
        config = WorkspaceConfig()
        self.assertEqual(config.table_surface_z, -0.0024)
        self.assertEqual(config.block_center_z, 0.0336)

    def test_task_objects_stay_at_least_thirty_centimetres_from_robot(self):
        config = WorkspaceConfig()
        self.assertEqual(config.bin_center, (0.310, -0.050))
        self.assertGreaterEqual(math.hypot(*config.bin_center), 0.300)
        self.assertEqual(config.stage1_radius, (0.350, 0.370))
        self.assertEqual(config.stage2_radius, (0.300, 0.380))

    def assert_valid_pose(self, pose: np.ndarray, stage: str, config: WorkspaceConfig) -> None:
        self.assertEqual(pose.shape, (7,))
        self.assertEqual(pose.dtype, np.float64)
        x, y, z, qw, qx, qy, qz = pose
        self.assertGreaterEqual(x, config.x_bounds[0])
        self.assertLessEqual(x, config.x_bounds[1])
        self.assertGreaterEqual(y, config.y_bounds[0])
        self.assertLessEqual(y, config.y_bounds[1])
        self.assertEqual(z, config.block_center_z)
        self.assertEqual(qx, 0.0)
        self.assertEqual(qy, 0.0)
        self.assertAlmostEqual(qw * qw + qz * qz, 1.0, places=12)

        radius = math.hypot(x, y)
        radius_bounds = config.stage1_radius if stage == "stage1" else config.stage2_radius
        self.assertGreaterEqual(radius, radius_bounds[0])
        self.assertLessEqual(radius, radius_bounds[1])
        self.assertFalse(
            abs(x - config.bin_center[0]) < config.bin_exclusion_half_extent
            and abs(y - config.bin_center[1]) < config.bin_exclusion_half_extent
        )

        yaw = 2.0 * math.atan2(qz, qw)
        half_extent = config.block_half_size[0] * (abs(math.cos(yaw)) + abs(math.sin(yaw)))
        self.assertGreaterEqual(x - half_extent, config.table_center[0] - config.table_half_size[0])
        self.assertLessEqual(x + half_extent, config.table_center[0] + config.table_half_size[0])
        self.assertGreaterEqual(y - half_extent, config.table_center[1] - config.table_half_size[1])
        self.assertLessEqual(y + half_extent, config.table_center[1] + config.table_half_size[1])

    def test_stage_samples_obey_workspace_constraints(self):
        """Catches accepting a pose outside the radial, bin, or table footprint limits."""
        config = WorkspaceConfig()
        for stage, seed in (("stage1", 31415), ("stage2", 92653)):
            rng = np.random.default_rng(seed)
            for _ in range(5_000):
                self.assert_valid_pose(sample_block_pose(rng, stage, config), stage, config)

    def test_seeded_generators_produce_identical_pose_sequences(self):
        """Catches sampling from global randomness instead of the supplied generator."""
        first = np.random.default_rng(1234)
        second = np.random.default_rng(1234)
        for stage in ("stage1", "stage2", "stage1"):
            np.testing.assert_array_equal(sample_block_pose(first, stage), sample_block_pose(second, stage))

    def test_unknown_stage_is_rejected(self):
        """Catches silently treating an unsupported spawn stage as a valid stage."""
        with self.assertRaises(ValueError):
            sample_block_pose(np.random.default_rng(1), "stage3")  # type: ignore[arg-type]

    def test_impossible_workspace_raises_after_configured_attempt_limit(self):
        """Catches an unbounded rejection loop when no candidate can be valid."""
        config = WorkspaceConfig(
            x_bounds=(0.100, 0.100),
            y_bounds=(-0.100, -0.100),
            max_attempts=3,
        )
        with self.assertRaises(RuntimeError):
            sample_block_pose(np.random.default_rng(1), "stage1", config)


if __name__ == "__main__":
    unittest.main()
