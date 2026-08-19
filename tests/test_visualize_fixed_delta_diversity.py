import unittest

import numpy as np

from visualize_fixed_delta_diversity import trajectory_diversity_stats


class FixedDeltaDiversityVisualizationTest(unittest.TestCase):
    def test_stats_measure_joint_spread_and_executed_action_perturbation(self):
        base_joint = np.zeros((4, 6), dtype=np.float32)
        changed_joint = base_joint.copy()
        changed_joint[2, 1] = 0.02
        teacher = np.full((4, 6), 0.10, dtype=np.float32)
        first_executed = teacher.copy()
        second_executed = teacher.copy()
        second_executed[1, 3] = 0.13

        stats = trajectory_diversity_stats(
            [
                {
                    "joint_pos": base_joint,
                    "teacher_action": teacher,
                    "executed_action": first_executed,
                },
                {
                    "joint_pos": changed_joint,
                    "teacher_action": teacher,
                    "executed_action": second_executed,
                },
            ]
        )

        self.assertAlmostEqual(stats["max_joint_spread_rad"], 0.02, places=6)
        self.assertAlmostEqual(stats["max_action_perturbation"], 0.03, places=6)
        self.assertGreater(stats["mean_joint_spread_rad"], 0.0)

    def test_stats_require_matching_trajectory_shapes(self):
        action = np.zeros((4, 6), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "same trajectory shape"):
            trajectory_diversity_stats(
                [
                    {
                        "joint_pos": np.zeros((4, 6), dtype=np.float32),
                        "teacher_action": action,
                        "executed_action": action,
                    },
                    {
                        "joint_pos": np.zeros((3, 6), dtype=np.float32),
                        "teacher_action": np.zeros((3, 6), dtype=np.float32),
                        "executed_action": np.zeros((3, 6), dtype=np.float32),
                    },
                ]
            )


if __name__ == "__main__":
    unittest.main()
