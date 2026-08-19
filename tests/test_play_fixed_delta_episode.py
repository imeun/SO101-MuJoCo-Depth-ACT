import tempfile
import unittest
from pathlib import Path

import numpy as np

from delta_depth_dataset import DeltaDepthEpisodeWriter
from play_fixed_delta_episode import load_delta_episode


class FixedDeltaEpisodePlaybackTest(unittest.TestCase):
    def test_loader_returns_the_requested_episode_and_all_control_arrays(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            frames = 3
            top = np.full((frames, 240, 320), 650, dtype=np.uint16)
            side = np.full((frames, 240, 320), 700, dtype=np.uint16)
            joints = np.zeros((frames, 6), dtype=np.float32)
            velocity = np.zeros((frames, 6), dtype=np.float32)
            goal = np.full((frames, 6), 0.01, dtype=np.float32)
            delta = goal - joints
            teacher_action = np.full((frames, 6), 0.2, dtype=np.float32)
            executed_action = np.full((frames, 6), 0.22, dtype=np.float32)
            phase = np.zeros(frames, dtype=np.uint8)
            timestamp = np.arange(frames, dtype=np.float64) * 0.034
            DeltaDepthEpisodeWriter(root).save_episode(
                top,
                side,
                joints,
                velocity,
                goal,
                delta,
                teacher_action,
                executed_action,
                phase,
                timestamp,
                seed=7,
                initial_joint_pos=np.zeros(6),
                cube_position=np.array([0.32, 0.155, 0.0336]),
                bin_position=np.array([0.31, -0.05, 0.0]),
            )

            entry, arrays = load_delta_episode(root, 0)

            self.assertEqual(entry["seed"], 7)
            self.assertEqual(arrays["executed_action"].shape, (3, 6))
            np.testing.assert_array_equal(arrays["phase_id"], phase)

    def test_loader_rejects_an_unknown_episode_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            DeltaDepthEpisodeWriter(root)
            with self.assertRaisesRegex(ValueError, "episode index 9"):
                load_delta_episode(root, 9)


if __name__ == "__main__":
    unittest.main()
