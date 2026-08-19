import tempfile
import unittest
from pathlib import Path

import numpy as np

from delta_depth_dataset import DeltaDepthEpisodeWriter, load_delta_manifest


def episode_arrays(frames: int = 3):
    top = np.full((frames, 240, 320), 650, dtype=np.uint16)
    side = np.full((frames, 240, 320), 700, dtype=np.uint16)
    joints = np.zeros((frames, 6), dtype=np.float32)
    velocity = np.zeros((frames, 6), dtype=np.float32)
    teacher_goal = np.full((frames, 6), 0.01, dtype=np.float32)
    delta = teacher_goal - joints
    teacher_action = np.full((frames, 6), 0.2, dtype=np.float32)
    executed_action = teacher_action.copy()
    phase = np.zeros(frames, dtype=np.uint8)
    timestamp = np.arange(frames, dtype=np.float64) * 0.034
    return top, side, joints, velocity, teacher_goal, delta, teacher_action, executed_action, phase, timestamp


class DeltaDepthDatasetTest(unittest.TestCase):
    def test_writer_saves_immediate_delta_and_fixed_scene_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "delta"
            writer = DeltaDepthEpisodeWriter(root)
            path = writer.save_episode(
                *episode_arrays(),
                seed=7,
                initial_joint_pos=np.zeros(6),
                cube_position=np.array([0.32, 0.155, 0.0336]),
                bin_position=np.array([0.31, -0.05, 0.0]),
            )

            manifest = load_delta_manifest(root)
            self.assertEqual(manifest["target_type"], "immediate_joint_delta")
            self.assertEqual(manifest["initial_conditions"], "fixed")
            self.assertEqual(len(manifest["episodes"]), 1)
            self.assertEqual(len(manifest["episodes"][0]["trajectory_sha256"]), 64)
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_allclose(
                    archive["delta_target_rad"],
                    archive["teacher_goal_pos"] - archive["joint_pos"],
                )
                self.assertEqual(archive["top_depth_mm"].dtype, np.uint16)
                self.assertEqual(archive["joint_velocity"].shape, (3, 6))

    def test_writer_rejects_inconsistent_delta(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            arrays = list(episode_arrays())
            arrays[5] = np.zeros((3, 6), dtype=np.float32)
            writer = DeltaDepthEpisodeWriter(Path(temp_dir) / "delta")
            with self.assertRaisesRegex(ValueError, "delta_target_rad"):
                writer.save_episode(
                    *arrays,
                    seed=7,
                    initial_joint_pos=np.zeros(6),
                    cube_position=np.array([0.32, 0.155, 0.0336]),
                    bin_position=np.array([0.31, -0.05, 0.0]),
                )

    def test_writer_rejects_duplicate_trajectory_with_a_different_seed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = DeltaDepthEpisodeWriter(Path(temp_dir) / "delta")
            metadata = {
                "initial_joint_pos": np.zeros(6),
                "cube_position": np.array([0.32, 0.155, 0.0336]),
                "bin_position": np.array([0.31, -0.05, 0.0]),
            }
            writer.save_episode(*episode_arrays(), seed=7, **metadata)
            with self.assertRaisesRegex(ValueError, "duplicate trajectory"):
                writer.save_episode(*episode_arrays(), seed=8, **metadata)

    def test_executed_noise_alone_does_not_make_a_trajectory_unique(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = DeltaDepthEpisodeWriter(Path(temp_dir) / "delta")
            metadata = {
                "initial_joint_pos": np.zeros(6),
                "cube_position": np.array([0.32, 0.155, 0.0336]),
                "bin_position": np.array([0.31, -0.05, 0.0]),
            }
            writer.save_episode(*episode_arrays(), seed=7, **metadata)
            changed = list(episode_arrays())
            changed[7] = np.full((3, 6), 0.25, dtype=np.float32)
            with self.assertRaisesRegex(ValueError, "duplicate trajectory"):
                writer.save_episode(*changed, seed=8, **metadata)

    def test_writer_rejects_a_changed_initial_scene(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = DeltaDepthEpisodeWriter(Path(temp_dir) / "delta")
            metadata = {
                "initial_joint_pos": np.zeros(6),
                "cube_position": np.array([0.32, 0.155, 0.0336]),
                "bin_position": np.array([0.31, -0.05, 0.0]),
            }
            writer.save_episode(*episode_arrays(), seed=7, **metadata)
            changed = list(episode_arrays())
            changed[2] = changed[2].copy()
            changed[2][1, 0] = 0.02
            changed[4] = changed[2] + changed[5]
            with self.assertRaisesRegex(ValueError, "fixed initial conditions"):
                writer.save_episode(
                    *changed,
                    seed=8,
                    initial_joint_pos=np.full(6, 0.01),
                    cube_position=metadata["cube_position"],
                    bin_position=metadata["bin_position"],
                )

    def test_near_position_mode_accepts_changed_object_positions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = DeltaDepthEpisodeWriter(
                Path(temp_dir) / "delta",
                initial_conditions="near_position",
            )
            writer.save_episode(
                *episode_arrays(),
                seed=7,
                initial_joint_pos=np.zeros(6),
                cube_position=np.array([0.32, 0.155, 0.0336]),
                bin_position=np.array([0.31, -0.05, 0.0]),
            )
            changed = list(episode_arrays())
            changed[2] = changed[2].copy()
            changed[2][1:, 0] = 0.02
            changed[4] = changed[2] + changed[5]
            writer.save_episode(
                *changed,
                seed=8,
                initial_joint_pos=np.zeros(6),
                cube_position=np.array([0.335, 0.140, 0.0336]),
                bin_position=np.array([0.305, -0.042, 0.0]),
            )
            manifest = load_delta_manifest(Path(temp_dir) / "delta")
            self.assertEqual(manifest["initial_conditions"], "near_position")
            self.assertEqual(len(manifest["episodes"]), 2)

    def test_near_position_mode_still_rejects_changed_robot_home(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = DeltaDepthEpisodeWriter(
                Path(temp_dir) / "delta",
                initial_conditions="near_position",
            )
            writer.save_episode(
                *episode_arrays(),
                seed=7,
                initial_joint_pos=np.zeros(6),
                cube_position=np.array([0.32, 0.155, 0.0336]),
                bin_position=np.array([0.31, -0.05, 0.0]),
            )
            changed = list(episode_arrays())
            changed[2] = changed[2].copy()
            changed[2][1:, 0] = 0.02
            changed[4] = changed[2] + changed[5]
            with self.assertRaisesRegex(ValueError, "robot home"):
                writer.save_episode(
                    *changed,
                    seed=8,
                    initial_joint_pos=np.full(6, 0.01),
                    cube_position=np.array([0.335, 0.140, 0.0336]),
                    bin_position=np.array([0.305, -0.042, 0.0]),
                )


if __name__ == "__main__":
    unittest.main()
