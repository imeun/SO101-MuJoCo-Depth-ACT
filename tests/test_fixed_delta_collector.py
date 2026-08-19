import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import collect_fixed_delta_depth_dataset as collector
from delta_depth_dataset import load_delta_manifest
from so101_ball_bins_env import SO101BallBinsEnv


class FakeDepthRenderer:
    def __init__(self, model, *, camera_name):
        self.camera_name = camera_name

    def render(self, data):
        value = 0.65 if self.camera_name == "top" else 0.75
        return np.full((240, 320), value, dtype=np.float32)

    def close(self):
        pass


class FixedDeltaCollectorTest(unittest.TestCase):
    def test_candidate_seeds_skip_episodes_already_saved(self):
        candidates = collector.candidate_seeds(100, {100, 102})
        self.assertEqual([next(candidates) for _ in range(3)], [101, 103, 104])

    def test_perturbation_is_seeded_bounded_and_does_not_change_gripper(self):
        first = collector.SmoothActionPerturbation(seed=7, std=0.03, limit=0.08, correlation=0.9)
        second = collector.SmoothActionPerturbation(seed=7, std=0.03, limit=0.08, correlation=0.9)
        clean = np.zeros(6, dtype=np.float32)
        a = first(clean, "pregrasp", 0)
        b = second(clean, "pregrasp", 0)
        np.testing.assert_array_equal(a, b)
        self.assertLessEqual(float(np.max(np.abs(a[:5]))), 0.08)
        self.assertEqual(float(a[5]), 0.0)

    def test_nearby_scene_is_seeded_and_stays_inside_requested_offsets(self):
        environment = SO101BallBinsEnv(spawn_stage="stage1")
        try:
            environment.reset(seed=7)
            cube_a, bin_a = collector.place_nearby_scene(
                environment, seed=99, cube_jitter=0.02, bin_jitter=0.01
            )
            environment.reset(seed=7)
            cube_b, bin_b = collector.place_nearby_scene(
                environment, seed=99, cube_jitter=0.02, bin_jitter=0.01
            )
        finally:
            environment.close()
        np.testing.assert_array_equal(cube_a, cube_b)
        np.testing.assert_array_equal(bin_a, bin_b)
        self.assertLessEqual(float(np.max(np.abs(cube_a[:2] - collector.BASE_CUBE_POSITION))), 0.02)
        self.assertLessEqual(float(np.max(np.abs(bin_a[:2] - collector.BASE_BIN_POSITION))), 0.01)

    def test_collects_unique_successes_with_identical_initial_conditions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "fixed_delta"
            args = collector.parse_args(
                [
                    "--episodes", "2",
                    "--out-dir", str(root),
                    "--seed", "100",
                    "--max-attempts", "8",
                    "--perturbation-std", "0.02",
                    "--perturbation-limit", "0.05",
                    "--cube-jitter", "0",
                    "--bin-jitter", "0",
                ]
            )
            with mock.patch.object(collector, "TopDepthRenderer", FakeDepthRenderer):
                summary = collector.collect_episodes(args)

            manifest = load_delta_manifest(root)
            self.assertEqual(summary["saved_episodes"], 2)
            self.assertEqual(summary["duplicate_trajectories"], 0)
            self.assertEqual(len(manifest["episodes"]), 2)
            self.assertEqual(len(manifest["provenance"]["scene_xml_sha256"]), 64)
            self.assertEqual(manifest["provenance"]["cameras"]["top"]["resolution"], [640, 480])
            first, second = manifest["episodes"]
            self.assertNotEqual(first["trajectory_sha256"], second["trajectory_sha256"])
            np.testing.assert_array_equal(first["initial_joint_pos"], second["initial_joint_pos"])
            np.testing.assert_array_equal(first["cube_position"], second["cube_position"])
            np.testing.assert_array_equal(first["bin_position"], second["bin_position"])

    def test_default_collection_varies_positions_but_keeps_clean_actions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "near_delta"
            args = collector.parse_args([
                "--episodes", "2",
                "--out-dir", str(root),
                "--seed", "7",
                "--max-attempts", "20",
            ])
            with mock.patch.object(collector, "TopDepthRenderer", FakeDepthRenderer):
                summary = collector.collect_episodes(args)

            manifest = load_delta_manifest(root)
            self.assertEqual(manifest["initial_conditions"], "near_position")
            self.assertEqual(summary["perturbation_std"], 0.0)
            first, second = manifest["episodes"]
            self.assertFalse(np.array_equal(first["cube_position"], second["cube_position"]))
            self.assertFalse(np.array_equal(first["bin_position"], second["bin_position"]))
            for entry in manifest["episodes"]:
                with np.load(root / entry["file"], allow_pickle=False) as archive:
                    np.testing.assert_array_equal(archive["teacher_action"], archive["executed_action"])


if __name__ == "__main__":
    unittest.main()
