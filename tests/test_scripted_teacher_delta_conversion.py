import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import collect_scripted_teacher_delta_depth_dataset as converter
from delta_depth_dataset import load_delta_manifest
from teacher_dataset import TeacherEpisodeWriter


class FakeDepthRenderer:
    def __init__(self, model, *, camera_name):
        self.value = 0.6 if camera_name == "top" else 0.7

    def render(self, data):
        return np.full((240, 320), self.value, dtype=np.float32)

    def close(self):
        pass


class ScriptedTeacherDeltaConversionTest(unittest.TestCase):
    def test_replays_success_source_into_act_delta_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "teacher"
            output = Path(temp_dir) / "act"
            writer = TeacherEpisodeWriter(source)
            provenance = {
                "kind": "scripted_teacher",
                "controller_version": "waypoint-v3-full-cycle",
                "curriculum": "near",
                "requested_count": 1,
                "seed_start": 7,
                "max_steps": 1100,
                "capture_stride": 3,
                "cube_jitter": 0.02,
                "yaw_range_degrees": 0.0,
                "randomize_bin": True,
                "bin_jitter": 0.01,
                "mode": "teacher",
            }
            run_id = writer.register_run(provenance)
            writer.save_episode(
                np.full((1, 240, 320), 600, dtype=np.uint16),
                np.zeros((1, 6), dtype=np.float32),
                np.zeros((1, 6), dtype=np.float32),
                seed=7,
                success=True,
                source="teacher",
                run_id=run_id,
                episode_key=f"{run_id}:7",
            )
            args = converter.parse_args([
                "--source-dataset", str(source),
                "--out-dir", str(output),
                "--max-episodes", "1",
            ])
            with mock.patch.object(converter, "TopDepthRenderer", FakeDepthRenderer):
                summary = converter.replay_successes(args)

            self.assertEqual(summary["saved_episodes"], 1)
            manifest = load_delta_manifest(output)
            self.assertEqual(manifest["initial_conditions"], "near_position")
            record = manifest["episodes"][0]
            with np.load(output / record["file"], allow_pickle=False) as archive:
                self.assertEqual(archive["top_depth_mm"].shape[0], 970)
                self.assertEqual(int(archive["top_depth_mm"][0, 0, 0]), 600)
                self.assertEqual(int(archive["side_depth_mm"][0, 0, 0]), 700)
                np.testing.assert_allclose(
                    archive["delta_target_rad"],
                    archive["teacher_goal_pos"] - archive["joint_pos"],
                )
                np.testing.assert_array_equal(archive["teacher_action"], archive["executed_action"])


if __name__ == "__main__":
    unittest.main()
