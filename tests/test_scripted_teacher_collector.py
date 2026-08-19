import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import collect_scripted_teacher_depth_dataset as collector
from teacher_dataset import load_manifest


class FakeDepthRenderer:
    def __init__(self, model):
        self.closed = False

    def render(self, data):
        return np.full((240, 320), 0.65, dtype=np.float32)

    def close(self):
        self.closed = True


class ScriptedTeacherCollectorTest(unittest.TestCase):
    def test_parser_accepts_curriculum_and_sampling_options(self):
        args = collector.parse_args(
            [
                "--episodes", "2", "--out-dir", "dataset", "--curriculum", "near",
                "--cube-jitter", "0.02", "--yaw-range-deg", "5", "--randomize-bin",
                "--bin-jitter", "0.01", "--capture-stride", "3", "--max-attempts", "8",
            ]
        )
        self.assertEqual(args.episodes, 2)
        self.assertEqual(args.curriculum, "near")
        self.assertEqual(args.capture_stride, 3)
        self.assertTrue(args.randomize_bin)

    def test_collects_only_successful_scripted_episode_in_existing_dataset_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            args = collector.parse_args(
                [
                    "--episodes", "1", "--out-dir", str(root), "--curriculum", "fixed",
                    "--capture-stride", "3", "--max-attempts", "1", "--seed", "7",
                ]
            )
            with mock.patch.object(collector, "TopDepthRenderer", FakeDepthRenderer):
                summary = collector.collect_episodes(args)

            manifest = load_manifest(root)
            self.assertEqual(summary["saved_episodes"], 1)
            self.assertEqual(summary["failed_attempts"], 0)
            self.assertEqual(len(manifest["episodes"]), 1)
            episode = root / manifest["episodes"][0]["file"]
            with np.load(episode, allow_pickle=False) as archive:
                self.assertEqual(archive["depth_mm"].shape[1:], (240, 320))
                self.assertEqual(archive["joint_pos"].shape[1:], (6,))
                self.assertEqual(archive["action"].shape[1:], (6,))
                self.assertEqual(str(archive["source"]), "teacher")
                self.assertTrue(bool(archive["success"]))
                self.assertGreater(archive["depth_mm"].shape[0], 300)
                self.assertLessEqual(archive["depth_mm"].shape[0], 367)

    def test_failed_scripted_attempt_is_logged_but_not_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            args = collector.parse_args(
                [
                    "--episodes", "1", "--out-dir", str(root), "--curriculum", "fixed",
                    "--max-attempts", "1", "--seed", "99",
                ]
            )
            failed_info = {
                "is_success": False,
                "phase": "grasp",
                "has_grasped": False,
                "has_lifted": False,
                "distance_to_goal": 0.2,
            }
            with (
                mock.patch.object(collector, "TopDepthRenderer", FakeDepthRenderer),
                mock.patch.object(collector, "execute_waypoint_episode", return_value=failed_info),
            ):
                with self.assertRaisesRegex(RuntimeError, "collected 0/1"):
                    collector.collect_episodes(args)

            manifest = load_manifest(root)
            self.assertEqual(manifest["episodes"], [])
            failure_logs = list((root / "runs").glob("*/failed_attempts.json"))
            self.assertEqual(len(failure_logs), 1)
            self.assertIn('"seed": 99', failure_logs[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
