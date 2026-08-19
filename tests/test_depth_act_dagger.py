import tempfile
import unittest
from pathlib import Path

import numpy as np

from collect_fixed_delta_depth_dataset import place_nearby_scene
from collect_depth_act_dagger import collect_episodes, parse_args
from delta_depth_dataset import load_delta_manifest
from so101_ball_bins_env import SO101BallBinsEnv
from so101_depth_act_dagger import InterventionController, StateAwareWaypointTeacher


class InterventionControllerTest(unittest.TestCase):
    def test_takeover_uses_hysteresis_and_minimum_duration(self):
        controller = InterventionController(
            joint_scale=np.ones(2),
            trigger_threshold=0.10,
            release_threshold=0.04,
            minimum_teacher_steps=3,
        )

        self.assertFalse(controller.update(np.array([0.02, 0.0]), np.zeros(2)))
        self.assertTrue(controller.update(np.array([0.20, 0.0]), np.zeros(2)))
        self.assertTrue(controller.update(np.zeros(2), np.zeros(2)))
        self.assertTrue(controller.update(np.zeros(2), np.zeros(2)))
        self.assertFalse(controller.update(np.zeros(2), np.zeros(2)))
        self.assertEqual(controller.intervention_count, 1)
        self.assertEqual(controller.teacher_steps, 3)


class StateAwareWaypointTeacherTest(unittest.TestCase):
    def test_teacher_completes_pick_place_and_return_home(self):
        environment = SO101BallBinsEnv(spawn_stage="stage1", max_steps=1600)
        try:
            environment.reset(seed=10000)
            place_nearby_scene(
                environment,
                seed=10000,
                cube_jitter=0.020,
                bin_jitter=0.010,
            )
            teacher = StateAwareWaypointTeacher(environment)
            ever_success = False

            for _ in range(1500):
                action, goal, phase = teacher.command()
                self.assertEqual(action.shape, (6,))
                self.assertEqual(goal.shape, (6,))
                self.assertIsInstance(phase, str)
                _, _, _, truncated, info = environment.step(action)
                ever_success |= bool(info["is_success"])
                teacher.observe(info)
                if teacher.complete or truncated or info["is_failure"]:
                    break

            self.assertTrue(ever_success)
            self.assertTrue(teacher.complete)
            np.testing.assert_allclose(
                environment.joint_positions(),
                environment.home_qpos,
                atol=0.08,
            )
        finally:
            environment.close()


class DAggerCollectorTest(unittest.TestCase):
    def test_teacher_only_episode_is_saved_in_delta_depth_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "dagger"
            args = parse_args(
                [
                    "--teacher-only",
                    "--out-dir",
                    str(output),
                    "--episodes",
                    "1",
                    "--max-attempts",
                    "2",
                    "--max-steps",
                    "1600",
                    "--cube-jitter",
                    "0.02",
                    "--bin-jitter",
                    "0.01",
                ]
            )
            summary = collect_episodes(args)

            manifest = load_delta_manifest(output)
            self.assertEqual(summary["saved_episodes"], 1)
            self.assertEqual(len(manifest["episodes"]), 1)
            with np.load(output / manifest["episodes"][0]["file"], allow_pickle=False) as episode:
                np.testing.assert_allclose(episode["teacher_action"], episode["executed_action"])
                np.testing.assert_allclose(
                    episode["delta_target_rad"],
                    episode["teacher_goal_pos"] - episode["joint_pos"],
                )


if __name__ == "__main__":
    unittest.main()
