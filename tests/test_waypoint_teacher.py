import unittest

import numpy as np

from play_waypoint_teacher import (
    build_waypoints,
    configure_viewer_camera,
    execute_waypoint_episode,
    playback_delay_seconds,
    randomize_bin_position,
    randomize_cube_pose,
    run_waypoint_episode,
)
from so101_ball_bins_env import SO101BallBinsEnv


class WaypointTeacherTest(unittest.TestCase):
    def test_top_view_selects_the_model_top_camera(self):
        env = SO101BallBinsEnv(spawn_stage="stage1")

        class Camera:
            type = None
            fixedcamid = -1

        class Viewer:
            cam = Camera()

        try:
            viewer = Viewer()
            configure_viewer_camera(viewer, env.model, "top")
            self.assertEqual(viewer.cam.type, 2)
            self.assertGreaterEqual(viewer.cam.fixedcamid, 0)
        finally:
            env.close()

    def test_playback_speed_scales_the_control_period(self):
        self.assertAlmostEqual(playback_delay_seconds(0.002, 17, 2.0), 0.017)
        with self.assertRaises(ValueError):
            playback_delay_seconds(0.002, 17, 0.0)

    def test_waypoints_are_finite_valid_joint_targets(self):
        env = SO101BallBinsEnv(spawn_stage="stage1", max_steps=1100)
        try:
            env.reset(seed=7)
            waypoints = build_waypoints(env)
            self.assertEqual(
                set(waypoints),
                {
                    "open",
                    "pregrasp",
                    "grasp",
                    "close",
                    "hold",
                    "lift",
                    "bin",
                    "release",
                    "settle",
                    "retreat",
                    "home",
                    "home_hold",
                },
            )
            for target in waypoints.values():
                self.assertEqual(target.shape, (6,))
                self.assertTrue(np.all(np.isfinite(target)))
                self.assertTrue(np.all(target >= env.task_ctrl_low))
                self.assertTrue(np.all(target <= env.task_ctrl_high))
        finally:
            env.close()

    def test_headless_stage1_waypoint_episode_succeeds(self):
        result = run_waypoint_episode(seed=7, render=False, spawn_stage="stage1")
        self.assertTrue(result["is_success"])
        self.assertTrue(result["has_grasped"])
        self.assertTrue(result["has_lifted"])

    def test_executor_reports_every_applied_action(self):
        env = SO101BallBinsEnv(spawn_stage="stage1", max_steps=1100)
        actions = []
        try:
            env.reset(seed=7)
            result = execute_waypoint_episode(env, on_step=lambda action, phase: actions.append((action, phase)))
            final_joint_positions = env.joint_positions().copy()
            final_control_target = env.ctrl.copy()
        finally:
            env.close()
        self.assertTrue(result["is_success"])
        self.assertGreater(len(actions), 100)
        self.assertTrue(all(action.shape == (6,) for action, _ in actions))
        self.assertTrue(all(action.dtype == np.float32 for action, _ in actions))
        self.assertTrue(all(np.all(np.isfinite(action)) for action, _ in actions))
        self.assertTrue(all(np.all((-1.0 <= action) & (action <= 1.0)) for action, _ in actions))
        phases = [phase for _, phase in actions]
        hold_indices = [index for index, phase in enumerate(phases) if phase == "hold"]
        self.assertTrue(hold_indices)
        self.assertLessEqual(len(hold_indices), 5)
        self.assertEqual(phases[hold_indices[-1] + 1], "lift")
        phase_starts = {phase: phases.index(phase) for phase in set(phases)}
        self.assertLess(phase_starts["release"], phase_starts["settle"])
        self.assertLess(phase_starts["settle"], phase_starts["retreat"])
        self.assertLess(phase_starts["retreat"], phase_starts["home"])
        self.assertLess(phase_starts["home"], phase_starts["home_hold"])
        self.assertEqual(phases[-1], "home_hold")
        np.testing.assert_allclose(final_control_target, env.home_qpos, atol=1e-6)
        np.testing.assert_allclose(final_joint_positions, env.home_qpos, atol=1e-3)

    def test_executor_reports_clean_and_transformed_actions_before_execution(self):
        env = SO101BallBinsEnv(spawn_stage="stage1", max_steps=1100)
        reported = []
        transformed_steps = 0

        def transform(action, phase, step):
            nonlocal transformed_steps
            transformed_steps += 1
            result = action.copy()
            if step == 0:
                result[0] = np.clip(result[0] + 0.02, -1.0, 1.0)
            return result

        try:
            env.reset(seed=7)
            result = execute_waypoint_episode(
                env,
                action_transform=transform,
                on_control_step=lambda clean, executed, phase: reported.append(
                    (clean.copy(), executed.copy(), phase)
                ),
            )
        finally:
            env.close()

        self.assertTrue(result["is_success"])
        self.assertEqual(transformed_steps, len(reported))
        self.assertGreater(len(reported), 100)
        self.assertFalse(np.array_equal(reported[0][0], reported[0][1]))
        self.assertTrue(all(clean.shape == executed.shape == (6,) for clean, executed, _ in reported))

    def test_bin_randomization_is_deterministic_and_avoids_the_cube(self):
        first = SO101BallBinsEnv(spawn_stage="stage2")
        second = SO101BallBinsEnv(spawn_stage="stage2")
        try:
            first.reset(seed=20)
            second.reset(seed=20)
            first_position = randomize_bin_position(first, np.random.default_rng(99), jitter=0.03)
            second_position = randomize_bin_position(second, np.random.default_rng(99), jitter=0.03)
            np.testing.assert_array_equal(first_position, second_position)
            self.assertGreater(np.linalg.norm(first_position - first._cube_position()[:2]), 0.10)
            np.testing.assert_allclose(first._target_position()[:2], first_position, atol=1e-6)
        finally:
            first.close()
            second.close()

    def test_cube_randomization_is_deterministic_and_respects_requested_ranges(self):
        first = SO101BallBinsEnv(spawn_stage="stage1")
        second = SO101BallBinsEnv(spawn_stage="stage1")
        try:
            first.reset(seed=20)
            second.reset(seed=20)
            first_pose = randomize_cube_pose(
                first, np.random.default_rng(101), xy_jitter=0.02, yaw_range_degrees=10.0
            )
            second_pose = randomize_cube_pose(
                second, np.random.default_rng(101), xy_jitter=0.02, yaw_range_degrees=10.0
            )
            np.testing.assert_array_equal(first_pose, second_pose)
            self.assertTrue(np.all(np.abs(first_pose[:2] - np.array([0.32, 0.155])) <= 0.02))
            yaw = 2.0 * np.arctan2(first_pose[6], first_pose[3])
            self.assertLessEqual(abs(np.degrees(yaw)), 10.0)
        finally:
            first.close()
            second.close()


if __name__ == "__main__":
    unittest.main()
