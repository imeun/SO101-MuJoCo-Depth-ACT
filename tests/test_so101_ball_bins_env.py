import unittest

import mujoco
import numpy as np

from so101_ball_bins_env import SO101BallBinsEnv
from so101_workspace import WorkspaceConfig


class SO101BallBinsEnvTest(unittest.TestCase):
    def _place_cube(self, env: SO101BallBinsEnv, position: tuple[float, float, float]) -> None:
        env.data.qpos[env.cube_qpos_id : env.cube_qpos_id + 7] = np.array(
            [*position, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        env.data.qvel[env.cube_qvel_id : env.cube_qvel_id + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)

    def test_default_environment_is_single_task_with_expected_frame_skip(self):
        """Catches retaining the transitional task selector or old 10-frame default."""
        env = SO101BallBinsEnv()
        try:
            self.assertEqual(env.frame_skip, 17)
            self.assertFalse(hasattr(env, "task_mode"))
            self.assertFalse(hasattr(env, "current_task"))
            self.assertFalse(hasattr(env, "round_goal_site_id"))
            self.assertFalse(hasattr(env, "_sample_task"))
        finally:
            env.close()

    def test_reset_returns_finite_vector_observation(self):
        """Catches reset emitting data outside its declared observation contract."""
        env = SO101BallBinsEnv()
        obs, info = env.reset(seed=7)
        try:
            self.assertEqual(obs.shape, env.observation_space.shape)
            self.assertTrue(np.all(np.isfinite(obs)))
            np.testing.assert_array_equal(obs[-5:], [1.0, 0.0, 0.0, 0.0, 0.0])
        finally:
            env.close()

    def test_reset_clears_all_task_phase_history(self):
        env = SO101BallBinsEnv()
        try:
            env.phase = env.PHASE_RELEASE
            env.has_grasped = True
            env.has_lifted = True
            env.capture_ready_streak = 9
            env.bilateral_contact_streak = 9
            _, info = env.reset(seed=7)
            self.assertEqual(env.phase, env.PHASE_APPROACH)
            self.assertFalse(env.has_grasped)
            self.assertFalse(env.has_lifted)
            self.assertEqual(env.capture_ready_streak, 0)
            self.assertEqual(env.bilateral_contact_streak, 0)
            self.assertEqual(info["phase"], "approach")
        finally:
            env.close()

    def test_home_pose_starts_with_side_grasp_wrist_orientation(self):
        """Catches resetting the wrist to the vertical pinch orientation."""
        env = SO101BallBinsEnv()
        try:
            env.reset(seed=7)
            wrist_roll_index = env.JOINT_NAMES.index("wrist_roll")
            self.assertAlmostEqual(float(env.home_qpos[wrist_roll_index]), np.pi / 2.0, places=6)
            self.assertGreater(env._side_grasp_alignment(), 0.85)
            gripper_index = env.JOINT_NAMES.index("gripper")
            self.assertAlmostEqual(float(env.home_qpos[gripper_index]), env.GRIPPER_CLOSED_POSITION)
            np.testing.assert_allclose(env.home_qpos[1:4], [-1.6, 1.3, 1.2], atol=1e-6)
            self.assertLess(np.linalg.norm(env.data.site_xpos[env.gripper_site_id, :2]), 0.21)
            self.assertLess(float(env.data.site_xpos[env.gripper_site_id, 2]), 0.07)
        finally:
            env.close()

    def test_grasp_center_is_between_the_finger_pads(self):
        env = SO101BallBinsEnv()
        try:
            env.reset(seed=7)
            frame_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
            center_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
            self.assertEqual(env.gripper_site_id, center_id)
            frame_position = env.data.site_xpos[frame_id]
            frame_rotation = env.data.site_xmat[frame_id].reshape(3, 3)
            expected_center = frame_position + 0.0175 * frame_rotation[:, 2]
            np.testing.assert_allclose(env.data.site_xpos[center_id], expected_center, atol=1e-6)
            self.assertEqual(env.model.geom_type[env.fixed_jaw_geom_id], mujoco.mjtGeom.mjGEOM_BOX)
            self.assertEqual(env.model.geom_type[env.moving_jaw_geom_id], mujoco.mjtGeom.mjGEOM_BOX)
        finally:
            env.close()

    def test_scene_matches_measured_block_bin_and_camera(self):
        env = SO101BallBinsEnv()
        try:
            cube = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
            cube_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
            cube_free = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
            camera = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
            side_depth_camera = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "side_depth")
            side_depth_mount = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "side_depth_camera_mount")
            right_wall = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "right_workspace_wall")
            fixed_jaw = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_collision")
            moving_jaw = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_collision")
            square_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "square_bin")
            square_goal = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "square_bin_goal")
            round_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "round_bin")
            round_material = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_MATERIAL, "round_bin_mat")

            self.assertNotEqual(cube, -1)
            self.assertNotEqual(cube_body, -1)
            self.assertNotEqual(cube_free, -1)
            self.assertNotEqual(camera, -1)
            self.assertNotEqual(side_depth_camera, -1)
            self.assertNotEqual(side_depth_mount, -1)
            self.assertNotEqual(right_wall, -1)
            self.assertNotEqual(fixed_jaw, -1)
            self.assertNotEqual(moving_jaw, -1)
            self.assertNotEqual(square_body, -1)
            self.assertNotEqual(square_goal, -1)
            self.assertEqual(round_body, -1)
            self.assertEqual(round_material, -1)
            self.assertEqual(env.model.geom_type[cube], mujoco.mjtGeom.mjGEOM_BOX)
            np.testing.assert_allclose(env.model.geom_size[cube], [0.0175, 0.0175, 0.035], atol=1e-6)
            np.testing.assert_allclose(env.model.body_pos[cube_body, 2], 0.0336, atol=1e-6)
            np.testing.assert_allclose(env.model.body_mass[cube_body], 0.025, atol=1e-6)
            mujoco.mj_forward(env.model, env.data)
            np.testing.assert_allclose(env.data.cam_xpos[camera], [0.280, 0.000, 0.6176], atol=1e-6)
            camera_orientation = env.data.cam_xmat[camera].reshape(3, 3)
            np.testing.assert_allclose(-camera_orientation[:, 2], [0.0, 0.0, -1.0], atol=1e-6)
            np.testing.assert_allclose(
                env.data.cam_xpos[side_depth_camera],
                [0.2740173, 0.4300, 0.1526],
                atol=1e-6,
            )
            expected_view = np.array([0.300, 0.000, 0.1526]) - env.data.cam_xpos[side_depth_camera]
            expected_view /= np.linalg.norm(expected_view)
            side_orientation = env.data.cam_xmat[side_depth_camera].reshape(3, 3)
            np.testing.assert_allclose(-side_orientation[:, 2], expected_view, atol=1e-6)
            self.assertAlmostEqual(float((-side_orientation[:, 2])[2]), 0.0, places=6)
            self.assertAlmostEqual(
                float(env.model.cam_fovy[side_depth_camera]),
                63.891862067260504,
                places=5,
            )

            np.testing.assert_allclose(env.model.geom_size[right_wall], [0.2725, 0.005, 0.2025], atol=1e-6)
            self.assertAlmostEqual(
                float(env.model.geom_pos[right_wall, 1] + env.model.geom_size[right_wall, 1]),
                -0.220,
                places=6,
            )
            self.assertAlmostEqual(
                float(env.model.geom_pos[right_wall, 0] - env.model.geom_size[right_wall, 0]),
                -0.0309827,
                places=6,
            )
            self.assertAlmostEqual(
                float(env.model.geom_pos[right_wall, 2] - env.model.geom_size[right_wall, 2]),
                -0.0024,
                places=6,
            )
            self.assertFalse(bool(env.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD))
            self.assertGreaterEqual(int(env.model.opt.iterations), 80)
            self.assertGreaterEqual(float(env.model.geom_friction[fixed_jaw, 0]), 2.0)
            self.assertGreaterEqual(float(env.model.geom_friction[moving_jaw, 0]), 2.0)
            np.testing.assert_allclose(env.model.body_pos[square_body, :2], [0.310, -0.050], atol=1e-6)
            np.testing.assert_allclose(env.model.body_pos[square_body, 2], -0.0024, atol=1e-6)
            np.testing.assert_allclose(env.model.site_rgba[square_goal], [0.0, 0.0, 1.0, 0.0], atol=1e-6)

            table = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
            self.assertNotEqual(table, -1)
            self.assertAlmostEqual(
                float(env.model.geom_pos[table, 2] + env.model.geom_size[table, 2]),
                -0.0024,
                places=6,
            )

            for name, expected_position in (
                ("front", [0.0, 0.036, 0.0375]),
                ("back", [0.0, -0.036, 0.0375]),
            ):
                wall = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"square_bin_wall_{name}")
                self.assertNotEqual(wall, -1)
                np.testing.assert_allclose(env.model.geom_size[wall], [0.037, 0.001, 0.0375], atol=1e-6)
                np.testing.assert_allclose(env.model.geom_pos[wall], expected_position, atol=1e-6)
            for name, expected_position in (
                ("left", [-0.036, 0.0, 0.0375]),
                ("right", [0.036, 0.0, 0.0375]),
            ):
                wall = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"square_bin_wall_{name}")
                self.assertNotEqual(wall, -1)
                np.testing.assert_allclose(env.model.geom_size[wall], [0.001, 0.037, 0.0375], atol=1e-6)
                np.testing.assert_allclose(env.model.geom_pos[wall], expected_position, atol=1e-6)

            floor = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "square_bin_floor")
            self.assertNotEqual(floor, -1)
            np.testing.assert_allclose(env.model.geom_size[floor], [0.035, 0.035, 0.001], atol=1e-6)
        finally:
            env.close()

    def test_seeded_resets_are_deterministic_and_use_valid_workspace_poses(self):
        """Catches reset bypassing the validated sampler or ignoring the Gym seed."""
        env = SO101BallBinsEnv(spawn_stage="stage2")
        try:
            env.reset(seed=27)
            first_pose = env.data.qpos[env.cube_qpos_id : env.cube_qpos_id + 7].copy()
            env.reset(seed=27)
            second_pose = env.data.qpos[env.cube_qpos_id : env.cube_qpos_id + 7].copy()
            np.testing.assert_array_equal(first_pose, second_pose)

            x, y = first_pose[:2]
            config = WorkspaceConfig()
            self.assertGreaterEqual(np.hypot(x, y), config.stage2_radius[0])
            self.assertLessEqual(np.hypot(x, y), config.stage2_radius[1])
            self.assertFalse(abs(x - 0.310) < 0.065 and abs(y + 0.050) < 0.065)
            self.assertEqual(first_pose[2], 0.0336)
        finally:
            env.close()

    def test_joint_positions_are_float32_six_vector_copy(self):
        """Catches exposing a live qpos view or an incomplete joint observation."""
        env = SO101BallBinsEnv()
        try:
            env.reset(seed=3)
            positions = env.joint_positions()
            self.assertEqual(positions.shape, (6,))
            self.assertEqual(positions.dtype, np.float32)
            original = float(env.data.qpos[env.qpos_ids[0]])
            positions[0] += 1.0
            self.assertEqual(float(env.data.qpos[env.qpos_ids[0]]), original)
        finally:
            env.close()

    def test_teacher_observation_is_privileged_observation_copy(self):
        """Catches teacher observation exposing a mutable alias instead of the full observation."""
        env = SO101BallBinsEnv()
        try:
            env.reset(seed=3)
            observation = env.teacher_observation()
            np.testing.assert_array_equal(observation, env._get_obs())
            observation[0] += 1.0
            self.assertNotEqual(observation[0], env._get_obs()[0])
        finally:
            env.close()

    def test_step_accepts_normalized_action_and_reports_settled_info(self):
        """Catches finite action handling or required settled-success telemetry being removed."""
        env = SO101BallBinsEnv()
        obs, _ = env.reset(seed=11)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        try:
            self.assertEqual(next_obs.shape, obs.shape)
            self.assertTrue(np.isfinite(reward))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIn("is_success", info)
            self.assertIn("distance_to_goal", info)
            self.assertIn("cube_z", info)
            self.assertIn("settle_steps", info)
            self.assertIn("has_grasped", info)
            self.assertIn("has_lifted", info)
            self.assertIn("fixed_jaw_contact", info)
            self.assertIn("moving_jaw_contact", info)
            self.assertIn("side_grasp_alignment", info)
            self.assertIn("side_approach_alignment", info)
            self.assertIn("grasp_height_error", info)
            self.assertIn("capture_ready", info)
            self.assertIn("gripper_close_progress", info)
            self.assertIn("phase", info)
            self.assertIn("cube_bottom_z", info)
            self.assertIn("spawn_stage", info)
            self.assertNotIn("task", info)
        finally:
            env.close()

    def test_control_target_matches_the_absolute_position_applied_by_step(self):
        env = SO101BallBinsEnv()
        try:
            env.reset(seed=11)
            action = np.array([0.2, -0.3, 0.1, -0.2, 0.0, 0.4], dtype=np.float32)
            expected_target = env.control_target(action)

            env.step(action)

            np.testing.assert_allclose(env.ctrl, expected_target, rtol=0.0, atol=1e-7)
            self.assertEqual(expected_target.shape, (6,))
            self.assertTrue(np.all(np.isfinite(expected_target)))
            self.assertFalse(np.shares_memory(expected_target, env.ctrl))
        finally:
            env.close()

    def test_success_requires_fifteen_consecutive_valid_placement_steps(self):
        """Catches success becoming true before 15 stable, in-bin, contact-free policy steps."""
        env = SO101BallBinsEnv()
        try:
            env.reset(seed=4)
            env.has_lifted = True
            self._place_cube(env, (0.310, -0.050, 0.0346))
            for expected_steps in range(1, 15):
                self.assertFalse(env._update_settled_success())
                self.assertEqual(env.settle_steps, expected_steps)
            self.assertTrue(env._update_settled_success())
            self.assertEqual(env.settle_steps, 15)
        finally:
            env.close()

    def test_pushed_block_in_bin_is_not_success_without_prior_lift(self):
        """Catches a policy earning success by sliding through a bin wall."""
        env = SO101BallBinsEnv()
        try:
            env.reset(seed=4)
            self._place_cube(env, (0.310, -0.050, 0.0346))
            for _ in range(20):
                self.assertFalse(env._update_settled_success())
            self.assertFalse(env.has_lifted)
            self.assertEqual(env.settle_steps, 0)
        finally:
            env.close()

    def test_grasp_then_clearance_sets_lift_history(self):
        env = SO101BallBinsEnv()
        try:
            env.reset(seed=4)
            env.has_grasped = True
            env.phase = env.PHASE_LIFT
            env._jaw_contact_state = lambda: (True, True)
            self._place_cube(env, (0.200, 0.100, 0.085))
            env._update_manipulation_state()
            self.assertTrue(env.has_lifted)
        finally:
            env.close()

    def test_capture_and_grasp_require_three_consecutive_policy_steps(self):
        """Catches transient alignment or fingertip contact advancing the task phase."""
        env = SO101BallBinsEnv()
        try:
            env.reset(seed=4)
            env._pregrasp_ready = lambda: True
            for _ in range(2):
                env._update_manipulation_state()
                self.assertEqual(env.phase, env.PHASE_APPROACH)
            env._update_manipulation_state()
            self.assertEqual(env.phase, env.PHASE_CAPTURE)

            env._capture_still_valid = lambda: True
            env._capture_ready = lambda: True
            env._jaw_contact_state = lambda: (True, False)
            env._update_manipulation_state()
            self.assertFalse(env.has_grasped)

            env._jaw_contact_state = lambda: (True, True)
            for _ in range(2):
                env._update_manipulation_state()
                self.assertFalse(env.has_grasped)
            env._update_manipulation_state()
            self.assertTrue(env.has_grasped)
            self.assertTrue(env.just_grasped)
            self.assertEqual(env.phase, env.PHASE_LIFT)

            env.just_grasped = False
            env._update_manipulation_state()
            self.assertFalse(env.just_grasped)
        finally:
            env.close()

    def test_bilateral_capture_is_recognized_even_after_phase_falls_back(self):
        env = SO101BallBinsEnv()
        try:
            env.reset(seed=7)
            env.ctrl[5] = 0.5
            env.phase = env.PHASE_APPROACH
            env._capture_ready = lambda: True
            env._pregrasp_ready = lambda: False
            env._jaw_contact_state = lambda: (True, True)
            for _ in range(env.REQUIRED_STREAK):
                env._update_manipulation_state()
            self.assertTrue(env.has_grasped)
            self.assertEqual(env.phase, env.PHASE_LIFT)
            self.assertLess(env.grasp_hold_ctrl, env.ctrl[5])
        finally:
            env.close()

    def test_pregrasp_target_is_eight_centimeters_before_cube(self):
        env = SO101BallBinsEnv()
        try:
            env.reset(seed=4)
            cube = env._cube_position()
            target = env._pregrasp_position()
            radial = cube[:2] / np.linalg.norm(cube[:2])
            np.testing.assert_allclose(target[:2], cube[:2] - radial * 0.08, atol=1e-6)
            self.assertAlmostEqual(float(target[2]), float(cube[2]), places=6)
        finally:
            env.close()

    def test_side_grasp_alignment_scores_horizontal_jaw_axis(self):
        """Catches rewarding a vertical top-down pinch for the upright block."""
        env = SO101BallBinsEnv()
        try:
            self.assertAlmostEqual(env._horizontal_alignment_score(np.array([1.0, 0.0, 0.0])), 1.0)
            self.assertAlmostEqual(env._horizontal_alignment_score(np.array([0.0, 1.0, 0.0])), 1.0)
            self.assertAlmostEqual(env._horizontal_alignment_score(np.array([0.0, 0.0, 1.0])), 0.0)
        finally:
            env.close()

    def test_side_approach_requires_horizontal_axis_facing_cube(self):
        """Catches a top-down or backwards tool pose receiving side-approach reward."""
        env = SO101BallBinsEnv()
        try:
            to_cube = np.array([1.0, 0.0, 0.0])
            self.assertAlmostEqual(env._side_approach_score(np.array([1.0, 0.0, 0.0]), to_cube), 1.0)
            self.assertAlmostEqual(env._side_approach_score(np.array([0.0, 0.0, -1.0]), to_cube), 0.0)
            self.assertAlmostEqual(env._side_approach_score(np.array([-1.0, 0.0, 0.0]), to_cube), 0.0)
        finally:
            env.close()

    def test_capture_ready_uses_position_not_redundant_orientation_thresholds(self):
        env = SO101BallBinsEnv()
        try:
            self.assertTrue(env._capture_ready_from_metrics(0.024, 0.014, 0.81, 0.81))
            self.assertFalse(env._capture_ready_from_metrics(0.026, 0.014, 0.81, 0.81))
            self.assertFalse(env._capture_ready_from_metrics(0.024, 0.016, 0.81, 0.81))
            self.assertTrue(env._capture_ready_from_metrics(0.024, 0.014, 0.0, 0.0))
        finally:
            env.close()

    def test_gripper_progress_and_shaping_reward_close_only_when_capture_ready(self):
        env = SO101BallBinsEnv()
        try:
            self.assertAlmostEqual(env._gripper_close_progress_from_position(env.GRIPPER_OPEN_POSITION), 0.0)
            self.assertAlmostEqual(env._gripper_close_progress_from_position(env.GRIPPER_CLOSED_POSITION), 1.0)
            self.assertLess(env._gripper_shaping_reward(False, 0.8, 0.2), 0.0)
            self.assertGreater(env._gripper_shaping_reward(False, 0.2, 0.8), 0.0)
            self.assertGreater(env._gripper_shaping_reward(True, 0.8, 0.2), 0.0)
            self.assertEqual(env._gripper_shaping_reward(True, 0.8, 0.8), 0.0)
        finally:
            env.close()

    def test_progress_reward_is_zero_when_state_does_not_improve(self):
        env = SO101BallBinsEnv()
        try:
            self.assertEqual(env._potential_progress(0.75, 0.75, 4.0), 0.0)
            self.assertGreater(env._potential_progress(0.80, 0.75, 4.0), 0.0)
            self.assertLess(env._potential_progress(0.70, 0.75, 4.0), 0.0)
        finally:
            env.close()

    def test_approach_potential_strongly_rewards_one_centimeter_distance_improvement(self):
        env = SO101BallBinsEnv()
        try:
            far = env._approach_potential_from_metrics(0.11, 1.0, 1.0, 1.0)
            near = env._approach_potential_from_metrics(0.10, 1.0, 1.0, 1.0)
            reward = env._potential_progress(near, far, 4.0)
            self.assertGreaterEqual(reward, 0.20 - 1e-9)
        finally:
            env.close()

    def test_approach_potential_ignores_orientation_until_position_is_reached(self):
        first = SO101BallBinsEnv._approach_potential_from_metrics(0.10, 0.0, 0.0, 0.0)
        second = SO101BallBinsEnv._approach_potential_from_metrics(0.10, 1.0, 1.0, 1.0)
        self.assertEqual(first, second)

    def test_task_control_limits_exclude_inverted_arm_postures(self):
        env = SO101BallBinsEnv()
        try:
            self.assertLessEqual(env.task_ctrl_high[1], 0.60)
            self.assertGreaterEqual(env.task_ctrl_low[2], 0.20)
            self.assertGreaterEqual(env.task_ctrl_low[4], 1.10)
            self.assertLessEqual(env.task_ctrl_high[4], 2.00)
        finally:
            env.close()

    def test_approach_cannot_close_gripper(self):
        env = SO101BallBinsEnv()
        try:
            env.reset(seed=7)
            env.ctrl[5] = 0.5
            env.data.ctrl[:] = env.ctrl
            env.phase = env.PHASE_APPROACH
            env.step(np.array([0, 0, 0, 0, 0, -1], dtype=np.float32))
            self.assertGreaterEqual(env.ctrl[5], 0.5)
        finally:
            env.close()

    def test_capture_phase_returns_to_approach_when_pregrasp_pose_is_lost(self):
        env = SO101BallBinsEnv()
        try:
            env.reset(seed=4)
            env.phase = env.PHASE_CAPTURE
            env._capture_still_valid = lambda: False
            env._jaw_contact_state = lambda: (False, False)
            env._update_manipulation_state()
            self.assertEqual(env.phase, env.PHASE_APPROACH)
            self.assertEqual(env.bilateral_contact_streak, 0)
        finally:
            env.close()

    def test_stationary_approach_does_not_collect_pose_reward(self):
        env = SO101BallBinsEnv()
        try:
            env.reset(seed=4)
            env.previous_approach_potential = env._approach_potential()
            env.previous_cube_xy = env._cube_position()[:2].copy()
            env.previous_close_progress = env._gripper_close_progress()
            reward = env._reward(env._info())
            self.assertLessEqual(reward, -0.01)
        finally:
            env.close()

    def test_invalid_placement_or_gripper_contact_resets_settled_success(self):
        """Catches accepting a cube outside the bin, above its rim, or touching the gripper."""
        env = SO101BallBinsEnv()
        try:
            env.reset(seed=4)
            env.has_lifted = True
            self._place_cube(env, (0.310, -0.050, 0.0346))
            for _ in range(4):
                self.assertFalse(env._update_settled_success())

            self._place_cube(env, (0.370, -0.050, 0.0346))
            self.assertFalse(env._update_settled_success())
            self.assertEqual(env.settle_steps, 0)

            self._place_cube(env, (0.310, -0.050, 0.0476))
            self.assertFalse(env._update_settled_success())
            self.assertEqual(env.settle_steps, 0)

            gripper_position = tuple(env.data.site_xpos[env.gripper_site_id])
            self._place_cube(env, gripper_position)
            contacts = [
                (int(env.data.contact[index].geom1), int(env.data.contact[index].geom2))
                for index in range(env.data.ncon)
            ]
            self.assertTrue(
                any(
                    env.cube_geom_id in pair
                    and (pair[0] in env.gripper_geom_ids or pair[1] in env.gripper_geom_ids)
                    for pair in contacts
                )
            )
            self.assertFalse(env._update_settled_success())
            self.assertEqual(env.settle_steps, 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
