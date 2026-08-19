from __future__ import annotations

from pathlib import Path
import os

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from so101_workspace import SpawnStage, WorkspaceConfig, sample_block_pose


class SO101BallBinsEnv(gym.Env):
    """State-based MuJoCo environment for SO101 cube-to-bin training."""

    metadata = {"render_modes": []}

    JOINT_NAMES = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    TABLE_SURFACE_Z = WorkspaceConfig().table_surface_z
    GRIPPER_CLOSED_POSITION = -0.17453
    GRIPPER_OPEN_POSITION = 1.0
    CAPTURE_DISTANCE = 0.025
    CAPTURE_HEIGHT_ERROR = 0.015
    CAPTURE_ALIGNMENT = 0.80
    PHASE_APPROACH = 0
    PHASE_CAPTURE = 1
    PHASE_LIFT = 2
    PHASE_TRANSPORT = 3
    PHASE_RELEASE = 4
    PHASE_NAMES = ("approach", "capture", "lift", "transport", "release")
    REQUIRED_STREAK = 3

    def __init__(
        self,
        spawn_stage: SpawnStage = "stage1",
        xml_path: str | Path | None = None,
        max_steps: int = 300,
        frame_skip: int = 17,
        action_scale: float = 0.035,
    ):
        super().__init__()
        self.spawn_stage = spawn_stage
        self.xml_path = Path(xml_path) if xml_path is not None else Path(__file__).with_name("scene_ball_bins.xml")
        self.model = self._load_model(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.action_scale = action_scale
        self.step_count = 0
        self.settle_steps = 0
        self.has_grasped = False
        self.has_lifted = False
        self.just_grasped = False
        self.just_lifted = False
        self.just_failed = False
        self.just_pregrasp_reached = False
        self.phase = self.PHASE_APPROACH
        self.capture_ready_streak = 0
        self.bilateral_contact_streak = 0
        self.grasp_hold_ctrl = self.GRIPPER_CLOSED_POSITION

        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.JOINT_NAMES],
            dtype=np.int32,
        )
        self.qpos_ids = np.array([self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids], dtype=np.int32)
        self.qvel_ids = np.array([self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids], dtype=np.int32)

        self.cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self.cube_qpos_id = int(self.model.jnt_qposadr[self.cube_joint_id])
        self.cube_qvel_id = int(self.model.jnt_dofadr[self.cube_joint_id])
        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
        self.square_goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "square_bin_goal")
        self.fixed_jaw_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_collision"
        )
        self.moving_jaw_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_collision"
        )
        self.gripper_geom_ids = self._gripper_geom_ids()

        self.ctrl_low, self.ctrl_high = self._control_limits()
        self.task_ctrl_low = np.maximum(
            self.ctrl_low,
            np.array([-1.70, -1.70, 0.20, -0.80, 1.10, self.GRIPPER_CLOSED_POSITION]),
        )
        self.task_ctrl_high = np.minimum(
            self.ctrl_high,
            np.array([1.70, 0.60, 1.67, 1.50, 2.00, self.GRIPPER_OPEN_POSITION]),
        )
        self.home_qpos = np.array(
            [0.0, -1.6, 1.3, 1.2, np.pi / 2.0, self.GRIPPER_CLOSED_POSITION],
            dtype=np.float64,
        )
        self.ctrl = np.clip(self.home_qpos.copy(), self.ctrl_low, self.ctrl_high)
        self.previous_close_progress = 0.0
        self.previous_open_progress = 1.0
        self.previous_approach_potential = 0.0
        self.previous_capture_potential = 0.0
        self.previous_transport_potential = 0.0
        self.previous_cube_xy = np.zeros(2, dtype=np.float64)
        self.previous_cube_bottom_z = self.TABLE_SURFACE_Z

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = self._get_obs().shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    @staticmethod
    def _load_model(xml_path: Path):
        xml_path = xml_path.resolve()
        old_cwd = Path.cwd()
        try:
            # MuJoCo on Windows can fail on non-ASCII absolute XML paths.
            # Loading by filename from the XML directory also keeps relative includes/assets correct.
            os.chdir(xml_path.parent)
            return mujoco.MjModel.from_xml_path(xml_path.name)
        finally:
            os.chdir(old_cwd)

    def _control_limits(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.empty(self.model.nu, dtype=np.float64)
        high = np.empty(self.model.nu, dtype=np.float64)
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if bool(self.model.actuator_ctrllimited[actuator_id]):
                low[actuator_id], high[actuator_id] = self.model.actuator_ctrlrange[actuator_id]
            elif bool(self.model.jnt_limited[joint_id]):
                low[actuator_id], high[actuator_id] = self.model.jnt_range[joint_id]
            else:
                low[actuator_id], high[actuator_id] = -np.pi, np.pi
        return low, high

    def _gripper_geom_ids(self) -> frozenset[int]:
        gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        gripper_bodies = set()
        for body_id in range(self.model.nbody):
            ancestor_id = body_id
            while ancestor_id != 0:
                if ancestor_id == gripper_body_id:
                    gripper_bodies.add(body_id)
                    break
                ancestor_id = int(self.model.body_parentid[ancestor_id])
        return frozenset(
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in gripper_bodies
        )

    def _cube_position(self) -> np.ndarray:
        return self.data.geom_xpos[self.cube_geom_id].copy()

    def _target_position(self) -> np.ndarray:
        return self.data.site_xpos[self.square_goal_site_id].copy()

    def _get_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[self.qpos_ids]
        joint_vel = self.data.qvel[self.qvel_ids]
        cube_pos = self._cube_position()
        cube_vel = self.data.qvel[self.cube_qvel_id : self.cube_qvel_id + 3]
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        target_pos = self._target_position()
        obs = np.concatenate(
            [
                joint_pos,
                joint_vel,
                self.ctrl,
                gripper_pos,
                cube_pos,
                cube_vel,
                target_pos,
                self._phase_one_hot(),
            ]
        )
        return obs.astype(np.float32)

    def _phase_one_hot(self) -> np.ndarray:
        phase = np.zeros(len(self.PHASE_NAMES), dtype=np.float64)
        phase[self.phase] = 1.0
        return phase

    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qpos_ids].astype(np.float32).copy()

    def teacher_observation(self) -> np.ndarray:
        return self._get_obs().copy()

    def _jaw_contact_state(self) -> tuple[bool, bool]:
        fixed_contact = False
        moving_contact = False
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom_ids = (int(contact.geom1), int(contact.geom2))
            if self.cube_geom_id not in geom_ids:
                continue
            fixed_contact = fixed_contact or self.fixed_jaw_geom_id in geom_ids
            moving_contact = moving_contact or self.moving_jaw_geom_id in geom_ids
        return fixed_contact, moving_contact

    def _cube_has_gripper_contact(self) -> bool:
        return any(self._jaw_contact_state())

    @staticmethod
    def _horizontal_alignment_score(axis: np.ndarray) -> float:
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-9:
            return 0.0
        vertical_component = float(np.clip(abs(axis[2]) / norm, 0.0, 1.0))
        return float(np.sqrt(max(0.0, 1.0 - vertical_component**2)))

    def _side_grasp_alignment(self) -> float:
        jaw_axis = self.data.geom_xpos[self.moving_jaw_geom_id] - self.data.geom_xpos[self.fixed_jaw_geom_id]
        return self._horizontal_alignment_score(jaw_axis)

    @staticmethod
    def _side_approach_score(approach_axis: np.ndarray, to_cube: np.ndarray) -> float:
        axis_norm = float(np.linalg.norm(approach_axis))
        if axis_norm <= 1e-9:
            return 0.0
        axis_xy = approach_axis[:2]
        axis_xy_norm = float(np.linalg.norm(axis_xy))
        horizontal_score = axis_xy_norm / axis_norm
        target_xy = to_cube[:2]
        target_xy_norm = float(np.linalg.norm(target_xy))
        if axis_xy_norm <= 1e-9:
            return 0.0
        if target_xy_norm <= 1e-9:
            return float(horizontal_score)
        facing_score = float(np.dot(axis_xy / axis_xy_norm, target_xy / target_xy_norm))
        return float(horizontal_score * max(0.0, facing_score))

    def _side_approach_alignment(self) -> float:
        site_rotation = self.data.site_xmat[self.gripper_site_id].reshape(3, 3)
        approach_axis = site_rotation[:, 0]
        to_cube = self._cube_position() - self.data.site_xpos[self.gripper_site_id]
        return self._side_approach_score(approach_axis, to_cube)

    def _pregrasp_position(self) -> np.ndarray:
        cube_pos = self._cube_position()
        radial = cube_pos[:2]
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm <= 1e-9:
            direction = np.array([1.0, 0.0], dtype=np.float64)
        else:
            direction = radial / radial_norm
        target = cube_pos.copy()
        target[:2] -= direction * 0.080
        return target

    def _pregrasp_ready(self) -> bool:
        target = self._pregrasp_position()
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        return bool(
            np.linalg.norm(gripper_pos - target) <= 0.030
            and abs(gripper_pos[2] - target[2]) <= 0.020
            and self._side_approach_alignment() >= 0.75
            and self._side_grasp_alignment() >= 0.75
            and self._gripper_close_progress() <= 0.20
        )

    @classmethod
    def _capture_ready_from_metrics(
        cls,
        distance: float,
        height_error: float,
        side_approach_alignment: float,
        side_grasp_alignment: float,
    ) -> bool:
        del side_approach_alignment, side_grasp_alignment
        return bool(distance <= cls.CAPTURE_DISTANCE and height_error <= cls.CAPTURE_HEIGHT_ERROR)

    def _capture_ready(self) -> bool:
        cube_pos = self._cube_position()
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        return self._capture_ready_from_metrics(
            float(np.linalg.norm(gripper_pos - cube_pos)),
            float(abs(gripper_pos[2] - cube_pos[2])),
            self._side_approach_alignment(),
            self._side_grasp_alignment(),
        )

    def _capture_still_valid(self) -> bool:
        cube_pos = self._cube_position()
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        return bool(
            np.linalg.norm(gripper_pos - cube_pos) <= 0.120
            and abs(gripper_pos[2] - cube_pos[2]) <= 0.040
        )

    @classmethod
    def _gripper_close_progress_from_position(cls, position: float) -> float:
        span = cls.GRIPPER_OPEN_POSITION - cls.GRIPPER_CLOSED_POSITION
        return float(np.clip((cls.GRIPPER_OPEN_POSITION - position) / span, 0.0, 1.0))

    def _gripper_close_progress(self) -> float:
        gripper_index = self.JOINT_NAMES.index("gripper")
        return self._gripper_close_progress_from_position(float(self.data.qpos[self.qpos_ids[gripper_index]]))

    @staticmethod
    def _gripper_shaping_reward(capture_ready: bool, close_progress: float, previous_progress: float) -> float:
        closing_delta = close_progress - previous_progress
        if capture_ready:
            return float(8.0 * max(0.0, closing_delta) - 2.0 * max(0.0, -closing_delta))
        return float(
            2.0 * max(0.0, -closing_delta)
            - 4.0 * max(0.0, closing_delta)
            - 0.02 * close_progress
        )

    @staticmethod
    def _potential_progress(current: float, previous: float, scale: float) -> float:
        return float(scale * (current - previous))

    @staticmethod
    def _approach_potential_from_metrics(
        distance: float,
        height_alignment: float,
        side_approach_alignment: float,
        side_grasp_alignment: float,
    ) -> float:
        del height_alignment, side_approach_alignment, side_grasp_alignment
        return float(-5.0 * distance)

    def _approach_potential(self) -> float:
        target = self._pregrasp_position()
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        distance = float(np.linalg.norm(gripper_pos - target))
        height_error = float(abs(gripper_pos[2] - target[2]))
        height_alignment = float(np.exp(-25.0 * height_error))
        return self._approach_potential_from_metrics(
            distance,
            height_alignment,
            self._side_approach_alignment(),
            self._side_grasp_alignment(),
        )

    def _capture_potential(self) -> float:
        cube_pos = self._cube_position()
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        distance = float(np.linalg.norm(gripper_pos - cube_pos))
        height_error = float(abs(gripper_pos[2] - cube_pos[2]))
        height_alignment = float(np.exp(-25.0 * height_error))
        return float(
            -10.0 * distance
            + 0.25 * height_alignment
            + 0.25 * self._side_approach_alignment()
            + 0.25 * self._side_grasp_alignment()
        )

    def _transport_potential(self) -> float:
        cube_pos = self._cube_position()
        goal_distance = float(np.linalg.norm(cube_pos[:2] - self._target_position()[:2]))
        wall_clearance = float(
            np.clip(
                (self._cube_bottom_z() - (self.TABLE_SURFACE_Z + 0.015)) / 0.075,
                0.0,
                1.0,
            )
        )
        return float(-20.0 * goal_distance + 5.0 * wall_clearance)

    def _release_ready(self) -> bool:
        goal_distance = float(np.linalg.norm(self._cube_position()[:2] - self._target_position()[:2]))
        clears_rim = self._cube_bottom_z() >= self.TABLE_SURFACE_Z + 0.085
        return bool(self.has_lifted and goal_distance <= 0.015 and clears_rim)

    def _dropped_outside_bin(self) -> bool:
        if not self.has_lifted:
            return False
        goal_distance = float(np.linalg.norm(self._cube_position()[:2] - self._target_position()[:2]))
        on_table = self._cube_bottom_z() <= self.TABLE_SURFACE_Z + 0.005
        return bool(on_table and goal_distance > 0.030)

    def _cube_corners(self) -> np.ndarray:
        center = self.data.geom_xpos[self.cube_geom_id]
        rotation = self.data.geom_xmat[self.cube_geom_id].reshape(3, 3)
        half_size = self.model.geom_size[self.cube_geom_id]
        local_corners = np.array(
            [
                [x, y, z]
                for x in (-half_size[0], half_size[0])
                for y in (-half_size[1], half_size[1])
                for z in (-half_size[2], half_size[2])
            ],
            dtype=np.float64,
        )
        return local_corners @ rotation.T + center

    def _cube_bottom_z(self) -> float:
        return float(np.min(self._cube_corners()[:, 2]))

    def _update_manipulation_state(self) -> None:
        fixed_contact, moving_contact = self._jaw_contact_state()
        bilateral_contact = fixed_contact and moving_contact

        if not self.has_grasped and bilateral_contact and self._capture_ready():
            self.bilateral_contact_streak += 1
            if self.bilateral_contact_streak >= self.REQUIRED_STREAK:
                self.has_grasped = True
                self.just_grasped = True
                self.grasp_hold_ctrl = max(
                    self.GRIPPER_CLOSED_POSITION,
                    float(self.ctrl[5]) - 0.12,
                )
                self.phase = self.PHASE_LIFT
                return
        elif not bilateral_contact:
            self.bilateral_contact_streak = 0

        if self.phase == self.PHASE_APPROACH:
            self.capture_ready_streak = self.capture_ready_streak + 1 if self._pregrasp_ready() else 0
            if self.capture_ready_streak >= self.REQUIRED_STREAK:
                self.phase = self.PHASE_CAPTURE
                self.bilateral_contact_streak = 0
                self.just_pregrasp_reached = True
        elif self.phase == self.PHASE_CAPTURE:
            if not bilateral_contact and not self._capture_still_valid():
                self.phase = self.PHASE_APPROACH
                self.capture_ready_streak = 0
                self.bilateral_contact_streak = 0
                return
        elif self.phase == self.PHASE_LIFT:
            if bilateral_contact and self._cube_bottom_z() >= self.TABLE_SURFACE_Z + 0.015:
                self.has_lifted = True
                self.just_lifted = True
                self.phase = self.PHASE_TRANSPORT
        elif self.phase == self.PHASE_TRANSPORT and self._release_ready():
            self.phase = self.PHASE_RELEASE

        if self._dropped_outside_bin():
            self.just_failed = True

    def _valid_settled_placement(self) -> bool:
        corners = self._cube_corners()
        tolerance = 1e-6
        bin_x, bin_y = self._target_position()[:2]
        in_bin = bool(
            np.all(corners[:, 0] >= bin_x - 0.035 - tolerance)
            and np.all(corners[:, 0] <= bin_x + 0.035 + tolerance)
            and np.all(corners[:, 1] >= bin_y - 0.035 - tolerance)
            and np.all(corners[:, 1] <= bin_y + 0.035 + tolerance)
        )
        below_rim = bool(np.max(corners[:, 2]) <= self.TABLE_SURFACE_Z + 0.080)
        linear_velocity = self.data.qvel[self.cube_qvel_id : self.cube_qvel_id + 3]
        angular_velocity = self.data.qvel[self.cube_qvel_id + 3 : self.cube_qvel_id + 6]
        stable = bool(np.linalg.norm(linear_velocity) < 0.05 and np.linalg.norm(angular_velocity) < 0.05)
        return self.has_lifted and in_bin and below_rim and stable and not self._cube_has_gripper_contact()

    def _update_settled_success(self) -> bool:
        if self._valid_settled_placement():
            self.settle_steps += 1
        else:
            self.settle_steps = 0
        return self.settle_steps >= 15

    def _info(self) -> dict:
        cube_pos = self._cube_position()
        target_pos = self._target_position()
        fixed_contact, moving_contact = self._jaw_contact_state()
        distance = float(np.linalg.norm(cube_pos[:2] - target_pos[:2]))
        grasp_height_error = float(abs(self.data.site_xpos[self.gripper_site_id, 2] - cube_pos[2]))
        capture_ready = self._capture_ready()
        success = bool(self.settle_steps >= 15)
        return {
            "is_success": success,
            "distance_to_goal": distance,
            "cube_z": float(cube_pos[2]),
            "cube_bottom_z": self._cube_bottom_z(),
            "settle_steps": self.settle_steps,
            "has_grasped": self.has_grasped,
            "has_lifted": self.has_lifted,
            "fixed_jaw_contact": fixed_contact,
            "moving_jaw_contact": moving_contact,
            "side_grasp_alignment": self._side_grasp_alignment(),
            "side_approach_alignment": self._side_approach_alignment(),
            "grasp_height_error": grasp_height_error,
            "capture_ready": capture_ready,
            "gripper_close_progress": self._gripper_close_progress(),
            "phase": self.PHASE_NAMES[self.phase],
            "is_failure": self.just_failed,
            "spawn_stage": self.spawn_stage,
        }

    def _reward(self, info: dict) -> float:
        cube_pos = self._cube_position()
        close_progress = self._gripper_close_progress()
        open_progress = 1.0 - close_progress
        approach_potential = self._approach_potential()
        capture_potential = self._capture_potential()
        transport_potential = self._transport_potential()
        cube_xy_motion = float(np.linalg.norm(cube_pos[:2] - self.previous_cube_xy))
        cube_bottom_z = self._cube_bottom_z()
        lift_delta = cube_bottom_z - self.previous_cube_bottom_z

        reward = -0.01
        if self.phase == self.PHASE_APPROACH:
            reward += self._potential_progress(approach_potential, self.previous_approach_potential, 4.0)
            reward += self._gripper_shaping_reward(False, close_progress, self.previous_close_progress)
            reward -= 200.0 * cube_xy_motion
        elif self.phase == self.PHASE_CAPTURE:
            reward += self._potential_progress(capture_potential, self.previous_capture_potential, 4.0)
            reward += self._gripper_shaping_reward(
                self._capture_ready(), close_progress, self.previous_close_progress
            )
        elif self.phase == self.PHASE_LIFT:
            reward += 200.0 * max(0.0, lift_delta) - 50.0 * max(0.0, -lift_delta)
        elif self.phase == self.PHASE_TRANSPORT:
            reward += self._potential_progress(transport_potential, self.previous_transport_potential, 1.0)
        elif self.phase == self.PHASE_RELEASE:
            opening_delta = open_progress - self.previous_open_progress
            reward += 10.0 * max(0.0, opening_delta) - 2.0 * max(0.0, -opening_delta)

        if self.just_pregrasp_reached:
            reward += 5.0
        if self.just_grasped:
            reward += 15.0
        if self.just_lifted:
            reward += 20.0
        if self.just_failed:
            reward -= 20.0
        if info["is_success"]:
            reward += 100.0

        self.previous_close_progress = close_progress
        self.previous_open_progress = open_progress
        self.previous_approach_potential = approach_potential
        self.previous_capture_potential = capture_potential
        self.previous_transport_potential = transport_potential
        self.previous_cube_xy = cube_pos[:2].copy()
        self.previous_cube_bottom_z = cube_bottom_z
        return float(reward)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.settle_steps = 0
        self.has_grasped = False
        self.has_lifted = False
        self.just_grasped = False
        self.just_lifted = False
        self.just_failed = False
        self.just_pregrasp_reached = False
        self.phase = self.PHASE_APPROACH
        self.capture_ready_streak = 0
        self.bilateral_contact_streak = 0
        self.grasp_hold_ctrl = self.GRIPPER_CLOSED_POSITION

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_ids] = self.home_qpos
        self.ctrl = np.clip(self.home_qpos.copy(), self.ctrl_low, self.ctrl_high)
        self.data.ctrl[:] = self.ctrl

        self.data.qpos[self.cube_qpos_id : self.cube_qpos_id + 7] = sample_block_pose(
            self.np_random, self.spawn_stage
        )
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.previous_close_progress = self._gripper_close_progress()
        self.previous_open_progress = 1.0 - self.previous_close_progress
        self.previous_approach_potential = self._approach_potential()
        self.previous_capture_potential = self._capture_potential()
        self.previous_transport_potential = self._transport_potential()
        self.previous_cube_xy = self._cube_position()[:2].copy()
        self.previous_cube_bottom_z = self._cube_bottom_z()
        return self._get_obs(), self._info()

    def control_target(self, action: np.ndarray) -> np.ndarray:
        """Return the absolute six-joint position target for a normalized action."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (len(self.JOINT_NAMES),) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite vector with shape (6,)")
        action = np.clip(action, -1.0, 1.0)
        requested_ctrl = self.ctrl + action * self.action_scale
        if (
            self.phase in (self.PHASE_APPROACH, self.PHASE_CAPTURE)
            and not self._capture_ready()
            and action[5] < 0.0
        ):
            requested_ctrl[5] = self.ctrl[5]
        if self.phase in (self.PHASE_LIFT, self.PHASE_TRANSPORT):
            requested_ctrl[5] = min(requested_ctrl[5], self.grasp_hold_ctrl)
        return np.clip(requested_ctrl, self.task_ctrl_low, self.task_ctrl_high).copy()

    def step(self, action):
        self.step_count += 1
        self.just_grasped = False
        self.just_lifted = False
        self.just_failed = False
        self.just_pregrasp_reached = False
        self.ctrl = self.control_target(action)

        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self.ctrl
            mujoco.mj_step(self.model, self.data)

        self._update_manipulation_state()

        self._update_settled_success()
        info = self._info()
        reward = self._reward(info)
        terminated = bool(info["is_success"] or info["is_failure"])
        truncated = bool(self.step_count >= self.max_steps)
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        pass
