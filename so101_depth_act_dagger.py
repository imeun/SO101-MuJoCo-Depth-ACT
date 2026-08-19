from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from play_waypoint_teacher import build_waypoints
from so101_ball_bins_env import SO101BallBinsEnv


@dataclass(frozen=True)
class TeacherPhase:
    name: str
    minimum_steps: int
    maximum_steps: int
    tolerance: float = 0.05


TEACHER_PHASES = (
    TeacherPhase("open", 12, 80),
    TeacherPhase("pregrasp", 20, 180),
    TeacherPhase("grasp", 20, 180),
    TeacherPhase("close", 8, 100),
    TeacherPhase("hold", 5, 20),
    TeacherPhase("lift", 20, 220),
    TeacherPhase("bin", 20, 280),
    TeacherPhase("release", 20, 180),
    TeacherPhase("settle", 20, 100),
    TeacherPhase("retreat", 15, 140),
    TeacherPhase("home", 20, 180),
    TeacherPhase("home_hold", 30, 60),
)


class InterventionController:
    def __init__(
        self,
        *,
        joint_scale: np.ndarray,
        trigger_threshold: float = 0.04,
        release_threshold: float = 0.015,
        minimum_teacher_steps: int = 20,
    ):
        scale = np.asarray(joint_scale, dtype=np.float64)
        if scale.ndim != 1 or scale.size == 0 or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("joint_scale must be a finite positive vector")
        if (
            not math.isfinite(trigger_threshold)
            or not math.isfinite(release_threshold)
            or trigger_threshold <= 0.0
            or release_threshold < 0.0
            or release_threshold >= trigger_threshold
        ):
            raise ValueError("intervention thresholds must satisfy 0 <= release < trigger")
        if minimum_teacher_steps <= 0:
            raise ValueError("minimum_teacher_steps must be positive")
        self.joint_scale = scale
        self.trigger_threshold = float(trigger_threshold)
        self.release_threshold = float(release_threshold)
        self.minimum_teacher_steps = int(minimum_teacher_steps)
        self.active = False
        self.current_teacher_steps = 0
        self.teacher_steps = 0
        self.intervention_count = 0
        self.last_error = 0.0

    def reset(self) -> None:
        self.active = False
        self.current_teacher_steps = 0
        self.teacher_steps = 0
        self.intervention_count = 0
        self.last_error = 0.0

    def update(self, policy_target: np.ndarray, teacher_target: np.ndarray) -> bool:
        policy = np.asarray(policy_target, dtype=np.float64)
        teacher = np.asarray(teacher_target, dtype=np.float64)
        if policy.shape != self.joint_scale.shape or teacher.shape != self.joint_scale.shape:
            raise ValueError("policy and teacher targets must match joint_scale")
        if not np.all(np.isfinite(policy)) or not np.all(np.isfinite(teacher)):
            raise ValueError("policy and teacher targets must be finite")
        self.last_error = float(np.max(np.abs(policy - teacher) / self.joint_scale))

        if self.active:
            if self.current_teacher_steps >= self.minimum_teacher_steps and self.last_error <= self.release_threshold:
                self.active = False
                self.current_teacher_steps = 0
                return False
            self.current_teacher_steps += 1
            self.teacher_steps += 1
            return True

        if self.last_error >= self.trigger_threshold:
            self.active = True
            self.current_teacher_steps = 1
            self.teacher_steps += 1
            self.intervention_count += 1
            return True
        return False


class StateAwareWaypointTeacher:
    """Waypoint expert that waits for observed progress and retries failed grasps."""

    def __init__(self, environment: SO101BallBinsEnv):
        self.environment = environment
        self.waypoints = build_waypoints(environment)
        self.phase_index = 0
        self.phase_steps = 0
        self.retry_count = 0
        self.complete = False

    @property
    def phase(self) -> TeacherPhase:
        return TEACHER_PHASES[self.phase_index]

    def _target_reached(self) -> bool:
        target = self.waypoints[self.phase.name]
        position = self.environment.joint_positions().astype(np.float64)
        if self.phase.name in {"close", "hold"}:
            return bool(self.environment.has_grasped)
        arm_error = float(np.max(np.abs(position[:5] - target[:5])))
        if self.phase.name in {"release", "settle", "retreat"}:
            gripper_error = abs(float(position[5] - target[5]))
            return arm_error <= self.phase.tolerance and gripper_error <= 0.10
        return arm_error <= self.phase.tolerance and abs(float(position[5] - target[5])) <= 0.10

    def _advance(self) -> None:
        if self.phase_index == len(TEACHER_PHASES) - 1:
            self.complete = True
            return
        self.phase_index += 1
        self.phase_steps = 0

    def _retry_grasp(self) -> None:
        self.phase_index = 0
        self.phase_steps = 0
        self.retry_count += 1
        self.waypoints = build_waypoints(self.environment)

    def command(self) -> tuple[np.ndarray, np.ndarray, str]:
        if self.complete:
            raise RuntimeError("teacher episode is complete")
        target = self.waypoints[self.phase.name]
        action = np.clip(
            (target - self.environment.ctrl) / self.environment.action_scale,
            -1.0,
            1.0,
        ).astype(np.float32)
        goal = self.environment.control_target(action).astype(np.float32)
        return action, goal, self.phase.name

    def observe(self, info: dict) -> None:
        if self.complete:
            return
        self.phase_steps += 1
        name = self.phase.name

        if name == "close" and self.phase_steps >= self.phase.maximum_steps and not info.get("has_grasped"):
            self._retry_grasp()
            return
        if name == "lift" and self.phase_steps >= self.phase.maximum_steps and not info.get("has_lifted"):
            self._retry_grasp()
            return

        ready = self._target_reached()
        if name == "close":
            ready = bool(info.get("has_grasped"))
        elif name == "lift":
            ready = bool(info.get("has_lifted")) and ready
        elif name == "release":
            ready = ready and self.environment._gripper_close_progress() <= 0.20
        elif name == "settle":
            ready = self.phase_steps >= self.phase.minimum_steps and bool(info.get("is_success"))
        elif name in {"hold", "home_hold"}:
            ready = self.phase_steps >= self.phase.minimum_steps

        if self.phase_steps >= self.phase.minimum_steps and ready:
            self._advance()
        elif self.phase_steps >= self.phase.maximum_steps:
            if name in {"open", "pregrasp", "grasp", "close", "hold", "lift"}:
                self._retry_grasp()
            else:
                self._advance()
