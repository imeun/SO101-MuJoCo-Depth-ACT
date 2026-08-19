import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim2real_joint_mapping import (
    JOINT_NAMES,
    JointMapping,
    observation_to_real_degrees,
    read_degrees_from_robot_bus,
)


class JointMappingTest(unittest.TestCase):
    def test_direct_degree_to_radian_mapping(self):
        mapping = JointMapping.identity()
        actual = mapping.real_degrees_to_sim_radians([0, 90, -90, 180, -180, 45])
        expected = np.array([0, math.pi / 2, -math.pi / 2, math.pi, -math.pi, math.pi / 4])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_sign_and_offset_mapping_round_trips(self):
        mapping = JointMapping(
            signs=np.array([1, -1, 1, -1, 1, -1], dtype=np.float64),
            offset_rad=np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6], dtype=np.float64),
        )
        real_degrees = np.array([5, 10, -15, 20, -25, 30], dtype=np.float64)
        sim_radians = mapping.real_degrees_to_sim_radians(real_degrees)
        restored = mapping.sim_radians_to_real_degrees(sim_radians)
        np.testing.assert_allclose(restored, real_degrees, atol=1e-12)

    def test_observation_is_ordered_by_joint_names(self):
        observation = {
            "gripper.pos": 6.0,
            "wrist_roll.pos": 5.0,
            "wrist_flex.pos": 4.0,
            "elbow_flex.pos": 3.0,
            "shoulder_lift.pos": 2.0,
            "shoulder_pan.pos": 1.0,
        }
        np.testing.assert_array_equal(
            observation_to_real_degrees(observation),
            np.arange(1.0, len(JOINT_NAMES) + 1.0),
        )

    def test_json_save_and_load_preserves_mapping(self):
        mapping = JointMapping(
            signs=np.array([1, -1, 1, 1, -1, 1], dtype=np.float64),
            offset_rad=np.linspace(0.0, 0.5, 6),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            mapping.save(path, metadata={"port": "/dev/ttyACM0"})
            loaded = JointMapping.load(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        np.testing.assert_array_equal(loaded.signs, mapping.signs)
        np.testing.assert_allclose(loaded.offset_rad, mapping.offset_rad)
        self.assertEqual(payload["port"], "/dev/ttyACM0")
        self.assertEqual(payload["joint_names"], list(JOINT_NAMES))

    def test_invalid_sign_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "signs"):
            JointMapping(signs=np.ones(6) * 0.5, offset_rad=np.zeros(6))

    def test_read_only_capture_bypasses_robot_connect_and_configuration(self):
        class FakeBus:
            def __init__(self):
                self.calls = []

            def connect(self):
                self.calls.append(("connect",))

            def disconnect(self, disable_torque):
                self.calls.append(("disconnect", disable_torque))

        class FakeRobot:
            def __init__(self):
                self.bus = FakeBus()

            def connect(self, calibrate=False):
                raise AssertionError("robot.connect() would configure and write motor registers")

            def get_observation(self):
                return {f"{joint}.pos": float(index) for index, joint in enumerate(JOINT_NAMES)}

        robot = FakeRobot()
        result = read_degrees_from_robot_bus(robot)

        np.testing.assert_array_equal(result, np.arange(6, dtype=np.float64))
        self.assertEqual(robot.bus.calls, [("connect",), ("disconnect", False)])


if __name__ == "__main__":
    unittest.main()
