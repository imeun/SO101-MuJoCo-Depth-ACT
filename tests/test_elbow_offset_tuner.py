import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tune_elbow_offset_with_robot import (
    adjusted_mapping,
    adjusted_mapping_many,
    parse_args,
    save_adjusted_mapping,
    save_adjusted_mapping_many,
)
from sweep_joint_calibration import AffineJointMapping


class ElbowOffsetTunerTest(unittest.TestCase):
    def setUp(self):
        self.mapping = AffineJointMapping(
            scale_rad_per_real_unit=np.array([np.pi / 180] * 5 + [0.02]),
            offset_rad=np.zeros(6),
            real_period=np.array([360.0] * 5 + [0.0]),
            real_reference=np.zeros(6),
        )

    def test_positive_trim_adjusts_selected_wrist_flex_only(self):
        sim_pose = np.array([0.0, -1.2, 1.0, 0.8, 1.5, 0.1])
        original_real = self.mapping.sim_to_real(sim_pose)
        adjusted = adjusted_mapping(
            self.mapping,
            joint_name="wrist_flex",
            trim_degrees=1.0,
        )
        adjusted_real = adjusted.sim_to_real(sim_pose)

        self.assertAlmostEqual(adjusted.offset_rad[3], np.deg2rad(1.0))
        self.assertAlmostEqual(adjusted_real[3] - original_real[3], -1.0)
        np.testing.assert_allclose(adjusted_real[[0, 1, 2, 4, 5]], original_real[[0, 1, 2, 4, 5]])

    def test_save_preserves_swept_limits_and_records_trim(self):
        payload = {
            "version": 4,
            "joint_names": [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper",
            ],
            "scale_rad_per_real_unit": [np.pi / 180] * 5 + [0.02],
            "offset_rad": [0.0] * 6,
            "real_period": [360.0] * 5 + [0.0],
            "real_reference": [0.0] * 6,
            "real_at_sim_low": [-100.0] * 6,
            "real_at_sim_high": [100.0] * 6,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.json"
            output = Path(temp_dir) / "adjusted.json"
            source.write_text(json.dumps(payload), encoding="utf-8")

            save_adjusted_mapping(
                source,
                output,
                joint_name="wrist_flex",
                trim_degrees=1.5,
            )
            saved = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(saved["real_at_sim_low"], payload["real_at_sim_low"])
            self.assertAlmostEqual(saved["offset_rad"][3], np.deg2rad(1.5))
            self.assertEqual(saved["manual_joint_name"], "wrist_flex")
            self.assertEqual(saved["manual_joint_trim_degrees"], 1.5)

    def test_multiple_joint_trims_are_applied_independently(self):
        adjusted = adjusted_mapping_many(
            self.mapping,
            {"elbow_flex": 2.0, "wrist_flex": 12.0},
        )

        self.assertAlmostEqual(adjusted.offset_rad[2], np.deg2rad(2.0))
        self.assertAlmostEqual(adjusted.offset_rad[3], np.deg2rad(12.0))
        np.testing.assert_allclose(adjusted.offset_rad[[0, 1, 4, 5]], 0.0)

    def test_multiple_joint_trims_are_saved_together(self):
        payload = {
            "version": 4,
            "joint_names": list((
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper",
            )),
            "scale_rad_per_real_unit": [np.pi / 180] * 5 + [0.02],
            "offset_rad": [0.0] * 6,
            "real_period": [360.0] * 5 + [0.0],
            "real_reference": [0.0] * 6,
            "real_at_sim_low": [-100.0] * 6,
            "real_at_sim_high": [100.0] * 6,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.json"
            output = Path(temp_dir) / "adjusted.json"
            source.write_text(json.dumps(payload), encoding="utf-8")

            save_adjusted_mapping_many(
                source,
                output,
                joint_trims_degrees={"elbow_flex": 2.0, "wrist_flex": 12.0},
            )
            saved = json.loads(output.read_text(encoding="utf-8"))

            self.assertAlmostEqual(saved["offset_rad"][2], np.deg2rad(2.0))
            self.assertAlmostEqual(saved["offset_rad"][3], np.deg2rad(12.0))
            self.assertEqual(
                saved["manual_joint_trims_degrees"],
                {"elbow_flex": 2.0, "wrist_flex": 12.0},
            )

    def test_default_trim_limit_is_thirty_degrees(self):
        args = parse_args([])
        self.assertEqual(args.max_absolute_trim, 30.0)
