import unittest
import os
from pathlib import Path

import mujoco
import numpy as np

from so101_depth import (
    DepthConfig,
    TopDepthRenderer,
    augment_depth_sequence,
    depth_to_millimetres,
    millimetres_to_depth,
    preprocess_depth,
    resize_depth_nearest,
)


class DepthConfigTest(unittest.TestCase):
    def test_rejects_invalid_render_and_augmentation_ranges(self):
        """Catches accepting dimensions, depth ranges, or probabilities outside the public contract."""
        invalid_kwargs = (
            {"width": 0},
            {"height": -1},
            {"near_m": 1.0, "far_m": 1.0},
            {"near_m": 1.0, "far_m": 0.2},
            {"invalid_fill_m": 0.19},
            {"invalid_fill_m": 1.01},
            {"noise_std_range_m": (0.005, 0.001)},
            {"noise_std_range_m": (-0.001, 0.005)},
            {"invalid_pixel_probability": -0.01},
            {"invalid_pixel_probability": 1.01},
            {"frame_bias_range_m": (0.003, -0.003)},
            {"depth_scale_range": (1.01, 0.99)},
            {"invalid_pixel_probability_range": (0.02, 0.01)},
            {"edge_dropout_probability_range": (-0.1, 0.2)},
            {"edge_dropout_probability_range": (0.1, 1.1)},
            {"hole_count_range": (3, 1)},
            {"hole_size_range_px": (20, 3)},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DepthConfig(**kwargs)


class ResizeDepthNearestTest(unittest.TestCase):
    def test_repeats_hand_derived_nearest_pixels_as_float32_copy(self):
        """Catches selecting an incorrect source pixel, returning the wrong layout, or exposing an input view."""
        depth = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

        resized = resize_depth_nearest(depth, width=6, height=4)

        expected = np.array(
            [
                [1, 1, 2, 2, 3, 3],
                [1, 1, 2, 2, 3, 3],
                [4, 4, 5, 5, 6, 6],
                [4, 4, 5, 5, 6, 6],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(resized, expected)
        self.assertEqual(resized.shape, (4, 6))
        self.assertEqual(resized.dtype, np.float32)
        resized[0, 0] = 99.0
        self.assertEqual(depth[0, 0], 1.0)

    def test_same_shape_returns_a_copy(self):
        """Catches the resize fast path returning a mutable alias of the input depth image."""
        depth = np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32)

        resized = resize_depth_nearest(depth, width=2, height=2)

        np.testing.assert_array_equal(resized, depth)
        resized[0, 0] = 9.0
        self.assertEqual(depth[0, 0], 0.3)

    def test_rejects_non_image_and_non_numeric_depth_inputs(self):
        """Catches accepting empty, non-2D, boolean, or object arrays as depth images."""
        invalid_depths = (
            np.array([], dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((2, 0), dtype=np.float32),
            np.ones((2, 2, 1), dtype=np.float32),
            np.ones((2, 2), dtype=bool),
            np.array([["0.5"]], dtype=object),
        )
        for depth in invalid_depths:
            with self.subTest(shape=depth.shape, dtype=depth.dtype), self.assertRaises(ValueError):
                resize_depth_nearest(depth, width=2, height=2)


class PreprocessDepthTest(unittest.TestCase):
    def test_replaces_invalid_clips_and_normalizes_to_chw_float32(self):
        """Catches invalid pixels leaking through, clipping after normalization, or an incorrect model tensor layout."""
        depth = np.array([[np.nan, np.inf, 0.0], [-1.0, 0.5, 1.2]], dtype=np.float64)
        config = DepthConfig(width=3, height=2, near_m=0.2, far_m=1.0, invalid_fill_m=0.8)

        processed = preprocess_depth(depth, config)

        expected = np.array([[[0.75, 0.75, 0.75], [0.75, 0.375, 1.0]]], dtype=np.float32)
        np.testing.assert_array_equal(processed, expected)
        self.assertEqual(processed.shape, (1, 2, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertGreaterEqual(float(processed.min()), 0.0)
        self.assertLessEqual(float(processed.max()), 1.0)

    def test_augmentation_is_seeded_and_requires_a_generator(self):
        """Catches global/random augmentation or allowing unseeded augmentation calls."""
        depth = np.full((2, 2), 0.6, dtype=np.float32)
        config = DepthConfig(
            width=2,
            height=2,
            near_m=0.2,
            far_m=1.0,
            invalid_fill_m=0.9,
            noise_std_range_m=(0.002, 0.004),
            invalid_pixel_probability=0.5,
        )

        with self.assertRaises(ValueError):
            preprocess_depth(depth, config, augment=True)

        first = preprocess_depth(depth, config, rng=np.random.default_rng(917), augment=True)
        second = preprocess_depth(depth, config, rng=np.random.default_rng(917), augment=True)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, preprocess_depth(depth, config)))

    def test_rejects_invalid_preprocess_depth_input(self):
        """Catches preprocessing silently coercing an invalid image instead of rejecting it."""
        with self.assertRaises(ValueError):
            preprocess_depth(np.array([0.3, 0.4], dtype=np.float32))

    def test_sequence_augmentation_is_seeded_structured_and_preserves_shape(self):
        sequence = np.full((8, 24, 32), 0.65, dtype=np.float32)
        sequence[:, :, 16:] = 0.45
        config = DepthConfig(
            width=32,
            height=24,
            frame_bias_range_m=(-0.003, 0.003),
            depth_scale_range=(0.995, 1.005),
            invalid_pixel_probability_range=(0.01, 0.02),
            edge_dropout_probability_range=(0.2, 0.2),
            hole_count_range=(2, 2),
            hole_size_range_px=(3, 6),
        )

        first = augment_depth_sequence(sequence, config, rng=np.random.default_rng(17))
        second = augment_depth_sequence(sequence, config, rng=np.random.default_rng(17))

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (8, 1, 24, 32))
        self.assertEqual(first.dtype, np.float32)
        invalid_value = (config.invalid_fill_m - config.near_m) / (config.far_m - config.near_m)
        invalid = np.isclose(first[:, 0], invalid_value)
        self.assertTrue(np.any(invalid))
        self.assertTrue(np.any(np.all(invalid, axis=0)))

    def test_sequence_augmentation_rejects_non_sequence_input(self):
        with self.assertRaises(ValueError):
            augment_depth_sequence(
                np.full((24, 32), 0.65, dtype=np.float32),
                DepthConfig(width=32, height=24),
                rng=np.random.default_rng(1),
            )


class DepthMillimetreConversionTest(unittest.TestCase):
    def test_converts_valid_metres_and_maps_invalid_values_to_zero(self):
        """Catches truncating millimetres or retaining non-finite and non-positive depth values."""
        depth = np.array([[0.580, 0.0004, np.nan], [np.inf, -0.1, 100.0]], dtype=np.float64)

        converted = depth_to_millimetres(depth)

        expected = np.array([[580, 0, 0], [0, 0, 65535]], dtype=np.uint16)
        np.testing.assert_array_equal(converted, expected)
        self.assertEqual(converted.dtype, np.uint16)

    def test_converts_uint16_millimetres_to_float32_metres(self):
        """Catches the shared metric conversion using the wrong scale, dtype, or invalid-zero convention."""
        depth_mm = np.array([[0, 580], [1000, 65535]], dtype=np.uint16)

        converted = millimetres_to_depth(depth_mm)

        expected = np.array([[0.0, 0.58], [1.0, 65.535]], dtype=np.float32)
        np.testing.assert_allclose(converted, expected, rtol=0.0, atol=1e-6)
        self.assertEqual(converted.dtype, np.float32)

    def test_rejects_invalid_conversion_input_dimensions_and_dtypes(self):
        """Catches accepting non-image metre input or storage formats other than uint16 millimetres."""
        with self.assertRaises(ValueError):
            depth_to_millimetres(np.array([0.58], dtype=np.float32))
        for invalid_mm in (
            np.array([[580]], dtype=np.int32),
            np.array([[580.0]], dtype=np.float32),
            np.array([580], dtype=np.uint16),
        ):
            with self.subTest(shape=invalid_mm.shape, dtype=invalid_mm.dtype), self.assertRaises(ValueError):
                millimetres_to_depth(invalid_mm)


class TopDepthRendererIntegrationTest(unittest.TestCase):
    @staticmethod
    def load_scene_ball_bins_model() -> mujoco.MjModel:
        """Load includes from their XML directory while preserving the caller's working directory."""
        original_cwd = Path.cwd()
        xml_directory = Path(__file__).resolve().parents[1]
        try:
            os.chdir(xml_directory)
            return mujoco.MjModel.from_xml_path("scene_ball_bins.xml")
        finally:
            os.chdir(original_cwd)

    def test_real_top_camera_returns_independent_metric_depth_images(self):
        """Catches non-metric, aliased, blank, or physically implausible depth from the real top camera."""
        model = self.load_scene_ball_bins_model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = TopDepthRenderer(model)
        try:
            first = renderer.render(data)
            second = renderer.render(data)
            self.assertEqual(first.shape, (240, 320))
            self.assertEqual(first.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(first)))
            self.assertTrue(np.all(first > 0.0))
            self.assertGreater(float(first.max() - first.min()), 0.001)
            self.assertGreaterEqual(float(np.median(first)), 0.35)
            self.assertLessEqual(float(np.median(first)), 0.75)
            self.assertTrue(np.any((first >= 0.50) & (first <= 0.70)))
            self.assertFalse(np.shares_memory(first, second))
            first[0, 0] = 123.0
            self.assertNotEqual(second[0, 0], 123.0)
        finally:
            renderer.close()

    def test_side_depth_camera_renders_metric_workspace_and_wall(self):
        model = self.load_scene_ball_bins_model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = TopDepthRenderer(model, camera_name="side_depth")
        try:
            depth = renderer.render(data)
            self.assertEqual(depth.shape, (240, 320))
            self.assertEqual(depth.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(depth)))
            self.assertGreater(float(np.ptp(depth)), 0.05)
            self.assertGreater(float(np.median(depth)), 0.1)
            self.assertLess(float(np.median(depth)), 1.0)
        finally:
            renderer.close()

    def test_simulated_depth_cameras_use_measured_d435_intrinsics(self):
        model = self.load_scene_ball_bins_model()
        expected = {
            "top": {
                "focal": [382.2572326660156, 382.2572326660156],
                "principal_offset": [1.12847900390625, 1.9199676513671875],
                "fovy": 64.24527021290076,
            },
            "side_depth": {
                "focal": [384.8880310058594, 384.8880310058594],
                "principal_offset": [7.94189453125, 1.1730499267578125],
                "fovy": 63.891862067260504,
            },
        }
        for camera_name, intrinsics in expected.items():
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            np.testing.assert_array_equal(model.cam_resolution[camera_id], [640, 480])
            np.testing.assert_allclose(model.cam_sensorsize[camera_id], [640, 480], atol=1e-6)
            np.testing.assert_allclose(
                model.cam_intrinsic[camera_id, :2], intrinsics["focal"], atol=1e-5
            )
            np.testing.assert_allclose(
                model.cam_intrinsic[camera_id, 2:], intrinsics["principal_offset"], atol=1e-5
            )
            self.assertAlmostEqual(float(model.cam_fovy[camera_id]), intrinsics["fovy"], places=5)

    def test_top_depth_camera_matches_measured_pose_and_vertical_plan_orientation(self):
        model = self.load_scene_ball_bins_model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "top_depth_camera_mount")
        housing_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "top_depth_camera_body")
        axis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "top_depth_camera_axis")

        self.assertNotEqual(body_id, -1)
        self.assertNotEqual(housing_id, -1)
        self.assertEqual(axis_id, -1)
        self.assertEqual(int(model.cam_bodyid[camera_id]), body_id)
        np.testing.assert_allclose(data.cam_xpos[camera_id], [0.280, 0.0, 0.6176], atol=1e-6)
        orientation = data.cam_xmat[camera_id].reshape(3, 3)
        np.testing.assert_allclose(-orientation[:, 2], [0.0, 0.0, -1.0], atol=1e-6)
        body_orientation = data.xmat[body_id].reshape(3, 3)
        np.testing.assert_allclose(body_orientation[:, 0], [1.0, 0.0, 0.0], atol=1e-6)
        self.assertEqual(int(model.geom_contype[housing_id]), 0)

    def test_missing_camera_is_rejected_before_rendering(self):
        """Catches creating a renderer that cannot resolve its requested MuJoCo camera."""
        model = self.load_scene_ball_bins_model()

        with self.assertRaisesRegex(ValueError, "camera"):
            TopDepthRenderer(model, camera_name="missing_camera")

    def test_render_after_close_is_rejected_and_close_is_idempotent(self):
        """Catches a closed renderer forwarding into MuJoCo or failing during repeated cleanup."""
        model = self.load_scene_ball_bins_model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = TopDepthRenderer(model)
        renderer.close()
        renderer.close()

        with self.assertRaisesRegex(RuntimeError, "closed"):
            renderer.render(data)


if __name__ == "__main__":
    unittest.main()
