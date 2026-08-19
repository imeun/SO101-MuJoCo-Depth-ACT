import unittest

import numpy as np

from preview_realsense_depth import depth_to_display_u8


class RealSenseDepthPreviewTest(unittest.TestCase):
    def test_depth_display_preserves_invalid_pixels_and_scales_range(self):
        depth = np.array([[0, 500, 1000, 1500, 2000]], dtype=np.uint16)

        display = depth_to_display_u8(depth, max_depth_mm=1500)

        self.assertEqual(display.dtype, np.uint8)
        self.assertEqual(display.shape, depth.shape)
        self.assertEqual(int(display[0, 0]), 0)
        self.assertEqual(int(display[0, 1]), 85)
        self.assertEqual(int(display[0, 2]), 170)
        self.assertEqual(int(display[0, 3]), 255)
        self.assertEqual(int(display[0, 4]), 255)

    def test_depth_display_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            depth_to_display_u8(np.zeros((2, 2), dtype=np.float32), max_depth_mm=1000)
        with self.assertRaises(ValueError):
            depth_to_display_u8(np.zeros((2, 2), dtype=np.uint16), max_depth_mm=0)


if __name__ == "__main__":
    unittest.main()
