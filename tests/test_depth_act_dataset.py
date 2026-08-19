import tempfile
import unittest
from pathlib import Path

import numpy as np

from delta_depth_dataset import DeltaDepthEpisodeWriter
from depth_act_dataset import (
    DepthACTEpisodeDataset,
    EpisodeBatchSampler,
    delta_episode_records,
    split_delta_records,
)


def episode_arrays(frames: int = 4):
    top = np.full((frames, 240, 320), 600, dtype=np.uint16)
    side = np.full((frames, 240, 320), 700, dtype=np.uint16)
    joint_pos = np.arange(frames, dtype=np.float32)[:, None] * np.ones((1, 6), dtype=np.float32)
    velocity = np.ones((frames, 6), dtype=np.float32)
    teacher_goal = joint_pos + np.arange(1, frames + 1, dtype=np.float32)[:, None] * 0.1
    delta = teacher_goal - joint_pos
    action = np.zeros((frames, 6), dtype=np.float32)
    phase = np.arange(frames, dtype=np.uint8)
    timestamp = np.arange(frames, dtype=np.float64) * 0.034
    return top, side, joint_pos, velocity, teacher_goal, delta, action, action, phase, timestamp


PROVENANCE = {
    "scene_xml_sha256": "a" * 64,
    "cameras": {
        "top": {"resolution": [640, 480], "focalpixel": [382.0, 382.0]},
        "side_depth": {"resolution": [640, 480], "focalpixel": [385.0, 385.0]},
    },
}


class DepthACTDatasetTest(unittest.TestCase):
    def _save(self, root: Path, seed: int, frames: int = 4):
        writer = DeltaDepthEpisodeWriter(root, provenance=PROVENANCE)
        arrays = list(episode_arrays(frames))
        arrays[2] = arrays[2].copy()
        arrays[2][1:, 0] += np.float32(seed * 1e-4)
        arrays[4] = arrays[4].copy()
        arrays[4][1:, 0] += np.float32(seed * 1e-4)
        arrays[5] = arrays[4] - arrays[2]
        writer.save_episode(
            *arrays,
            seed=seed,
            initial_joint_pos=np.zeros(6),
            cube_position=np.array([0.32, 0.155, 0.0336]),
            bin_position=np.array([0.31, -0.05, 0.0]),
        )

    def test_chunk_starts_at_current_teacher_goal_and_pads_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            self._save(root, seed=0)
            dataset = DepthACTEpisodeDataset(delta_episode_records(root), chunk_size=3)
            self.assertEqual(dataset.episode_offsets, (0, 4))

            first = dataset[0]
            expected = np.stack([
                np.full(6, 0.1, dtype=np.float32),
                np.full(6, 1.2, dtype=np.float32),
                np.full(6, 2.3, dtype=np.float32),
            ])
            np.testing.assert_allclose(first["delta_chunk"].numpy(), expected)
            np.testing.assert_array_equal(first["chunk_mask"].numpy(), [True, True, True])

            last = dataset[3]
            np.testing.assert_allclose(last["delta_chunk"].numpy()[0], np.full(6, 0.4), atol=1e-6)
            np.testing.assert_allclose(last["delta_chunk"].numpy()[1:], 0.0)
            np.testing.assert_array_equal(last["chunk_mask"].numpy(), [True, False, False])

    def test_split_is_by_episode_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            for seed in range(10, 14):
                self._save(root, seed=seed)
            records = delta_episode_records(root)
            train_a, validation_a = split_delta_records(records, validation_fraction=0.25, seed=31)
            train_b, validation_b = split_delta_records(records, validation_fraction=0.25, seed=31)
            self.assertEqual([r.index for r in train_a], [r.index for r in train_b])
            self.assertEqual([r.index for r in validation_a], [r.index for r in validation_b])
            self.assertEqual(len(train_a), 3)
            self.assertEqual(len(validation_a), 1)

    def test_writer_rejects_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            self._save(root, seed=7)
            changed = dict(PROVENANCE)
            changed["scene_xml_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "provenance"):
                DeltaDepthEpisodeWriter(root, provenance=changed)

    def test_episode_batch_sampler_never_mixes_episodes(self):
        sampler = EpisodeBatchSampler(
            (0, 3, 8),
            batch_size=2,
            shuffle_episodes=False,
            shuffle_frames=False,
            seed=31,
        )
        self.assertEqual(list(sampler), [[0, 1], [2], [3, 4], [5, 6], [7]])
        self.assertEqual(len(sampler), 5)


if __name__ == "__main__":
    unittest.main()
