import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from delta_depth_dataset import DeltaDepthEpisodeWriter
from so101_depth_act import DepthACTPolicy, load_depth_act_checkpoint, predict_delta_chunk
from so101_depth import DepthConfig
from train_depth_act import build_record_splits, main, parse_args


def arrays(seed: int, frames: int = 4):
    rng = np.random.default_rng(seed)
    top = np.full((frames, 240, 320), 600, dtype=np.uint16)
    side = np.full((frames, 240, 320), 700, dtype=np.uint16)
    qpos = rng.normal(0, 0.01, (frames, 6)).astype(np.float32)
    qvel = np.zeros_like(qpos)
    goal = qpos + 0.01
    delta = goal - qpos
    action = np.zeros_like(qpos)
    phase = np.zeros(frames, dtype=np.uint8)
    timestamp = np.arange(frames, dtype=np.float64) * 0.034
    return top, side, qpos, qvel, goal, delta, action, action, phase, timestamp


class DepthACTPipelineTest(unittest.TestCase):
    def test_rollout_defaults_match_near_position_collection(self):
        args = parse_args(["--dataset", "dataset", "--output-dir", "run"])
        self.assertEqual(args.rollout_cube_jitter, 0.02)
        self.assertEqual(args.rollout_bin_jitter, 0.01)

    def test_one_epoch_training_saves_reloadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            writer = DeltaDepthEpisodeWriter(dataset)
            for seed in (1, 2):
                writer.save_episode(
                    *arrays(seed),
                    seed=seed,
                    initial_joint_pos=np.zeros(6),
                    cube_position=np.array([0.32, 0.155, 0.0336]),
                    bin_position=np.array([0.31, -0.05, 0.0]),
                )
            output = root / "run"
            main([
                "--dataset", str(dataset),
                "--output-dir", str(output),
                "--epochs", "1",
                "--batch-size", "2",
                "--chunk-size", "3",
                "--d-model", "32",
                "--nhead", "4",
                "--encoder-layers", "1",
                "--decoder-layers", "1",
                "--backbone-width", "4",
                "--num-workers", "0",
                "--rollout-eval-episodes", "0",
                "--device", "cpu",
            ])
            checkpoint = output / "last_checkpoint.pt"
            self.assertTrue(checkpoint.is_file())
            policy, payload = load_depth_act_checkpoint(str(checkpoint))
            self.assertEqual(payload["epoch"], 1)
            chunk = predict_delta_chunk(
                policy.eval(),
                np.full((240, 320), 0.6, dtype=np.float32),
                np.full((240, 320), 0.7, dtype=np.float32),
                np.zeros(6, dtype=np.float32),
                np.zeros(6, dtype=np.float32),
                depth_config=DepthConfig(),
                device=torch.device("cpu"),
            )
            self.assertEqual(chunk.shape, (3, 6))
            self.assertTrue(np.all(np.isfinite(chunk)))

    def test_additional_dataset_is_repeated_only_in_training_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "base"
            dagger = root / "dagger"
            metadata = {
                "initial_joint_pos": np.zeros(6),
                "cube_position": np.array([0.32, 0.155, 0.0336]),
                "bin_position": np.array([0.31, -0.05, 0.0]),
            }
            for dataset, seeds in ((base, (1, 2, 3, 4)), (dagger, (11, 12, 13, 14))):
                writer = DeltaDepthEpisodeWriter(dataset)
                for seed in seeds:
                    writer.save_episode(*arrays(seed), seed=seed, **metadata)

            train, validation, sources = build_record_splits(
                base,
                [dagger],
                additional_repeat=3,
                seed=31,
            )

            base_train = [record for record in train if record.root == base.resolve()]
            dagger_train = [record for record in train if record.root == dagger.resolve()]
            dagger_validation = [record for record in validation if record.root == dagger.resolve()]
            self.assertEqual(len(base_train), 3)
            self.assertEqual(len(dagger_train), 9)
            self.assertEqual(len(dagger_validation), 1)
            self.assertEqual(sources[1]["train_episodes"], 3)
            self.assertEqual(sources[1]["train_repeats"], 3)

    def test_finetune_loads_checkpoint_and_records_mixed_dataset_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "base"
            dagger = root / "dagger"
            metadata = {
                "initial_joint_pos": np.zeros(6),
                "cube_position": np.array([0.32, 0.155, 0.0336]),
                "bin_position": np.array([0.31, -0.05, 0.0]),
            }
            for dataset, seeds in ((base, (1, 2)), (dagger, (11, 12))):
                writer = DeltaDepthEpisodeWriter(dataset)
                for seed in seeds:
                    writer.save_episode(*arrays(seed), seed=seed, **metadata)

            policy = DepthACTPolicy(
                chunk_size=3,
                d_model=32,
                nhead=4,
                encoder_layers=1,
                decoder_layers=1,
                dim_feedforward=64,
                backbone_channels=(4, 8, 16, 32),
            )
            source_checkpoint = root / "source.pt"
            torch.save(
                {
                    "architecture_version": policy.architecture_version,
                    "architecture_config": policy.architecture_config(),
                    "model_state_dict": policy.state_dict(),
                    "depth_config": asdict(DepthConfig()),
                },
                source_checkpoint,
            )
            output = root / "finetune"
            main(
                [
                    "--dataset",
                    str(base),
                    "--additional-dataset",
                    str(dagger),
                    "--additional-repeat",
                    "2",
                    "--pretrained-checkpoint",
                    str(source_checkpoint),
                    "--output-dir",
                    str(output),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--num-workers",
                    "0",
                    "--rollout-eval-episodes",
                    "0",
                    "--device",
                    "cpu",
                ]
            )

            payload = torch.load(output / "last_checkpoint.pt", map_location="cpu", weights_only=False)
            self.assertEqual(payload["initialized_from"], str(source_checkpoint.resolve()))
            self.assertEqual(payload["dataset_roots"], [str(base.resolve()), str(dagger.resolve())])
            self.assertEqual(payload["architecture_config"]["chunk_size"], 3)


if __name__ == "__main__":
    unittest.main()
