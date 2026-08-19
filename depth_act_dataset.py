from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from delta_depth_dataset import load_delta_manifest
from so101_depth import DepthConfig, millimetres_to_depth, preprocess_depth


@dataclass(frozen=True)
class DeltaEpisodeRecord:
    root: Path
    index: int
    path: Path
    frames: int
    seed: int


def delta_episode_records(root: str | Path) -> list[DeltaEpisodeRecord]:
    dataset_root = Path(root).resolve()
    manifest = load_delta_manifest(dataset_root)
    return [
        DeltaEpisodeRecord(
            root=dataset_root,
            index=int(entry["index"]),
            path=dataset_root / entry["file"],
            frames=int(entry["frames"]),
            seed=int(entry["seed"]),
        )
        for entry in manifest["episodes"]
    ]


def split_delta_records(
    records: Sequence[DeltaEpisodeRecord],
    *,
    validation_fraction: float = 0.1,
    seed: int = 31,
) -> tuple[list[DeltaEpisodeRecord], list[DeltaEpisodeRecord]]:
    if len(records) < 2:
        raise ValueError("at least two episodes are required")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be within (0, 1)")
    indices = np.random.default_rng(seed).permutation(len(records))
    validation_count = max(1, int(round(len(records) * validation_fraction)))
    validation_indices = set(indices[:validation_count].tolist())
    train = [record for index, record in enumerate(records) if index not in validation_indices]
    validation = [record for index, record in enumerate(records) if index in validation_indices]
    return train, validation


class EpisodeBatchSampler(Sampler[list[int]]):
    """Yield deterministic batches without mixing frames from different episodes."""

    def __init__(
        self,
        episode_offsets: Sequence[int],
        *,
        batch_size: int,
        shuffle_episodes: bool,
        shuffle_frames: bool,
        seed: int,
    ):
        offsets = tuple(episode_offsets)
        if len(offsets) < 2 or offsets[0] != 0 or any(
            not isinstance(offset, (int, np.integer)) for offset in offsets
        ):
            raise ValueError("episode_offsets must start at zero and contain at least one episode")
        if any(end <= start for start, end in zip(offsets, offsets[1:])):
            raise ValueError("episode_offsets must delimit positive-length episodes")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self._episode_offsets = tuple(int(offset) for offset in offsets)
        self._batch_size = batch_size
        self._shuffle_episodes = bool(shuffle_episodes)
        self._shuffle_frames = bool(shuffle_frames)
        self._seed = int(seed)

    def __iter__(self) -> Iterator[list[int]]:
        episode_positions = list(range(len(self._episode_offsets) - 1))
        rng = np.random.default_rng(self._seed)
        if self._shuffle_episodes:
            rng.shuffle(episode_positions)
        for episode_position in episode_positions:
            start = self._episode_offsets[episode_position]
            stop = self._episode_offsets[episode_position + 1]
            frame_indices = list(range(start, stop))
            if self._shuffle_frames:
                rng.shuffle(frame_indices)
            for batch_start in range(0, len(frame_indices), self._batch_size):
                yield frame_indices[batch_start : batch_start + self._batch_size]

    def __len__(self) -> int:
        return sum(
            (stop - start + self._batch_size - 1) // self._batch_size
            for start, stop in zip(self._episode_offsets, self._episode_offsets[1:])
        )


class DepthACTEpisodeDataset(Dataset):
    """Lazy frame dataset that constructs cumulative future-delta action chunks."""

    def __init__(
        self,
        records: Sequence[DeltaEpisodeRecord],
        *,
        chunk_size: int = 30,
        depth_config: DepthConfig = DepthConfig(),
        training: bool = False,
        augmentation_seed: int = 31,
    ):
        if not records:
            raise ValueError("records must not be empty")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not isinstance(depth_config, DepthConfig):
            raise ValueError("depth_config must be a DepthConfig")
        self.records = list(records)
        self.chunk_size = int(chunk_size)
        self.depth_config = depth_config
        self.training = bool(training)
        self.augmentation_seed = int(augmentation_seed)
        self.epoch = 0
        self._ends = np.cumsum([record.frames for record in self.records]).tolist()
        self._cached_path: Path | None = None
        self._cached: dict[str, np.ndarray] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._ends[-1]

    @property
    def episode_offsets(self) -> tuple[int, ...]:
        return (0, *self._ends)

    def _location(self, index: int) -> tuple[DeltaEpisodeRecord, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = bisect_right(self._ends, index)
        start = 0 if episode_index == 0 else self._ends[episode_index - 1]
        return self.records[episode_index], index - start

    def _load(self, path: Path) -> dict[str, np.ndarray]:
        if path != self._cached_path:
            with np.load(path, allow_pickle=False) as archive:
                self._cached = {name: archive[name] for name in archive.files}
            self._cached_path = path
        assert self._cached is not None
        return self._cached

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record, frame = self._location(index)
        arrays = self._load(record.path)
        joint_pos = arrays["joint_pos"][frame].astype(np.float32, copy=True)
        joint_velocity = arrays["joint_velocity"][frame].astype(np.float32, copy=True)
        proprio = np.concatenate([joint_pos, joint_velocity]).astype(np.float32)

        chunk = np.zeros((self.chunk_size, 6), dtype=np.float32)
        mask = np.zeros(self.chunk_size, dtype=np.bool_)
        available = min(self.chunk_size, record.frames - frame)
        goals = arrays["teacher_goal_pos"][frame : frame + available]
        chunk[:available] = goals - joint_pos[None, :]
        mask[:available] = True

        def normalized_depth(name: str) -> np.ndarray:
            depth = millimetres_to_depth(arrays[name][frame])
            stream = 0 if name == "top_depth_mm" else 1
            rng = np.random.default_rng(self.augmentation_seed + self.epoch * 1_000_003 + index * 2 + stream)
            return preprocess_depth(depth, self.depth_config, rng=rng, augment=self.training)

        return {
            "top_depth": torch.from_numpy(normalized_depth("top_depth_mm")),
            "side_depth": torch.from_numpy(normalized_depth("side_depth_mm")),
            "proprio": torch.from_numpy(proprio),
            "joint_pos": torch.from_numpy(joint_pos),
            "delta_chunk": torch.from_numpy(chunk),
            "chunk_mask": torch.from_numpy(mask),
            "phase_id": torch.tensor(int(arrays["phase_id"][frame]), dtype=torch.long),
        }
