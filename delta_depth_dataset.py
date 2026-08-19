from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


DEPTH_SHAPE = (240, 320)
VECTOR_DIM = 6
CONTROL_HZ = 1.0 / (0.002 * 17)
PHASE_NAMES = (
    "open",
    "pregrasp",
    "grasp",
    "close",
    "hold",
    "lift",
    "bin",
    "release",
    "settle",
    "retreat",
    "home",
    "home_hold",
)
PHASE_TO_ID = {name: index for index, name in enumerate(PHASE_NAMES)}


DELTA_DATASET_VERSION = 2


INITIAL_CONDITION_MODES = ("fixed", "near_position")


def _empty_manifest(
    provenance: dict[str, Any] | None = None,
    initial_conditions: str = "fixed",
) -> dict[str, Any]:
    return {
        "version": DELTA_DATASET_VERSION,
        "depth_shape": list(DEPTH_SHAPE),
        "joint_dim": VECTOR_DIM,
        "control_hz": CONTROL_HZ,
        "target_type": "immediate_joint_delta",
        "initial_conditions": initial_conditions,
        "phase_names": list(PHASE_NAMES),
        "provenance": provenance,
        "episodes": [],
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _finite_vector(value: np.ndarray, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return array


def trajectory_digest(
    joint_pos: np.ndarray,
    delta_target_rad: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for value in (joint_pos, delta_target_rad):
        quantized = np.round(np.asarray(value, dtype=np.float64), decimals=5).astype("<f4")
        digest.update(np.asarray(quantized.shape, dtype="<i8").tobytes())
        digest.update(quantized.tobytes(order="C"))
    return digest.hexdigest()


def load_delta_manifest(root: str | Path) -> dict[str, Any]:
    dataset_root = Path(root).resolve()
    path = dataset_root / "manifest.json"
    if not path.is_file():
        return _empty_manifest()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("delta-depth manifest cannot be read") from error
    expected = {
        "version",
        "depth_shape",
        "joint_dim",
        "control_hz",
        "target_type",
        "initial_conditions",
        "phase_names",
        "provenance",
        "episodes",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise ValueError("delta-depth manifest is malformed")
    if (
        manifest["version"] != DELTA_DATASET_VERSION
        or manifest["depth_shape"] != list(DEPTH_SHAPE)
        or manifest["joint_dim"] != VECTOR_DIM
        or not np.isclose(manifest["control_hz"], CONTROL_HZ)
        or manifest["target_type"] != "immediate_joint_delta"
        or manifest["initial_conditions"] not in INITIAL_CONDITION_MODES
        or manifest["phase_names"] != list(PHASE_NAMES)
        or (manifest["provenance"] is not None and not isinstance(manifest["provenance"], dict))
        or not isinstance(manifest["episodes"], list)
    ):
        raise ValueError("delta-depth manifest schema is incompatible")
    required = {
        "index",
        "file",
        "frames",
        "seed",
        "trajectory_sha256",
        "initial_joint_pos",
        "cube_position",
        "bin_position",
    }
    digests: set[str] = set()
    validated: list[dict[str, Any]] = []
    for entry in manifest["episodes"]:
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError("delta-depth episode metadata is malformed")
        index = entry["index"]
        canonical = f"episodes/episode_{index:06d}.npz" if isinstance(index, int) else ""
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or entry["file"] != canonical
            or isinstance(entry["frames"], bool)
            or not isinstance(entry["frames"], int)
            or entry["frames"] <= 0
            or isinstance(entry["seed"], bool)
            or not isinstance(entry["seed"], int)
            or not isinstance(entry["trajectory_sha256"], str)
            or len(entry["trajectory_sha256"]) != 64
        ):
            raise ValueError("delta-depth episode metadata is invalid")
        _finite_vector(entry["initial_joint_pos"], "initial_joint_pos", VECTOR_DIM)
        _finite_vector(entry["cube_position"], "cube_position", 3)
        _finite_vector(entry["bin_position"], "bin_position", 3)
        if entry["trajectory_sha256"] in digests:
            raise ValueError("delta-depth manifest contains a duplicate trajectory")
        episode_path = (dataset_root / entry["file"]).resolve()
        try:
            episode_path.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError("delta-depth episode path escapes the dataset") from error
        if not episode_path.is_file():
            raise ValueError(f"delta-depth episode file is missing: {entry['file']}")
        digests.add(entry["trajectory_sha256"])
        validated.append(dict(entry))
    result = dict(manifest)
    result["episodes"] = validated
    return result


class DeltaDepthEpisodeWriter:
    def __init__(
        self,
        root: str | Path,
        *,
        provenance: dict[str, Any] | None = None,
        initial_conditions: str = "fixed",
    ):
        if initial_conditions not in INITIAL_CONDITION_MODES:
            raise ValueError(f"initial_conditions must be one of {INITIAL_CONDITION_MODES}")
        self.root = Path(root).resolve()
        self.episodes_dir = self.root / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.exists():
            atomic_json(self.manifest_path, _empty_manifest(provenance, initial_conditions))
        self.manifest = load_delta_manifest(self.root)
        if self.manifest["initial_conditions"] != initial_conditions:
            raise ValueError("dataset initial_conditions mode does not match the requested mode")
        if provenance is not None and self.manifest["provenance"] != provenance:
            raise ValueError("dataset provenance does not match the current scene and cameras")

    @staticmethod
    def _validate_arrays(arrays: tuple[np.ndarray, ...]) -> int:
        (
            top_depth_mm,
            side_depth_mm,
            joint_pos,
            joint_velocity,
            teacher_goal_pos,
            delta_target_rad,
            teacher_action,
            executed_action,
            phase_id,
            timestamp_s,
        ) = arrays
        for name, value in (("top_depth_mm", top_depth_mm), ("side_depth_mm", side_depth_mm)):
            if not isinstance(value, np.ndarray) or value.dtype != np.uint16 or value.ndim != 3:
                raise ValueError(f"{name} must be uint16 with shape (T, 240, 320)")
            if value.shape[1:] != DEPTH_SHAPE:
                raise ValueError(f"{name} must have shape (T, 240, 320)")
        frames = top_depth_mm.shape[0]
        if frames <= 0 or side_depth_mm.shape[0] != frames:
            raise ValueError("depth streams must have the same positive frame count")
        for name, value in (
            ("joint_pos", joint_pos),
            ("joint_velocity", joint_velocity),
            ("teacher_goal_pos", teacher_goal_pos),
            ("delta_target_rad", delta_target_rad),
            ("teacher_action", teacher_action),
            ("executed_action", executed_action),
        ):
            if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != (frames, VECTOR_DIM):
                raise ValueError(f"{name} must be finite float32 with shape (T, 6)")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
        if not np.allclose(delta_target_rad, teacher_goal_pos - joint_pos, rtol=1e-5, atol=1e-6):
            raise ValueError("delta_target_rad must equal teacher_goal_pos - joint_pos")
        if np.any(np.abs(teacher_action) > 1.0) or np.any(np.abs(executed_action) > 1.0):
            raise ValueError("teacher_action and executed_action must stay within [-1, 1]")
        if (
            not isinstance(phase_id, np.ndarray)
            or phase_id.dtype != np.uint8
            or phase_id.shape != (frames,)
            or np.any(phase_id >= len(PHASE_NAMES))
        ):
            raise ValueError("phase_id must be valid uint8 with shape (T,)")
        if (
            not isinstance(timestamp_s, np.ndarray)
            or timestamp_s.dtype != np.float64
            or timestamp_s.shape != (frames,)
            or not np.all(np.isfinite(timestamp_s))
            or timestamp_s[0] < 0.0
            or (frames > 1 and not np.all(np.diff(timestamp_s) > 0.0))
        ):
            raise ValueError("timestamp_s must be strictly increasing float64 with shape (T,)")
        return frames

    def save_episode(
        self,
        top_depth_mm: np.ndarray,
        side_depth_mm: np.ndarray,
        joint_pos: np.ndarray,
        joint_velocity: np.ndarray,
        teacher_goal_pos: np.ndarray,
        delta_target_rad: np.ndarray,
        teacher_action: np.ndarray,
        executed_action: np.ndarray,
        phase_id: np.ndarray,
        timestamp_s: np.ndarray,
        *,
        seed: int,
        initial_joint_pos: np.ndarray,
        cube_position: np.ndarray,
        bin_position: np.ndarray,
    ) -> Path:
        arrays = (
            top_depth_mm,
            side_depth_mm,
            joint_pos,
            joint_velocity,
            teacher_goal_pos,
            delta_target_rad,
            teacher_action,
            executed_action,
            phase_id,
            timestamp_s,
        )
        frames = self._validate_arrays(arrays)
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        initial = _finite_vector(initial_joint_pos, "initial_joint_pos", VECTOR_DIM)
        cube = _finite_vector(cube_position, "cube_position", 3)
        bin_position = _finite_vector(bin_position, "bin_position", 3)
        self.manifest = load_delta_manifest(self.root)
        if self.manifest["episodes"]:
            reference = self.manifest["episodes"][0]
            compared = [("initial_joint_pos", initial)]
            if self.manifest["initial_conditions"] == "fixed":
                compared.extend((("cube_position", cube), ("bin_position", bin_position)))
            for name, value in compared:
                if not np.allclose(value, np.asarray(reference[name]), rtol=0.0, atol=1e-7):
                    if name == "initial_joint_pos" and self.manifest["initial_conditions"] == "near_position":
                        raise ValueError("episode changes the fixed robot home pose")
                    raise ValueError("episode violates fixed initial conditions")
        digest = trajectory_digest(joint_pos, delta_target_rad)
        if any(entry["trajectory_sha256"] == digest for entry in self.manifest["episodes"]):
            raise ValueError("duplicate trajectory")
        index = 0 if not self.manifest["episodes"] else max(entry["index"] for entry in self.manifest["episodes"]) + 1
        relative = f"episodes/episode_{index:06d}.npz"
        destination = self.root / relative
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=self.episodes_dir,
                prefix=f".episode_{index:06d}.",
                suffix=".npz",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(
                    handle,
                    top_depth_mm=top_depth_mm,
                    side_depth_mm=side_depth_mm,
                    joint_pos=joint_pos,
                    joint_velocity=joint_velocity,
                    teacher_goal_pos=teacher_goal_pos,
                    delta_target_rad=delta_target_rad,
                    teacher_action=teacher_action,
                    executed_action=executed_action,
                    phase_id=phase_id,
                    timestamp_s=timestamp_s,
                    seed=np.asarray(seed, dtype=np.int64),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            temporary = None
            entry = {
                "index": index,
                "file": relative,
                "frames": frames,
                "seed": int(seed),
                "trajectory_sha256": digest,
                "initial_joint_pos": initial.tolist(),
                "cube_position": cube.tolist(),
                "bin_position": bin_position.tolist(),
            }
            updated = dict(self.manifest)
            updated["episodes"] = list(self.manifest["episodes"]) + [entry]
            atomic_json(self.manifest_path, updated)
            self.manifest = updated
            return destination
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
