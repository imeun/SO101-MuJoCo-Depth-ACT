from __future__ import annotations

import copy
from contextlib import contextmanager
import errno
import hashlib
import json
import numbers
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any
import uuid

import numpy as np

if os.name == "nt":
    import msvcrt
else:
    import fcntl


DATASET_VERSION = 2
VALID_SOURCES = frozenset({"teacher", "smoke", "dagger"})
_DEPTH_SHAPE = (240, 320)
_JOINT_DIM = 6
_ACTION_DIM = 6
_INT64_INFO = np.iinfo(np.int64)
_CANONICAL_ARCHIVE = re.compile(r"episode_\d{6,}\.npz")
_LOCK_RETRY_SECONDS = 0.05
_LEGACY_RUN_ID = "legacy-v1"
_MANUAL_RUN_ID = "manual"


def _empty_manifest() -> dict[str, Any]:
    return {
        "version": DATASET_VERSION,
        "depth_shape": list(_DEPTH_SHAPE),
        "joint_dim": _JOINT_DIM,
        "action_dim": _ACTION_DIM,
        "runs": {},
        "episodes": [],
    }


def _is_integer(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _validate_int64(value: object, name: str) -> int:
    if not _is_integer(value):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < _INT64_INFO.min or parsed > _INT64_INFO.max:
        raise ValueError(f"{name} must be within the signed int64 range")
    return parsed


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _run_id(provenance: dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(provenance).encode("ascii")).hexdigest()
    return f"run-{digest[:24]}"


def _validate_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_run_provenance(provenance: object) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise ValueError("run provenance must be an object")
    kind = provenance.get("kind")
    if kind == "manual":
        if provenance != {"kind": "manual"}:
            raise ValueError("manual run provenance is malformed")
        return {"kind": "manual"}
    if kind == "legacy-v1":
        if provenance != {"kind": "legacy-v1"}:
            raise ValueError("legacy run provenance is malformed")
        return {"kind": "legacy-v1"}
    required = {
        "teacher": {
            "kind", "teacher_checkpoint_digest", "stage", "requested_count", "seed_start", "max_steps", "mode",
        },
        "scripted_teacher": {
            "kind", "controller_version", "curriculum", "requested_count", "seed_start", "max_steps",
            "capture_stride", "cube_jitter", "yaw_range_degrees", "randomize_bin", "bin_jitter", "mode",
        },
        "dagger": {
            "kind", "teacher_checkpoint_digest", "student_checkpoint_digest", "teacher_execute_probability",
            "stage", "requested_count", "seed_start", "max_steps", "mode",
        },
    }
    if kind not in required or set(provenance) != required[kind]:
        raise ValueError("run provenance is malformed")
    mode = provenance["mode"]
    if kind == "teacher" and mode not in {"teacher", "smoke"}:
        raise ValueError("teacher run mode is invalid")
    if kind == "scripted_teacher" and mode != "teacher":
        raise ValueError("scripted teacher run mode is invalid")
    if kind == "dagger" and mode != "dagger":
        raise ValueError("DAgger run mode is invalid")
    requested_count = provenance["requested_count"]
    max_steps = provenance["max_steps"]
    if not _is_integer(requested_count) or requested_count <= 0:
        raise ValueError("run requested_count must be positive")
    if not _is_integer(max_steps) or max_steps <= 0:
        raise ValueError("run max_steps must be positive")
    normalized: dict[str, Any] = {
        "kind": kind,
        "requested_count": int(requested_count),
        "seed_start": _validate_int64(provenance["seed_start"], "run seed_start"),
        "max_steps": int(max_steps),
        "mode": mode,
    }
    if kind == "scripted_teacher":
        controller_version = provenance["controller_version"]
        curriculum = provenance["curriculum"]
        capture_stride = provenance["capture_stride"]
        if not isinstance(controller_version, str) or not controller_version:
            raise ValueError("scripted teacher controller_version is invalid")
        if curriculum not in {"fixed", "near", "wide"}:
            raise ValueError("scripted teacher curriculum is invalid")
        if not _is_integer(capture_stride) or capture_stride <= 0:
            raise ValueError("scripted teacher capture_stride must be positive")
        normalized.update(
            {
                "controller_version": controller_version,
                "curriculum": curriculum,
                "capture_stride": int(capture_stride),
            }
        )
        for name in ("cube_jitter", "yaw_range_degrees", "bin_jitter"):
            value = provenance[name]
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value) or value < 0.0:
                raise ValueError(f"scripted teacher {name} must be finite and non-negative")
            normalized[name] = float(value)
        if not isinstance(provenance["randomize_bin"], bool):
            raise ValueError("scripted teacher randomize_bin must be a bool")
        normalized["randomize_bin"] = provenance["randomize_bin"]
        return normalized

    stage = provenance["stage"]
    if stage not in {"stage1", "stage2"}:
        raise ValueError("run stage is invalid")
    normalized["teacher_checkpoint_digest"] = _validate_digest(
        provenance["teacher_checkpoint_digest"], "teacher checkpoint digest"
    )
    normalized["stage"] = stage
    if kind == "dagger":
        probability = provenance["teacher_execute_probability"]
        if isinstance(probability, bool) or not isinstance(probability, numbers.Real) or not 0.0 <= probability <= 1.0:
            raise ValueError("DAgger teacher_execute_probability is invalid")
        normalized["student_checkpoint_digest"] = _validate_digest(
            provenance["student_checkpoint_digest"], "student checkpoint digest"
        )
        normalized["teacher_execute_probability"] = float(probability)
    return normalized


def _physical_identity(path: Path) -> tuple[int, int]:
    try:
        details = os.stat(path)
    except OSError as error:
        raise ValueError(f"manifest episode file is missing: {path.name}") from error
    if not stat.S_ISREG(details.st_mode):
        raise ValueError(f"manifest episode file is missing: {path.name}")
    return int(details.st_dev), int(details.st_ino)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@contextmanager
def _dataset_lock(root: Path):
    lock_path = root / ".teacher_dataset.lock"
    if os.path.lexists(lock_path) and not lock_path.is_file():
        raise ValueError("dataset lock exists but is not a regular file")
    with lock_path.open("a+b") as handle:
        resolved_lock = lock_path.resolve()
        try:
            resolved_lock.relative_to(root)
        except ValueError as error:
            raise ValueError("dataset lock escapes the dataset root") from error
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())

        if os.name == "nt":
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    time.sleep(_LOCK_RETRY_SECONDS)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _quarantine_orphan_archives(root: Path, episodes_dir: Path, manifest: dict[str, Any]) -> None:
    referenced = {Path(entry["file"]).name for entry in manifest["episodes"]}
    orphans = sorted(
        path
        for path in episodes_dir.iterdir()
        if path.is_file() and _CANONICAL_ARCHIVE.fullmatch(path.name) and path.name not in referenced
    )
    if not orphans:
        return

    quarantine_path = root / "orphaned_episodes"
    quarantine_path.mkdir(parents=True, exist_ok=True)
    quarantine_dir = quarantine_path.resolve()
    try:
        quarantine_dir.relative_to(root)
    except ValueError as error:
        raise ValueError("orphan quarantine escapes the dataset root") from error

    for orphan in orphans:
        destination = quarantine_dir / f"{orphan.stem}.orphan-{uuid.uuid4().hex}{orphan.suffix}"
        os.link(orphan, destination)
        try:
            orphan.unlink()
        except BaseException:
            destination.unlink(missing_ok=True)
            raise


def _episode_path(root: Path, file_name: str) -> Path:
    candidate = (root / file_name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("manifest episode file escapes the dataset root") from error
    return candidate


def _validate_manifest(manifest: object, root: Path) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if not _is_integer(manifest.get("version")):
        raise ValueError("unsupported dataset version")
    if manifest["version"] == 1:
        manifest = copy.deepcopy(manifest)
        manifest["version"] = DATASET_VERSION
        manifest["runs"] = {_LEGACY_RUN_ID: {"kind": "legacy-v1"}}
        legacy_episodes = manifest.get("episodes")
        if isinstance(legacy_episodes, list):
            for entry in legacy_episodes:
                if isinstance(entry, dict):
                    entry["run_id"] = _LEGACY_RUN_ID
                    entry["episode_key"] = f"{_LEGACY_RUN_ID}:{entry.get('index')}"
    expected = _empty_manifest()
    for key in ("version", "depth_shape", "joint_dim", "action_dim", "runs", "episodes"):
        if key not in manifest:
            raise ValueError(f"manifest is missing {key}")
    if not _is_integer(manifest["version"]) or manifest["version"] != expected["version"]:
        raise ValueError("unsupported dataset version")
    if (
        not isinstance(manifest["depth_shape"], list)
        or len(manifest["depth_shape"]) != len(_DEPTH_SHAPE)
        or any(not _is_integer(value) for value in manifest["depth_shape"])
        or manifest["depth_shape"] != expected["depth_shape"]
    ):
        raise ValueError("manifest depth_shape is incompatible")
    if (
        not _is_integer(manifest["joint_dim"])
        or not _is_integer(manifest["action_dim"])
        or manifest["joint_dim"] != expected["joint_dim"]
        or manifest["action_dim"] != expected["action_dim"]
    ):
        raise ValueError("manifest dimensions are incompatible")
    episodes = manifest["episodes"]
    if not isinstance(episodes, list):
        raise ValueError("manifest episodes must be a list")
    runs = manifest["runs"]
    if not isinstance(runs, dict):
        raise ValueError("manifest runs must be an object")
    validated_runs: dict[str, dict[str, Any]] = {}
    for run_id, provenance in runs.items():
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("manifest run ID is invalid")
        normalized_provenance = _validate_run_provenance(provenance)
        validated_runs[run_id] = normalized_provenance

    indices: set[int] = set()
    files: set[str] = set()
    physical_files: set[tuple[int, int]] = set()
    episode_keys: set[str] = set()
    run_seeds: set[tuple[str, int]] = set()
    validated_entries: list[dict[str, Any]] = []
    required = {"index", "file", "frames", "seed", "success", "source", "run_id", "episode_key"}
    for entry in episodes:
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError("manifest episode entry is malformed")
        index = entry["index"]
        file_name = entry["file"]
        frames = entry["frames"]
        seed = entry["seed"]
        success = entry["success"]
        source = entry["source"]
        run_id = entry["run_id"]
        episode_key = entry["episode_key"]
        if not _is_integer(index) or index < 0:
            raise ValueError("manifest episode index must be a nonnegative integer")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError("manifest episode file must be a nonempty string")
        canonical_file = f"episodes/episode_{index:06d}.npz"
        if file_name != canonical_file:
            raise ValueError(f"manifest episode file must use canonical path {canonical_file}")
        if not _is_integer(frames) or frames <= 0:
            raise ValueError("manifest episode frames must be a positive integer")
        seed = _validate_int64(seed, "manifest episode seed")
        if not isinstance(success, bool) or not isinstance(source, str) or source not in VALID_SOURCES:
            raise ValueError("manifest episode success or source is invalid")
        if source == "teacher" and not success:
            raise ValueError("teacher manifest episodes must be successful")
        if not isinstance(run_id, str) or run_id not in validated_runs:
            raise ValueError("manifest episode run ID is invalid")
        if not isinstance(episode_key, str) or not episode_key:
            raise ValueError("manifest episode key is invalid")
        if index in indices:
            raise ValueError("manifest contains duplicate episode indices or files")
        episode_file = _episode_path(root, file_name)
        normalized_file = os.path.normcase(str(episode_file))
        if normalized_file in files:
            raise ValueError("manifest contains duplicate episode indices or files")
        physical_identity = _physical_identity(episode_file)
        if physical_identity in physical_files:
            raise ValueError("manifest contains duplicate physical episode files")
        if episode_key in episode_keys:
            raise ValueError("manifest contains duplicate episode keys")
        if (run_id, seed) in run_seeds:
            raise ValueError("manifest contains duplicate run seeds")
        indices.add(int(index))
        files.add(normalized_file)
        physical_files.add(physical_identity)
        episode_keys.add(episode_key)
        run_seeds.add((run_id, seed))
        validated_entries.append(
            {
                "index": int(index),
                "file": file_name,
                "frames": int(frames),
                "seed": int(seed),
                "success": success,
                "source": source,
                "run_id": run_id,
                "episode_key": episode_key,
            }
        )
    return {
        "version": DATASET_VERSION,
        "depth_shape": list(_DEPTH_SHAPE),
        "joint_dim": _JOINT_DIM,
        "action_dim": _ACTION_DIM,
        "runs": validated_runs,
        "episodes": validated_entries,
    }


def load_manifest(root: str | Path) -> dict:
    """Load an independently mutable validated dataset manifest."""
    dataset_root = Path(root).resolve()
    manifest_path = dataset_root / "manifest.json"
    if not os.path.lexists(manifest_path):
        return _empty_manifest()
    if not manifest_path.is_file():
        raise ValueError("manifest.json exists but is not a regular file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("manifest cannot be read as JSON") from error
    return copy.deepcopy(_validate_manifest(manifest, dataset_root))


class TeacherEpisodeWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        episodes_path = self.root / "episodes"
        episodes_path.mkdir(parents=True, exist_ok=True)
        self.episodes_dir = episodes_path.resolve()
        try:
            self.episodes_dir.relative_to(self.root)
        except ValueError as error:
            raise ValueError("episodes directory escapes the dataset root") from error
        self.manifest_path = self.root / "manifest.json"
        with _dataset_lock(self.root):
            if not os.path.lexists(self.manifest_path):
                _atomic_write_json(self.manifest_path, _empty_manifest())
            self._manifest = load_manifest(self.root)
            raw_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if raw_manifest.get("version") != DATASET_VERSION:
                _atomic_write_json(self.manifest_path, self._manifest)
            _quarantine_orphan_archives(self.root, self.episodes_dir, self._manifest)

    def register_run(self, provenance: dict[str, Any], *, run_id: str | None = None) -> str:
        """Register an immutable collection definition and return its stable identifier."""
        normalized_provenance = _validate_run_provenance(provenance)
        if normalized_provenance["kind"] in {"manual", "legacy-v1"}:
            raise ValueError("manual and legacy run provenance cannot be registered")
        if run_id is None:
            run_id = _run_id(normalized_provenance)
        elif not isinstance(run_id, str) or not run_id or run_id in {_MANUAL_RUN_ID, _LEGACY_RUN_ID}:
            raise ValueError("run_id is invalid")
        with _dataset_lock(self.root):
            self._manifest = load_manifest(self.root)
            _quarantine_orphan_archives(self.root, self.episodes_dir, self._manifest)
            existing = self._manifest["runs"].get(run_id)
            if existing is not None:
                if existing != normalized_provenance:
                    raise ValueError("conflicting run definition")
                return run_id
            updated_manifest = copy.deepcopy(self._manifest)
            updated_manifest["runs"][run_id] = normalized_provenance
            _atomic_write_json(self.manifest_path, updated_manifest)
            self._manifest = updated_manifest
        return run_id

    @staticmethod
    def _validate_episode(
        depth_mm: np.ndarray,
        joint_pos: np.ndarray,
        action: np.ndarray,
        *,
        seed: int,
        success: bool,
        source: str,
    ) -> int:
        if not isinstance(depth_mm, np.ndarray) or depth_mm.dtype != np.uint16 or depth_mm.ndim != 3:
            raise ValueError("depth_mm must be a uint16 array with shape (T, 240, 320)")
        if depth_mm.shape[1:] != _DEPTH_SHAPE:
            raise ValueError("depth_mm must have shape (T, 240, 320)")
        if not isinstance(joint_pos, np.ndarray) or joint_pos.dtype != np.float32 or joint_pos.ndim != 2:
            raise ValueError("joint_pos must be a float32 array with shape (T, 6)")
        if joint_pos.shape[1:] != (_JOINT_DIM,):
            raise ValueError("joint_pos must have shape (T, 6)")
        if not isinstance(action, np.ndarray) or action.dtype != np.float32 or action.ndim != 2:
            raise ValueError("action must be a float32 array with shape (T, 6)")
        if action.shape[1:] != (_ACTION_DIM,):
            raise ValueError("action must have shape (T, 6)")
        frames = depth_mm.shape[0]
        if frames <= 0 or joint_pos.shape[0] != frames or action.shape[0] != frames:
            raise ValueError("episode arrays must have the same positive frame count")
        if not np.all(np.isfinite(joint_pos)) or not np.all(np.isfinite(action)):
            raise ValueError("joint_pos and action must be finite")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError("action values must be within [-1, 1]")
        _validate_int64(seed, "seed")
        if not isinstance(success, bool):
            raise ValueError("success must be a bool")
        if not isinstance(source, str) or source not in VALID_SOURCES:
            raise ValueError("source is invalid")
        if source == "teacher" and not success:
            raise ValueError("teacher episodes must be successful")
        return int(frames)

    def _write_archive(self, path: Path, payload: dict[str, np.ndarray]) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b", dir=self.episodes_dir, prefix=f".{path.stem}.", suffix=".npz", delete=False
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, **payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, path)
            try:
                temporary.unlink()
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def save_episode(
        self,
        depth_mm: np.ndarray,
        joint_pos: np.ndarray,
        action: np.ndarray,
        *,
        seed: int,
        success: bool,
        source: str = "teacher",
        run_id: str | None = None,
        episode_key: str | None = None,
    ) -> Path:
        frames = self._validate_episode(depth_mm, joint_pos, action, seed=seed, success=success, source=source)
        if (run_id is None) != (episode_key is None):
            raise ValueError("run_id and episode_key must be supplied together")
        if run_id is not None and (not isinstance(run_id, str) or not run_id):
            raise ValueError("run_id must be a nonempty string")
        if episode_key is not None and (not isinstance(episode_key, str) or not episode_key):
            raise ValueError("episode_key must be a nonempty string")
        with _dataset_lock(self.root):
            self._manifest = load_manifest(self.root)
            _quarantine_orphan_archives(self.root, self.episodes_dir, self._manifest)
            next_index = 0 if not self._manifest["episodes"] else max(entry["index"] for entry in self._manifest["episodes"]) + 1
            if run_id is None:
                run_id = _MANUAL_RUN_ID
                episode_key = f"{_MANUAL_RUN_ID}:{next_index}"
                if run_id not in self._manifest["runs"]:
                    self._manifest = copy.deepcopy(self._manifest)
                    self._manifest["runs"][run_id] = {"kind": "manual"}
            if run_id not in self._manifest["runs"]:
                raise ValueError("episode references an unregistered run")
            if any(entry["episode_key"] == episode_key for entry in self._manifest["episodes"]):
                raise ValueError("duplicate episode key")
            if any(entry["run_id"] == run_id and entry["seed"] == int(seed) for entry in self._manifest["episodes"]):
                raise ValueError("duplicate run seed")
            relative_file = f"episodes/episode_{next_index:06d}.npz"
            episode_path = self.episodes_dir / f"episode_{next_index:06d}.npz"
            entry = {
                "index": next_index,
                "file": relative_file,
                "frames": frames,
                "seed": int(seed),
                "success": success,
                "source": source,
                "run_id": run_id,
                "episode_key": episode_key,
            }
            payload = {
                "depth_mm": depth_mm,
                "joint_pos": joint_pos,
                "action": action,
                "seed": np.asarray(seed, dtype=np.int64),
                "success": np.asarray(success, dtype=np.bool_),
                "source": np.asarray(source),
            }
            self._write_archive(episode_path, payload)
            updated_manifest = copy.deepcopy(self._manifest)
            updated_manifest["episodes"].append(entry)
            try:
                _atomic_write_json(self.manifest_path, updated_manifest)
            except BaseException:
                episode_path.unlink(missing_ok=True)
                raise
            self._manifest = updated_manifest
            return episode_path


def training_episode_files(root: str | Path) -> list[Path]:
    dataset_root = Path(root).resolve()
    manifest = load_manifest(dataset_root)
    return [
        _episode_path(dataset_root, entry["file"])
        for entry in manifest["episodes"]
        if entry["source"] in {"teacher", "dagger"}
    ]
