from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import mujoco
import numpy as np

from play_waypoint_teacher import (
    BASE_BIN_POSITION,
    execute_waypoint_episode,
    randomize_bin_position,
    randomize_cube_pose,
)
from so101_ball_bins_env import SO101BallBinsEnv
from so101_depth import TopDepthRenderer, depth_to_millimetres
from teacher_dataset import TeacherEpisodeWriter, load_manifest


CONTROLLER_VERSION = "waypoint-v3-full-cycle"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect successful top-depth demonstrations from the waypoint teacher."
    )
    parser.add_argument("--episodes", type=_positive_int, default=1_000)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--curriculum", choices=["fixed", "near", "wide"], default="near")
    parser.add_argument("--cube-jitter", type=_nonnegative_float, default=0.02)
    parser.add_argument("--yaw-range-deg", type=_nonnegative_float, default=0.0)
    parser.add_argument("--randomize-bin", action="store_true")
    parser.add_argument("--bin-jitter", type=_nonnegative_float, default=0.01)
    parser.add_argument("--capture-stride", type=_positive_int, default=3)
    parser.add_argument("--max-steps", type=_positive_int, default=1100)
    parser.add_argument("--seed", type=int, default=2_000)
    parser.add_argument("--max-attempts", type=_positive_int)
    args = parser.parse_args(argv)
    if args.max_attempts is not None and args.max_attempts < args.episodes:
        parser.error("--max-attempts must be at least --episodes")
    return args


def _write_json_atomic(path: Path, payload: dict) -> None:
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


def _load_failures(path: Path, run_id: str) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("failed-attempt log cannot be read") from error
    if not isinstance(payload, dict) or payload.get("run_id") != run_id or not isinstance(payload.get("attempts"), list):
        raise ValueError("failed-attempt log conflicts with this run")
    records = payload["attempts"]
    if any(not isinstance(item, dict) or not isinstance(item.get("seed"), int) for item in records):
        raise ValueError("failed-attempt log is malformed")
    if len({item["seed"] for item in records}) != len(records):
        raise ValueError("failed-attempt log contains duplicate seeds")
    return records


def _failure_record(seed: int, info: dict | None, reason: str) -> dict:
    info = info or {}
    return {
        "seed": seed,
        "reason": reason,
        "phase": str(info.get("phase", "setup")),
        "has_grasped": bool(info.get("has_grasped", False)),
        "has_lifted": bool(info.get("has_lifted", False)),
        "distance_to_goal": float(info.get("distance_to_goal", -1.0)),
    }


def _summary(
    args: argparse.Namespace,
    dataset_root: Path,
    run_id: str,
    saved: int,
    failures: list[dict],
) -> dict:
    attempts = saved + len(failures)
    return {
        "requested_episodes": args.episodes,
        "saved_episodes": saved,
        "attempts": attempts,
        "failed_attempts": len(failures),
        "success_rate": float(saved / attempts) if attempts else 0.0,
        "curriculum": args.curriculum,
        "cube_jitter": args.cube_jitter,
        "yaw_range_degrees": args.yaw_range_deg,
        "randomize_bin": args.randomize_bin,
        "bin_jitter": args.bin_jitter,
        "capture_stride": args.capture_stride,
        "seed_start": args.seed,
        "dataset_root": str(dataset_root),
        "run_id": run_id,
    }


def collect_episodes(args: argparse.Namespace) -> dict:
    max_attempts = args.max_attempts if args.max_attempts is not None else args.episodes * 5
    if max_attempts < args.episodes:
        raise ValueError("max_attempts must be at least episodes")

    dataset_root = Path(args.out_dir).resolve()
    provenance = {
        "kind": "scripted_teacher",
        "controller_version": CONTROLLER_VERSION,
        "curriculum": args.curriculum,
        "requested_count": args.episodes,
        "seed_start": args.seed,
        "max_steps": args.max_steps,
        "capture_stride": args.capture_stride,
        "cube_jitter": args.cube_jitter,
        "yaw_range_degrees": args.yaw_range_deg,
        "randomize_bin": args.randomize_bin,
        "bin_jitter": args.bin_jitter,
        "mode": "teacher",
    }
    writer = TeacherEpisodeWriter(dataset_root)
    run_id = writer.register_run(provenance)
    run_dir = dataset_root / "runs" / run_id
    failure_path = run_dir / "failed_attempts.json"
    failures = _load_failures(failure_path, run_id)

    manifest = load_manifest(dataset_root)
    existing = [entry for entry in manifest["episodes"] if entry["run_id"] == run_id]
    if any(entry["source"] != "teacher" or not entry["success"] for entry in existing):
        raise ValueError("existing run episodes conflict with scripted teacher provenance")
    if len(existing) > args.episodes:
        raise ValueError("existing run has more episodes than requested")

    attempted_seeds = {entry["seed"] for entry in existing} | {entry["seed"] for entry in failures}
    if any(seed < args.seed or seed >= args.seed + max_attempts for seed in attempted_seeds):
        raise ValueError("existing attempt seeds are outside this run's range")
    if len(existing) == args.episodes:
        result = _summary(args, dataset_root, run_id, len(existing), failures)
        _write_json_atomic(run_dir / "summary.json", result)
        _write_json_atomic(dataset_root / "collection_summary.json", result)
        return result

    spawn_stage = "stage2" if args.curriculum == "wide" else "stage1"
    environment: SO101BallBinsEnv | None = None
    renderer: TopDepthRenderer | None = None
    try:
        environment = SO101BallBinsEnv(spawn_stage=spawn_stage, max_steps=args.max_steps)
        renderer = TopDepthRenderer(environment.model)
        bin_body_id = mujoco.mj_name2id(environment.model, mujoco.mjtObj.mjOBJ_BODY, "square_bin")
        saved = len(existing)

        for attempt_seed in range(args.seed, args.seed + max_attempts):
            if attempt_seed in attempted_seeds:
                continue

            environment.model.body_pos[bin_body_id, :2] = BASE_BIN_POSITION
            environment.reset(seed=attempt_seed)
            scene_rng = np.random.default_rng(attempt_seed)
            if args.curriculum == "near":
                randomize_cube_pose(
                    environment,
                    scene_rng,
                    xy_jitter=args.cube_jitter,
                    yaw_range_degrees=args.yaw_range_deg,
                )
            if args.randomize_bin:
                randomize_bin_position(environment, scene_rng, jitter=args.bin_jitter)

            depths: list[np.ndarray] = []
            joints: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            control_step = 0

            def capture(action: np.ndarray, phase: str) -> None:
                nonlocal control_step
                if control_step % args.capture_stride == 0:
                    depths.append(depth_to_millimetres(renderer.render(environment.data)))
                    joints.append(environment.joint_positions().astype(np.float32, copy=True))
                    actions.append(np.asarray(action, dtype=np.float32).reshape(6).copy())
                control_step += 1

            info: dict | None = None
            reason = "task_failed"
            try:
                info = execute_waypoint_episode(environment, on_step=capture)
                success = bool(info["is_success"])
            except (RuntimeError, np.linalg.LinAlgError, FloatingPointError) as error:
                success = False
                reason = f"{type(error).__name__}: {error}"

            attempted_seeds.add(attempt_seed)
            if not success:
                failures.append(_failure_record(attempt_seed, info, reason))
                _write_json_atomic(failure_path, {"run_id": run_id, "attempts": failures})
                print(
                    f"saved={saved}/{args.episodes} attempt={len(attempted_seeds)}/{max_attempts} "
                    f"seed={attempt_seed} success=False"
                )
                continue

            if not depths:
                raise RuntimeError("successful episode did not capture any frames")
            writer.save_episode(
                np.stack(depths).astype(np.uint16, copy=False),
                np.stack(joints).astype(np.float32, copy=False),
                np.stack(actions).astype(np.float32, copy=False),
                seed=attempt_seed,
                success=True,
                source="teacher",
                run_id=run_id,
                episode_key=f"{run_id}:{attempt_seed}",
            )
            saved += 1
            print(
                f"saved={saved}/{args.episodes} attempt={len(attempted_seeds)}/{max_attempts} "
                f"seed={attempt_seed} frames={len(actions)} success=True"
            )
            if saved == args.episodes:
                break

        result = _summary(args, dataset_root, run_id, saved, failures)
        _write_json_atomic(run_dir / "summary.json", result)
        _write_json_atomic(dataset_root / "collection_summary.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if saved != args.episodes:
            raise RuntimeError(
                f"collected {saved}/{args.episodes} successful episodes within {max_attempts} attempts"
            )
        return result
    finally:
        try:
            if renderer is not None:
                renderer.close()
        finally:
            if environment is not None:
                environment.close()


def main(argv: list[str] | None = None) -> None:
    collect_episodes(parse_args(argv))


if __name__ == "__main__":
    main()
