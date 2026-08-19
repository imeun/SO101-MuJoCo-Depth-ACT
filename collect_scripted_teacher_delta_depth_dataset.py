from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from collect_fixed_delta_depth_dataset import scene_provenance
from delta_depth_dataset import CONTROL_HZ, PHASE_TO_ID, DeltaDepthEpisodeWriter, atomic_json, load_delta_manifest
from play_waypoint_teacher import (
    BASE_BIN_POSITION,
    execute_waypoint_episode,
    randomize_bin_position,
    randomize_cube_pose,
)
from so101_ball_bins_env import SO101BallBinsEnv
from so101_depth import TopDepthRenderer, depth_to_millimetres
from teacher_dataset import load_manifest


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay successful scripted-teacher episodes into the dual-depth ACT delta format."
    )
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-episodes", type=_positive_int)
    return parser.parse_args(argv)


def _source_entries(root: Path) -> tuple[dict, list[dict]]:
    manifest = load_manifest(root)
    entries = [
        entry for entry in manifest["episodes"]
        if entry["source"] == "teacher"
        and entry["success"]
        and manifest["runs"][entry["run_id"]].get("kind") == "scripted_teacher"
    ]
    entries.sort(key=lambda entry: entry["index"])
    if not entries:
        raise ValueError("source dataset has no successful scripted-teacher episodes")
    seeds = [entry["seed"] for entry in entries]
    if len(seeds) != len(set(seeds)):
        raise ValueError("selected scripted-teacher episodes contain duplicate seeds across runs")
    return manifest, entries


def _place_source_scene(
    environment: SO101BallBinsEnv,
    source_entry: dict,
    provenance: dict,
) -> None:
    bin_body_id = mujoco.mj_name2id(environment.model, mujoco.mjtObj.mjOBJ_BODY, "square_bin")
    environment.model.body_pos[bin_body_id, :2] = BASE_BIN_POSITION
    environment.reset(seed=source_entry["seed"])
    scene_rng = np.random.default_rng(source_entry["seed"])
    if provenance["curriculum"] == "near":
        randomize_cube_pose(
            environment,
            scene_rng,
            xy_jitter=provenance["cube_jitter"],
            yaw_range_degrees=provenance["yaw_range_degrees"],
        )
    if provenance["randomize_bin"]:
        randomize_bin_position(environment, scene_rng, jitter=provenance["bin_jitter"])


def replay_successes(args: argparse.Namespace) -> dict:
    source_root = Path(args.source_dataset).resolve()
    output_root = Path(args.out_dir).resolve()
    if source_root == output_root:
        raise ValueError("source and output datasets must use different directories")
    source_manifest, entries = _source_entries(source_root)
    if args.max_episodes is not None:
        entries = entries[: args.max_episodes]
    selected_seeds = {entry["seed"] for entry in entries}

    varied_scene = any(
        source_manifest["runs"][entry["run_id"]]["curriculum"] != "fixed"
        or source_manifest["runs"][entry["run_id"]]["randomize_bin"]
        for entry in entries
    )
    initial_conditions = "near_position" if varied_scene else "fixed"
    first_provenance = source_manifest["runs"][entries[0]["run_id"]]
    first_stage = "stage2" if first_provenance["curriculum"] == "wide" else "stage1"
    initial_environment = SO101BallBinsEnv(
        spawn_stage=first_stage,
        max_steps=first_provenance["max_steps"],
    )
    try:
        writer = DeltaDepthEpisodeWriter(
            output_root,
            provenance=scene_provenance(initial_environment),
            initial_conditions=initial_conditions,
        )
    finally:
        initial_environment.close()
    existing_manifest = load_delta_manifest(output_root)
    existing_seeds = {entry["seed"] for entry in existing_manifest["episodes"]}
    if not existing_seeds.issubset(selected_seeds):
        raise ValueError("output dataset contains seeds outside the selected source episodes")

    environment = None
    top_renderer = None
    side_renderer = None
    active_configuration = None
    failures: list[dict] = []
    replayed_now = 0
    try:
        for source_entry in entries:
            seed = source_entry["seed"]
            if seed in existing_seeds:
                continue
            provenance = source_manifest["runs"][source_entry["run_id"]]
            spawn_stage = "stage2" if provenance["curriculum"] == "wide" else "stage1"
            configuration = (spawn_stage, provenance["max_steps"])
            if configuration != active_configuration:
                if top_renderer is not None:
                    top_renderer.close()
                if side_renderer is not None:
                    side_renderer.close()
                if environment is not None:
                    environment.close()
                environment = SO101BallBinsEnv(spawn_stage=spawn_stage, max_steps=provenance["max_steps"])
                top_renderer = TopDepthRenderer(environment.model, camera_name="top")
                side_renderer = TopDepthRenderer(environment.model, camera_name="side_depth")
                active_configuration = configuration

            _place_source_scene(environment, source_entry, provenance)
            initial_joint_pos = environment.joint_positions().copy()
            cube_position = environment._cube_position().copy()
            bin_position = environment._target_position().copy()
            top_frames = []
            side_frames = []
            joint_positions = []
            joint_velocities = []
            teacher_goals = []
            delta_targets = []
            teacher_actions = []
            executed_actions = []
            phases = []
            timestamps = []
            control_period_s = 1.0 / CONTROL_HZ

            def capture(clean_action: np.ndarray, executed_action: np.ndarray, phase: str) -> None:
                if phase not in PHASE_TO_ID:
                    raise ValueError(f"unknown scripted phase: {phase}")
                joint_pos = environment.joint_positions().astype(np.float32, copy=True)
                joint_velocity = (
                    np.zeros(6, dtype=np.float32)
                    if not joint_positions
                    else ((joint_pos - joint_positions[-1]) / np.float32(control_period_s)).astype(np.float32)
                )
                teacher_goal = environment.control_target(clean_action).astype(np.float32, copy=True)
                top_frames.append(depth_to_millimetres(top_renderer.render(environment.data)))
                side_frames.append(depth_to_millimetres(side_renderer.render(environment.data)))
                joint_positions.append(joint_pos)
                joint_velocities.append(joint_velocity)
                teacher_goals.append(teacher_goal)
                delta_targets.append((teacher_goal - joint_pos).astype(np.float32))
                teacher_actions.append(clean_action.astype(np.float32, copy=True))
                executed_actions.append(executed_action.astype(np.float32, copy=True))
                phases.append(PHASE_TO_ID[phase])
                timestamps.append(len(timestamps) * control_period_s)

            reason = "task_failed"
            try:
                info = execute_waypoint_episode(environment, on_control_step=capture)
                success = bool(info["is_success"])
            except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                success = False
                reason = f"{type(error).__name__}: {error}"
            if not success:
                failures.append({
                    "source_episode_index": source_entry["index"],
                    "seed": seed,
                    "reason": reason,
                })
                print(
                    f"saved={len(existing_seeds)}/{len(entries)} source_episode={source_entry['index']} "
                    f"seed={seed} success=False",
                    flush=True,
                )
                continue

            writer.save_episode(
                np.stack(top_frames).astype(np.uint16, copy=False),
                np.stack(side_frames).astype(np.uint16, copy=False),
                np.stack(joint_positions).astype(np.float32, copy=False),
                np.stack(joint_velocities).astype(np.float32, copy=False),
                np.stack(teacher_goals).astype(np.float32, copy=False),
                np.stack(delta_targets).astype(np.float32, copy=False),
                np.stack(teacher_actions).astype(np.float32, copy=False),
                np.stack(executed_actions).astype(np.float32, copy=False),
                np.asarray(phases, dtype=np.uint8),
                np.asarray(timestamps, dtype=np.float64),
                seed=seed,
                initial_joint_pos=initial_joint_pos,
                cube_position=cube_position,
                bin_position=bin_position,
            )
            existing_seeds.add(seed)
            replayed_now += 1
            print(
                f"saved={len(existing_seeds)}/{len(entries)} source_episode={source_entry['index']} "
                f"seed={seed} frames={len(timestamps)} success=True",
                flush=True,
            )
    finally:
        if top_renderer is not None:
            top_renderer.close()
        if side_renderer is not None:
            side_renderer.close()
        if environment is not None:
            environment.close()

    summary = {
        "source_dataset": str(source_root),
        "output_dataset": str(output_root),
        "selected_source_episodes": len(entries),
        "saved_episodes": len(existing_seeds & selected_seeds),
        "replayed_now": replayed_now,
        "failed_replays": failures,
        "camera_names": ["top", "side_depth"],
        "capture_stride": 1,
        "control_hz": CONTROL_HZ,
        "target_type": "cumulative_future_joint_delta_generated_by_loader",
        "initial_conditions": initial_conditions,
    }
    atomic_json(output_root / "replay_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if summary["saved_episodes"] != len(entries):
        raise RuntimeError(
            f"converted {summary['saved_episodes']}/{len(entries)} successful source episodes"
        )
    return summary


def main(argv: list[str] | None = None) -> None:
    replay_successes(parse_args(argv))


if __name__ == "__main__":
    main()
