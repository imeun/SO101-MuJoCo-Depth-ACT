from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import time

import mujoco.viewer
import numpy as np
import torch

from collect_fixed_delta_depth_dataset import place_nearby_scene
from so101_ball_bins_env import SO101BallBinsEnv
from so101_depth import DepthConfig, TopDepthRenderer
from so101_depth_act import (
    JointVelocityEstimator,
    TemporalActionEnsembler,
    load_depth_act_checkpoint,
    predict_delta_chunk,
)
from train_depth_act import resolve_device


def _checkpoint(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"ACT checkpoint does not exist: {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visually play a dual-depth ACT checkpoint in MuJoCo.")
    parser.add_argument("--model", required=True, type=_checkpoint)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--max-steps", type=int, default=1100)
    parser.add_argument("--episodes", type=int, default=0, help="0 keeps replaying until the viewer closes")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ensemble-decay", type=float, default=0.08)
    parser.add_argument("--cube-jitter", type=float, default=0.020)
    parser.add_argument("--bin-jitter", type=float, default=0.010)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cube_jitter < 0 or args.bin_jitter < 0:
        raise ValueError("scene jitter must be non-negative")
    device = resolve_device(args.device)
    policy, payload = load_depth_act_checkpoint(str(args.model), map_location="cpu")
    policy = policy.to(device).eval()
    allowed = {field.name for field in fields(DepthConfig)}
    depth_config = DepthConfig(**{key: value for key, value in payload["depth_config"].items() if key in allowed})
    environment = SO101BallBinsEnv(spawn_stage="stage1", max_steps=args.max_steps)
    top_renderer = TopDepthRenderer(environment.model, camera_name="top")
    side_renderer = TopDepthRenderer(environment.model, camera_name="side_depth")
    episode = 0
    try:
        with mujoco.viewer.launch_passive(environment.model, environment.data) as viewer:
            while viewer.is_running() and (args.episodes == 0 or episode < args.episodes):
                episode_seed = args.seed + episode
                environment.reset(seed=episode_seed)
                place_nearby_scene(
                    environment,
                    seed=episode_seed,
                    cube_jitter=args.cube_jitter,
                    bin_jitter=args.bin_jitter,
                )
                ensemble = TemporalActionEnsembler(
                    chunk_size=policy.config.chunk_size,
                    decay=args.ensemble_decay,
                )
                velocity_estimator = JointVelocityEstimator(
                    control_period_s=environment.model.opt.timestep * environment.frame_skip
                )
                terminated = truncated = False
                step = 0
                info = {}
                while viewer.is_running() and not (terminated or truncated):
                    started = time.perf_counter()
                    qpos = environment.joint_positions()
                    qvel = velocity_estimator.update(qpos)
                    delta = predict_delta_chunk(
                        policy,
                        top_renderer.render(environment.data),
                        side_renderer.render(environment.data),
                        qpos,
                        qvel,
                        depth_config=depth_config,
                        device=device,
                    )
                    absolute_chunk = np.clip(
                        qpos[None, :] + delta,
                        environment.task_ctrl_low,
                        environment.task_ctrl_high,
                    )
                    target = ensemble.add_and_get(absolute_chunk)
                    action = np.clip(
                        (target - environment.ctrl) / environment.action_scale,
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                    _, _, terminated, truncated, info = environment.step(action)
                    if step % 30 == 0 or terminated or truncated:
                        print(
                            f"episode={episode + 1} step={step} phase={info.get('phase')} "
                            f"grasped={bool(info.get('has_grasped'))} lifted={bool(info.get('has_lifted'))} "
                            f"success={bool(info.get('is_success'))} "
                            f"inference_ms={(time.perf_counter() - started) * 1000:.1f}",
                            flush=True,
                        )
                    step += 1
                    viewer.sync()
                    time.sleep(max(0.0, environment.model.opt.timestep * environment.frame_skip))
                print(
                    f"episode={episode + 1} seed={args.seed + episode} "
                    f"success={bool(info.get('is_success'))}",
                    flush=True,
                )
                episode += 1
    except KeyboardInterrupt:
        print("Playback interrupted.")
    finally:
        top_renderer.close()
        side_renderer.close()
        environment.close()


if __name__ == "__main__":
    main()
