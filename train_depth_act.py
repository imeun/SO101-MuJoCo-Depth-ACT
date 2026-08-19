from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, fields
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from delta_depth_dataset import load_delta_manifest
from collect_fixed_delta_depth_dataset import place_nearby_scene
from depth_act_dataset import (
    DepthACTEpisodeDataset,
    EpisodeBatchSampler,
    delta_episode_records,
    split_delta_records,
)
from measure_realsense_depth_noise import depth_config_from_profiles
from so101_ball_bins_env import SO101BallBinsEnv
from so101_depth import DepthConfig, TopDepthRenderer
from so101_depth_act import (
    DepthACTPolicy,
    JointVelocityEstimator,
    TemporalActionEnsembler,
    depth_act_loss,
    load_depth_act_checkpoint,
    predict_delta_chunk,
)


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SO101 dual-depth ACT policy.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--additional-dataset", type=Path, action="append", default=[])
    parser.add_argument("--additional-repeat", type=_positive_int, default=3)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=_positive_int, default=30)
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=_positive_int, default=30)
    parser.add_argument("--d-model", type=_positive_int, default=256)
    parser.add_argument("--nhead", type=_positive_int, default=8)
    parser.add_argument("--encoder-layers", type=_positive_int, default=4)
    parser.add_argument("--decoder-layers", type=_positive_int, default=2)
    parser.add_argument("--dim-feedforward", type=_positive_int, default=1024)
    parser.add_argument("--backbone-width", type=_positive_int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--patience", type=_positive_int, default=8)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=4)
    parser.add_argument("--log-interval", type=_positive_int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--top-noise-profile", type=Path)
    parser.add_argument("--side-noise-profile", type=Path)
    parser.add_argument("--rollout-eval-episodes", type=_nonnegative_int, default=3)
    parser.add_argument("--rollout-max-steps", type=_positive_int, default=1100)
    parser.add_argument("--rollout-seed", type=int, default=30000)
    parser.add_argument("--rollout-cube-jitter", type=float, default=0.020)
    parser.add_argument("--rollout-bin-jitter", type=float, default=0.010)
    args = parser.parse_args(argv)
    if (args.top_noise_profile is None) != (args.side_noise_profile is None):
        parser.error("both noise profiles must be provided together")
    if args.d_model % args.nhead != 0 or args.d_model % 4 != 0:
        parser.error("--d-model must be divisible by --nhead and four")
    if args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error("invalid weight decay or dropout")
    if args.rollout_cube_jitter < 0 or args.rollout_bin_jitter < 0:
        parser.error("rollout scene jitter must be non-negative")
    return args


def build_record_splits(
    base_dataset: str | Path,
    additional_datasets: list[str | Path],
    *,
    additional_repeat: int,
    seed: int,
):
    if additional_repeat <= 0:
        raise ValueError("additional_repeat must be positive")
    roots = [Path(base_dataset).resolve(), *(Path(path).resolve() for path in additional_datasets)]
    if len(set(roots)) != len(roots):
        raise ValueError("base and additional dataset paths must be unique")

    base_manifest = load_delta_manifest(roots[0])
    train_records = []
    validation_records = []
    sources = []
    for index, root in enumerate(roots):
        manifest = load_delta_manifest(root)
        if manifest["provenance"] != base_manifest["provenance"]:
            raise ValueError(f"dataset scene/camera provenance does not match base dataset: {root}")
        source_train, source_validation = split_delta_records(
            delta_episode_records(root),
            seed=seed + index,
        )
        repeats = 1 if index == 0 else additional_repeat
        train_records.extend(source_train * repeats)
        validation_records.extend(source_validation)
        sources.append(
            {
                "root": str(root),
                "episodes": len(manifest["episodes"]),
                "train_episodes": len(source_train),
                "validation_episodes": len(source_validation),
                "train_repeats": repeats,
            }
        )
    return train_records, validation_records, sources


def resolve_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _depth_config(args: argparse.Namespace, checkpoint_payload: dict | None = None) -> DepthConfig:
    if args.top_noise_profile is None:
        if checkpoint_payload is not None and "depth_config" in checkpoint_payload:
            allowed = {field.name for field in fields(DepthConfig)}
            return DepthConfig(
                **{
                    key: value
                    for key, value in checkpoint_payload["depth_config"].items()
                    if key in allowed
                }
            )
        return DepthConfig()
    profiles = []
    for path in (args.top_noise_profile, args.side_noise_profile):
        try:
            profiles.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read RealSense profile: {path}") from error
    return depth_config_from_profiles(profiles)


def _joint_delta_scale() -> np.ndarray:
    environment = SO101BallBinsEnv(spawn_stage="stage1")
    try:
        return np.maximum(environment.task_ctrl_high - environment.task_ctrl_low, 0.1).astype(np.float32)
    finally:
        environment.close()


def _proprio_stats(records) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(12, dtype=np.float64)
    square = np.zeros(12, dtype=np.float64)
    count = 0
    for record in records:
        with np.load(record.path, allow_pickle=False) as archive:
            values = np.concatenate(
                [archive["joint_pos"].astype(np.float64), archive["joint_velocity"].astype(np.float64)],
                axis=1,
            )
        total += values.sum(axis=0)
        square += np.square(values).sum(axis=0)
        count += values.shape[0]
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _run_epoch(
    model: DepthACTPolicy,
    loader: DataLoader,
    device: torch.device,
    delta_scale: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer | None,
    log_interval: int,
    epoch: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "delta_loss": 0.0, "velocity_loss": 0.0, "zero_delta_baseline": 0.0}
    samples = 0
    started = time.perf_counter()
    context = nullcontext() if training else torch.inference_mode()
    with context:
        for batch_index, batch in enumerate(loader, start=1):
            top = batch["top_depth"].to(device, non_blocking=True)
            side = batch["side_depth"].to(device, non_blocking=True)
            proprio = batch["proprio"].to(device, non_blocking=True)
            target = batch["delta_chunk"].to(device, non_blocking=True)
            mask = batch["chunk_mask"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(top, side, proprio)
            loss, metrics = depth_act_loss(prediction, target, mask, delta_scale=delta_scale)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            count = top.shape[0]
            for name in totals:
                totals[name] += metrics[name] * count
            samples += count
            if training and batch_index % log_interval == 0:
                ratio = metrics["loss"] / max(metrics["zero_delta_baseline"], 1e-12)
                print(
                    f"epoch={epoch} batch={batch_index}/{len(loader)} loss={metrics['loss']:.6f} "
                    f"zero_ratio={ratio:.3f} elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    return {name: value / max(samples, 1) for name, value in totals.items()}


def evaluate_rollouts(
    model: DepthACTPolicy,
    depth_config: DepthConfig,
    device: torch.device,
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    cube_jitter: float,
    bin_jitter: float,
) -> dict[str, float]:
    if episodes == 0:
        return {"success_rate": 0.0, "lift_rate": 0.0, "grasp_rate": 0.0}
    success = grasp = lift = 0
    environment = SO101BallBinsEnv(spawn_stage="stage1", max_steps=max_steps)
    top_renderer = TopDepthRenderer(environment.model, camera_name="top")
    side_renderer = TopDepthRenderer(environment.model, camera_name="side_depth")
    try:
        model.eval()
        for episode in range(episodes):
            episode_seed = seed + episode
            environment.reset(seed=episode_seed)
            place_nearby_scene(
                environment,
                seed=episode_seed,
                cube_jitter=cube_jitter,
                bin_jitter=bin_jitter,
            )
            ensemble = TemporalActionEnsembler(chunk_size=model.config.chunk_size)
            velocity_estimator = JointVelocityEstimator(
                control_period_s=environment.model.opt.timestep * environment.frame_skip
            )
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                qpos = environment.joint_positions()
                qvel = velocity_estimator.update(qpos)
                delta = predict_delta_chunk(
                    model,
                    top_renderer.render(environment.data),
                    side_renderer.render(environment.data),
                    qpos,
                    qvel,
                    depth_config=depth_config,
                    device=device,
                )
                absolute = np.clip(qpos[None, :] + delta, environment.task_ctrl_low, environment.task_ctrl_high)
                target = ensemble.add_and_get(absolute)
                action = np.clip((target - environment.ctrl) / environment.action_scale, -1.0, 1.0)
                _, _, terminated, truncated, info = environment.step(action.astype(np.float32))
            success += int(bool(info.get("is_success")))
            grasp += int(bool(info.get("has_grasped")))
            lift += int(bool(info.get("has_lifted")))
    finally:
        top_renderer.close()
        side_renderer.close()
        environment.close()
    return {
        "success_rate": success / episodes,
        "lift_rate": lift / episodes,
        "grasp_rate": grasp / episodes,
    }


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    _set_seed(args.seed)
    device = resolve_device(args.device)
    pretrained_payload = None
    pretrained_model = None
    if args.pretrained_checkpoint is not None:
        checkpoint_path = args.pretrained_checkpoint.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise ValueError(f"pretrained checkpoint does not exist: {checkpoint_path}")
        pretrained_model, pretrained_payload = load_depth_act_checkpoint(
            str(checkpoint_path),
            map_location="cpu",
        )
    train_records, validation_records, dataset_sources = build_record_splits(
        args.dataset,
        args.additional_dataset,
        additional_repeat=args.additional_repeat,
        seed=args.seed,
    )
    depth_config = _depth_config(args, pretrained_payload)
    effective_chunk_size = (
        pretrained_model.config.chunk_size if pretrained_model is not None else args.chunk_size
    )
    train_dataset = DepthACTEpisodeDataset(
        train_records,
        chunk_size=effective_chunk_size,
        depth_config=depth_config,
        training=True,
        augmentation_seed=args.seed,
    )
    validation_dataset = DepthACTEpisodeDataset(
        validation_records,
        chunk_size=effective_chunk_size,
        depth_config=depth_config,
        training=False,
    )
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=EpisodeBatchSampler(
            train_dataset.episode_offsets,
            batch_size=args.batch_size,
            shuffle_episodes=True,
            shuffle_frames=True,
            seed=args.seed,
        ),
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=EpisodeBatchSampler(
            validation_dataset.episode_offsets,
            batch_size=args.batch_size,
            shuffle_episodes=False,
            shuffle_frames=False,
            seed=args.seed,
        ),
        **loader_kwargs,
    )
    if pretrained_model is None:
        widths = tuple(args.backbone_width * (2**index) for index in range(4))
        model = DepthACTPolicy(
            chunk_size=args.chunk_size,
            d_model=args.d_model,
            nhead=args.nhead,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            backbone_channels=widths,
        ).to(device)
    else:
        model = pretrained_model.to(device)
        print(
            f"initialized_from={args.pretrained_checkpoint.resolve()} "
            f"chunk_size={effective_chunk_size}",
            flush=True,
        )
    proprio_mean, proprio_std = _proprio_stats(train_records)
    model.set_proprio_stats(proprio_mean, proprio_std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    delta_scale_np = _joint_delta_scale()
    delta_scale = torch.from_numpy(delta_scale_np).to(device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    best_validation = math.inf
    best_rollout_rank = (-1.0, -1.0, -1.0, -math.inf)
    stale_epochs = 0
    history = []
    manifest = load_delta_manifest(args.dataset)

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_metrics = _run_epoch(
            model, train_loader, device, delta_scale, optimizer=optimizer,
            log_interval=args.log_interval, epoch=epoch,
        )
        validation_metrics = _run_epoch(
            model, validation_loader, device, delta_scale, optimizer=None,
            log_interval=args.log_interval, epoch=epoch,
        )
        rollout = evaluate_rollouts(
            model, depth_config, device,
            episodes=args.rollout_eval_episodes,
            seed=args.rollout_seed,
            max_steps=args.rollout_max_steps,
            cube_jitter=args.rollout_cube_jitter,
            bin_jitter=args.rollout_bin_jitter,
        )
        ratio = validation_metrics["loss"] / max(validation_metrics["zero_delta_baseline"], 1e-12)
        print(
            f"epoch={epoch}/{args.epochs} train={train_metrics['loss']:.6f} "
            f"validation={validation_metrics['loss']:.6f} zero_ratio={ratio:.3f} "
            f"rollout_success={rollout['success_rate']:.3f} lift={rollout['lift_rate']:.3f} "
            f"grasp={rollout['grasp_rate']:.3f}",
            flush=True,
        )
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics, "rollout": rollout}
        history.append(record)
        payload = {
            "architecture_version": model.architecture_version,
            "architecture_config": model.architecture_config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "depth_config": asdict(depth_config),
            "delta_scale": delta_scale_np.tolist(),
            "proprio_mean": proprio_mean.tolist(),
            "proprio_std": proprio_std.tolist(),
            "dataset_root": str(Path(args.dataset).resolve()),
            "dataset_roots": [source["root"] for source in dataset_sources],
            "dataset_sources": dataset_sources,
            "dataset_provenance": manifest["provenance"],
            "home_joint_pos": manifest["episodes"][0]["initial_joint_pos"],
            "initialized_from": (
                None
                if args.pretrained_checkpoint is None
                else str(args.pretrained_checkpoint.expanduser().resolve())
            ),
            "metrics": record,
        }
        _atomic_torch_save(payload, output / "last_checkpoint.pt")
        _atomic_torch_save(payload, output / "checkpoints" / f"epoch_{epoch:04d}.pt")
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            stale_epochs = 0
            _atomic_torch_save(payload, output / "best_checkpoint.pt")
        else:
            stale_epochs += 1
        if args.rollout_eval_episodes > 0:
            rank = (
                rollout["success_rate"], rollout["lift_rate"], rollout["grasp_rate"],
                -validation_metrics["loss"],
            )
            if rank > best_rollout_rank:
                best_rollout_rank = rank
                _atomic_torch_save(payload, output / "best_rollout_checkpoint.pt")
        (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if stale_epochs >= args.patience:
            print(f"early_stop epoch={epoch} patience={args.patience}", flush=True)
            break
    return history[-1]


if __name__ == "__main__":
    main()
