from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from delta_depth_dataset import CONTROL_HZ
from play_fixed_delta_episode import load_delta_episode
from so101_ball_bins_env import SO101BallBinsEnv


COLORS = ((25, 118, 210), (239, 83, 80), (67, 160, 71), (255, 167, 38), (123, 31, 162))
BACKGROUND = (245, 247, 250)
PANEL = (255, 255, 255)
GRID = (221, 226, 232)
TEXT = (30, 38, 48)
MUTED = (95, 107, 121)


def trajectory_diversity_stats(episodes: list[dict[str, np.ndarray]]) -> dict[str, float]:
    if len(episodes) < 2:
        raise ValueError("at least two episodes are required")
    shapes = {tuple(np.asarray(episode["joint_pos"]).shape) for episode in episodes}
    if len(shapes) != 1 or next(iter(shapes))[1:] != (6,):
        raise ValueError("episodes must have the same trajectory shape (T, 6)")

    joints = np.stack([np.asarray(episode["joint_pos"], dtype=np.float64) for episode in episodes])
    spread = np.ptp(joints, axis=0)
    perturbations = np.stack(
        [
            np.asarray(episode["executed_action"], dtype=np.float64)
            - np.asarray(episode["teacher_action"], dtype=np.float64)
            for episode in episodes
        ]
    )
    return {
        "max_joint_spread_rad": float(np.max(spread)),
        "mean_joint_spread_rad": float(np.mean(spread)),
        "max_action_perturbation": float(np.max(np.abs(perturbations))),
        "mean_action_perturbation": float(np.mean(np.abs(perturbations))),
    }


def replay_gripper_path(entry: dict, arrays: dict[str, np.ndarray]) -> np.ndarray:
    env = SO101BallBinsEnv(
        spawn_stage="stage1",
        max_steps=max(int(entry["frames"]) + 50, 1_100),
    )
    try:
        bin_body_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "square_bin",
        )
        env.model.body_pos[bin_body_id, :2] = np.asarray(entry["bin_position"][:2])
        env.reset(seed=int(entry["seed"]))
        cube_position = np.asarray(entry["cube_position"], dtype=np.float64)
        env.data.qpos[env.cube_qpos_id : env.cube_qpos_id + 7] = np.array(
            [*cube_position, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        env.data.qvel[env.cube_qvel_id : env.cube_qvel_id + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)

        path = []
        for action in arrays["executed_action"]:
            path.append(env.data.site_xpos[env.gripper_site_id].copy())
            env.step(action)
        return np.asarray(path, dtype=np.float64)
    finally:
        env.close()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "arialbd.ttf" if bold else "arial.ttf"
    path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / filename
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, subtitle: str) -> None:
    draw.rounded_rectangle(box, radius=7, fill=PANEL, outline=(229, 233, 238), width=1)
    draw.text((box[0] + 20, box[1] + 16), title, font=_font(22, bold=True), fill=TEXT)
    draw.text((box[0] + 20, box[1] + 47), subtitle, font=_font(14), fill=MUTED)


def _bounds(values: list[np.ndarray], padding: float = 0.08) -> tuple[float, float]:
    low = min(float(np.min(value)) for value in values)
    high = max(float(np.max(value)) for value in values)
    if math.isclose(low, high):
        return low - 1.0, high + 1.0
    margin = (high - low) * padding
    return low - margin, high + margin


def _line_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    series: list[np.ndarray],
    *,
    y_bounds: tuple[float, float] | None = None,
) -> None:
    left, top, right, bottom = box
    for index in range(5):
        y = int(top + index * (bottom - top) / 4)
        draw.line((left, y, right, y), fill=GRID, width=1)
    low, high = y_bounds if y_bounds is not None else _bounds(series)
    for series_index, values in enumerate(series):
        array = np.asarray(values, dtype=np.float64)
        x = np.linspace(left, right, len(array))
        y = bottom - (array - low) / max(high - low, 1e-12) * (bottom - top)
        points = [(int(px), int(py)) for px, py in zip(x, y, strict=True)]
        draw.line(points, fill=COLORS[series_index % len(COLORS)], width=3)
    draw.text((left, top - 20), f"{high:.3f}", font=_font(12), fill=MUTED)
    draw.text((left, bottom + 4), f"{low:.3f}", font=_font(12), fill=MUTED)


def render_diversity_dashboard(
    entries: list[dict],
    episodes: list[dict[str, np.ndarray]],
    gripper_paths: list[np.ndarray],
    output: str | Path,
) -> dict[str, float]:
    stats = trajectory_diversity_stats(episodes)
    image = Image.new("RGB", (1500, 940), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((55, 35), "Fixed-scene dataset diversity", font=_font(36, bold=True), fill=TEXT)
    draw.text(
        (55, 83),
        "Object, bin and initial pose are fixed. Colored lines show control-induced trajectory variation.",
        font=_font(18),
        fill=MUTED,
    )

    cards = (
        ("Episodes", str(len(episodes))),
        ("Max joint spread", f"{np.degrees(stats['max_joint_spread_rad']):.3f} deg"),
        ("Mean joint spread", f"{np.degrees(stats['mean_joint_spread_rad']):.4f} deg"),
        ("Max action noise", f"{stats['max_action_perturbation']:.4f}"),
    )
    for index, (label, value) in enumerate(cards):
        x = 55 + index * 350
        draw.rounded_rectangle((x, 125, x + 320, 210), radius=7, fill=PANEL, outline=(229, 233, 238))
        draw.text((x + 18, 142), label, font=_font(14), fill=MUTED)
        draw.text((x + 18, 168), value, font=_font(25, bold=True), fill=TEXT)

    top_path_panel = (55, 240, 735, 570)
    height_panel = (765, 240, 1445, 570)
    joint_panel = (55, 600, 735, 905)
    action_panel = (765, 600, 1445, 905)
    _panel(draw, top_path_panel, "Gripper path: top view", "Axes auto-zoomed; units are millimetres")
    _panel(draw, height_panel, "Gripper height", "Height over the complete 33 s manipulation cycle (mm)")
    _panel(draw, joint_panel, "Joint trajectory difference", "Maximum absolute difference from episode 0 across six joints (deg)")
    _panel(draw, action_panel, "Injected control perturbation", "Maximum |executed action - teacher action| across six joints")

    plot = (95, 315, 695, 525)
    x_values = [path[:, 0] * 1000.0 for path in gripper_paths]
    y_values = [path[:, 1] * 1000.0 for path in gripper_paths]
    x_low, x_high = _bounds(x_values)
    y_low, y_high = _bounds(y_values)
    for index in range(5):
        gx = int(plot[0] + index * (plot[2] - plot[0]) / 4)
        gy = int(plot[1] + index * (plot[3] - plot[1]) / 4)
        draw.line((gx, plot[1], gx, plot[3]), fill=GRID)
        draw.line((plot[0], gy, plot[2], gy), fill=GRID)
    for episode_index, (x_path, y_path) in enumerate(zip(x_values, y_values, strict=True)):
        px = plot[0] + (x_path - x_low) / (x_high - x_low) * (plot[2] - plot[0])
        py = plot[3] - (y_path - y_low) / (y_high - y_low) * (plot[3] - plot[1])
        draw.line(
            [(int(x), int(y)) for x, y in zip(px, py, strict=True)],
            fill=COLORS[episode_index % len(COLORS)],
            width=3,
        )
    draw.text((plot[0], plot[3] + 8), f"x {x_low:.1f}..{x_high:.1f} mm", font=_font(12), fill=MUTED)
    draw.text((plot[2] - 150, plot[3] + 8), f"y {y_low:.1f}..{y_high:.1f} mm", font=_font(12), fill=MUTED)

    height_series = [path[:, 2] * 1000.0 for path in gripper_paths]
    _line_chart(draw, (805, 315, 1405, 525), height_series)

    reference = episodes[0]["joint_pos"]
    joint_difference = [
        np.degrees(np.max(np.abs(episode["joint_pos"] - reference), axis=1))
        for episode in episodes
    ]
    _line_chart(draw, (95, 675, 695, 850), joint_difference, y_bounds=(0.0, max(0.1, _bounds(joint_difference)[1])))

    action_noise = [
        np.max(np.abs(episode["executed_action"] - episode["teacher_action"]), axis=1)
        for episode in episodes
    ]
    _line_chart(draw, (805, 675, 1405, 850), action_noise, y_bounds=(0.0, max(0.01, _bounds(action_noise)[1])))

    for index, entry in enumerate(entries):
        x = 1130 + (index % 3) * 100
        draw.line((x, 96, x + 24, 96), fill=COLORS[index % len(COLORS)], width=4)
        draw.text((x + 30, 87), f"ep {entry['index']}", font=_font(13), fill=TEXT)

    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    print(json.dumps({"output": str(destination), **stats}, indent=2, sort_keys=True), flush=True)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize trajectory diversity in a fixed-delta dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("artifacts/fixed_delta_diversity.png"))
    parser.add_argument("--open", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if len(args.episode_indices) < 2:
        raise ValueError("select at least two episode indices")
    loaded = [load_delta_episode(args.dataset, index) for index in args.episode_indices]
    entries = [entry for entry, _ in loaded]
    episodes = [arrays for _, arrays in loaded]
    paths = [replay_gripper_path(entry, arrays) for entry, arrays in loaded]
    render_diversity_dashboard(entries, episodes, paths, args.output)
    if args.open:
        os.startfile(Path(args.output).resolve())


if __name__ == "__main__":
    main()
