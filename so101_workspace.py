from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


SpawnStage = Literal["stage1", "stage2"]


@dataclass(frozen=True)
class WorkspaceConfig:
    x_bounds: tuple[float, float] = (0.050, 0.400)
    y_bounds: tuple[float, float] = (-0.250, 0.250)
    stage1_radius: tuple[float, float] = (0.350, 0.370)
    stage2_radius: tuple[float, float] = (0.300, 0.380)
    bin_center: tuple[float, float] = (0.310, -0.050)
    bin_exclusion_half_extent: float = 0.065
    table_center: tuple[float, float] = (0.220, 0.000)
    table_half_size: tuple[float, float] = (0.340, 0.260)
    table_surface_z: float = -0.0024
    block_half_size: tuple[float, float, float] = (0.0175, 0.0175, 0.035)
    block_center_z: float = 0.0336
    max_attempts: int = 10_000


def sample_block_pose(
    rng: np.random.Generator,
    stage: SpawnStage,
    config: WorkspaceConfig = WorkspaceConfig(),
) -> np.ndarray:
    """Return [x, y, z, qw, qx, qy, qz] as float64."""
    if stage == "stage1":
        x, y = 0.320, 0.155
        radius = float(np.hypot(x, y))
        on_table = bool(
            config.table_center[0] - config.table_half_size[0] <= x - config.block_half_size[0]
            and x + config.block_half_size[0] <= config.table_center[0] + config.table_half_size[0]
            and config.table_center[1] - config.table_half_size[1] <= y - config.block_half_size[1]
            and y + config.block_half_size[1] <= config.table_center[1] + config.table_half_size[1]
        )
        valid = bool(
            config.x_bounds[0] <= x <= config.x_bounds[1]
            and config.y_bounds[0] <= y <= config.y_bounds[1]
            and config.stage1_radius[0] <= radius <= config.stage1_radius[1]
            and not (
                abs(x - config.bin_center[0]) < config.bin_exclusion_half_extent
                and abs(y - config.bin_center[1]) < config.bin_exclusion_half_extent
            )
            and on_table
        )
        if not valid:
            raise RuntimeError("fixed stage1 block pose is outside the configured workspace")
        return np.array([x, y, config.block_center_z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elif stage == "stage2":
        radius_bounds = config.stage2_radius
    else:
        raise ValueError(f"unknown spawn stage: {stage}")

    table_x_bounds = (
        config.table_center[0] - config.table_half_size[0],
        config.table_center[0] + config.table_half_size[0],
    )
    table_y_bounds = (
        config.table_center[1] - config.table_half_size[1],
        config.table_center[1] + config.table_half_size[1],
    )
    half_width = config.block_half_size[0]

    for _ in range(config.max_attempts):
        x = rng.uniform(*config.x_bounds)
        y = rng.uniform(*config.y_bounds)
        yaw = rng.uniform(0.0, 2.0 * np.pi)
        radius = float(np.hypot(x, y))
        footprint_half_extent = half_width * (abs(np.cos(yaw)) + abs(np.sin(yaw)))

        if not radius_bounds[0] <= radius <= radius_bounds[1]:
            continue
        if (
            abs(x - config.bin_center[0]) < config.bin_exclusion_half_extent
            and abs(y - config.bin_center[1]) < config.bin_exclusion_half_extent
        ):
            continue
        if not (
            table_x_bounds[0] <= x - footprint_half_extent
            and x + footprint_half_extent <= table_x_bounds[1]
            and table_y_bounds[0] <= y - footprint_half_extent
            and y + footprint_half_extent <= table_y_bounds[1]
        ):
            continue

        half_yaw = yaw / 2.0
        return np.array(
            [x, y, config.block_center_z, np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
            dtype=np.float64,
        )

    raise RuntimeError(f"could not sample a valid {stage} block pose after {config.max_attempts} attempts")
