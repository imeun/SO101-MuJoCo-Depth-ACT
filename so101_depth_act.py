from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from so101_depth import DepthConfig, preprocess_depth


@dataclass(frozen=True)
class DepthACTConfig:
    chunk_size: int = 30
    d_model: int = 256
    nhead: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 2
    dim_feedforward: int = 1024
    dropout: float = 0.1
    backbone_channels: tuple[int, int, int, int] = (64, 128, 256, 512)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(value)
        value = F.relu(self.bn1(self.conv1(value)), inplace=True)
        value = self.bn2(self.conv2(value))
        return F.relu(value + residual, inplace=True)


class OneChannelResNet18(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int]):
        super().__init__()
        if len(channels) != 4 or any(channel <= 0 for channel in channels):
            raise ValueError("backbone_channels must contain four positive values")
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        stages = []
        in_channels = channels[0]
        for stage_index, out_channels in enumerate(channels):
            stride = 1 if stage_index == 0 else 2
            stages.append(nn.Sequential(
                BasicBlock(in_channels, out_channels, stride=stride),
                BasicBlock(out_channels, out_channels),
            ))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.out_channels = channels[-1]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(value))


def _spatial_encoding(height: int, width: int, channels: int, device, dtype) -> torch.Tensor:
    if channels % 4 != 0:
        raise ValueError("d_model must be divisible by four")
    quarter = channels // 4
    omega = torch.exp(
        torch.arange(quarter, device=device, dtype=dtype) * (-math.log(10_000.0) / max(quarter - 1, 1))
    )
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)[:, None] * omega[None, :]
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)[:, None] * omega[None, :]
    y_encoding = torch.cat([torch.sin(y), torch.cos(y)], dim=1)[:, None, :].expand(-1, width, -1)
    x_encoding = torch.cat([torch.sin(x), torch.cos(x)], dim=1)[None, :, :].expand(height, -1, -1)
    return torch.cat([y_encoding, x_encoding], dim=2).reshape(1, height * width, channels)


class DepthACTPolicy(nn.Module):
    architecture_version = "so101-dual-depth-act-v1"

    def __init__(
        self,
        *,
        chunk_size: int = 30,
        d_model: int = 256,
        nhead: int = 8,
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        backbone_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
    ):
        super().__init__()
        self.config = DepthACTConfig(
            chunk_size=chunk_size,
            d_model=d_model,
            nhead=nhead,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            backbone_channels=tuple(backbone_channels),
        )
        if chunk_size <= 0 or d_model % nhead != 0 or d_model % 4 != 0:
            raise ValueError("invalid ACT dimensions")
        self.backbone = OneChannelResNet18(tuple(backbone_channels))
        self.visual_projection = nn.Conv2d(self.backbone.out_channels, d_model, 1)
        self.camera_embedding = nn.Parameter(torch.zeros(2, 1, d_model))
        self.register_buffer("proprio_mean", torch.zeros(12))
        self.register_buffer("proprio_std", torch.ones(12))
        self.proprio_encoder = nn.Sequential(nn.Linear(12, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers, norm=nn.LayerNorm(d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers, norm=nn.LayerNorm(d_model))
        self.action_queries = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        self.action_head = nn.Linear(d_model, 6)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def architecture_config(self) -> dict:
        result = asdict(self.config)
        result["backbone_channels"] = list(result["backbone_channels"])
        return result

    def set_proprio_stats(self, mean: np.ndarray | torch.Tensor, std: np.ndarray | torch.Tensor) -> None:
        mean_tensor = torch.as_tensor(mean, dtype=self.proprio_mean.dtype, device=self.proprio_mean.device)
        std_tensor = torch.as_tensor(std, dtype=self.proprio_std.dtype, device=self.proprio_std.device)
        if mean_tensor.shape != (12,) or std_tensor.shape != (12,) or torch.any(std_tensor <= 0):
            raise ValueError("proprio statistics must be finite positive vectors with shape (12,)")
        if not torch.all(torch.isfinite(mean_tensor)) or not torch.all(torch.isfinite(std_tensor)):
            raise ValueError("proprio statistics must be finite")
        self.proprio_mean.copy_(mean_tensor)
        self.proprio_std.copy_(std_tensor)

    def _visual_tokens(self, depth: torch.Tensor, camera_index: int) -> torch.Tensor:
        features = self.visual_projection(self.backbone(depth))
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        position = _spatial_encoding(height, width, channels, features.device, features.dtype)
        return tokens + position + self.camera_embedding[camera_index]

    def forward(self, top_depth: torch.Tensor, side_depth: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        if top_depth.ndim != 4 or side_depth.shape != top_depth.shape or top_depth.shape[1] != 1:
            raise ValueError("depth inputs must have matching shape (B, 1, H, W)")
        if proprio.shape != (top_depth.shape[0], 12):
            raise ValueError("proprio must have shape (B, 12)")
        memory_input = torch.cat([
            self._visual_tokens(top_depth, 0),
            self._visual_tokens(side_depth, 1),
            self.proprio_encoder((proprio - self.proprio_mean) / self.proprio_std).unsqueeze(1),
        ], dim=1)
        memory = self.encoder(memory_input)
        queries = self.action_queries.unsqueeze(0).expand(top_depth.shape[0], -1, -1)
        return self.action_head(self.decoder(queries, memory))


def decode_delta_chunk(current_joint_pos: torch.Tensor, delta_chunk: torch.Tensor) -> torch.Tensor:
    if current_joint_pos.ndim != 2 or delta_chunk.ndim != 3:
        raise ValueError("expected current positions (B, J) and delta chunk (B, K, J)")
    if current_joint_pos.shape[0] != delta_chunk.shape[0] or current_joint_pos.shape[1] != delta_chunk.shape[2]:
        raise ValueError("joint dimensions do not match")
    return current_joint_pos[:, None, :] + delta_chunk


def depth_act_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    velocity_weight: float = 0.1,
    delta_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have matching shape (B, K, J)")
    if mask.shape != prediction.shape[:2]:
        raise ValueError("mask must have shape (B, K)")
    if delta_scale is None:
        scale = torch.ones(prediction.shape[-1], device=prediction.device, dtype=prediction.dtype)
    else:
        scale = delta_scale.to(device=prediction.device, dtype=prediction.dtype)
        if scale.shape != (prediction.shape[-1],) or torch.any(scale <= 0):
            raise ValueError("delta_scale must contain one positive value per joint")
    prediction_scaled = prediction / scale
    target_scaled = target / scale
    valid = mask.to(prediction.dtype)
    motion = target_scaled.norm(dim=-1)
    motion_weight = 0.25 + 0.75 * torch.clamp(motion / 0.02, 0.0, 1.0)
    joint_weight = torch.ones(prediction.shape[-1], device=prediction.device, dtype=prediction.dtype)
    joint_weight[-1] = 1.5
    element = F.smooth_l1_loss(prediction_scaled, target_scaled, reduction="none", beta=0.02)
    weight = valid[:, :, None] * motion_weight[:, :, None] * joint_weight[None, None, :]
    delta_loss = (element * weight).sum() / weight.sum().clamp_min(1.0)

    if prediction.shape[1] > 1:
        pair_mask = (mask[:, 1:] & mask[:, :-1]).to(prediction.dtype)
        pred_velocity = prediction_scaled[:, 1:] - prediction_scaled[:, :-1]
        target_velocity = target_scaled[:, 1:] - target_scaled[:, :-1]
        velocity_element = F.smooth_l1_loss(pred_velocity, target_velocity, reduction="none", beta=0.01)
        velocity_loss = (velocity_element * pair_mask[:, :, None]).sum() / (
            pair_mask.sum() * prediction.shape[-1]
        ).clamp_min(1.0)
    else:
        velocity_loss = prediction.sum() * 0.0
    total = delta_loss + float(velocity_weight) * velocity_loss

    zero_element = F.smooth_l1_loss(torch.zeros_like(target_scaled), target_scaled, reduction="none", beta=0.02)
    zero_baseline = (zero_element * weight).sum() / weight.sum().clamp_min(1.0)
    metrics = {
        "loss": float(total.detach()),
        "delta_loss": float(delta_loss.detach()),
        "velocity_loss": float(velocity_loss.detach()),
        "zero_delta_baseline": float(zero_baseline.detach()),
    }
    return total, metrics


def predict_delta_chunk(
    policy: DepthACTPolicy,
    top_depth_m: np.ndarray,
    side_depth_m: np.ndarray,
    joint_pos: np.ndarray,
    joint_velocity: np.ndarray,
    *,
    depth_config: DepthConfig,
    device: torch.device,
) -> np.ndarray:
    qpos = np.asarray(joint_pos, dtype=np.float32)
    qvel = np.asarray(joint_velocity, dtype=np.float32)
    if qpos.shape != (6,) or qvel.shape != (6,) or not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
        raise ValueError("joint position and velocity must be finite vectors with shape (6,)")
    top = preprocess_depth(np.asarray(top_depth_m), depth_config, augment=False)
    side = preprocess_depth(np.asarray(side_depth_m), depth_config, augment=False)
    proprio = np.concatenate([qpos, qvel]).astype(np.float32)
    with torch.inference_mode():
        prediction = policy(
            torch.from_numpy(top).unsqueeze(0).to(device),
            torch.from_numpy(side).unsqueeze(0).to(device),
            torch.from_numpy(proprio).unsqueeze(0).to(device),
        )[0]
    result = prediction.detach().cpu().numpy().astype(np.float32)
    if result.shape != (policy.config.chunk_size, 6) or not np.all(np.isfinite(result)):
        raise RuntimeError("policy produced an invalid action chunk")
    return result


class JointVelocityEstimator:
    """Estimate joint velocity exactly as the recorded dataset does."""

    def __init__(self, *, control_period_s: float, joint_count: int = 6):
        if not math.isfinite(control_period_s) or control_period_s <= 0.0:
            raise ValueError("control_period_s must be finite and positive")
        if joint_count <= 0:
            raise ValueError("joint_count must be positive")
        self.control_period_s = float(control_period_s)
        self.joint_count = int(joint_count)
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def update(self, joint_pos: np.ndarray) -> np.ndarray:
        current = np.asarray(joint_pos, dtype=np.float32)
        if current.shape != (self.joint_count,) or not np.all(np.isfinite(current)):
            raise ValueError(f"joint_pos must be finite with shape ({self.joint_count},)")
        if self._previous is None:
            velocity = np.zeros_like(current)
        else:
            velocity = (current - self._previous) / np.float32(self.control_period_s)
        self._previous = current.copy()
        return velocity.astype(np.float32, copy=False)


class TemporalActionEnsembler:
    def __init__(self, *, chunk_size: int, decay: float = 0.08):
        if chunk_size <= 0 or decay < 0.0:
            raise ValueError("invalid temporal ensemble configuration")
        self.chunk_size = int(chunk_size)
        self.decay = float(decay)
        self.time = 0
        self._predictions: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)

    def reset(self) -> None:
        self.time = 0
        self._predictions.clear()

    def _current(self) -> np.ndarray:
        candidates = self._predictions.pop(self.time, [])
        if not candidates:
            raise RuntimeError("no action prediction is available for the current control step")
        ages = np.asarray([self.time - origin for origin, _ in candidates], dtype=np.float64)
        weights = np.exp(-self.decay * ages)
        values = np.stack([value for _, value in candidates])
        return np.average(values, axis=0, weights=weights)

    def add_and_get(self, absolute_chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(absolute_chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[0] != self.chunk_size or not np.all(np.isfinite(chunk)):
            raise ValueError(f"absolute_chunk must be finite with shape ({self.chunk_size}, J)")
        for offset, action in enumerate(chunk):
            self._predictions[self.time + offset].append((self.time, action.copy()))
        result = self._current()
        self.time += 1
        return result

    def advance_without_prediction(self) -> np.ndarray:
        result = self._current()
        self.time += 1
        return result


def load_depth_act_checkpoint(path: str, *, map_location: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("architecture_version") != DepthACTPolicy.architecture_version:
        raise ValueError("checkpoint is not a compatible dual-depth ACT policy")
    config = dict(checkpoint["architecture_config"])
    config["backbone_channels"] = tuple(config["backbone_channels"])
    policy = DepthACTPolicy(**config)
    policy.load_state_dict(checkpoint["model_state_dict"])
    return policy, checkpoint
