"""Non-spiking heads that consume the encoder's multi-scale membrane features."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_gn_gelu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(max(1, cout // 8), cout),
        nn.GELU(),
    )


class TopDownFusion(nn.Module):
    """FPN-lite: 1x1 laterals + top-down upsample-add + 3x3 smoothing.

    Fuses the encoder's /1../8 membrane features into one full-resolution map.
    Input sizes are MVSEC-odd (260x346), so upsampling resizes to each lateral's
    exact shape instead of assuming clean /2 factors.
    """

    def __init__(self, in_channels: tuple[int, ...], out_channels: int) -> None:
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.smooth = nn.ModuleList([_conv_gn_gelu(out_channels, out_channels) for _ in in_channels[:-1]])

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = self.laterals[-1](feats[-1])
        for i in reversed(range(len(feats) - 1)):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="nearest")
            x = x + self.laterals[i](feats[i])
            x = self.smooth[i](x)
        return x


class EventHead(nn.Module):
    """Future-event prediction head (F3-style pretraining objective).

    Predicts occupancy logits for the next `pred_bins` time bins, per polarity,
    at full resolution: (B, K, 2, H, W).
    """

    def __init__(self, in_channels: tuple[int, ...], fused_channels: int = 64, pred_bins: int = 5) -> None:
        super().__init__()
        self.pred_bins = pred_bins
        self.fuse = TopDownFusion(in_channels, fused_channels)
        self.head = nn.Sequential(
            _conv_gn_gelu(fused_channels, fused_channels),
            nn.Conv2d(fused_channels, pred_bins * 2, 1),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        logits = self.head(self.fuse(feats))
        b, _, h, w = logits.shape
        return logits.view(b, self.pred_bins, 2, h, w)


class DepthHead(nn.Module):
    """Metric depth head, bounded to [depth_min, depth_max] via sigmoid in log
    space (stable under SiLog loss)."""

    def __init__(
        self,
        in_channels: tuple[int, ...],
        fused_channels: int = 64,
        depth_min: float = 0.5,
        depth_max: float = 80.0,
    ) -> None:
        super().__init__()
        self.log_min = math.log(depth_min)
        self.log_max = math.log(depth_max)
        self.fuse = TopDownFusion(in_channels, fused_channels)
        self.refine = nn.Sequential(
            _conv_gn_gelu(fused_channels, fused_channels),
            _conv_gn_gelu(fused_channels, fused_channels),
        )
        self.head = nn.Conv2d(fused_channels, 1, 1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        log_depth = self.head(self.refine(self.fuse(feats))).squeeze(1)
        log_depth = self.log_min + (self.log_max - self.log_min) * torch.sigmoid(log_depth)
        return torch.exp(log_depth)
