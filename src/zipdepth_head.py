"""ZipDepth (ECCV'26, 6.1M params) as the depth head over the SNN representation.

Pipeline: events -> MultiScaleSpikingEncoder -> /1../8 membrane features
-> TopDownFusion (full-res fused map) -> ZipDepth -> metric depth.

ZipDepth's 3-ch RGB stem is replaced with a stem matching the fused feature
channels; every other weight can be initialized from the distilled
zipdepth_base.pth. Drop-in replacement for heads.DepthHead:
forward(feats) -> (B, H, W) depth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .heads import TopDownFusion

ZIPDEPTH_ROOT = Path(__file__).resolve().parents[2] / "ZipDepth"
if str(ZIPDEPTH_ROOT) not in sys.path:
    sys.path.insert(0, str(ZIPDEPTH_ROOT))

from zipdepth.model.architecture import ConvBN, ZipDepth  # noqa: E402


class ZipDepthHead(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, ...],
        fused_channels: int = 64,
        depth_min: float = 1.0,
        depth_max: float = 80.0,
        variant: str = "base",
        pretrained_ckpt: str | None = None,
    ) -> None:
        super().__init__()
        self.depth_min = depth_min
        self.fuse = TopDownFusion(in_channels, fused_channels)
        self.net = ZipDepth(variant=variant)

        if pretrained_ckpt:
            state = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
            state = {k: v for k, v in state.items()
                     if not k.startswith("encoder.stem_half.") and k not in ("mean", "std")}
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            leftover = [k for k in missing if not k.startswith("encoder.stem_half.") and k not in ("mean", "std")]
            if leftover or unexpected:
                raise RuntimeError(f"ZipDepth load mismatch: missing={leftover} unexpected={unexpected}")

        # Replace the 3-ch RGB stem with one matching the fused feature map,
        # and neutralize the ImageNet normalization buffers.
        stem_out = self.net.encoder.stem_half.conv.out_channels
        self.net.encoder.stem_half = ConvBN(fused_channels, stem_out, k=3, s=2)
        self.net.mean = torch.zeros(1, fused_channels, 1, 1)
        self.net.std = torch.ones(1, fused_channels, 1, 1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = self.fuse(feats)
        # ZipDepth output is ReLU'd (>= 0); shift by depth_min so SiLog's
        # log(pred) is always defined. Upper bound is learned, not clamped.
        return self.depth_min + self.net(x).squeeze(1)
