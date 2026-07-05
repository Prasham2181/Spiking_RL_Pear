"""Full models: future-event pretraining, depth, and the honest CNN baseline."""
from __future__ import annotations

import torch
import torch.nn as nn

from .cells import LIFState
from .encoder import MultiScaleSpikingEncoder
from .heads import DepthHead, EventHead
from .losses import voxel_focal_loss


class SpikingFutureModel(nn.Module):
    """Pretraining: at every step t, predict occupancy of the next K bins.

    Trained with TBPTT: the driver loop calls forward_chunk() over consecutive
    step ranges of the same sequence, backpropagating per chunk and detaching
    state between chunks.
    """

    def __init__(
        self,
        stage_channels: tuple[int, ...] = (32, 64, 96, 128),
        fused_channels: int = 64,
        pred_bins: int = 5,
        decay_init: float = 0.9,
        threshold: float = 1.0,
        count_clip: float = 4.0,
        spike_reg_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.encoder = MultiScaleSpikingEncoder(
            in_channels=2, stage_channels=stage_channels, decay_init=decay_init, threshold=threshold
        )
        self.event_head = EventHead(self.encoder.stage_channels, fused_channels, pred_bins)
        self.pred_bins = pred_bins
        self.count_clip = count_clip
        self.spike_reg_weight = spike_reg_weight

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Dense scenes reach 50+ counts/pixel/bin, which saturates LIF neurons.
        return torch.clamp(x, 0.0, self.count_clip)

    def forward_chunk(
        self,
        bins: torch.Tensor,
        t_start: int,
        t_end: int,
        states: list[LIFState | None] | None,
        warmup: int = 0,
    ) -> tuple[torch.Tensor | None, dict[str, float], list[LIFState]]:
        """bins: (B, T_in + K, 2, H, W) raw event counts.

        Steps through bins[:, t_start:t_end]; for each step t >= warmup the
        target is the binarized occupancy of bins[:, t+1 : t+1+K].
        """
        K = self.pred_bins
        losses = []
        rate_sum = 0.0
        tp = fp = fn = 0.0
        for t in range(t_start, t_end):
            feats, states, spike_rate = self.encoder.step(self._normalize(bins[:, t]), states)
            rate_sum += float(spike_rate.detach())
            if t < warmup:
                continue
            logits = self.event_head(feats)
            target = (bins[:, t + 1 : t + 1 + K] > 0).to(logits.dtype)
            losses.append(voxel_focal_loss(logits, target) + self.spike_reg_weight * spike_rate)
            with torch.no_grad():
                pred = logits > 0
                gt = target > 0.5
                tp += float((pred & gt).sum())
                fp += float((pred & ~gt).sum())
                fn += float((~pred & gt).sum())

        loss = torch.stack(losses).mean() if losses else None
        f1 = 2 * tp / max(2 * tp + fp + fn, 1.0)
        metrics = {"f1": f1, "spike_rate": rate_sum / max(t_end - t_start, 1)}
        return loss, metrics, states


class SpikingDepthModel(nn.Module):
    """Pretrained spiking encoder + depth head over multi-scale membrane features.

    forward() runs a whole context window (training / cold start). step() is
    the streaming path: one new time bin -> updated depth, constant compute.
    """

    def __init__(
        self,
        stage_channels: tuple[int, ...] = (32, 64, 96, 128),
        fused_channels: int = 64,
        decay_init: float = 0.9,
        threshold: float = 1.0,
        count_clip: float = 4.0,
        depth_min: float = 0.5,
        depth_max: float = 80.0,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = MultiScaleSpikingEncoder(
            in_channels=2, stage_channels=stage_channels, decay_init=decay_init, threshold=threshold
        )
        self.depth_head = DepthHead(self.encoder.stage_channels, fused_channels, depth_min, depth_max)
        self.count_clip = count_clip
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def load_encoder(self, ckpt_path: str, device: str = "cpu") -> None:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)["model"]
        enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(enc_state, strict=True)

    def train(self, mode: bool = True) -> "SpikingDepthModel":
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode(self, bins: torch.Tensor, states):
        x = torch.clamp(bins, 0.0, self.count_clip)
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(x, states)
        return self.encoder(x, states)

    def forward(self, bins: torch.Tensor, states=None) -> tuple[torch.Tensor, list[LIFState]]:
        """bins: (B, T, 2, H, W) -> depth (B, H, W) from the final step's features."""
        feats, states, _ = self._encode(bins, states)
        return self.depth_head(feats), states

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, states) -> tuple[torch.Tensor, list[LIFState]]:
        """Streaming inference: one bin (B, 2, H, W) -> depth (B, H, W)."""
        x = torch.clamp(x_t, 0.0, self.count_clip)
        feats, states, _ = self.encoder.step(x, states)
        return self.depth_head(feats), states


def _double_conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.GELU(),
    )


class UNetDepthBaseline(nn.Module):
    """Control experiment: plain U-Net on the flattened voxel grid.

    Same data, same loss, same output bounds — no spiking, no recurrence. If
    the spiking pipeline can't beat this, the representation isn't earning its
    compute.
    """

    def __init__(
        self,
        input_bins: int,
        channels: tuple[int, ...] = (32, 64, 96, 128),
        depth_min: float = 0.5,
        depth_max: float = 80.0,
        count_clip: float = 4.0,
    ) -> None:
        super().__init__()
        import math

        self.log_min = math.log(depth_min)
        self.log_max = math.log(depth_max)
        self.count_clip = count_clip

        self.enc = nn.ModuleList()
        cin = input_bins * 2
        for c in channels:
            self.enc.append(_double_conv(cin, c))
            cin = c
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.bottleneck = _double_conv(channels[-1], channels[-1] * 2)
        self.dec = nn.ModuleList()
        cin = channels[-1] * 2
        for c in reversed(channels):
            self.dec.append(_double_conv(cin + c, c))
            cin = c
        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, bins: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        b, t, c, h, w = bins.shape
        x = torch.clamp(bins, 0.0, self.count_clip).view(b, t * c, h, w)
        skips = []
        for layer in self.enc:
            x = layer(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for layer, skip in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = layer(torch.cat([x, skip], dim=1))
        log_depth = self.head(x).squeeze(1)
        log_depth = self.log_min + (self.log_max - self.log_min) * torch.sigmoid(log_depth)
        return torch.exp(log_depth)
