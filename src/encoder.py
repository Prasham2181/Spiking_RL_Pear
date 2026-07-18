"""Multi-scale spiking encoder with streaming state.

Fixes the two structural problems shared by the earlier Not_optimised and
SpikingEvents encoders:

1. Spatial hierarchy: stages run at /1, /2, /4, /8 (strided ConvLIF), so the
   representation carries the global context dense depth needs — instead of a
   single full-resolution feature map with a ~15px receptive field.
2. Analog readout: each stage exposes its pre-reset membrane potential as the
   feature for downstream heads. Spikes (binary) still drive stage-to-stage
   propagation, but heads never see bare spike counts.

The encoder is stateful: `step()` consumes ONE time bin and returns features +
new state. At deployment each new event slice costs a single step (constant
compute), which is the actual latency argument for SNNs on a Jetson.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cells import ConvLIF, LIFState


class MultiScaleSpikingEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        stage_channels: tuple[int, ...] = (32, 64, 96, 128),
        decay_init: float = 0.9,
        threshold: float = 1.0,
        recurrent: bool = True,
    ) -> None:
        super().__init__()
        channels = [in_channels, *stage_channels]
        strides = [1] + [2] * (len(stage_channels) - 1)
        self.stages = nn.ModuleList(
            [
                ConvLIF(
                    channels[i],
                    channels[i + 1],
                    stride=strides[i],
                    decay_init=decay_init,
                    threshold=threshold,
                    recurrent=recurrent,
                )
                for i in range(len(stage_channels))
            ]
        )
        self.stage_channels = tuple(stage_channels)

    def step(
        self, x_t: torch.Tensor, states: list[LIFState | None] | None
    ) -> tuple[list[torch.Tensor], list[LIFState], torch.Tensor]:
        """One time bin. x_t: (B, C, H, W).

        Returns (features fine->coarse, new states, mean spike rate).
        Features are pre-reset membranes: analog, one per scale.
        """
        if states is None:
            states = [None] * len(self.stages)
        feats: list[torch.Tensor] = []
        new_states: list[LIFState] = []
        spike_rates = []
        x = x_t
        for stage, state in zip(self.stages, states):
            spike, mem, new_state = stage(x, state)
            feats.append(mem)
            new_states.append(new_state)
            spike_rates.append(spike.mean())
            x = spike
        return feats, new_states, torch.stack(spike_rates).mean()

    def forward(
        self, bins: torch.Tensor, states: list[LIFState | None] | None = None
    ) -> tuple[list[torch.Tensor], list[LIFState], torch.Tensor]:
        """Convenience: run a whole (B, T, C, H, W) sequence, return final-step
        features. Training loops that need per-step outputs should call step().
        """
        feats: list[torch.Tensor] = []
        rates = []
        for t in range(bins.shape[1]):
            feats, states, rate = self.step(bins[:, t], states)
            rates.append(rate)
        return feats, states, torch.stack(rates).mean()
