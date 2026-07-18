"""Spiking building blocks with explicit, streamable state (snntorch-based).

Every cell takes (input, state) and returns (spike, membrane, new_state) so the
same code path serves both training (unrolled over T bins) and streaming
inference on-device (one bin at a time, state carried across calls).
"""
from __future__ import annotations

from dataclasses import dataclass

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate


@dataclass
class LIFState:
    mem: torch.Tensor
    spike: torch.Tensor

    def detach(self) -> "LIFState":
        return LIFState(mem=self.mem.detach(), spike=self.spike.detach())


class ConvLIF(nn.Module):
    """Convolutional LIF layer with recurrent spike feedback.

    - GroupNorm on the input current (per-sample: consistent across timesteps,
      unlike BatchNorm whose statistics mix activity levels between steps).
    - snntorch Leaky neuron with learnable per-channel decay (beta) and soft
      reset (subtract), ATan surrogate gradient.

    Returns the PRE-reset membrane alongside the spikes: spikes propagate to
    the next spiking layer, while the analog membrane is the readout for
    dense-regression heads (the StereoSpike trick — binary spike counts alone
    are too coarse for depth).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        decay_init: float = 0.9,
        threshold: float = 1.0,
        recurrent: bool = True,
    ) -> None:
        super().__init__()
        self.ff = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.rec = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False) if recurrent else None
        self.norm = nn.GroupNorm(max(1, out_channels // 8), out_channels)
        # beta shaped (C, 1, 1) so it broadcasts against (B, C, H, W) membranes
        self.lif = snn.Leaky(
            beta=torch.full((out_channels, 1, 1), decay_init),
            threshold=threshold,
            learn_beta=True,
            reset_mechanism="subtract",
            spike_grad=surrogate.atan(alpha=2.0),
        )
        self.threshold = threshold

    def forward(self, x: torch.Tensor, state: LIFState | None) -> tuple[torch.Tensor, torch.Tensor, LIFState]:
        current = self.ff(x)
        if state is None:
            zeros = torch.zeros_like(current)
            state = LIFState(mem=zeros, spike=zeros)
        if self.rec is not None:
            current = current + self.rec(state.spike)
        current = self.norm(current)

        spike, mem_post = self.lif(current, state.mem)  # mem_post is after subtract-reset
        mem_pre = mem_post + spike * self.threshold     # undo the reset for the analog readout
        return spike, mem_pre, LIFState(mem=mem_post, spike=spike)


def detach_states(states: list[LIFState | None] | None) -> list[LIFState | None] | None:
    """Cut the autograd graph at a TBPTT boundary while keeping the dynamics."""
    if states is None:
        return None
    return [s.detach() if s is not None else None for s in states]
