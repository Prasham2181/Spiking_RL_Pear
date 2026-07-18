"""Deployment wrapper for the streaming depth path.

`SpikingDepthModel.step()` carries state as a list of LIFState dataclasses,
which is convenient for training but opaque to ONNX/TensorRT. This wrapper
flattens the state into plain tensors so the whole streaming update is one
traceable graph:

    depth, *state = net(event_bin, *state)

On the quadrotor this is the entire per-frame workload: one event bin
(bin_ms of events) in, one metric depth map + updated state out.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cells import LIFState
from .models import SpikingDepthModel


class StreamingDepthNet(nn.Module):
    def __init__(self, model: SpikingDepthModel) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.depth_head = model.depth_head
        self.count_clip = model.count_clip
        self.n_stages = len(model.encoder.stages)

    def forward(self, x_t: torch.Tensor, *state: torch.Tensor):
        """x_t: (B, 2, H, W) event counts; state: 2 tensors (mem, spike) per stage."""
        states = None
        if state:
            states = [LIFState(mem=state[2 * i], spike=state[2 * i + 1]) for i in range(self.n_stages)]
        x = torch.clamp(x_t, 0.0, self.count_clip)
        feats, new_states, _ = self.encoder.step(x, states)
        depth = self.depth_head(feats)
        flat: list[torch.Tensor] = []
        for s in new_states:
            flat.extend([s.mem, s.spike])
        return (depth, *flat)

    @torch.no_grad()
    def init_state(self, batch: int, H: int, W: int, device="cpu") -> tuple[torch.Tensor, ...]:
        """Zero state tensors with the correct per-stage shapes (found by a dry run)."""
        dummy = torch.zeros(batch, 2, H, W, device=device)
        _, states, _ = self.encoder.step(dummy, None)
        return tuple(torch.zeros_like(t) for s in states for t in (s.mem, s.spike))
