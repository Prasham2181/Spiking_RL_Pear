"""MVSEC datasets for streaming SNN training.

HDF5 layout (MVSEC):
  *_data.hdf5:  davis/<side>/events            (N, 4) [x, y, t, p]
  *_gt.hdf5:    davis/<side>/depth_image_rect  (F, H, W) float, NaN = no LiDAR
                davis/<side>/depth_image_rect_ts (F,)

Both datasets return SEQUENCES of time bins (T, 2, H, W) rather than a single
window, because the model is stateful: pretraining unrolls with TBPTT, and
depth training warms the membrane state up over the whole context window.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_paths(data_conf: dict) -> list[str]:
    """Join the config's `root` with its relative `paths` (pretraining files)."""
    root = Path(data_conf.get("root", "."))
    return [str(root / p) for p in data_conf["paths"]]


def resolve_pairs(data_conf: dict) -> list[tuple[str, str]]:
    """Join the config's `root` with its relative (data, gt) `pairs`."""
    root = Path(data_conf.get("root", "."))
    return [(str(root / a), str(root / b)) for a, b in data_conf["pairs"]]


def voxelize(ev: np.ndarray, t_start: float, bin_s: float, n_bins: int, H: int, W: int) -> np.ndarray:
    """Bin an event slice into (n_bins, 2, H, W) counts. ev: (N,4) [x,y,t,p].
    Channel 0 = positive polarity, channel 1 = negative."""
    volume = np.zeros((n_bins, 2, H, W), dtype=np.float32)
    if len(ev) == 0:
        return volume
    x = ev[:, 0].astype(np.int32)
    y = ev[:, 1].astype(np.int32)
    bin_idx = np.clip(((ev[:, 2] - t_start) / bin_s).astype(np.int32), 0, n_bins - 1)
    ch = (ev[:, 3] < 0).astype(np.int32)
    np.add.at(volume, (bin_idx, ch, y, x), 1.0)
    return volume


def _load_events(path: str, camera_side: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Load events, rebased to t=0. Returns (events_f32, ts_contiguous, t0).

    MVSEC timestamps are epoch-scale (~1.5e9); float32 spacing there is 128s,
    so rebase in float64 FIRST, then cast.
    """
    with h5py.File(path, "r") as f:
        events = f[f"davis/{camera_side}/events"][:]
    t0 = float(events[0, 2])
    events[:, 2] -= t0
    events = events.astype(np.float32)
    # Contiguous copy: searchsorted on a strided column view copies every call.
    ts = np.ascontiguousarray(events[:, 2])
    return events, ts, t0


def _split_range(n: int, split: str, train_frac: float, val_frac: float, test_frac: float) -> tuple[int, int]:
    """Time-ordered, non-overlapping index ranges (no temporal leakage)."""
    starts = {"train": 0.0, "val": train_frac, "test": train_frac + val_frac}
    fracs = {"train": train_frac, "val": val_frac, "test": test_frac}
    lo = int(starts[split] * n)
    return lo, lo + int(fracs[split] * n)


class MVSECPretrainDataset(Dataset):
    """Sliding-window sequences for future-event pretraining.

    Each sample is (T_in + pred_bins) consecutive bins of `bin_ms` each; the
    model steps through the first T_in and, at each step t, predicts occupancy
    of bins t+1 .. t+pred_bins.
    """

    def __init__(
        self,
        hdf5_paths: list[str],
        bin_ms: float = 5.0,
        T_in: int = 20,
        pred_bins: int = 5,
        stride_s: float = 0.05,
        H: int = 260,
        W: int = 346,
        camera_side: str = "left",
        train_frac: float = 0.8,
        val_frac: float = 0.2,
        test_frac: float = 0.0,
        split: str = "train",
    ) -> None:
        assert split in ("train", "val", "test")
        self.bin_s = bin_ms / 1000.0
        self.n_bins = T_in + pred_bins
        self.H, self.W = H, W

        window_s = self.n_bins * self.bin_s
        self.samples: list[tuple[np.ndarray, np.ndarray, float]] = []
        for path in hdf5_paths:
            events, ts, _ = _load_events(path, camera_side)
            t_min, t_max = float(ts[0]), float(ts[-1])
            usable = (t_max - window_s) - t_min
            starts = np.arange(t_min, t_min + usable, stride_s)
            lo, hi = _split_range(len(starts), split, train_frac, val_frac, test_frac)
            # All entries share references to the SAME arrays — no copies.
            for t in starts[lo:hi]:
                self.samples.append((events, ts, float(t)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        events, ts, t_start = self.samples[idx]
        # float32 queries: a float64 query upcasts the entire ts array.
        i0 = int(np.searchsorted(ts, np.float32(t_start), side="left"))
        i1 = int(np.searchsorted(ts, np.float32(t_start + self.n_bins * self.bin_s), side="left"))
        bins = voxelize(events[i0:i1], t_start, self.bin_s, self.n_bins, self.H, self.W)
        return torch.from_numpy(bins)


class MVSECDepthDataset(Dataset):
    """One sample per LiDAR depth frame: the T_in bins of events immediately
    preceding the frame, plus the depth map and its valid mask."""

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        bin_ms: float = 5.0,
        T_in: int = 20,
        H: int = 260,
        W: int = 346,
        camera_side: str = "left",
        max_depth: float | None = 80.0,
        min_depth: float = 0.1,
        train_frac: float = 0.8,
        val_frac: float = 0.2,
        test_frac: float = 0.0,
        split: str = "train",
    ) -> None:
        assert split in ("train", "val", "test")
        self.bin_s = bin_ms / 1000.0
        self.T_in = T_in
        self.H, self.W = H, W
        self.max_depth = max_depth
        self.min_depth = min_depth

        context_s = T_in * self.bin_s
        self.samples: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
        for data_path, gt_path in pairs:
            events, ts, t0 = _load_events(data_path, camera_side)
            with h5py.File(gt_path, "r") as f:
                depth = f[f"davis/{camera_side}/depth_image_rect"][:]
                depth_ts = f[f"davis/{camera_side}/depth_image_rect_ts"][:] - t0

            usable = np.where(depth_ts - context_s >= ts[0])[0]
            lo, hi = _split_range(len(usable), split, train_frac, val_frac, test_frac)
            for k in usable[lo:hi]:
                self.samples.append((events, ts, float(depth_ts[k]) - context_s, depth[k]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        events, ts, t_start, depth = self.samples[idx]
        i0 = int(np.searchsorted(ts, np.float32(t_start), side="left"))
        i1 = int(np.searchsorted(ts, np.float32(t_start + self.T_in * self.bin_s), side="left"))
        bins = voxelize(events[i0:i1], t_start, self.bin_s, self.T_in, self.H, self.W)

        depth = torch.from_numpy(depth.astype(np.float32))
        valid = torch.isfinite(depth) & (depth > self.min_depth)
        if self.max_depth is not None:
            valid = valid & (depth <= self.max_depth)
        depth = torch.where(valid, depth, torch.ones_like(depth))  # keep NaN out of the graph
        return torch.from_numpy(bins), depth, valid
