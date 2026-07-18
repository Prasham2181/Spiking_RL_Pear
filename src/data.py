"""M3ED (and legacy MVSEC) datasets for streaming SNN training.

HDF5 layout (M3ED):
  *_data.h5:      prophesee/<side>/{x,y,t,p}   1-D event columns, t int64 us
                                               from ~0, p in {0,1} (0 = neg)
                  prophesee/<side>/ms_map_idx  (M,) event index at each ms —
                                               used for O(1) window slicing
  *_depth_gt.h5:  depth/prophesee/left         (F, 720, 1280) float32, NaN holes
                  ts                           (F,) int64 us, same timebase

M3ED files are 6-128 GB (up to 20e9 events), so unlike the MVSEC classes the
M3ED datasets never load events into RAM: each sample slices its window from
HDF5 via ms_map_idx, with file handles opened lazily per dataloader worker.
Native 1280x720 is downscaled by integer `downscale` (default 2 -> 640x360).

Both datasets return SEQUENCES of time bins (T, 2, H, W) rather than a single
window, because the model is stateful: pretraining unrolls with TBPTT, and
depth training warms the membrane state up over the whole context window.
"""
from __future__ import annotations

import os
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
    # Channel by polarity: MVSEC uses {-1, 1}, M3ED uses {0, 1} — <= 0 is negative in both.
    ch = (ev[:, 3] <= 0).astype(np.int32)
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


class _LazyH5:
    """Per-worker lazy HDF5 handle. h5py handles must not cross fork(), so the
    file is (re)opened whenever the accessing pid differs from the opener's."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._file: h5py.File | None = None
        self._pid: int | None = None

    def __call__(self) -> h5py.File:
        if self._file is None or self._pid != os.getpid():
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._file


def _m3ed_window(f: h5py.File, msmap: np.ndarray, camera_side: str,
                 t_ms: int, window_ms: int, downscale: int) -> np.ndarray:
    """Slice events in [t_ms, t_ms + window_ms) as (N, 4) float32
    [x, y, t_s_rebased, p], coordinates integer-downscaled."""
    g = f[f"prophesee/{camera_side}"]
    i0, i1 = int(msmap[t_ms]), int(msmap[t_ms + window_ms])
    ev = np.empty((i1 - i0, 4), dtype=np.float32)
    ev[:, 0] = g["x"][i0:i1] // downscale
    ev[:, 1] = g["y"][i0:i1] // downscale
    # Rebase to the window start in int64 us BEFORE the float32 cast.
    ev[:, 2] = (g["t"][i0:i1] - np.int64(t_ms) * 1000) * np.float64(1e-6)
    ev[:, 3] = g["p"][i0:i1]
    return ev


class M3EDPretrainDataset(Dataset):
    """M3ED counterpart of MVSECPretrainDataset: sliding-window sequences of
    (T_in + pred_bins) bins, sliced lazily from HDF5 via ms_map_idx."""

    def __init__(
        self,
        hdf5_paths: list[str],
        bin_ms: float = 5.0,
        T_in: int = 20,
        pred_bins: int = 5,
        stride_s: float = 0.5,
        H: int = 360,
        W: int = 640,
        camera_side: str = "left",
        downscale: int = 2,
        train_frac: float = 0.8,
        val_frac: float = 0.2,
        test_frac: float = 0.0,
        split: str = "train",
    ) -> None:
        assert split in ("train", "val", "test")
        self.bin_s = bin_ms / 1000.0
        self.n_bins = T_in + pred_bins
        self.window_ms = int(round(self.n_bins * bin_ms))
        self.H, self.W = H, W
        self.camera_side = camera_side
        self.downscale = downscale

        self.files = [_LazyH5(p) for p in hdf5_paths]
        self.msmaps: list[np.ndarray] = []
        self.samples: list[tuple[int, int]] = []  # (file_idx, start_ms)
        stride_ms = max(int(round(stride_s * 1000)), 1)
        for fi, path in enumerate(hdf5_paths):
            with h5py.File(path, "r") as f:
                msmap = f[f"prophesee/{camera_side}/ms_map_idx"][:]
            self.msmaps.append(msmap)
            starts = np.arange(0, len(msmap) - 1 - self.window_ms, stride_ms)
            lo, hi = _split_range(len(starts), split, train_frac, val_frac, test_frac)
            self.samples.extend((fi, int(t)) for t in starts[lo:hi])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        fi, t_ms = self.samples[idx]
        ev = _m3ed_window(self.files[fi](), self.msmaps[fi], self.camera_side,
                          t_ms, self.window_ms, self.downscale)
        bins = voxelize(ev, 0.0, self.bin_s, self.n_bins, self.H, self.W)
        return torch.from_numpy(bins)


class M3EDDepthDataset(Dataset):
    """One sample per GT depth frame (~10 Hz; thin with frame_stride): the
    T_in bins of events immediately preceding the frame, plus the depth map
    (downscaled to match the events) and its valid mask."""

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        bin_ms: float = 5.0,
        T_in: int = 20,
        H: int = 360,
        W: int = 640,
        camera_side: str = "left",
        downscale: int = 2,
        frame_stride: int = 1,
        max_depth: float | None = 80.0,
        min_depth: float = 1.0,
        train_frac: float = 0.8,
        val_frac: float = 0.2,
        test_frac: float = 0.0,
        split: str = "train",
    ) -> None:
        assert split in ("train", "val", "test")
        self.bin_s = bin_ms / 1000.0
        self.T_in = T_in
        self.context_ms = int(round(T_in * bin_ms))
        self.H, self.W = H, W
        self.camera_side = camera_side
        self.downscale = downscale
        self.max_depth = max_depth
        self.min_depth = min_depth

        self.data_files = [_LazyH5(a) for a, _ in pairs]
        self.gt_files = [_LazyH5(b) for _, b in pairs]
        self.msmaps: list[np.ndarray] = []
        self.samples: list[tuple[int, int, int]] = []  # (pair_idx, frame_idx, frame_ms)
        for pi, (data_path, gt_path) in enumerate(pairs):
            with h5py.File(data_path, "r") as f:
                msmap = f[f"prophesee/{camera_side}/ms_map_idx"][:]
            self.msmaps.append(msmap)
            with h5py.File(gt_path, "r") as f:
                depth_ts = f["ts"][:]
            frame_ms = depth_ts // 1000  # events and depth share the us timebase
            usable = np.where((frame_ms - self.context_ms >= 0)
                              & (frame_ms < len(msmap) - 1))[0][::frame_stride]
            lo, hi = _split_range(len(usable), split, train_frac, val_frac, test_frac)
            self.samples.extend((pi, int(k), int(frame_ms[k])) for k in usable[lo:hi])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pi, k, frame_ms = self.samples[idx]
        ev = _m3ed_window(self.data_files[pi](), self.msmaps[pi], self.camera_side,
                          frame_ms - self.context_ms, self.context_ms, self.downscale)
        bins = voxelize(ev, 0.0, self.bin_s, self.T_in, self.H, self.W)

        d = self.gt_files[pi]()["depth/prophesee/left"][k][:: self.downscale, :: self.downscale]
        depth = torch.from_numpy(d.astype(np.float32))
        valid = torch.isfinite(depth) & (depth > self.min_depth)
        if self.max_depth is not None:
            valid = valid & (depth <= self.max_depth)
        depth = torch.where(valid, depth, torch.ones_like(depth))  # keep NaN out of the graph
        return torch.from_numpy(bins), depth, valid
