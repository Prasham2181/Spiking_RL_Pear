"""Continuous streaming depth over a real MVSEC sequence — the quadrotor mode.

Warms the membrane state up, then feeds one bin_ms event slice at a time:
every step yields a full depth map (constant compute). Reports per-step
latency, compares against each LiDAR GT frame inside the window, and writes a
GIF of the depth stream.

Usage:
    python stream_depth.py --ckpt checkpoints/depth/snn_finetune/best.pt \
        --data mvsec/indoor_flying2_data-002.hdf5 --gt mvsec/indoor_flying2_gt-003.hdf5 \
        --t-start 20 --duration 5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

from eval_depth import build_model
from src.data import _load_events, voxelize
from src.losses import depth_metric_sums


def colorize(depth: np.ndarray, depth_max: float) -> np.ndarray:
    x = np.clip(depth / depth_max, 0.0, 1.0)
    return (cm.magma_r(x)[..., :3] * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True, help="MVSEC *_data.hdf5 (events)")
    parser.add_argument("--gt", default=None, help="MVSEC *_gt.hdf5 (optional, for metrics)")
    parser.add_argument("--t-start", type=float, default=20.0, help="seconds into the recording")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds to stream")
    parser.add_argument("--gif", default=None, help="output GIF path (default: alongside ckpt)")
    parser.add_argument("--gif-stride", type=int, default=4, help="write every Nth bin to the GIF")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, conf, is_baseline = build_model(payload, device)
    if is_baseline:
        raise SystemExit("streaming mode needs the spiking model (the U-Net baseline has no state)")
    d, m = conf["data"], conf["model"]
    H, W, bin_s = d["H"], d["W"], d["bin_ms"] / 1000.0
    warmup_bins = d["T_in"]

    events, ts, t0 = _load_events(args.data, d["camera_side"])
    gt_depth = gt_ts = None
    if args.gt:
        with h5py.File(args.gt, "r") as f:
            gt_depth = f[f"davis/{d['camera_side']}/depth_image_rect"][:]
            gt_ts = f[f"davis/{d['camera_side']}/depth_image_rect_ts"][:] - t0

    def bin_at(t: float) -> torch.Tensor:
        i0 = int(np.searchsorted(ts, np.float32(t), side="left"))
        i1 = int(np.searchsorted(ts, np.float32(t + bin_s), side="left"))
        vol = voxelize(events[i0:i1], t, bin_s, 1, H, W)[0]  # (2, H, W)
        return torch.from_numpy(vol).unsqueeze(0).to(device)

    # ---- warm up state on the T_in bins before t_start
    states = None
    t = args.t_start - warmup_bins * bin_s
    with torch.no_grad():
        for _ in range(warmup_bins):
            _, states = model.step(bin_at(t), states)
            t += bin_s

    # ---- stream
    n_bins = int(args.duration / bin_s)
    frames: list[np.ndarray] = []
    latencies: list[float] = []
    sums = {"abs_rel": 0.0, "sq_err": 0.0, "d1": 0.0, "n": 0.0}
    n_gt = 0
    with torch.no_grad():
        for k in range(n_bins):
            x_t = bin_at(t)
            tic = time.perf_counter()
            depth, states = model.step(x_t, states)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - tic) * 1000.0)

            if gt_depth is not None:
                # GT frames whose timestamp falls inside this bin
                for j in np.where((gt_ts >= t) & (gt_ts < t + bin_s))[0]:
                    target = torch.from_numpy(gt_depth[j].astype(np.float32)).to(device)
                    valid = torch.isfinite(target) & (target > d["min_depth"]) & (target <= d["max_depth"])
                    target = torch.where(valid, target, torch.ones_like(target))
                    for kk, v in depth_metric_sums(depth[0], target, valid).items():
                        sums[kk] += v
                    n_gt += 1
            if k % args.gif_stride == 0:
                frames.append(colorize(depth[0].cpu().numpy(), m["depth_max"]))
            t += bin_s

    lat = sorted(latencies)
    print(f"streamed {n_bins} bins ({args.duration:.1f}s of events) on {device}")
    print(f"latency per update: mean {sum(lat) / len(lat):.2f} ms   median {lat[len(lat) // 2]:.2f} ms   "
          f"p95 {lat[int(0.95 * len(lat))]:.2f} ms   -> {1000.0 * len(lat) / sum(lat):.0f} Hz")
    if n_gt:
        n_px = max(sums["n"], 1.0)
        print(f"vs {n_gt} LiDAR frames: abs_rel {sums['abs_rel'] / n_px:.4f}   "
              f"rmse {(sums['sq_err'] / n_px) ** 0.5:.3f} m   d1 {sums['d1'] / n_px:.4f}")

    gif_path = Path(args.gif) if args.gif else Path(args.ckpt).parent / "stream.gif"
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 duration=int(bin_s * args.gif_stride * 1000), loop=0)
    print(f"depth stream: {gif_path} ({len(imgs)} frames)")


if __name__ == "__main__":
    main()
