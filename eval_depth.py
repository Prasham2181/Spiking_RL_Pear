"""Evaluate a trained depth checkpoint: metrics on a split + sample visualizations.

Usage:
    python eval_depth.py --ckpt checkpoints/depth/snn_finetune/best.pt
    python eval_depth.py --ckpt checkpoints/depth/baseline_unet/best.pt --split val --viz 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import M3EDDepthDataset, resolve_pairs
from src.losses import depth_metric_sums
from src.models import SpikingDepthModel, UNetDepthBaseline


def build_model(payload: dict, device: torch.device):
    conf = payload["conf"]
    m, d = conf["model"], conf["data"]
    is_baseline = payload.get("tag") == "baseline_unet" or any(
        k.startswith("bottleneck") for k in payload["model"]
    )
    if is_baseline:
        model = UNetDepthBaseline(
            input_bins=d["T_in"], channels=tuple(m["stage_channels"]),
            depth_min=m["depth_min"], depth_max=m["depth_max"], count_clip=m["count_clip"],
        )
    else:
        model = SpikingDepthModel(
            stage_channels=tuple(m["stage_channels"]), fused_channels=m["fused_channels"],
            decay_init=m["decay_init"], threshold=m["threshold"], count_clip=m["count_clip"],
            depth_min=m["depth_min"], depth_max=m["depth_max"],
        )
        if payload.get("tag", "").endswith("_zipdepth"):
            from src.zipdepth_head import ZipDepthHead

            model.depth_head = ZipDepthHead(
                in_channels=model.encoder.stage_channels, fused_channels=m["fused_channels"],
                depth_min=m["depth_min"], depth_max=m["depth_max"],
            )
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), conf, is_baseline


def save_viz(bins, pred, target, valid, out_path: Path, depth_max: float) -> None:
    events = bins.sum(dim=(0, 1)).cpu().numpy()  # total event count image
    pred = pred.cpu().numpy()
    gt = np.where(valid.cpu().numpy(), target.cpu().numpy(), np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(events, cmap="gray_r")
    axes[0].set_title("events (context window)")
    im1 = axes[1].imshow(pred, cmap="magma_r", vmin=0, vmax=depth_max)
    axes[1].set_title("predicted depth [m]")
    im2 = axes[2].imshow(gt, cmap="magma_r", vmin=0, vmax=depth_max)
    axes[2].set_title("LiDAR GT (sparse)")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--viz", type=int, default=8, help="number of sample visualizations to save")
    parser.add_argument("--out", default=None, help="output dir (default: alongside the checkpoint)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, conf, is_baseline = build_model(payload, device)
    d = conf["data"]
    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = M3EDDepthDataset(
        pairs=resolve_pairs(d), bin_ms=d["bin_ms"], T_in=d["T_in"],
        H=d["H"], W=d["W"], camera_side=d["camera_side"],
        downscale=d.get("downscale", 2), frame_stride=d.get("frame_stride", 1),
        max_depth=d["max_depth"], min_depth=d["min_depth"],
        train_frac=d["train_frac"], val_frac=d["val_frac"], test_frac=d["test_frac"],
        split=args.split,
    )
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)
    print(f"{args.split} samples: {len(ds)}  model: {'baseline U-Net' if is_baseline else 'spiking'}")

    sums = {"abs_rel": 0.0, "sq_err": 0.0, "d1": 0.0, "n": 0.0}
    n_saved = 0
    with torch.no_grad():
        for bins, depth, valid in loader:
            bins = bins.to(device)
            depth = depth.to(device)
            valid = valid.to(device)
            pred = model(bins) if is_baseline else model(bins)[0]
            for k, v in depth_metric_sums(pred, depth, valid).items():
                sums[k] += v
            while n_saved < args.viz and n_saved < len(bins):
                i = n_saved
                save_viz(bins[i], pred[i], depth[i], valid[i],
                         out_dir / f"sample_{n_saved:03d}.png", conf["model"]["depth_max"])
                n_saved += 1

    n_px = max(sums["n"], 1.0)
    print(f"abs_rel {sums['abs_rel'] / n_px:.4f}   "
          f"rmse {(sums['sq_err'] / n_px) ** 0.5:.3f} m   "
          f"d1 {sums['d1'] / n_px:.4f}   ({int(n_px):,} valid px)")
    print(f"visualizations: {out_dir}")


if __name__ == "__main__":
    main()
