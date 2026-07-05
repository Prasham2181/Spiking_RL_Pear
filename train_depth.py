"""Depth training on MVSEC LiDAR frames.

Usage:
    # spiking model from a pretrained encoder (frozen probe)
    python train_depth.py --conf configs/depth.yaml --encoder-ckpt checkpoints/pretrain/best.pt

    # fine-tune the encoder end-to-end
    python train_depth.py --conf configs/depth.yaml --encoder-ckpt checkpoints/pretrain/best.pt --finetune

    # from scratch (no pretraining)
    python train_depth.py --conf configs/depth.yaml

    # honest CNN control on the same data
    python train_depth.py --conf configs/depth.yaml --baseline
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data import MVSECDepthDataset, resolve_pairs
from src.losses import depth_metric_sums, silog_loss
from src.models import SpikingDepthModel, UNetDepthBaseline


def run_epoch(model, loader, conf, device, optimizer=None, epoch=0, is_baseline=False):
    training = optimizer is not None
    model.train(training)
    lam = conf["training"]["silog_lambda"]
    log_interval = conf["training"].get("log_interval", 50)

    total_loss = 0.0
    sums = {"abs_rel": 0.0, "sq_err": 0.0, "d1": 0.0, "n": 0.0}
    for step, (bins, depth, valid) in enumerate(loader, start=1):
        bins = bins.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            pred = model(bins) if is_baseline else model(bins)[0]
            loss = silog_loss(pred, depth, valid, lam)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.detach())
        for k, v in depth_metric_sums(pred.detach(), depth, valid).items():
            sums[k] += v
        if training and step % log_interval == 0:
            print(f"epoch {epoch} step {step}/{len(loader)} loss {total_loss / step:.4f}", flush=True)

    n_px = max(sums["n"], 1.0)
    return {
        "loss": total_loss / max(len(loader), 1),
        "abs_rel": sums["abs_rel"] / n_px,
        "rmse": (sums["sq_err"] / n_px) ** 0.5,
        "d1": sums["d1"] / n_px,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="configs/depth.yaml")
    parser.add_argument("--encoder-ckpt", default=None, help="pretrained SpikingFutureModel checkpoint")
    parser.add_argument("--finetune", action="store_true", help="train the encoder too (lower lr)")
    parser.add_argument("--baseline", action="store_true", help="train the U-Net control instead")
    parser.add_argument("--tag", default=None, help="checkpoint subdirectory name")
    args = parser.parse_args()
    with open(args.conf) as f:
        conf = yaml.safe_load(f)

    device = torch.device(conf["training"]["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(conf["training"].get("seed", 403))

    d = conf["data"]
    common = dict(
        pairs=resolve_pairs(d), bin_ms=d["bin_ms"], T_in=d["T_in"],
        H=d["H"], W=d["W"], camera_side=d["camera_side"],
        max_depth=d["max_depth"], min_depth=d["min_depth"],
        train_frac=d["train_frac"], val_frac=d["val_frac"], test_frac=d["test_frac"],
    )
    train_ds = MVSECDepthDataset(**common, split="train")
    val_ds = MVSECDepthDataset(**common, split="val")
    print(f"train samples: {len(train_ds)}  val samples: {len(val_ds)}")

    t = conf["training"]
    m = conf["model"]
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                              num_workers=t["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False,
                            num_workers=t["num_workers"], pin_memory=True)

    if args.baseline:
        model = UNetDepthBaseline(
            input_bins=d["T_in"], channels=tuple(m["stage_channels"]),
            depth_min=m["depth_min"], depth_max=m["depth_max"], count_clip=m["count_clip"],
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 1e-4))
        tag = args.tag or "baseline_unet"
    else:
        freeze = args.encoder_ckpt is not None and not args.finetune
        model = SpikingDepthModel(
            stage_channels=tuple(m["stage_channels"]), fused_channels=m["fused_channels"],
            decay_init=m["decay_init"], threshold=m["threshold"], count_clip=m["count_clip"],
            depth_min=m["depth_min"], depth_max=m["depth_max"], freeze_encoder=freeze,
        ).to(device)
        if args.encoder_ckpt:
            model.load_encoder(args.encoder_ckpt, device=str(device))
            print(f"loaded encoder from {args.encoder_ckpt} (frozen={freeze})")
        if args.finetune and args.encoder_ckpt:
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.encoder.parameters(), "lr": t["finetune_lr"]},
                    {"params": model.depth_head.parameters(), "lr": t["lr"]},
                ],
                weight_decay=t.get("weight_decay", 1e-4),
            )
            tag = args.tag or "snn_finetune"
        else:
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=t["lr"], weight_decay=t.get("weight_decay", 1e-4),
            )
            tag = args.tag or ("snn_frozen_probe" if freeze else "snn_scratch")

    print(f"run: {tag}  params: {sum(p.numel() for p in model.parameters()):,}")
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=t.get("lr_end_factor", 0.1), total_iters=t["epochs"]
    )

    ckpt_dir = Path(t["checkpoint_dir"]) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir / "metrics.jsonl"
    best_val = float("inf")

    for epoch in range(t["epochs"]):
        tic = time.time()
        train_m = run_epoch(model, train_loader, conf, device, optimizer, epoch, args.baseline)
        val_m = run_epoch(model, val_loader, conf, device, is_baseline=args.baseline)
        scheduler.step()
        print(
            f"epoch {epoch} done in {time.time() - tic:.0f}s  "
            f"train loss {train_m['loss']:.4f} | val loss {val_m['loss']:.4f} "
            f"abs_rel {val_m['abs_rel']:.4f} rmse {val_m['rmse']:.3f} d1 {val_m['d1']:.4f}",
            flush=True,
        )
        with metrics_path.open("a") as f:
            f.write(json.dumps({"epoch": epoch, "train": train_m, "val": val_m,
                                "lr": scheduler.get_last_lr()[0]}) + "\n")
        payload = {"epoch": epoch, "model": model.state_dict(), "conf": conf, "tag": tag}
        torch.save(payload, ckpt_dir / "last.pt")
        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            torch.save(payload, ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
