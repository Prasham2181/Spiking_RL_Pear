"""Future-event pretraining with TBPTT.

Usage:
    python train_pretrain.py --conf configs/pretrain.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.cells import detach_states
from src.data import M3EDPretrainDataset, resolve_paths
from src.models import SpikingFutureModel


def run_epoch(model, loader, conf, device, optimizer=None, epoch=0, log_interval=50):
    training = optimizer is not None
    model.train(training)
    T_in = conf["data"]["T_in"]
    tbptt = conf["training"]["tbptt"]
    warmup = conf["training"]["warmup_steps"]

    total_loss = total_f1 = total_rate = 0.0
    n_steps = 0
    for step, bins in enumerate(loader, start=1):
        bins = bins.to(device, non_blocking=True)
        states = None
        sample_loss = sample_f1 = sample_rate = 0.0
        n_chunks = 0
        if training:
            optimizer.zero_grad(set_to_none=True)
        for t0 in range(0, T_in, tbptt):
            t1 = min(t0 + tbptt, T_in)
            if training:
                loss, metrics, states = model.forward_chunk(bins, t0, t1, states, warmup)
                if loss is not None:
                    loss.backward()  # grads accumulate across chunks; one step per batch
            else:
                with torch.no_grad():
                    loss, metrics, states = model.forward_chunk(bins, t0, t1, states, warmup)
            states = detach_states(states)
            if loss is not None:
                sample_loss += float(loss.detach())
                sample_f1 += metrics["f1"]
                sample_rate += metrics["spike_rate"]
                n_chunks += 1
        if training:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += sample_loss / max(n_chunks, 1)
        total_f1 += sample_f1 / max(n_chunks, 1)
        total_rate += sample_rate / max(n_chunks, 1)
        n_steps += 1
        if training and step % log_interval == 0:
            print(
                f"epoch {epoch} step {step}/{len(loader)} "
                f"loss {total_loss / n_steps:.4f} f1 {total_f1 / n_steps:.4f} "
                f"spike_rate {total_rate / n_steps:.4f}",
                flush=True,
            )
    n = max(n_steps, 1)
    return {"loss": total_loss / n, "f1": total_f1 / n, "spike_rate": total_rate / n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="configs/pretrain.yaml")
    args = parser.parse_args()
    with open(args.conf) as f:
        conf = yaml.safe_load(f)

    device = torch.device(conf["training"]["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(conf["training"].get("seed", 403))

    d = conf["data"]
    common = dict(
        hdf5_paths=resolve_paths(d), bin_ms=d["bin_ms"], T_in=d["T_in"], pred_bins=d["pred_bins"],
        stride_s=d["stride_s"], H=d["H"], W=d["W"], camera_side=d["camera_side"],
        downscale=d.get("downscale", 2),
        train_frac=d["train_frac"], val_frac=d["val_frac"], test_frac=d["test_frac"],
    )
    train_ds = M3EDPretrainDataset(**common, split="train")
    val_ds = M3EDPretrainDataset(**common, split="val")
    print(f"train samples: {len(train_ds)}  val samples: {len(val_ds)}")

    t = conf["training"]
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                              num_workers=t["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False,
                            num_workers=t["num_workers"], pin_memory=True)

    m = conf["model"]
    model = SpikingFutureModel(
        stage_channels=tuple(m["stage_channels"]), fused_channels=m["fused_channels"],
        pred_bins=d["pred_bins"], decay_init=m["decay_init"], threshold=m["threshold"],
        count_clip=m["count_clip"], spike_reg_weight=m["spike_reg_weight"],
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=t.get("lr_end_factor", 0.1), total_iters=t["epochs"]
    )

    ckpt_dir = Path(t["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir / "metrics.jsonl"
    best_val = float("inf")

    for epoch in range(t["epochs"]):
        tic = time.time()
        train_m = run_epoch(model, train_loader, conf, device, optimizer, epoch, t.get("log_interval", 50))
        val_m = run_epoch(model, val_loader, conf, device)
        scheduler.step()
        print(
            f"epoch {epoch} done in {time.time() - tic:.0f}s  "
            f"train loss {train_m['loss']:.4f} f1 {train_m['f1']:.4f} | "
            f"val loss {val_m['loss']:.4f} f1 {val_m['f1']:.4f}",
            flush=True,
        )
        with metrics_path.open("a") as f:
            f.write(json.dumps({"epoch": epoch, "train": train_m, "val": val_m,
                                "lr": scheduler.get_last_lr()[0]}) + "\n")
        payload = {"epoch": epoch, "model": model.state_dict(), "conf": conf}
        torch.save(payload, ckpt_dir / "last.pt")
        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            torch.save(payload, ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
