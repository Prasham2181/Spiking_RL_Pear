"""Latency benchmark: cold-start window vs streaming step (the deploy number).

The streaming number is the one that matters for the drone: after the state is
warm, each new event bin costs ONE encoder step + one decode, so depth updates
arrive every bin_ms milliseconds.

Usage:
    python benchmark.py --conf configs/depth.yaml            # CPU
    python benchmark.py --conf configs/depth.yaml --device cuda
    python benchmark.py --conf configs/depth.yaml --baseline # U-Net control
"""
from __future__ import annotations

import argparse
import time

import torch
import yaml

from src.models import SpikingDepthModel, UNetDepthBaseline


def time_fn(fn, warmup: int, iters: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return sorted(times)


def report(name: str, times: list[float]) -> None:
    mean = sum(times) / len(times)
    print(f"{name:24s} mean {mean:7.2f} ms   median {times[len(times) // 2]:7.2f} ms   "
          f"min {times[0]:7.2f} ms   ({1000.0 / mean:7.1f} Hz)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="configs/depth.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    with open(args.conf) as f:
        conf = yaml.safe_load(f)
    d, m = conf["data"], conf["model"]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    T_in, H, W = d["T_in"], d["H"], d["W"]
    bins = torch.rand(1, T_in, 2, H, W, device=device)

    if args.baseline:
        model = UNetDepthBaseline(
            input_bins=T_in, channels=tuple(m["stage_channels"]),
            depth_min=m["depth_min"], depth_max=m["depth_max"], count_clip=m["count_clip"],
        ).to(device).eval()
        print(f"U-Net baseline  params {sum(p.numel() for p in model.parameters()):,}  "
              f"device {device}  input (1, {T_in}, 2, {H}, {W})")
        with torch.no_grad():
            report("window forward", time_fn(lambda: model(bins), args.warmup, args.iters, device))
        print("\n(no streaming mode: a window CNN recomputes the full context every frame)")
        return

    model = SpikingDepthModel(
        stage_channels=tuple(m["stage_channels"]), fused_channels=m["fused_channels"],
        decay_init=m["decay_init"], threshold=m["threshold"], count_clip=m["count_clip"],
        depth_min=m["depth_min"], depth_max=m["depth_max"],
    ).to(device).eval()
    print(f"SpikingDepthModel  params {sum(p.numel() for p in model.parameters()):,}  "
          f"device {device}  input (1, {T_in}, 2, {H}, {W})")

    with torch.no_grad():
        report(f"cold start ({T_in} steps)", time_fn(lambda: model(bins), args.warmup, args.iters, device))

        # warm streaming state, then time single steps
        _, states = model(bins)
        x_t = bins[:, 0]

        def streaming_step():
            nonlocal states
            _, states = model.step(x_t, states)

        report("streaming step (1 bin)", time_fn(streaming_step, args.warmup, args.iters, device))
    print(f"\nstreaming = one depth update per {d['bin_ms']} ms event bin; "
          f"the achievable update rate is min(1000/bin_ms, streaming Hz above).")


if __name__ == "__main__":
    main()
