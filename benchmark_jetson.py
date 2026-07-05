"""Jetson-focused latency benchmark: encoder, decoder, and full pipeline
separately, for both cold-start (window) and streaming (one bin) paths.

Adds two things benchmark.py doesn't need on a dev box:
- optional `tegrastats` power/thermal sampling (Jetson-only, auto-skipped
  if the binary isn't found, e.g. when testing this script off-device)
- `--fp16` autocast, since Jetson's tensor cores make fp16 the realistic
  deployment precision, not fp32

Usage:
    python benchmark_jetson.py --conf configs/depth.yaml
    python benchmark_jetson.py --conf configs/depth.yaml --fp16
    python benchmark_jetson.py --conf configs/depth.yaml --baseline
    python benchmark_jetson.py --conf configs/depth.yaml --no-tegrastats
"""
from __future__ import annotations

import argparse
import contextlib
import re
import subprocess
import threading
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
    print(f"{name:32s} mean {mean:7.2f} ms   median {times[len(times) // 2]:7.2f} ms   "
          f"min {times[0]:7.2f} ms   ({1000.0 / mean:7.1f} Hz)")


class TegrastatsMonitor:
    """Samples `tegrastats` in the background for the benchmark's duration.

    Jetson-only; if the binary isn't on PATH (e.g. running this on a dev
    laptop to sanity-check the script), starts as a no-op.
    """

    _POWER_RE = re.compile(r"VDD_IN (\d+)mW")
    _GPU_TEMP_RE = re.compile(r"gpu@([\d.]+)C")
    _CPU_TEMP_RE = re.compile(r"cpu@([\d.]+)C")
    _GR3D_RE = re.compile(r"GR3D_FREQ (\d+)%")

    def __init__(self, interval_ms: int = 200) -> None:
        self.interval_ms = interval_ms
        self.proc: subprocess.Popen | None = None
        self.lines: list[str] = []
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except FileNotFoundError:
            self.proc = None
            return
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            if self._stop:
                break
            self.lines.append(line)

    def stop(self) -> None:
        self._stop = True
        if self.proc is None:
            return
        self.proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.proc.wait(timeout=2)

    def report(self) -> None:
        if self.proc is None:
            print("\ntegrastats not found on PATH - skipping power/thermal readout "
                  "(expected off-Jetson; on-device this should just work).")
            return
        if not self.lines:
            print("\ntegrastats ran but produced no samples (benchmark too short?).")
            return

        def extract(pattern: re.Pattern, cast) -> list:
            vals = []
            for line in self.lines:
                m = pattern.search(line)
                if m:
                    vals.append(cast(m.group(1)))
            return vals

        power = extract(self._POWER_RE, int)
        gpu_temp = extract(self._GPU_TEMP_RE, float)
        cpu_temp = extract(self._CPU_TEMP_RE, float)
        gr3d = extract(self._GR3D_RE, int)

        print(f"\ntegrastats over the full benchmark ({len(self.lines)} samples):")
        if power:
            print(f"  VDD_IN power   mean {sum(power) / len(power):6.0f} mW   max {max(power):6.0f} mW")
        if gr3d:
            print(f"  GR3D (GPU) util mean {sum(gr3d) / len(gr3d):5.0f}%   max {max(gr3d):5.0f}%")
        if gpu_temp:
            print(f"  GPU temp       mean {sum(gpu_temp) / len(gpu_temp):5.1f} C   max {max(gpu_temp):5.1f} C")
        if cpu_temp:
            print(f"  CPU temp       mean {sum(cpu_temp) / len(cpu_temp):5.1f} C   max {max(cpu_temp):5.1f} C")
        if not (power or gr3d or gpu_temp or cpu_temp):
            print("  (raw tegrastats output didn't match any known field patterns — "
                  "field names vary by L4T version; check a sample line below)")
            print(f"  sample: {self.lines[0].strip()}")


def bench_spiking(model: SpikingDepthModel, bins: torch.Tensor, warmup: int, iters: int,
                   device: torch.device) -> None:
    x = torch.clamp(bins, 0.0, model.count_clip)
    x_t_raw = bins[:, 0]

    # --- encoder only ---
    report("encoder cold start (window)", time_fn(lambda: model.encoder(x), warmup, iters, device))

    feats, states, _ = model.encoder(x)

    def enc_step():
        nonlocal states
        _, states, _ = model.encoder.step(x[:, 0], states)

    report("encoder streaming step", time_fn(enc_step, warmup, iters, device))

    # --- decoder only (fixed feats from above; isolates head cost) ---
    report("decoder (depth head)", time_fn(lambda: model.depth_head(feats), warmup, iters, device))

    # --- full pipeline ---
    report("full pipeline cold start", time_fn(lambda: model(bins), warmup, iters, device))

    _, pipe_states = model(bins)

    def pipe_step():
        nonlocal pipe_states
        _, pipe_states = model.step(x_t_raw, pipe_states)

    report("full pipeline streaming step", time_fn(pipe_step, warmup, iters, device))


def bench_baseline(model: UNetDepthBaseline, bins: torch.Tensor, warmup: int, iters: int,
                    device: torch.device) -> None:
    # No streaming path for a window CNN: every call recomputes full context.
    x, skips = model.encode(bins)
    report("encoder (down + bottleneck)", time_fn(lambda: model.encode(bins), warmup, iters, device))
    report("decoder (up + head)", time_fn(lambda: model.decode(x, skips), warmup, iters, device))
    report("full pipeline (window, no streaming)", time_fn(lambda: model(bins), warmup, iters, device))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="configs/depth.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="autocast fp16 (Jetson tensor cores)")
    parser.add_argument("--no-tegrastats", action="store_true", help="skip power/thermal sampling")
    parser.add_argument("--tegrastats-interval-ms", type=int, default=200)
    args = parser.parse_args()

    with open(args.conf) as f:
        conf = yaml.safe_load(f)
    d, m = conf["data"], conf["model"]

    cuda_ok = torch.cuda.is_available()
    device = torch.device(args.device if cuda_ok or args.device == "cpu" else "cpu")
    print(f"torch {torch.__version__}  cuda_available={cuda_ok}"
          + (f"  device_name={torch.cuda.get_device_name(0)}" if cuda_ok else ""))
    if args.device == "cuda" and not cuda_ok:
        print("WARNING: --device cuda requested but torch.cuda.is_available() is False; "
              "falling back to CPU. Check JETSON_SETUP.md if you expect a GPU build here.")

    if device.type == "cuda":
        # is_available() only means a CUDA driver was found, not that this
        # torch build shipped kernels for this GPU's compute capability
        # (e.g. PyPI torch wheels have crashed with "no kernel image is
        # available" on Jetson Orin's CC 8.7 - see JETSON_SETUP.md).
        try:
            (torch.zeros(1, device=device) + 1).item()
        except RuntimeError as e:
            print(f"WARNING: CUDA reports available but a real op failed ({e}); "
                  "falling back to CPU. See JETSON_SETUP.md for GPU remediation.")
            device = torch.device("cpu")

    print(f"device={device}  fp16={args.fp16}\n")

    T_in, H, W = d["T_in"], d["H"], d["W"]
    bins = torch.rand(1, T_in, 2, H, W, device=device)

    monitor = None
    if not args.no_tegrastats:
        monitor = TegrastatsMonitor(args.tegrastats_interval_ms)
        monitor.start()

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if args.fp16 else contextlib.nullcontext()
    )

    try:
        with torch.no_grad(), autocast_ctx:
            if args.baseline:
                model = UNetDepthBaseline(
                    input_bins=T_in, channels=tuple(m["stage_channels"]),
                    depth_min=m["depth_min"], depth_max=m["depth_max"], count_clip=m["count_clip"],
                ).to(device).eval()
                print(f"U-Net baseline  params {sum(p.numel() for p in model.parameters()):,}  "
                      f"input (1, {T_in}, 2, {H}, {W})\n")
                bench_baseline(model, bins, args.warmup, args.iters, device)
            else:
                model = SpikingDepthModel(
                    stage_channels=tuple(m["stage_channels"]), fused_channels=m["fused_channels"],
                    decay_init=m["decay_init"], threshold=m["threshold"], count_clip=m["count_clip"],
                    depth_min=m["depth_min"], depth_max=m["depth_max"],
                ).to(device).eval()
                print(f"SpikingDepthModel  params {sum(p.numel() for p in model.parameters()):,}  "
                      f"input (1, {T_in}, 2, {H}, {W})\n")
                bench_spiking(model, bins, args.warmup, args.iters, device)

        print(f"\nstreaming = one depth update per {d['bin_ms']} ms event bin; "
              f"the achievable update rate is min(1000/bin_ms, streaming Hz above).")
    finally:
        if monitor is not None:
            monitor.stop()
            monitor.report()


if __name__ == "__main__":
    main()
