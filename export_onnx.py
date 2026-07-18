"""Export the streaming depth step to ONNX for TensorRT on the Jetson.

The exported graph is the entire per-frame workload:
    inputs : event bin (1, 2, H, W) + one (mem, spike) pair per encoder stage
    outputs: depth (1, H, W) + the updated state tensors

On the Jetson, keep the state tensors device-resident between invocations
(TensorRT: bind output buffers of step k as input buffers of step k+1).

Usage:
    python export_onnx.py --ckpt checkpoints/depth/snn_finetune/best.pt --out depth_stream.onnx
"""
from __future__ import annotations

import argparse

import torch

from eval_depth import build_model
from src.streaming import StreamingDepthNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="depth_stream.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, conf, is_baseline = build_model(payload, torch.device("cpu"))
    if is_baseline:
        raise SystemExit("export targets the spiking streaming path; the U-Net baseline is a plain CNN")
    d = conf["data"]
    H, W = d["H"], d["W"]

    net = StreamingDepthNet(model).eval()
    state = net.init_state(1, H, W)
    x_t = torch.zeros(1, 2, H, W)

    state_names = [f"{kind}{i}" for i in range(net.n_stages) for kind in ("mem", "spike")]
    torch.onnx.export(
        net,
        (x_t, *state),
        args.out,
        input_names=["events", *[f"{n}_in" for n in state_names]],
        output_names=["depth", *[f"{n}_out" for n in state_names]],
        opset_version=args.opset,
        dynamo=False,
    )

    # parity check: ONNX Runtime vs PyTorch on random input
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        x = torch.rand(1, 2, H, W) * 3
        feeds = {"events": x.numpy()}
        feeds.update({f"{n}_in": s.numpy() for n, s in zip(state_names, state)})
        ort_depth = sess.run(["depth"], feeds)[0]
        with torch.no_grad():
            torch_depth = net(x, *state)[0]
        err = float(np.abs(ort_depth - torch_depth.numpy()).max())
        print(f"parity vs onnxruntime: max abs diff {err:.2e}")
    except ImportError:
        print("onnxruntime not installed — skipped parity check (pip install onnxruntime)")

    print(f"exported: {args.out}  (input {H}x{W}, {2 * net.n_stages} state tensors)")
    print("Jetson: trtexec --onnx=depth_stream.onnx --fp16, then ping-pong the state buffers")


if __name__ == "__main__":
    main()
