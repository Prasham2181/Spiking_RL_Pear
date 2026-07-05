"""Save a random-initialized checkpoint in the format eval_depth.py/export_onnx.py
expect, so the ONNX/TensorRT export path can be exercised before real training
finishes (weights are garbage; this only tests the export+runtime pipeline).

Usage:
    python make_synthetic_ckpt.py --conf configs/depth.yaml --out checkpoints/synthetic.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.models import SpikingDepthModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="configs/depth.yaml")
    parser.add_argument("--out", default="checkpoints/synthetic.pt")
    args = parser.parse_args()

    with open(args.conf) as f:
        conf = yaml.safe_load(f)
    m = conf["model"]

    model = SpikingDepthModel(
        stage_channels=tuple(m["stage_channels"]), fused_channels=m["fused_channels"],
        decay_init=m["decay_init"], threshold=m["threshold"], count_clip=m["count_clip"],
        depth_min=m["depth_min"], depth_max=m["depth_max"],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "conf": conf, "tag": "synthetic"}, out)
    print(f"wrote {out} (random weights, for export/runtime pipeline testing only)")


if __name__ == "__main__":
    main()
