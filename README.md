# Spiking_RL

Streaming spiking encoder for event cameras: self-supervised future-event
prediction (F3-style objective), downstream metric depth on MVSEC. Merges the
lessons from `Not_optimised` (Prasham) and `SpikingEvents` (Deepak) and fixes
the two structural problems both shared.

## Design decisions (and why)

1. **Membrane-potential readout, not spike counts.** Both earlier encoders
   averaged binary spike maps over T steps into the latent — ~2.5 bits per
   channel/pixel, a hard ceiling for dense regression. Here every stage exposes
   its pre-reset membrane (analog) to the heads; spikes only propagate between
   spiking layers. (StereoSpike, Rançon et al. 2022.)
2. **Multi-scale spiking encoder.** Stages at /1, /2, /4, /8 via strided
   ConvLIF, fused by an FPN-lite neck. Depth needs global context; a
   full-resolution-only encoder has a ~15 px receptive field and can't provide
   it, no matter how big the decoder is.
3. **Streaming state + TBPTT.** The encoder is stateful (`step()` = one time
   bin). Pretraining unrolls 20 bins with truncated BPTT; at deployment each
   new 5 ms bin costs ONE step + decode, so depth updates at up to 200 Hz with
   constant compute. This — not sparsity — is the honest SNN latency story on
   a Jetson: a window CNN must recompute its full context every frame.
4. **Fine time bins (5 ms).** With 10 ms × 5 bins the LIF dynamics barely
   mattered. 20 × 5 ms gives the recurrence something to encode.
5. **Honest baseline built in.** `--baseline` trains a plain U-Net on the same
   voxel grids, same loss, same output bounds. If the spiking model doesn't
   beat it, the representation isn't earning its compute.
6. **snntorch Leaky neurons** with learnable per-channel decay, soft reset,
   ATan surrogate; GroupNorm on input currents (BatchNorm statistics mix
   activity levels across timesteps — a known SNN training pathology).

## Layout

Spiking_RL is fully standalone — it imports nothing from the other project
folders. The only external dependency is the MVSEC data directory, pointed at
by `data.root` in the configs.

```text
src/cells.py      ConvLIF (snntorch Leaky + recurrent feedback), explicit LIFState
src/encoder.py    MultiScaleSpikingEncoder — step() for streaming, forward() for windows
src/heads.py      TopDownFusion (FPN-lite), EventHead (future occupancy), DepthHead
src/models.py     SpikingFutureModel (pretrain), SpikingDepthModel, UNetDepthBaseline
src/streaming.py  StreamingDepthNet — flat-tensor state wrapper for export/deployment
src/data.py       MVSEC pretrain + depth datasets (sequence-of-bins samples)
src/losses.py     focal loss (events), SiLog (depth), depth metric sums
train_pretrain.py TBPTT pretraining driver
train_depth.py    depth probe / finetune / scratch / baseline
eval_depth.py     metrics + side-by-side event/pred/GT visualizations
stream_depth.py   continuous streaming depth over a sequence (latency + GIF + metrics)
benchmark.py      cold-start vs streaming-step latency (synthetic input)
export_onnx.py    ONNX export of the streaming step for TensorRT on the Jetson
tests/smoke_test.py  synthetic end-to-end checks, no data needed
```

## Usage

Running on a Jetson Orin Nano? See [JETSON_SETUP.md](JETSON_SETUP.md) first —
`pip install torch` does not work on aarch64.

```bash
pip install -r requirements.txt

# 1. Pretrain the encoder on future-event prediction (events only, no GT)
python train_pretrain.py --conf configs/pretrain.yaml

# 2. Depth: frozen probe first (representation quality), then finetune (real number)
python train_depth.py --conf configs/depth.yaml --encoder-ckpt checkpoints/pretrain/best.pt
python train_depth.py --conf configs/depth.yaml --encoder-ckpt checkpoints/pretrain/best.pt --finetune

# 3. Controls
python train_depth.py --conf configs/depth.yaml                # scratch (no pretraining)
python train_depth.py --conf configs/depth.yaml --baseline     # plain U-Net on voxels

# 4. Evaluate + visualize a trained model
python eval_depth.py --ckpt checkpoints/depth/snn_finetune/best.pt --viz 12

# 5. Quadrotor mode: stream depth continuously over a real sequence
python stream_depth.py --ckpt checkpoints/depth/snn_finetune/best.pt \
    --data $MVSEC/indoor_flying2_data-002.hdf5 --gt $MVSEC/indoor_flying2_gt-003.hdf5 \
    --t-start 20 --duration 5

# 6. Latency
python benchmark.py --conf configs/depth.yaml --device cuda
python benchmark.py --conf configs/depth.yaml --baseline --device cuda

# 6b. Jetson: same, but split encoder/decoder/full-pipeline, +power/thermal
python benchmark_jetson.py --conf configs/depth.yaml
python benchmark_jetson.py --conf configs/depth.yaml --fp16
python benchmark_jetson.py --conf configs/depth.yaml --baseline

# 7. Export the streaming step for the Jetson (verified against onnxruntime)
python export_onnx.py --ckpt checkpoints/depth/snn_finetune/best.pt --out depth_stream.onnx
```

Set `data.root` in `configs/*.yaml` to the directory that contains your
`mvsec/` folder (e.g. `/home/psoni/Prasham_DR` on the cluster); all dataset
paths in the configs are relative to it.

## Experiments this is set up to answer

| Question | Runs to compare |
| --- | --- |
| Does pretraining help? | `snn_frozen_probe` + `snn_finetune` vs `snn_scratch` |
| Does the SNN earn its compute? | `snn_finetune` vs `baseline_unet` (accuracy) + `benchmark.py` streaming vs window (latency) |
| Is the representation richer than spike counts? | swap `mem_pre` for spikes in `encoder.py` step() and rerun the probe |

## Notes

- `depth_max` / `max_depth`: for indoor_flying-only runs set both to ~10 m —
  a 0.5–80 m sigmoid range wastes most of its resolution on depths that never
  occur indoors.
- MVSEC GT is sparse LiDAR with NaN holes; the valid mask handles it, metrics
  are computed on valid pixels only.
- Jetson later: export the streaming path (`encoder.step` + `DepthHead`) with
  membrane states as explicit I/O tensors for TensorRT.
- M3ED (has UAV/Falcon sequences): the F3 repo's loaders are the reference;
  `src/data.py` is deliberately loader-shaped so an M3ED dataset can slot in
  next to the MVSEC one.
