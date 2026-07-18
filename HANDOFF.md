# Spiking_RL_M3ed — cluster handoff brief

> **FORK NOTE (2026-07-09):** this is the M3ED variant of `../Spiking_RL`.
> Same model/objective; `src/data.py` adds `M3EDPretrainDataset`/`M3EDDepthDataset`
> (lazy HDF5 slicing via `ms_map_idx`, 1280x720 downscaled 2x to 640x360,
> polarity {0,1}), and the trainers/configs point at `m3ed/car_urban_day_*`.
> Rationale: MVSEC indoor targets are ~0.5% positive per 5 ms bin and the
> pretrain F1 stalls ~0.08-0.23; M3ED is ~7% per bin (measured), the regime
> where the F3 objective is known to work. Corrupt files (truncated downloads,
> excluded from configs): penno_big_loop_data.h5, schuylkill_tunnel_depth_gt.h5,
> ucity_big_loop_depth_gt.h5. Logs + checkpoints stay inside this folder.
> `stream_depth.py` still reads MVSEC-layout files — adapt before using it here.
> The MVSEC-specific facts below are kept for reference.

You are picking up a freshly built, smoke-tested research codebase. Read this
whole file before touching anything; it contains the design rationale, the
failure modes of two earlier attempts (do NOT reintroduce them), and the exact
run matrix that answers the research question.

## 1. Project context and goal

PEAR Lab project: train spiking neural networks (SNNs) on event-camera streams
to learn representations via self-supervised **future-event prediction**
(the objective from the "Fast Feature Fields" (F³) paper, which used a
multiresolution hash encoder — we replace it with a spiking encoder), with
**monocular metric depth** as the downstream task. End target: high-rate,
low-latency depth for **quadrotor navigation** on an NVIDIA Jetson Orin Nano.
Data: MVSEC (indoor_flying = hexacopter drone data; outdoor_day/night =
driving). M3ED (has UAV "Falcon" sequences) is a planned extension.

The thesis: SNNs fit event cameras not because of "sparsity" (which buys
nothing on GPU/Jetson — dense tensor simulation), but because a **stateful
recurrent encoder amortizes computation over time**: after warm-up, each new
5 ms slice of events costs ONE small forward step and yields a full depth map,
where any window-based CNN must recompute its whole context every frame.
Measured on laptop CPU: cold-start 20-step window ~775 ms vs streaming step
~98 ms (~8×). That constant-compute streaming property is the paper's claim.

## 2. History — two prior codebases and why their depth output was bad

Both live in the old repo (`Prasham_DR_code/Prasham_DR/`), for reference only.
This project imports NOTHING from them.

- `Not_optimised/` (Prasham): snntorch encoder, F³-style pretraining, frozen
  probe + large U-Net depth decoder.
- `Deepak/SpikingEvents/`: same encoder idea (hand-rolled ConvLIFCell), better
  harness (F³ dataloaders, SNN-vs-UNet control, CKA/effective-rank latent
  analysis, DepthAnythingV2 shared-decoder probing).

Shared structural flaws (diagnosed 2026-07-02, both fixed here):
1. **Spike-count latent**: both averaged binary spike maps over T timesteps
   into the representation → ~2.5 bits per channel/pixel — an information
   ceiling no decoder can recover from. Depth came out blurry / scene-mean.
2. **No spatial hierarchy**: full-resolution-only encoders, ~15 px receptive
   field. Monocular depth needs global context; the decoders had to invent it
   from scratch, so the pretrained representation contributed ~nothing.
3. **No persistent state**: membrane state reset every window; full recompute
   per inference → SNN strictly slower than an equivalent CNN.

## 3. What this codebase does differently (the three fixes)

1. **Membrane-potential readout** (`src/cells.py`): every ConvLIF returns
   spikes (binary, propagate to next spiking stage) AND the pre-reset membrane
   (analog, goes to heads). Heads never see bare spike counts. (StereoSpike
   trick, Rançon et al. 2022.)
2. **Multi-scale spiking encoder** (`src/encoder.py`): 4 ConvLIF stages at
   /1, /2, /4, /8 via stride-2 convs; heads fuse all scales with an FPN-lite
   neck (`src/heads.py:TopDownFusion`).
3. **Streaming state + TBPTT** (`src/models.py`, `train_pretrain.py`):
   explicit `LIFState` carried across time bins; pretraining unrolls 20 bins
   of 5 ms with truncated BPTT (chunks of 10, state detached at boundaries,
   grads accumulate, one optimizer step per batch); deployment path is
   `model.step(one_bin, state)`.

Other deliberate choices:
- snntorch `snn.Leaky` (user preference), learnable per-channel decay
  (beta shaped (C,1,1)), `reset_mechanism="subtract"`, ATan surrogate.
- **GroupNorm on input currents, never BatchNorm** — BatchNorm statistics mix
  activity levels across timesteps (known SNN pathology; Deepak's code had it).
- Event counts clamped to `count_clip=4` before the encoder (dense scenes hit
  50+ counts/px/bin and saturate LIF neurons).
- Depth output = sigmoid-bounded log-depth in [depth_min, depth_max]; SiLog loss.
- Pretraining loss = class-balanced focal loss on binarized future occupancy
  (events ~95-99% empty); spike-rate regularizer weight 1e-4.
- Encoder+EventHead train jointly in stage 1; EventHead is discarded after;
  DepthHead is fresh in stage 2.

## 4. Layout (all standalone; only external dependency = MVSEC data dir)

```text
src/cells.py      ConvLIF (snntorch Leaky + recurrent conv feedback), LIFState, detach_states
src/encoder.py    MultiScaleSpikingEncoder — step() streaming, forward() windows
src/heads.py      TopDownFusion, EventHead (pretrain only), DepthHead
src/models.py     SpikingFutureModel (forward_chunk = TBPTT unit), SpikingDepthModel, UNetDepthBaseline
src/streaming.py  StreamingDepthNet — flat-tensor state wrapper for ONNX/deployment
src/data.py       MVSECPretrainDataset, MVSECDepthDataset, voxelize, resolve_paths/resolve_pairs
src/losses.py     voxel_focal_loss, silog_loss, depth_metric_sums
train_pretrain.py / train_depth.py / eval_depth.py / stream_depth.py / benchmark.py / export_onnx.py
tests/smoke_test.py   synthetic end-to-end checks (no data needed)
configs/pretrain.yaml, configs/depth.yaml
```

MVSEC HDF5 facts (already handled in `src/data.py`, don't re-solve):
`davis/left/events` (N,4)=[x,y,t,p], epoch-scale float64 timestamps — rebase
to t=0 in float64 BEFORE casting float32; searchsorted queries must be float32
on a contiguous ts copy. GT: `davis/left/depth_image_rect(_ts)`, sparse LiDAR
with NaN holes — valid masks everywhere, metrics on valid px only. Splits are
time-ordered within each recording (no shuffling leakage).

## 5. Verified state (Windows laptop, CPU, torch 2.12 + snntorch)

- `python tests/smoke_test.py` — 5/5 pass: TBPTT trains; depth bounded;
  frozen probe isolates encoder grads; **streaming step == window forward
  bit-identical (0.00e+00)**; StreamingDepthNet wrapper identical too.
- `benchmark.py` at 260×346, T=20: 668,097 params; window 775 ms vs streaming
  step 98 ms on CPU.
- `export_onnx.py` end-to-end on a synthetic checkpoint: onnxruntime parity
  max diff 2.15e-06. (snntorch emits a benign shape-check TracerWarning;
  fixed camera resolution makes it irrelevant.)
- NOT yet done anywhere: training on real data. No real accuracy numbers exist.

## 6. Your job on the cluster, in order

0. Setup: `pip install -r requirements.txt` (+ `onnx onnxruntime` if exporting).
   Run `python tests/smoke_test.py` — must be 5/5 before anything else.
1. **Data wiring**: set `data.root` in both configs to the dir CONTAINING
   `mvsec/` (expected: `/home/psoni/Prasham_DR`). Verify every file in
   `configs/*.yaml` exists; fix names if the download layout differs.
2. **Config sanity for the first campaign**: recommend indoor_flying-only
   first (drop outdoor pairs), and set BOTH `data.max_depth` and
   `model.depth_max` to 10.0 in depth.yaml — a 0.5–80 m sigmoid range wastes
   its resolution indoors. Keep `bin_ms`/`T_in`/`stage_channels` identical
   between pretrain.yaml and depth.yaml (the encoder checkpoint must match).
3. **Pretrain**: `python train_pretrain.py --conf configs/pretrain.yaml`
   (SLURM GPU job; batch 8 fits comfortably — model is <1M params, memory is
   dominated by 20-step activations; raise batch if headroom).
   Healthy: focal loss drops steeply then flattens; **F1 plateaus ~0.4–0.7**
   (≈1.0 = task too easy → shrink pred window; ≈0 = broken); **spike_rate
   settles 0.05–0.2** (→0 = dead neurons: lower threshold or spike_reg;
   >0.5 = raise spike_reg).
4. **Depth run matrix** (4 runs, same config, tags auto-assigned):
   ```bash
   python train_depth.py --conf configs/depth.yaml --encoder-ckpt checkpoints/pretrain/best.pt              # snn_frozen_probe
   python train_depth.py --conf configs/depth.yaml --encoder-ckpt checkpoints/pretrain/best.pt --finetune   # snn_finetune
   python train_depth.py --conf configs/depth.yaml                                                          # snn_scratch
   python train_depth.py --conf configs/depth.yaml --baseline                                               # baseline_unet
   ```
5. **Evaluate & inspect**: `eval_depth.py --ckpt .../best.pt --viz 12` per run;
   `stream_depth.py --ckpt .../snn_finetune/best.pt --data <indoor_flying2_data> --gt <indoor_flying2_gt> --t-start 20 --duration 5`
   (per-update latency + GIF + vs-LiDAR metrics);
   `benchmark.py --conf configs/depth.yaml --device cuda` and `--baseline`.
6. **Success criteria** (val, within-sequence split — expect abs_rel roughly
   0.15–0.35, RMSE 1–2.5 m, d1 0.5–0.8 indoors if learning at all):
   - snn_finetune ≪ snn_scratch → pretraining works (the F³-with-SNN thesis).
   - snn_finetune within a few points of baseline_unet on accuracy WHILE
     streaming updates are ~8× cheaper → the paper result.
   - frozen_probe ≪ finetune is expected/fine (probe = representation
     diagnostic, not the headline).
   - If baseline_unet wins by a wide margin → suspect the readout/fusion; run
     the ablation: in `src/encoder.py` step(), append `spike` instead of `mem`
     to feats and rerun the probe — quantifies the membrane-readout claim.
7. **Report numbers honestly**: the split is within-sequence (first 80% train
   / last 20% val of each recording) — optimistic vs the literature's
   cross-sequence protocol (train flying 2+3, test flying 1). Before quoting
   against papers, add a sequence-level split (small change in
   `src/data.py::_split_range` usage — split by file instead of by time).

## 7. Backlog after the first campaign (in rough priority order)

- Port Deepak's **effective-rank / CKA latent analysis**
  (`Prasham_DR_code/Prasham_DR/Deepak/SpikingEvents/scripts/compare_representations.py`)
  to quantify membrane-vs-spike-count information content.
- **M3ED loader** (UAV Falcon sequences) — `src/data.py` is loader-shaped for
  it; F³ repo (`Prasham_DR_code/Prasham_DR/F3/fast-feature-fields/`) has the
  reference download/processing scripts.
- GPU-side voxelization (old repo's `_scatter_voxel` in
  `Not_optimised/src/data.py`) if dataloading becomes the bottleneck.
- Cross-sequence split for publication numbers.
- Longer-horizon / multi-scale prediction objective ablations (pred_bins,
  bin_ms, T_in sweeps).
- Jetson: `export_onnx.py` → `trtexec --fp16`, state buffers ping-pong between
  invocations; report per-update latency on the Orin Nano.
- F³ baseline comparison: same depth head on F³ features (F³ repo is vendored
  in the old folder).

## 8. Gotchas

- `forward_chunk` computes loss per step and backprops per TBPTT chunk; grads
  accumulate across chunks, ONE optimizer step per batch. Don't "fix" this
  into step-per-chunk without thinking.
- Streaming/window parity is a load-bearing invariant (deploy path must equal
  train path). If you touch cells/encoder, rerun smoke test — the parity
  asserts exist precisely to catch drift.
- Keep GroupNorm; do not swap in BatchNorm for a quick accuracy bump.
- `model.depth_max` must match `data.max_depth`, and depth.yaml's encoder
  section must match the pretrain checkpoint architecture exactly
  (load_encoder is strict=True on purpose).
- eval_depth auto-detects baseline-vs-spiking from the checkpoint; keep the
  `tag` field in payloads.
