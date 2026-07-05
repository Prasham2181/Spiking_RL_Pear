# Jetson Orin Nano setup (inference / test runs, no training)

## Current status: CPU confirmed working, GPU blocked (CC 8.7 kernel mismatch)

`torch==2.12.1+cu126` (plain `pip install torch`, no jetson-containers or
jetson-ai-lab index) installs cleanly on JetPack 6 / L4T R36.4.7 / CUDA 12.6 /
Python 3.10.12, and **`torch.cuda.is_available()` reports `True`** — but that
flag only means torch found a CUDA driver, not that it shipped kernels for
this GPU. Actual CUDA ops crash:

```text
CUDA error: no kernel image is available for execution on the device
```

This PyPI wheel was built for compute capability 8.0–8.6 and 9.0; Orin's GPU
is **CC 8.7**, which falls in the gap. `torch.cuda.is_available() == True` is
necessary but not sufficient — always sanity-check with a real op
(`torch.zeros(1, device="cuda") + 1`), not just the availability flag.

**What works right now:** CPU inference, ~1.6 Hz full-pipeline streaming
(608 ms/step) — see [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). Sufficient
for initial testing; not the deployment target.

**To get GPU working:** need a torch build compiled with `TORCH_CUDA_ARCH_LIST`
including 8.7 — i.e. an actual Jetson/NVIDIA wheel, not the generic PyPI one.
Attempted so far and blocked by *this device's* network access, not
necessarily true elsewhere — retry when network access allows:

- NVIDIA's redist server (`developer.download.nvidia.com/compute/redist/jp/v60/pytorch/`) — 404'd
- Jetson AI Lab pip index (`pypi.jetson-ai-lab.dev`) — DNS resolution failed
- `dusty-nv/jetson-containers` — not yet tried (needs Docker + network)

Full remediation steps and options: [JETSON_PYTORCH_INSTALL.md](JETSON_PYTORCH_INSTALL.md),
[GPU_SETUP_STATUS.md](GPU_SETUP_STATUS.md).

Environment reference (nvcc, driver, uname, repo path): [requirements_installed.txt](requirements_installed.txt).
Pip-installable freeze of the currently-installed (CPU-functional) versions: [requirements-jetson.lock.txt](requirements-jetson.lock.txt).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-jetson.lock.txt
python -c "import torch; print(torch.__version__); torch.zeros(1, device='cuda') + 1"  # expect the CC 8.7 error above, for now
python tests/smoke_test.py --device cpu 2>/dev/null || python tests/smoke_test.py       # must be 5/5 on CPU
```

---

## If plain `pip install torch` doesn't give you CUDA

Historically PyPI carried no Jetson (aarch64+CUDA) torch builds at all, so
older JetPack versions need one of the routes below. Confirm which situation
you're in first:

## 1. Confirm the exact JetPack / L4T version

```bash
dpkg-query --show nvidia-l4t-core        # e.g. nvidia-l4t-core 36.3.0-...
sudo apt-cache show nvidia-jetpack | grep Version   # e.g. 6.1
python3 --version                        # JetPack 6.x ships Python 3.10
```

The wheel you need is specific to the L4T/JetPack point release (6.0 vs 6.1
vs 6.2 differ), so don't skip this — the value goes into the URL/index below.

## 2. Install torch + torchvision matched to that version

Two supported routes, pick one:

**A. Jetson AI Lab pip index (fastest, no Docker)**

```bash
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
```

Swap `jp6/cu126` for the tag matching your JetPack/CUDA (check
https://pypi.jetson-ai-lab.dev/ for the current tag list — it's versioned by
JetPack release, and tags shift as new JetPack point releases ship).

**B. dusty-nv `jetson-containers` (most reliable, if A's tag doesn't match)**

```bash
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers
./install.sh
jetson-containers run $(autotag pytorch)
```

Gives a container with a torch build guaranteed to match your L4T version;
run the rest of this setup inside it.

## 3. Verify torch sees the GPU before touching this repo

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Must print `True`. If not, stop here — installing snntorch on top of a
broken torch just obscures the real problem.

## 4. Install the rest of the project

```bash
git clone https://github.com/Prasham2181/Spiking_RL_Pear.git
cd Spiking_RL_Pear
pip install -r requirements.txt   # skips torch on aarch64, installs the rest
pip install onnx onnxruntime-gpu  # if running export_onnx.py / TensorRT path
python tests/smoke_test.py        # must be 5/5 before anything else
```

`pip install snntorch` will see torch already satisfies its requirement and
won't try to replace it — if you ever see it pulling a second torch wheel,
stop and re-check step 3 rather than letting it "fix" your install.

## 5. What actually needs to run here

Per `HANDOFF.md` §7, the intended Jetson path is `export_onnx.py` (run
anywhere) → `trtexec --fp16` on-device → TensorRT inference, which needs no
torch at all. The steps above are for the broader case of also running
`eval_depth.py`, `stream_depth.py`, or `benchmark.py` directly via PyTorch on
the Jetson for test runs, per your call to install the full stack.
