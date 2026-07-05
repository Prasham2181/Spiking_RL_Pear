# Jetson PyTorch Installation Guide

## System Information

- **JetPack Version**: R36 (6.0)
- **Python Version**: 3.10.12 (cp310)
- **Device**: Jetson Orin (CC 8.7)
- **Architecture**: aarch64

---

## Issue

Currently installed PyTorch `2.12.1+cu126` lacks compute capability 8.7 kernels, blocking GPU inference.

Error: `CUDA error: no kernel image is available for execution on the device`

---

## Solution: Install NVIDIA-Optimized PyTorch

### Step 1: Check JetPack Compatibility

```bash
cat /etc/nv_tegra_release
# Output should start with: # R36 (release)
```

This maps to **JetPack 6.0** → **JP_VERSION=60**

### Step 2: Available PyTorch Versions for JetPack 6.0

NVIDIA provides official wheels compiled for Jetson with full CC 8.7 support.

**Available versions** (from NVIDIA redist server):
https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/

Common options:
- PyTorch 2.1.0 (recommended for stability)
- PyTorch 2.2.0
- PyTorch 2.3.0
- Later versions if available

### Step 3: Uninstall Current PyTorch

```bash
cd /home/pear/spiking_rl/Spiking_RL_Pear
source .venv/bin/activate

# Remove incompatible build
pip uninstall -y torch torchvision torchaudio
```

### Step 4: Install Jetson-Specific PyTorch

**Option A: PyTorch 2.1.0 (Recommended)**

```bash
pip install --no-cache-dir \
  https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.1.0-cp310-cp310-linux_aarch64.whl
```

**Option B: PyTorch 2.3.0 (Latest available)**

```bash
pip install --no-cache-dir \
  https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl
```

### Step 5: Verify Installation

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); x = torch.randn(10, 10, device='cuda'); print('GPU tensor ok:', x.device)"
```

Expected output (no warnings):
```
torch-2.1.0 (or 2.3.0)
CUDA available: True
GPU tensor ok: cuda:0
```

### Step 6: Test GPU Inference

```bash
cd /home/pear/spiking_rl/Spiking_RL_Pear
python benchmark_jetson.py --device cuda --warmup 1 --iters 1 --no-tegrastats
```

Should now run **without** the CC 8.7 error and show GPU latency timings.

---

## Troubleshooting

### If wheel not found (404)

The URL structure may have changed. Check available wheels:

```bash
curl -s https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/ 2>/dev/null | grep -i '.whl' | head -10
```

### If you get CUDA 12.1 compatibility errors

Some Jetson PyTorch versions expect CUDA 12.1. Install it first (if not present):

```bash
wget https://raw.githubusercontent.com/pytorch/pytorch/5c6af2b583709f6176898c017424dc9981023c28/.ci/docker/common/install_cusparselt.sh
export CUDA_VERSION=12.1
bash ./install_cusparselt.sh
```

### If CUDA is still unavailable after install

Rebuild PyTorch from source (advanced):

```bash
pip uninstall -y torch
git clone https://github.com/pytorch/pytorch.git --depth 1 --branch v2.1.0
cd pytorch
export TORCH_CUDA_ARCH_LIST="8.7"
pip install -v --no-cache-dir .
# ⚠️ This takes hours on Jetson. Use only as last resort.
```

---

## Performance Expectations (with GPU)

Once Jetson PyTorch is installed, you should see:

- **Encoder streaming**: ~15-30 ms per step (10-60x faster)
- **Full pipeline streaming**: ~30-50 ms per step (10-20x faster)  
- **GPU utilization**: 20-40% load on Orin

vs. current CPU:
- Encoder streaming: 290 ms
- Full pipeline streaming: 608 ms

---

## References

- [NVIDIA PyTorch Jetson Install Guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/)
- [NVIDIA Optimized Frameworks Release Notes](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)
- [Jetson Platform Download Center](https://developer.nvidia.com/jetson-downloads)
