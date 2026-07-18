#!/bin/bash
#SBATCH --mail-user=psoni@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -A nitin
# Turing supports long training runs: partition `long` allows up to 7-00:00:00
# (vs 24h on `short`).
#SBATCH -p long
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:L40S:1
#SBATCH -t 3-00:00:00
#SBATCH --mem 48G
#SBATCH --job-name="SRL-M3ED-zipdepth-ft"

#SBATCH --output=/home/psoni/Prasham_DR/Spiking_RL_M3ed/logs/srl_m3ed_zipdepth_ft_%j.log
#SBATCH --error=/home/psoni/Prasham_DR/Spiking_RL_M3ed/logs/srl_m3ed_zipdepth_ft_%j.err

set -euo pipefail

REPO=/home/psoni/Prasham_DR/Spiking_RL_M3ed
cd "$REPO" || exit 1
source /home/psoni/anaconda3/etc/profile.d/conda.sh
conda activate snn_ae

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Complete end-to-end training of the encoder-decoder model:
# events -> SNN encoder (init: pretrain job 2082680, val F1 0.50, lr 1e-4)
#        -> representation
#        -> ZipDepth decoder (init: distilled zipdepth_base.pth, lr 1e-3)
#        -> metric depth (SiLog on M3ED LiDAR GT)
python train_depth.py \
  --conf configs/depth.yaml \
  --encoder-ckpt checkpoints/pretrain/best.pt \
  --head zipdepth \
  --zipdepth-ckpt /home/psoni/Prasham_DR/ZipDepth/checkpoints/zipdepth_base.pth \
  --finetune

echo "ZipDepth finetune (end-to-end) training complete!"
