#!/bin/bash
#SBATCH --mail-user=psoni@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -A nitin
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:L40S:1
#SBATCH -t 24:00:00
#SBATCH --mem 48G
#SBATCH --job-name="SRL-M3ED-pretrain"

#SBATCH --output=/home/psoni/Prasham_DR/Spiking_RL_M3ed/logs/srl_m3ed_pretrain_%j.log
#SBATCH --error=/home/psoni/Prasham_DR/Spiking_RL_M3ed/logs/srl_m3ed_pretrain_%j.err

set -euo pipefail

REPO=/home/psoni/Prasham_DR/Spiking_RL_M3ed
CONF=configs/pretrain.yaml

mkdir -p /home/psoni/Prasham_DR/Spiking_RL_M3ed/logs
cd "$REPO" || exit 1

source /home/psoni/anaconda3/etc/profile.d/conda.sh
conda activate snn_ae

echo "=========================================="
nvidia-smi
echo "=========================================="
echo "Working directory: $(pwd)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
echo "Config: $CONF"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "Starting Spiking_RL_M3ed Stage 1 pretrain (M3ED car_urban_day)..."
python train_pretrain.py --conf "$CONF"

echo "Pretrain complete!"
