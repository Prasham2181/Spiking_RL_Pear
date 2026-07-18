#!/bin/bash
#SBATCH --mail-user=psoni@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -A nitin
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:L40S:1
#SBATCH -t 00:30:00
#SBATCH --mem 16G
#SBATCH --job-name="SRL-M3ED-eval-viz"

#SBATCH --output=/home/psoni/Prasham_DR/Spiking_RL_M3ed/logs/srl_m3ed_eval_viz_%j.log
#SBATCH --error=/home/psoni/Prasham_DR/Spiking_RL_M3ed/logs/srl_m3ed_eval_viz_%j.err

set -euo pipefail

REPO=/home/psoni/Prasham_DR/Spiking_RL_M3ed
cd "$REPO" || exit 1
source /home/psoni/anaconda3/etc/profile.d/conda.sh
conda activate snn_ae

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Save predicted-vs-GT depth map visualizations for both zipdepth runs.
python eval_depth.py --ckpt checkpoints/depth/snn_finetune_zipdepth/best.pt --split val --viz 12
python eval_depth.py --ckpt checkpoints/depth/snn_frozen_probe_zipdepth/best.pt --split val --viz 12

echo "Eval viz complete!"
