#!/bin/bash
#SBATCH --job-name=yuhao-sedd
#SBATCH --account=def-lincai_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

module load StdEnv/2023
source .venv/bin/activate

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

echo "Starting training at $(date)"

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

python -u src/train/train.py \
    worker=rorqual \
    model.scale_by_sigma=False

echo "Finished training at $(date)"
