#!/bin/bash
#SBATCH --job-name=yuhao-sedd-debug
#SBATCH --account=def-lincai_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load StdEnv/2023

source .venv/bin/activate

python src/train/train.py \
    worker=nibi-test \
    model.scale_by_sigma=False
