#!/bin/bash
#SBATCH --job-name=yuhao-sedd
#SBATCH --account=def-lincai

# 一台 GPU 节点上 4 块 H100（Fir 文档推荐的写法）
#SBATCH --gpus-per-node=h100:4

# GPU 节点一共就 48 核，全部给这个作业
#SBATCH --cpus-per-task=48

# 内存和时间按你需要调
#SBATCH --mem=128G
#SBATCH --time=24:00:00

#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"

module load StdEnv/2023

# 你用 uv 创建的 venv
source .venv/bin/activate

# 4 块 GPU 对应 4 个 spawn 出来的进程
python src/train/train.py \
    model.scale_by_sigma=False \
    worker.ngpus=4
