#!/bin/bash
#SBATCH --job-name=yuhao-sedd-debug
#SBATCH --account=def-lincai_gpu     # 用 scontrol 里看到的 GPU account
#SBATCH --gpus=h100:1                # 只要 1 块 H100
#SBATCH --cpus-per-task=6            # 对齐 Fir 文档，一个 CCD = 6 cores
#SBATCH --mem=32G                    # 内存给少一点
#SBATCH --time=00:30:00              # 最多跑 30 分钟，debug 用
#SBATCH --output=slurm_logs/%x-%j.out

# 保证在提交目录
cd "$SLURM_SUBMIT_DIR"

# Fir 基础环境
module load StdEnv/2023

# 你的 uv 虚拟环境
source .venv/bin/activate

# 跑训练：1 块 GPU，对应 1 个进程
# 如果你有调试用的 config 覆盖项，比如只跑几步，可以在这里一起加
python src/train/train.py \
    worker=fir \
    model.scale_by_sigma=False \
    worker.ngpus=1
