#!/bin/bash
#SBATCH --job-name=yuhao-sedd-debug
#SBATCH --account=def-lincai_gpu     # 用 scontrol 里看到的 GPU account
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:2                # 只要 2 块 H100
#SBATCH --cpus-per-task=6            # 对齐 Fir 文档，一个 CCD = 6 cores
#SBATCH --mem=32G                    # 内存给少一点
#SBATCH --time=00:20:00              # 最多跑 30 分钟，debug 用
#SBATCH --output=slurm_logs/%x-%j.out

# 保证在提交目录
cd "$SLURM_SUBMIT_DIR"

# Fir 基础环境
module load StdEnv/2023

# 你的 uv 虚拟环境
source ~/workspace/data-sythesis-research/.venv/bin/activate

# 跑训练：2 块 GPU，对应 2 个进程
# 如果你有调试用的 config 覆盖项，比如只跑几步，可以在这里一起加
python src/train/train.py \
    worker=fir-test \
    model.scale_by_sigma=False
