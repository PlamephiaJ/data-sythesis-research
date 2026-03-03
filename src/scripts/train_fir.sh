#!/bin/bash
#SBATCH --job-name=yuhao-sedd          # 作业名字，随便
#SBATCH --account=def-lincai          # 你的 allocation
# 请求 GPU：Fir 上要用 gpus= / gpus-per-node=，而不是 gres=
#SBATCH --gpus=h100:4                 # 1 块完整 H100-80GB
#SBATCH --cpus-per-task=128           # 给这一块 GPU 分 12 个 CPU 核心（48 核总共足够分 4 块）
#SBATCH --mem=128G                     # 内存自己估一个合理值
#SBATCH --time=24:00:00               # 最长跑 24 小时
#SBATCH --output=logs/%x-%j.out       # 日志输出文件

# 一般 sbatch 默认就在你提交的目录，这里稳妥一点
cd "$SLURM_SUBMIT_DIR"

# Fir 文档里建议加载 StdEnv，其他模块你 uv 已经搞定，就不用再 load python/cuda 了
module load StdEnv/2023

# 启用你用 uv sync 建好的虚拟环境（假设就在项目根目录 .venv）
source .venv/bin/activate

# 跑训练脚本：这里用 1 块 GPU，所以 worker.ngpus=1
python src/train/train.py \
    model.scale_by_sigma=False \
    worker.ngpus=4
