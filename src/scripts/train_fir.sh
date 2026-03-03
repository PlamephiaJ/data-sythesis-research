#!/bin/bash
#SBATCH --job-name=yuhao-sedd          # 作业名
#SBATCH --account=def-lincai         # <-- 按 Fir 文档/你PI的 account 改
#SBATCH --cpus-per-task=128            # CPU 核心数
#SBATCH --gres=gpu:4                 # <-- 申请 GPU 数量；如果想 4 卡就写 gpu:4
#SBATCH --mem=256G                    # 内存
#SBATCH --time=24:00:00              # 运行时间上限
#SBATCH --output=logs/%x-%j.out      # 输出日志
#SBATCH --mail-type=ALL              # 可选：邮件通知
#SBATCH --mail-user=yuhaoc@uvic.ca          # 可选：你的邮箱

# 1. Slurm 默认在提交目录，稳妥起见可以加上
cd "$SLURM_SUBMIT_DIR"

# 2. 加载模块（按实际情况改）
module load StdEnv/2023

# 3. 激活虚拟环境
source .venv/bin/activate

# 4. 运行训练
python src/train/train.py \
    model.scale_by_sigma=False \
    worker.ngpus=4
