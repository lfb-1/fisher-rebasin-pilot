#!/bin/bash
# Stage 1: Train 40 ResNet-20 models on CIFAR-10 (array job)
# Usage: sbatch --array=0-39 scripts/slurm_train.sh

#SBATCH --job-name=frb-train
#SBATCH --partition=sablab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/share/sablab/nfs04/users/fl453/fisher-rebasin-pilot/logs/train_%A_%a.log

set -euo pipefail

REPO_DIR="/share/sablab/nfs04/users/fl453/fisher-rebasin-pilot"
ENV_DIR="$REPO_DIR/env"
CKPT_DIR="$REPO_DIR/pilot_checkpoints"
SEED=${SLURM_ARRAY_TASK_ID}

echo "=== Training seed $SEED on $(hostname) ==="
nvidia-smi -L

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

cd "$REPO_DIR/src"

mkdir -p "$CKPT_DIR/seed${SEED}"

python cifar10_resnet20_train.py \
    --seed "$SEED" \
    --data-split both \
    --checkpoint-dir "$CKPT_DIR/seed${SEED}" \
    --wandb-mode disabled

echo "=== Seed $SEED training complete ==="
