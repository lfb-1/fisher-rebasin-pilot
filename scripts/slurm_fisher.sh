#!/bin/bash
# Stage 2: Compute diagonal Fisher for each model (array job)
# Depends on Stage 1 completing all 40 seeds
# Usage: sbatch --array=0-39 --dependency=afterok:<train_jobid> scripts/slurm_fisher.sh

#SBATCH --job-name=frb-fisher
#SBATCH --partition=sablab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/share/sablab/nfs04/users/fl453/fisher-rebasin-pilot/logs/fisher_%A_%a.log

set -euo pipefail

REPO_DIR="/share/sablab/nfs04/users/fl453/fisher-rebasin-pilot"
ENV_DIR="$REPO_DIR/env"
CKPT_DIR="$REPO_DIR/pilot_checkpoints"
FISHER_DIR="$REPO_DIR/pilot_fishers"
SEED=${SLURM_ARRAY_TASK_ID}
EPOCH=249  # final epoch (0-indexed, 250 epochs)

echo "=== Fisher for seed $SEED on $(hostname) ==="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

cd "$REPO_DIR/src"

mkdir -p "$FISHER_DIR"

CKPT="$CKPT_DIR/seed${SEED}/checkpoint${EPOCH}"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi

python compute_fisher.py \
    --checkpoint "$CKPT" \
    --output "$FISHER_DIR/seed${SEED}_fisher.pkl" \
    --num-samples 1000 \
    --seed 42

echo "=== Fisher seed $SEED complete ==="
