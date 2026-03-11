#!/bin/bash
# Stage 3: Weight matching + interpolation evaluation (array job, 20 pairs)
# Depends on Stage 2 completing all 40 Fishers
# Usage: sbatch --array=0-19 --dependency=afterok:<fisher_jobid> scripts/slurm_eval.sh

#SBATCH --job-name=frb-eval
#SBATCH --partition=sablab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/share/sablab/nfs04/users/fl453/fisher-rebasin-pilot/logs/eval_%A_%a.log

set -euo pipefail

REPO_DIR="/share/sablab/nfs04/users/fl453/fisher-rebasin-pilot"
ENV_DIR="$REPO_DIR/env"
CKPT_DIR="$REPO_DIR/pilot_checkpoints"
FISHER_DIR="$REPO_DIR/pilot_fishers"
RESULTS_DIR="$REPO_DIR/pilot_results"
PAIR_ID=${SLURM_ARRAY_TASK_ID}
EPOCH=249

# Each pair: seed_a = 2*pair_id, seed_b = 2*pair_id + 1
SEED_A=$((PAIR_ID * 2))
SEED_B=$((PAIR_ID * 2 + 1))

echo "=== Eval pair $PAIR_ID (seeds $SEED_A vs $SEED_B) on $(hostname) ==="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

cd "$REPO_DIR/src"

mkdir -p "$RESULTS_DIR"

# Verify inputs exist
for f in \
    "$CKPT_DIR/seed${SEED_A}/checkpoint${EPOCH}" \
    "$CKPT_DIR/seed${SEED_B}/checkpoint${EPOCH}" \
    "$FISHER_DIR/seed${SEED_A}_fisher.pkl" \
    "$FISHER_DIR/seed${SEED_B}_fisher.pkl"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f"
        exit 1
    fi
done

python run_matching_eval.py \
    --pair-id "$PAIR_ID" \
    --model-a "$CKPT_DIR/seed${SEED_A}/checkpoint${EPOCH}" \
    --model-b "$CKPT_DIR/seed${SEED_B}/checkpoint${EPOCH}" \
    --fisher-a "$FISHER_DIR/seed${SEED_A}_fisher.pkl" \
    --fisher-b "$FISHER_DIR/seed${SEED_B}_fisher.pkl" \
    --methods euclidean magnitude fisher_inner fisher_l2_sym activation_hybrid \
    --output-dir "$RESULTS_DIR" \
    --seed 0

echo "=== Pair $PAIR_ID complete ==="
