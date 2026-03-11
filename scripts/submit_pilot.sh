#!/bin/bash
# Master submission script — chains all 4 stages with SLURM dependencies
# Usage: bash scripts/submit_pilot.sh [--dry-run]
# 
# Stage 1: Train 40 models           (array 0-39, ~20 min each on GPU)
# Stage 2: Compute Fisher 40 models  (array 0-39, ~5 min each, after stage 1)
# Stage 3: Eval 20 pairs             (array 0-19, ~60-90 min each, after stage 2)
# Stage 4: Statistical analysis      (1 CPU job, after stage 3)

set -euo pipefail

REPO_DIR="/share/sablab/nfs04/users/fl453/fisher-rebasin-pilot"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "=== DRY RUN — no jobs submitted ==="
fi

# Create log dir
mkdir -p "$REPO_DIR/logs"

submit() {
    if [[ $DRY_RUN == 1 ]]; then
        echo "[DRY RUN] sbatch $*"
        echo "DUMMY_JOB_ID"
    else
        sbatch "$@" | awk '{print $NF}'
    fi
}

echo "=== Submitting Fisher Re-Basin Pilot ==="

# Stage 1: Train (40 array jobs, no dependency)
echo "Stage 1: Training 40 models..."
TRAIN_JID=$(submit \
    --array=0-39 \
    "$REPO_DIR/scripts/slurm_train.sh")
echo "  Train job: $TRAIN_JID"

# Stage 2: Fisher (40 array jobs, after ALL train jobs complete)
echo "Stage 2: Computing Fisher..."
FISHER_JID=$(submit \
    --array=0-39 \
    --dependency=afterok:"$TRAIN_JID" \
    "$REPO_DIR/scripts/slurm_fisher.sh")
echo "  Fisher job: $FISHER_JID"

# Stage 3: Eval (20 array jobs, after ALL Fisher jobs complete)
echo "Stage 3: Matching + Evaluation..."
EVAL_JID=$(submit \
    --array=0-19 \
    --dependency=afterok:"$FISHER_JID" \
    "$REPO_DIR/scripts/slurm_eval.sh")
echo "  Eval job: $EVAL_JID"

# Stage 4: Analysis (1 job, after ALL eval jobs complete)
echo "Stage 4: Statistical analysis..."
ANALYZE_JID=$(submit \
    --dependency=afterok:"$EVAL_JID" \
    "$REPO_DIR/scripts/slurm_analyze.sh")
echo "  Analyze job: $ANALYZE_JID"

echo ""
echo "=== Pipeline submitted ==="
echo "  Train:   $TRAIN_JID  (40 jobs)"
echo "  Fisher:  $FISHER_JID  (40 jobs)"
echo "  Eval:    $EVAL_JID   (20 jobs)"
echo "  Analyze: $ANALYZE_JID  (1 job)"
echo ""
echo "Monitor with: squeue -u fl453"
echo "Results will appear in: $REPO_DIR/pilot_results/"
