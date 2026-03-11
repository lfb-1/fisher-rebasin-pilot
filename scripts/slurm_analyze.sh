#!/bin/bash
# Stage 4: Statistical analysis (CPU-only, single job)
# Depends on Stage 3 completing all 20 pairs
# Usage: sbatch --dependency=afterok:<eval_jobid> scripts/slurm_analyze.sh

#SBATCH --job-name=frb-analyze
#SBATCH --partition=sablab
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/share/sablab/nfs04/users/fl453/fisher-rebasin-pilot/logs/analyze_%j.log

set -euo pipefail

REPO_DIR="/share/sablab/nfs04/users/fl453/fisher-rebasin-pilot"
ENV_DIR="$REPO_DIR/env"
RESULTS_DIR="$REPO_DIR/pilot_results"

echo "=== Statistical Analysis ==="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

cd "$REPO_DIR/src"

SUMMARY="$RESULTS_DIR/interpolation_summary.csv"
if [ ! -f "$SUMMARY" ]; then
    echo "ERROR: $SUMMARY not found — did all eval jobs finish?"
    exit 1
fi

# Count completed pairs
N_ROWS=$(tail -n +2 "$SUMMARY" | wc -l)
echo "Rows in summary CSV: $N_ROWS (expected 100 = 20 pairs × 5 methods)"

python analyze_results.py --input "$SUMMARY"

echo "=== Analysis complete. Results in $RESULTS_DIR ==="
