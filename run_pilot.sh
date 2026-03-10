#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/src"

CHECKPOINT_DIR="pilot_checkpoints"
FISHER_DIR="pilot_fishers"
RESULTS_DIR="pilot_results"
NUM_SEEDS=40
EPOCH=249  # final epoch (0-indexed, 250 epochs)
MAX_PARALLEL=${MAX_PARALLEL:-4}

echo "=== STAGE 1: Training ${NUM_SEEDS} models ==="
for seed in $(seq 0 $((NUM_SEEDS - 1))); do
    echo "Training seed $seed..."
    python cifar10_resnet20_train.py \
        --seed $seed \
        --data-split both \
        --checkpoint-dir "$CHECKPOINT_DIR/seed${seed}" \
        --wandb-mode disabled &

    # Limit parallelism
    if (( $(jobs -r | wc -l) >= $MAX_PARALLEL )); then
        wait -n
    fi
done
wait
echo "Stage 1 complete."

echo "=== STAGE 2: Computing Fisher ==="
for seed in $(seq 0 $((NUM_SEEDS - 1))); do
    echo "Fisher for seed $seed..."
    python compute_fisher.py \
        --checkpoint "$CHECKPOINT_DIR/seed${seed}/checkpoint${EPOCH}" \
        --output "$FISHER_DIR/seed${seed}_fisher.pkl" \
        --num-samples 1000 &

    if (( $(jobs -r | wc -l) >= $MAX_PARALLEL )); then
        wait -n
    fi
done
wait
echo "Stage 2 complete."

echo "=== STAGE 3: Matching + Evaluation ==="
for pair_id in $(seq 0 19); do
    seed_a=$((pair_id * 2))
    seed_b=$((pair_id * 2 + 1))
    echo "Pair $pair_id: seed $seed_a vs seed $seed_b"
    python run_matching_eval.py \
        --pair-id $pair_id \
        --model-a "$CHECKPOINT_DIR/seed${seed_a}/checkpoint${EPOCH}" \
        --model-b "$CHECKPOINT_DIR/seed${seed_b}/checkpoint${EPOCH}" \
        --fisher-a "$FISHER_DIR/seed${seed_a}_fisher.pkl" \
        --fisher-b "$FISHER_DIR/seed${seed_b}_fisher.pkl" \
        --output-dir "$RESULTS_DIR"
done
echo "Stage 3 complete."

echo "=== STAGE 4: Statistical Analysis ==="
python analyze_results.py --input "$RESULTS_DIR/interpolation_summary.csv"
echo "=== PILOT COMPLETE ==="
