#!/bin/bash

# Example script to evaluate ViLR on MMVP Benchmark

# Set GPU device (modify as needed, e.g., "0" for GPU 0, "1" for GPU 1, "0,1" for multiple GPUs)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# 1. Configuration
CHECKPOINT="/root/autodl-tmp/ViLR/training/checkpoints/debug_run/checkpoint-1250"
DATA_ROOT="/root/autodl-tmp/ViLR/data/MMVP"
OUTPUT_FILE="/root/autodl-tmp/ViLR/training/eval_results/mmvp_results.jsonl"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo "Checkpoint not found at $CHECKPOINT"
    # Try fallback
    CHECKPOINT=$(ls -d /root/autodl-tmp/ViLR/training/checkpoints/debug_run/checkpoint* | tail -n 1)
    if [ -z "$CHECKPOINT" ]; then
        echo "No checkpoint found. Please train first."
        exit 1
    fi
    echo "Using latest found: $CHECKPOINT"
fi

echo "Evaluating MMVP..."
echo "Model: $CHECKPOINT"
echo "Output: $OUTPUT_FILE"

# 2. Run Evaluation
# Note: Remove --force_lvr if your model can generate lvr tokens autonomously
python ../../training/src/evaluation/run_eval.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "mmvp" \
    --data_root "$DATA_ROOT" \
    --output_file "$OUTPUT_FILE" \
    --max_samples 10 \
    --force_lvr

echo "Done."
