#!/bin/bash

# ViLR Evaluation Script for BLINK Benchmark

# Set GPU device (modify as needed, e.g., "0" for GPU 0, "1" for GPU 1, "0,1" for multiple GPUs)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# 1. Configuration
# Default to debug run if not specified
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT="/root/autodl-tmp/ViLR/training/checkpoints/debug_run/checkpoint-1250"
fi

DATA_ROOT="/root/autodl-tmp/ViLR/data/BLINK"
OUTPUT_DIR="/root/autodl-tmp/ViLR/training/eval_results/blink"

# Tasks to evaluate
TASKS=("Counting" "Jigsaw" "Relative_Reflectance" "Spatial_Relation" "Object_Localization" "Relative_Depth" "IQ_Test")

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

echo "Evaluating BLINK Benchmark..."
echo "Model: $CHECKPOINT"
echo "Output Directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# 2. Run Evaluation Loop
for TASK in "${TASKS[@]}"; do
    echo "--------------------------------------------------"
    echo "Running Task: $TASK"
    OUTPUT_FILE="${OUTPUT_DIR}/${TASK}.jsonl"
    
    python /root/autodl-tmp/ViLR/training/src/evaluation/run_eval.py \
        --checkpoint "$CHECKPOINT" \
        --dataset "blink" \
        --data_root "$DATA_ROOT" \
        --task_name "$TASK" \
        --output_file "$OUTPUT_FILE" \
        --max_samples 10 \
        
    echo "Saved to: $OUTPUT_FILE"
done

echo "--------------------------------------------------"
echo "All Tasks Done."

