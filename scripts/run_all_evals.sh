#!/bin/bash

# ViLR Master Evaluation Script
# Runs evaluation on both Blink subsets and MMVP datasets

# Set GPU device
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# 1. Configuration
if [ -z "$CHECKPOINT" ]; then
    echo "Usage: CHECKPOINT=/path/to/checkpoint ./run_all_evals.sh"
    exit 1
fi

echo "=========================================================="
echo "Starting Master Evaluation"
echo "Model: $CHECKPOINT"
echo "=========================================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# 1. Run Blink Evaluation
echo ""
echo "[1/2] Running Blink Evaluation (Counting, Jigsaw, Relative_Reflectance, Spatial_Relation, Object_Localization, Relative_Depth, IQ_Test)..."
export CHECKPOINT=$CHECKPOINT
bash "$REPO_ROOT/evaluation/blink/run_eval_blink.sh"

# 2. Run MMVP Evaluation
echo ""
echo "[2/2] Running MMVP Evaluation..."
bash "$REPO_ROOT/evaluation/mmvp/run_eval_mmvp.sh"

echo ""
echo "=========================================================="
echo "Master Evaluation Complete!"
echo "=========================================================="
