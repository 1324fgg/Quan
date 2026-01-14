#!/bin/bash

# ViLR Viscot Dataset Construction One-Click Script
# Automates: Trajectory Extraction, VTOP Extraction, and Training Data Preparation

set -e

# --- Configuration ---
DATA_FILE=${DATA_FILE:-"/root/autodl-tmp/ViLR/data/viscot_train.json"}
IMAGE_DIR=${IMAGE_DIR:-"/root/autodl-tmp/ViLR/data/images"}
MODEL_PATH=${MODEL_PATH:-"/root/autodl-tmp/my_qwen_model/Qwen/Qwen2.5-VL-32B-Instruct"}
OUTPUT_DIR=${OUTPUT_DIR:-"/root/autodl-tmp/ViLR_OpenSource/output/viscot_construction"}
MAX_SAMPLES=${MAX_SAMPLES:-10}  # Small default for testing

TRAJ_OUTPUT="${OUTPUT_DIR}/attention"
VTOP_OUTPUT="${OUTPUT_DIR}/vtop"
FINAL_JSON="${OUTPUT_DIR}/viscot_training_final.json"

echo "=========================================================="
echo "Starting ViLR Viscot Dataset Construction"
echo "=========================================================="
echo "Data File: $DATA_FILE"
echo "Model Path: $MODEL_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================================="

mkdir -p "$OUTPUT_DIR"

# 1. Extract Trajectories (Attention) with Correctness Validation
echo "[Step 1/3] Extracting Trajectories and Validating Correctness..."
python /root/autodl-tmp/ViLR_OpenSource/data_construction/viscot/extract_viscot_trajectories.py \
    --data_file "$DATA_FILE" \
    --base_image_dir "$IMAGE_DIR" \
    --model_path "$MODEL_PATH" \
    --method attention \
    --max_samples "$MAX_SAMPLES" \
    --output_dir "$TRAJ_OUTPUT" \
    --enable_validation

# 2. Extract VTOP Tensors
echo "[Step 2/3] Extracting VTOP Tensors..."
python /root/autodl-tmp/ViLR_OpenSource/data_construction/viscot/extract_viscot_vtop.py \
    --data_file "$DATA_FILE" \
    --base_image_dir "$IMAGE_DIR" \
    --model_path "$MODEL_PATH" \
    --max_samples "$MAX_SAMPLES" \
    --output_dir "$VTOP_OUTPUT" \
    --enable_validation

# 3. Final Preparation: Concentration Verification and Merging
echo "[Step 3/3] Performing Concentration Verification and Merging..."
python /root/autodl-tmp/ViLR_OpenSource/data_construction/viscot/prepare_viscot_training_data.py \
    --traj_file "${TRAJ_OUTPUT}/trajectories.json" \
    --vtop_file "${VTOP_OUTPUT}/trajectories_vtop.json" \
    --original_data "$DATA_FILE" \
    --output_file "$FINAL_JSON" \
    --concentration_threshold 0.1 \
    --traj_root "$TRAJ_OUTPUT" \
    --vtop_root "$VTOP_OUTPUT"

echo "=========================================================="
echo "Construction Complete!"
echo "Final Training JSON: $FINAL_JSON"
echo "=========================================================="
