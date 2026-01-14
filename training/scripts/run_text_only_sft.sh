#!/bin/bash
set -x

# ============================================================================
# Text-Only SFT Training Script (Ablation Study)
# ============================================================================
# This script trains a standard Qwen2.5-VL model with text-only supervision.
# NO <lvr> tokens, NO v_top supervision, NO trajectory supervision.
# This serves as a baseline/ablation study to compare against ViLR method.
# ============================================================================

# Ensure we are in the training directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

export MODEL_PATH="/root/autodl-tmp/my_qwen_model/Qwen2.5-VL-3B-Instruct"
export DATA_JSON="/root/autodl-tmp/ViLR/trajectories/trajectories_vtop_enriched_correct_only.json"
export IMAGE_ROOT="/root/autodl-tmp/ViLR/data/Visual-CoT-full"
export OUTPUT_DIR="/root/autodl-tmp/ViLR/training/checkpoints/text_only_sft_baseline"

# Check if model path exists, else use HF Hub ID
if [ ! -d "$MODEL_PATH" ]; then
    echo "Local model not found at $MODEL_PATH, using HF ID Qwen/Qwen2.5-VL-3B-Instruct"
    MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
fi

echo "============================================"
echo "Text-Only SFT Training (Ablation Study)"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATA_JSON"
echo "Output: $OUTPUT_DIR"
echo "============================================"
echo "Configuration:"
echo "  - NO <lvr> tokens"
echo "  - NO v_top supervision"
echo "  - NO trajectory supervision"
echo "  - Standard text generation loss only"
echo "============================================"

python src/train_text_only.py \
    --model_name_or_path $MODEL_PATH \
    --data_json $DATA_JSON \
    --image_root $IMAGE_ROOT \
    --output_dir $OUTPUT_DIR \
    --min_pixels 200704 \
    --max_pixels 4194304 \
    --max_steps 800 \
    --freeze_vision True \
    --freeze_llm False \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --save_strategy "no" \
    --report_to "wandb" \
    --run_name "text-only-sft-baseline" \
    --remove_unused_columns False \
    --dataloader_num_workers 0 \
    --gradient_checkpointing True \
    --bf16 True \
    --max_samples 8765 \
    --do_train

echo ""
echo "============================================"
echo "Training Complete!"
echo "Checkpoint saved to: $OUTPUT_DIR"
echo "============================================"

