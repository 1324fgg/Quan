#!/bin/bash
set -x

# ============================================================================
# Two-Stage Loss-Based Bottleneck Training for ViLR
# ============================================================================
# This script implements a two-stage training strategy inspired by LIVR paper:
#
# Stage 1 (Bottleneck via Loss Control):
#   - CE Loss DISABLED, only V_top + Trajectory losses active
#   - Forces <lvr> tokens to learn visual representations because
#     only <lvr>-related losses drive the learning
#   - Similar effect to attention bottleneck, achieved through loss weighting
#
# Stage 2 (Joint):
#   - CE Loss RE-ENABLED, all losses active
#   - Model learns to generate text using pre-trained <lvr> representations
#   - <lvr> tokens now carry meaningful visual info from Stage 1
#
# Note: Direct attention mask modification is not compatible with Qwen2.5-VL's
# RoPE position calculation. Loss-based approach achieves similar effect.
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

export MODEL_PATH=${MODEL_PATH:-"/root/autodl-tmp/my_qwen_model/Qwen2.5-VL-3B-Instruct"}
export DATA_JSON=${DATA_JSON:-"/root/autodl-tmp/ViLR_OpenSource/output/viscot_construction/viscot_training_final.json"}
export IMAGE_ROOT=${IMAGE_ROOT:-"/root/autodl-tmp/ViLR/data/images"}
export VTOP_DIR=${VTOP_DIR:-"/root/autodl-tmp/ViLR_OpenSource/output/viscot_construction/vtop/tensors"}
export ATTN_DIR=${ATTN_DIR:-"/root/autodl-tmp/ViLR_OpenSource/output/viscot_construction/attention/tensors"}

# Output directories for each stage
STAGE1_OUTPUT=${STAGE1_OUTPUT:-"/root/autodl-tmp/ViLR_OpenSource/output/checkpoints/two_stage/stage1_bottleneck"}
STAGE2_OUTPUT=${STAGE2_OUTPUT:-"/root/autodl-tmp/ViLR_OpenSource/output/checkpoints/two_stage/stage2_joint"}

# Training steps for each stage (total = 800, split 50-50)
STAGE1_STEPS=400
STAGE2_STEPS=600

if [ ! -d "$MODEL_PATH" ]; then
    echo "Local model not found at $MODEL_PATH, using HF ID Qwen/Qwen2.5-VL-3B-Instruct"
    MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
fi

echo "============================================"
echo "Stage 1: Bottleneck Training (${STAGE1_STEPS} steps)"
echo "============================================"

python src/train.py \
    --model_name_or_path $MODEL_PATH \
    --data_json $DATA_JSON \
    --image_root $IMAGE_ROOT \
    --v_top_dir $VTOP_DIR \
    --attention_dir $ATTN_DIR \
    --output_dir $STAGE1_OUTPUT \
    --min_pixels 200704 \
    --max_pixels 4194304 \
    --max_steps $STAGE1_STEPS \
    --loss_scale_vtop 0.3 \
    --loss_scale_traj 0.3 \
    --training_stage 1 \
    --bottleneck_block_prompt True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.06 \
    --logging_steps 1 \
    --save_steps 200 \
    --save_total_limit 3 \
    --report_to "wandb" \
    --run_name "vilr-two-stage-s1-bottleneck" \
    --remove_unused_columns False \
    --dataloader_num_workers 0 \
    --gradient_checkpointing True \
    --bf16 True \
    --max_samples 8765 \
    --do_train

# Check if stage 1 completed successfully
if [ $? -ne 0 ]; then
    echo "Stage 1 failed! Exiting..."
    exit 1
fi

echo ""
echo "============================================"
echo "Stage 2: Joint Training (${STAGE2_STEPS} steps)"
echo "Loading checkpoint from Stage 1..."
echo "============================================"

# Find the latest checkpoint from stage 1
LATEST_CKPT=$(ls -td ${STAGE1_OUTPUT}/checkpoint-* 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "No checkpoint found in Stage 1 output! Using final model..."
    LATEST_CKPT=$STAGE1_OUTPUT
fi

echo "Resuming from: $LATEST_CKPT"

python src/train.py \
    --model_name_or_path $LATEST_CKPT \
    --data_json $DATA_JSON \
    --image_root $IMAGE_ROOT \
    --v_top_dir $VTOP_DIR \
    --attention_dir $ATTN_DIR \
    --output_dir $STAGE2_OUTPUT \
    --min_pixels 200704 \
    --max_pixels 4194304 \
    --max_steps $STAGE2_STEPS \
    --loss_scale_vtop 0.3 \
    --loss_scale_traj 0.3 \
    --training_stage 2 \
    --bottleneck_block_prompt True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --save_steps 200 \
    --save_total_limit 3 \
    --report_to "wandb" \
    --run_name "vilr-two-stage-s2-joint" \
    --remove_unused_columns False \
    --dataloader_num_workers 0 \
    --gradient_checkpointing True \
    --bf16 True \
    --max_samples 8765 \
    --do_train

echo ""
echo "============================================"
echo "Two-Stage Training Complete!"
echo "Stage 1 checkpoint: $STAGE1_OUTPUT"
echo "Stage 2 (final) checkpoint: $STAGE2_OUTPUT"
echo "============================================"

