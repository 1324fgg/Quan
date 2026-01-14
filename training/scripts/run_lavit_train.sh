#!/bin/bash
set -x

# Ensure we are in the training directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"


export DATA_JSON="/root/autodl-tmp/ViLR/trajectories/trajectories_vtop_enriched_all_final.json"   
export IMAGE_ROOT="/root/autodl-tmp/ViLR/data/Visual-CoT-full"
export VTOP_DIR="/root/autodl-tmp/ViLR/trajectories/viscot_vtop_only/tensors" # This dir is now handled by absolute paths in JSON
export ATTN_DIR="/root/autodl-tmp/ViLR/trajectories/bbox_attention"


# Check if model path exists, else use HF Hub ID
if [ ! -d "$MODEL_PATH" ]; then
    echo "Local model not found at $MODEL_PATH, using HF ID Qwen/Qwen2.5-VL-3B-Instruct"
    MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
fi

# ============================================================================
# Training Stage Control (Two-Stage Loss-Based Bottleneck Training)
# ============================================================================
# --training_stage:
#   0 = Original training (CE + V_top + Traj losses, standard method)
#   1 = Bottleneck stage (ONLY V_top + Traj losses, CE disabled!)
#       This forces <lvr> tokens to learn visual representations because
#       only <lvr>-related losses drive the learning.
#   2 = Joint stage (CE + V_top + Traj, with pre-trained <lvr> from stage 1)
#
# Why loss-based instead of attention mask?
#   Qwen2.5-VL uses attention_mask for RoPE position calculation, which
#   requires 2D mask. Direct 4D attention mask modification is not compatible.
#   Loss-based bottleneck achieves similar effect through loss weighting.
#
# Recommended workflow:
#   1. Run with --training_stage 1 for first half of training
#   2. Run with --training_stage 2 --resume_from_checkpoint <stage1_ckpt>
# Or set --training_stage 0 to use original method (backward compatible)
# ============================================================================

# Single Stage Training Configuration (Original Method)
# Training Stage Options:
#   0 = Original training (CE + V_top + Traj losses, standard single-stage method)
#   1 = Bottleneck stage (ONLY V_top + Traj losses, CE disabled)
#   2 = Joint stage (CE + V_top + Traj, requires pre-trained checkpoint from stage 1)
TRAINING_STAGE=0
BOTTLENECK_BLOCK_PROMPT=True  # Not used in stage 0, but kept for consistency
MODEL_PATH="/root/autodl-tmp/my_qwen_model/Qwen2.5-VL-3B-Instruct"  # Use base model for single-stage
OUTPUT_DIR="/root/autodl-tmp/ViLR/training/checkpoints/single_stage_4lavit-1000steps"

# Number of <lvr> tokens (default: 4, can try 6, 8, etc. for spatial reasoning tasks)
# Usage: Set NUM_LAVIT_TOKENS environment variable before running, or modify this line
# Example: export NUM_LAVIT_TOKENS=6 && bash run_lavit_train.sh
NUM_LAVIT_TOKENS=${NUM_LAVIT_TOKENS:-4}

# ============================================================================
# 消融实验配置 (Ablation Study Configuration)
# ============================================================================
# --use_trajectory_supervision:
#   True  = 使用trajectory监督（默认，创建traj_head并计算trajectory loss）
#   False = 禁用trajectory监督（消融实验，不创建traj_head，只使用V_top loss）
#
# 消融实验示例：
#   1. 完整模型（默认）：
#      --use_trajectory_supervision True --loss_scale_vtop 0.3 --loss_scale_traj 0.3
#   
#   2. 消融实验（无trajectory监督）：
#      --use_trajectory_supervision False --loss_scale_vtop 0.3 --loss_scale_traj 0.0
#
# 注意：当use_trajectory_supervision=False时，traj_head不会被创建，可以节省显存
# ============================================================================
USE_TRAJECTORY_SUPERVISION=${USE_TRAJECTORY_SUPERVISION:-True}

python src/train.py \
    --model_name_or_path $MODEL_PATH \
    --data_json $DATA_JSON \
    --image_root $IMAGE_ROOT \
    --v_top_dir $VTOP_DIR \
    --attention_dir $ATTN_DIR \
    --output_dir $OUTPUT_DIR \
    --min_pixels 200704 \
    --max_pixels 4194304 \
    --max_steps 1000 \
    --loss_scale_vtop 0.3 \
    --loss_scale_traj 0.3 \
    --training_stage $TRAINING_STAGE \
    --bottleneck_block_prompt $BOTTLENECK_BLOCK_PROMPT \
    --use_trajectory_supervision $USE_TRAJECTORY_SUPERVISION \
    --num_lavit_tokens $NUM_LAVIT_TOKENS \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --save_steps 200 \
    --save_total_limit 3 \
    --report_to "wandb" \
    --run_name "lavit-single-stage-4lavit-1000steps" \
    --remove_unused_columns False \
    --dataloader_num_workers 0 \
    --gradient_checkpointing True \
    --bf16 True \
    --max_samples 14567 \
    --do_train
