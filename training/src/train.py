import os
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    Qwen2VLProcessor
)

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modeling_lavit import LaViTQwen2VL, LaViTConfig
from dataset import LaViTDataset, LaViTCollator
from trainer import LaViTTrainer

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    v_top_dim: int = field(default=5120, metadata={"help": "Dimension of V_top features"})
    loss_scale_vtop: float = field(default=1.0, metadata={"help": "Weight for V_top loss"})
    loss_scale_traj: float = field(default=1.0, metadata={"help": "Weight for Trajectory loss"})
    freeze_vision: bool = field(default=True, metadata={"help": "Whether to freeze vision encoder"})
    freeze_llm: bool = field(default=False, metadata={"help": "Whether to freeze LLM"})
    use_trajectory_supervision: bool = field(
        default=True, 
        metadata={"help": "Ablation flag: whether to enable trajectory supervision (if False, traj_head is not created)."}
    )
    
    # Two-stage training configuration
    training_stage: int = field(
        default=0, 
        metadata={
            "help": "Training stage: 0=original (no bottleneck), 1=bottleneck (visual info flows only through <lvr>), 2=joint (standard mask with trained <lvr>)"
        }
    )
    bottleneck_block_prompt: bool = field(
        default=True,
        metadata={"help": "In stage 1, also block prompt tokens from seeing image tokens (prevents info leakage)"}
    )

@dataclass
class DataArguments:
    data_json: str = field(metadata={"help": "Path to enriched data json"})
    image_root: str = field(metadata={"help": "Path to image root directory"})
    v_top_dir: str = field(metadata={"help": "Path to V_top tensors directory"})
    attention_dir: str = field(metadata={"help": "Path to attention maps directory"})
    max_samples: Optional[int] = field(default=None, metadata={"help": "For debugging, truncate dataset"})
    min_pixels: Optional[int] = field(default=256*28*28, metadata={"help": "Minimum number of pixels for image"})
    max_pixels: Optional[int] = field(default=1280*28*28, metadata={"help": "Maximum number of pixels for image"})
    num_lavit_tokens: int = field(default=4, metadata={"help": "Number of <lvr> tokens to insert (default: 4, can try 6, 8, etc.)"})

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Fix boolean parsing issues: HfArgumentParser may parse string "False" as True.
    # We normalize all boolean-like fields manually.
    if isinstance(model_args.use_trajectory_supervision, str):
        model_args.use_trajectory_supervision = model_args.use_trajectory_supervision.lower() in ('true', '1', 'yes', 't')
    if isinstance(model_args.freeze_vision, str):
        model_args.freeze_vision = model_args.freeze_vision.lower() in ('true', '1', 'yes', 't')
    if isinstance(model_args.freeze_llm, str):
        model_args.freeze_llm = model_args.freeze_llm.lower() in ('true', '1', 'yes', 't')
    if isinstance(model_args.bottleneck_block_prompt, str):
        model_args.bottleneck_block_prompt = model_args.bottleneck_block_prompt.lower() in ('true', '1', 'yes', 't')

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    # 1. Load Processor
    try:
        processor = Qwen2VLProcessor.from_pretrained(
            model_args.model_name_or_path, 
            min_pixels=data_args.min_pixels, 
            max_pixels=data_args.max_pixels
        )
    except Exception as e:
        logger.warning(f"Failed to load processor from {model_args.model_name_or_path}: {e}")
        # Fallback or exit?
        # Qwen2VL usually needs 'Qwen/Qwen2.5-VL-3B-Instruct'
        raise e

    # Add <lvr1>, <lvr2>, <lvr3>, <lvr4> tokens
    special_tokens = ["<lvr1>", "<lvr2>", "<lvr3>", "<lvr4>"]
    tokens_to_add = [token for token in special_tokens if token not in processor.tokenizer.get_vocab()]
    if tokens_to_add:
        processor.tokenizer.add_tokens(tokens_to_add, special_tokens=True)
        logger.info(f"Added LaViT tokens to tokenizer: {tokens_to_add}")

    # 2. Dataset
    logger.info(f"Using {data_args.num_lavit_tokens} numbered LaViT tokens (<lvr1>, <lvr2>, etc.) per sample")
    dataset = LaViTDataset(
        data_json_path=data_args.data_json,
        image_root=data_args.image_root,
        v_top_dir=data_args.v_top_dir,
        attention_dir=data_args.attention_dir,
        processor=processor,
        max_samples=data_args.max_samples,
        num_lavit_tokens=data_args.num_lavit_tokens
    )
    
    # 3. Model
    # Load config first to inject custom args
    config = LaViTConfig.from_pretrained(model_args.model_name_or_path)
    config.v_top_dim = model_args.v_top_dim
    config.loss_scale_vtop = model_args.loss_scale_vtop
    config.loss_scale_traj = model_args.loss_scale_traj
    config.training_stage = model_args.training_stage
    config.bottleneck_block_prompt = model_args.bottleneck_block_prompt
    # Force override of use_trajectory_supervision so that ablation arguments take effect.
    # Important: must be set before loading the model config and must be a proper bool.
    use_traj_supervision = bool(model_args.use_trajectory_supervision)
    config.use_trajectory_supervision = use_traj_supervision
    
    # Print config summary to confirm that values are set as expected.
    print(f"\n{'='*60}")
    print(f"CONFIG SETTINGS:")
    print(f"  use_trajectory_supervision: {config.use_trajectory_supervision} (type: {type(config.use_trajectory_supervision)})")
    print(f"  model_args.use_trajectory_supervision: {model_args.use_trajectory_supervision} (type: {type(model_args.use_trajectory_supervision)})")
    print(f"{'='*60}\n")
    
    logger.info(f"Training Stage: {model_args.training_stage} (0=original, 1=bottleneck, 2=joint)")
    logger.info(f"Trajectory Supervision: {'Enabled' if config.use_trajectory_supervision else 'Disabled (Ablation)'}")
    logger.info(f"use_trajectory_supervision value: {config.use_trajectory_supervision} (type: {type(config.use_trajectory_supervision)})")
    
    # Load Model
    # Note: LaViTQwen2VL will load weights from Qwen2VL base
    # We might need to handle safe_serialization explicitly or allowed_missing_keys
    # because v_top_head and traj_head are new and random initialized.
    model = LaViTQwen2VL.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True 
    )
    
    # Resize embeddings for numbered LaViT tokens
    model.resize_token_embeddings(len(processor.tokenizer))
    
    # Freeze logic
    if model_args.freeze_vision:
        # Freeze 'visual' module in Qwen2VL
        for param in model.model.visual.parameters():
            param.requires_grad = False
        logger.info("Froze Vision Encoder")
        
    if model_args.freeze_llm:
        logger.info("Freezing LLM (text model) parameters...")
        for name, param in model.model.named_parameters():
            if "visual" not in name:
                param.requires_grad = False
        
        # Re-enable gradients for custom heads and embeddings (to train numbered LaViT tokens)
        for param in model.v_top_head.parameters():
            param.requires_grad = True
        if model.traj_head is not None:
            for param in model.traj_head.parameters():
                param.requires_grad = True
        model.get_input_embeddings().weight.requires_grad = True
        logger.info("Kept custom heads and embeddings trainable")

    # 4. Trainer
    collator = LaViTCollator(processor)
    
    trainer = LaViTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer # Trainer might expect this for saving
    )
    
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        # Also save the processor so that later stages can reload checkpoints safely.
        processor.save_pretrained(training_args.output_dir)
        
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

if __name__ == "__main__":
    main()
