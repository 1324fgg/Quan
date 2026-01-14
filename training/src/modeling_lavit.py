import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
# Switch to Qwen2_5_VL
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class LaViTConfig(Qwen2_5_VLConfig):
    def __init__(
        self, 
        v_top_dim=5120, 
        loss_scale_vtop=1.0, 
        loss_scale_traj=1.0,
        training_stage=0,  # 0=original, 1=bottleneck (loss-based), 2=joint
        bottleneck_block_prompt=True,  # Kept for future use
        use_trajectory_supervision=True,  # Ablation flag: whether to enable trajectory supervision
        **kwargs
    ):
        super().__init__(**kwargs)
        self.v_top_dim = v_top_dim
        self.loss_scale_vtop = loss_scale_vtop
        self.loss_scale_traj = loss_scale_traj
        self.training_stage = training_stage
        self.bottleneck_block_prompt = bottleneck_block_prompt
        self.use_trajectory_supervision = use_trajectory_supervision


# ============================================================================
# Two-Stage Training Strategy (Hard Bottleneck)
# ============================================================================
# 
# Stage 1 (Hard Bottleneck): 
#   - Answer tokens CANNOT see image tokens directly.
#   - Visual info MUST flow through <lvr> tokens.
#   - This forces the LLM to learn that <lvr> are the only source of truth 
#     for image content during answer generation.
#
# Stage 2 (Joint): 
#   - Restore standard attention mask.
#   - Fine-tune the whole system (Image + <lvr> + Text).
#
# Qwen2.5-VL compatibility: 
#   Qwen2.5-VL uses 2D attention_mask for RoPE calculation. To support 4D 
#   bottleneck masks, we pre-calculate RoPE indices using the original 2D mask 
#   and pass them to the base model, allowing the use of 4D masks in 
#   attention layers without breaking RoPE indexing.
#
# ============================================================================

def create_bottleneck_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vision_start_positions: torch.Tensor,  # [batch]
    vision_end_positions: torch.Tensor,    # [batch]
    lavit_start_positions: torch.Tensor,     # [batch]
    lavit_end_positions: torch.Tensor,       # [batch]
    block_prompt: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create hard bottleneck attention mask for Stage 1 training.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Start with causal mask (lower triangular)
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=device, dtype=dtype)
    ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).clone()
    
    # Apply original attention mask (handle padding)
    expanded_attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype)
    causal_mask = causal_mask * expanded_attn_mask
    
    # Apply bottleneck constraints
    for b in range(batch_size):
        vis_start = vision_start_positions[b].item()
        vis_end = vision_end_positions[b].item()
        lavit_start = lavit_start_positions[b].item()
        lavit_end = lavit_end_positions[b].item()
        
        img_start = vis_start + 1 
        img_end = vis_end          
        answer_start = lavit_end + 1
        
        # Rule 1: Answer tokens CANNOT see image tokens
        if answer_start < seq_len and img_start < img_end:
            causal_mask[b, 0, answer_start:, img_start:img_end] = 0
        
        # Rule 2: Prompt tokens CANNOT see image tokens
        if block_prompt:
            prompt_start = img_end
            prompt_end = lavit_start
            if prompt_start < prompt_end:
                causal_mask[b, 0, prompt_start:prompt_end, img_start:img_end] = 0
                
    return causal_mask

def convert_mask_to_4d(
    attention_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=device, dtype=dtype)
    ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).clone()
    expanded_attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype)
    return causal_mask * expanded_attn_mask

class VTopHead(nn.Module):
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        # Simple MLP Projection
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class TrajectoryHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Projection to create a "Query" vector for attention
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, lavit_state, image_features):
        """
        lavit_state: [Batch, Hidden]
        image_features: [Batch, N_patches, Hidden]
        
        Returns: [Batch, N_patches] (Unnormalized scores)
        """
        # [Batch, 1, Hidden]
        query = self.query_proj(lavit_state).unsqueeze(1)
        # [Batch, Hidden, N_patches]
        keys = image_features.transpose(1, 2)
        
        # Dot product attention: [Batch, 1, N_patches]
        scores = torch.matmul(query, keys).squeeze(1)
        return scores

from dataclasses import dataclass
from typing import Optional

@dataclass
class LaViTOutput(CausalLMOutputWithPast):
    loss_v_top: Optional[torch.FloatTensor] = None
    loss_traj: Optional[torch.FloatTensor] = None

class LaViTQwen2VL(Qwen2_5_VLForConditionalGeneration):
    config_class = LaViTConfig
    
    def __init__(self, config):
        # Ensure config has our custom attributes if loaded from base config
        if not hasattr(config, 'v_top_dim'): config.v_top_dim = 5120
        if not hasattr(config, 'loss_scale_vtop'): config.loss_scale_vtop = 1.0
        if not hasattr(config, 'loss_scale_traj'): config.loss_scale_traj = 1.0
        if not hasattr(config, 'training_stage'): config.training_stage = 0
        if not hasattr(config, 'bottleneck_block_prompt'): config.bottleneck_block_prompt = True
        # Note: use_trajectory_supervision should be set in train.py; here we only define a default.
        if not hasattr(config, 'use_trajectory_supervision'): 
            config.use_trajectory_supervision = True
        
        super().__init__(config)
        
        self.v_top_head = VTopHead(config.hidden_size, config.v_top_dim)
        # Ablation: create or skip traj_head depending on config.use_trajectory_supervision.
        # Ensure we always treat the flag as a strict bool.
        use_traj = bool(config.use_trajectory_supervision)
        print(f"\n{'='*60}")
        print(f"MODEL INIT: use_trajectory_supervision = {config.use_trajectory_supervision}")
        print(f"MODEL INIT: use_traj (bool) = {use_traj}")
        print(f"{'='*60}\n")
        if use_traj:
            self.traj_head = TrajectoryHead(config.hidden_size)
            print("MODEL INIT: Created traj_head")
        else:
            self.traj_head = None
            logger = logging.getLogger(__name__)
            logger.info(f"Trajectory head disabled (ablation study). traj_head = {self.traj_head}")
            print("MODEL INIT: traj_head = None (ablation study)")
        
        # Store training stage
        self.training_stage = config.training_stage
        self.bottleneck_block_prompt = config.bottleneck_block_prompt
    
    def set_training_stage(self, stage: int):
        """Dynamically switch training stage (useful for 2-stage training in single run)"""
        self.training_stage = stage
        self.config.training_stage = stage
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        image_grid_thw=None,
        # Custom arguments from Dataset
        v_top_tensor=None,
        traj_tensor=None,
        lavit_token_idx=None,
        # Position indices
        vision_start_idx=None,
        vision_end_idx=None,
        lavit_start_idx=None,
        # RoPE deltas (Qwen2.5-VL specific)
        rope_deltas=None,
        **kwargs
    ):
        # 1. Handle Stage 1 Hard Bottleneck
        modified_attention_mask = attention_mask
        actual_position_ids = position_ids
        actual_rope_deltas = rope_deltas

        if self.training_stage == 1 and self.training:
            # First, pre-calculate RoPE indices using the ORIGINAL 2D mask
            # This avoids the IndexError in Qwen2_5_VL's get_rope_index
            if actual_position_ids is None or actual_rope_deltas is None:
                actual_position_ids, actual_rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    None, # video_grid_thw
                    attention_mask,
                )
            
            # Now create the 4D hard bottleneck mask
            if (vision_start_idx is not None and vision_end_idx is not None 
                and lavit_start_idx is not None and lavit_token_idx is not None):
                
                modified_attention_mask = create_bottleneck_attention_mask(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    vision_start_positions=vision_start_idx,
                    vision_end_positions=vision_end_idx,
                    lavit_start_positions=lavit_start_idx,
                    lavit_end_positions=lavit_token_idx,
                    block_prompt=self.bottleneck_block_prompt,
                    dtype=self.dtype if hasattr(self, 'dtype') else torch.float32,
                )
        
        # Base Model Forward
        # Pass the pre-calculated position_ids and rope_deltas to skip internal get_rope_index
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=modified_attention_mask,
            position_ids=actual_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels, # Re-enable labels: Stage 1 now has hard mask, we WANT CE loss to force usage of <lvr>
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            rope_deltas=actual_rope_deltas,
            **kwargs
        )
        
        # If we are in inference or validation without special targets, return
        if v_top_tensor is None or lavit_token_idx is None:
            return outputs
            
        # 2. Extract Hidden State for <lvr>
        # lavit_token_idx is [Batch] or scalar? 
        # Logic: We need h_lavit for each sample in batch.
        
        # Last hidden state: [Batch, SeqLen, Hidden]
        last_hidden_state = outputs.hidden_states[-1]
        
        # Gather LaViT states
        # lavit_token_idx: [Batch] tensors of indices
        batch_size = last_hidden_state.shape[0]
        
        # Handle scalar lavit_token_idx (if passed from collator as tensor)
        if hasattr(lavit_token_idx, 'dim') and lavit_token_idx.dim() == 0:
             # Should be batch, but if collator stacked 0-dim tensors?
             pass 
             
        # Select: h_lavit = last_hidden_state[b, idx[b], :]
        # We construct indices
        batch_indices = torch.arange(batch_size, device=last_hidden_state.device)
        
        # Ensure lavit_token_idx is on correct device
        lavit_token_idx = lavit_token_idx.to(last_hidden_state.device)
        
        h_lavit = last_hidden_state[batch_indices, lavit_token_idx, :] # [Batch, Hidden]
        
        # 3. V_top Loss (Cosine)
        # v_pred: [Batch, v_top_dim]
        v_pred = self.v_top_head(h_lavit)
        
        v_top_target = v_top_tensor.to(h_lavit.device)
        if v_top_target.dim() == 3: # [Batch, N, D]
            v_top_target = v_top_target.mean(dim=1)
            
        # Cosine Loss: 1 - CosineSim
        loss_v = 1.0 - F.cosine_similarity(v_pred, v_top_target, dim=-1).mean()
        
        # 4. Trajectory Loss (attention distribution over image patches)
        # In ablation mode, traj_head may be disabled; in that case we skip trajectory loss.
        loss_t = torch.tensor(0.0, device=h_lavit.device)
        
        if self.traj_head is not None and self.config.use_trajectory_supervision:
            try:
                total_traj_loss = 0
                for b in range(batch_size):
                    h, w = image_grid_thw[b, 1], image_grid_thw[b, 2]
                    num_patches = h * w
                    
                    # Approximate: Image tokens start at beginning
                    img_feats = last_hidden_state[b, :num_patches, :] # [N, Hidden]
                    
                    # Predict scores: [1, N]
                    pred_scores = self.traj_head(h_lavit[b].unsqueeze(0), img_feats.unsqueeze(0)).squeeze(0) # [N]
                    pred_prob = F.log_softmax(pred_scores, dim=0)
                    
                    # Target: [N]
                    target_prob = traj_tensor[b].to(h_lavit.device)
                    
                    cur_loss = F.kl_div(pred_prob, target_prob, reduction='sum')
                    total_traj_loss += cur_loss
                
                loss_t = total_traj_loss / batch_size
                
            except Exception as e:
                pass

        # 5. Total Loss Calculation
        # In ablation mode, loss_scale_traj is expected to be 0 when traj_head is None,
        # but we still keep the formula general for safety.
        loss_custom = self.config.loss_scale_vtop * loss_v + self.config.loss_scale_traj * loss_t
        
        total_loss = loss_custom
        if outputs.loss is not None:
            total_loss = outputs.loss + loss_custom
        
        # One-time debug print to verify loss decomposition.
        if not hasattr(self, '_loss_debug_printed'):
            print(f"\n{'='*60}")
            print(f"LOSS CALCULATION:")
            print(f"  loss_v: {loss_v.item():.6f}")
            print(f"  loss_t: {loss_t.item():.6f}")
            print(f"  loss_scale_vtop: {self.config.loss_scale_vtop}")
            print(f"  loss_scale_traj: {self.config.loss_scale_traj}")
            print(f"  loss_custom: {loss_custom.item():.6f}")
            print(f"  outputs.loss: {outputs.loss.item() if outputs.loss is not None else None}")
            print(f"  total_loss: {total_loss.item():.6f}")
            print(f"{'='*60}\n")
            self._loss_debug_printed = True
            
        return LaViTOutput(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_v_top=loss_v,
            loss_traj=loss_t
        )

