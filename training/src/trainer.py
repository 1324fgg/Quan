from transformers import Trainer
import torch

class LaViTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for LaViT training.
        The model forward pass already computes total loss (CE + V_top + trajectory).
        """
        # Forward pass
        if num_items_in_batch is not None:
             outputs = model(**inputs, num_items_in_batch=num_items_in_batch)
        else:
             outputs = model(**inputs)
        
        # Model outputs LaViTOutput with total loss already computed
        total_loss = outputs.loss
        
        # Custom losses (loss_v_top, loss_traj) are already computed in the model.
        # They can be logged via callbacks if needed.
        return (total_loss, outputs) if return_outputs else total_loss
