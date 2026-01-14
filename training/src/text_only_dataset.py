import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class TextOnlyDataset(Dataset):
    """
    Dataset for text-only SFT (ablation study).
    No <lvr> tokens, no v_top, no trajectory supervision.
    Just standard image + question -> answer training.
    """
    def __init__(
        self, 
        data_json_path, 
        image_root, 
        processor,
        max_samples=None
    ):
        """
        Args:
            data_json_path: Path to enriched data json (same format as ViLRDataset)
            image_root: Root directory for images
            processor: Qwen2-VL Processor
        """
        self.image_root = image_root
        self.processor = processor
        
        # Load data
        logger.info(f"Loading text-only dataset from {data_json_path}")
        with open(data_json_path, 'r') as f:
            full_data = json.load(f)
            self.data = full_data['results']
            
        if max_samples:
            self.data = self.data[:max_samples]
            
        logger.info(f"Loaded {len(self.data)} samples for text-only SFT")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Text & Image
        question = item['question']
        ground_truth = item.get('ground_truth_enriched', '') 
        
        image_path = os.path.join(self.image_root, item['image_relative_path'])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise e

        # Construct Conversation (standard format, NO <lvr> tokens)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": ground_truth  # Direct answer, no lvr tokens
            }
        ]
        
        # Apply prompt template
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text_input],
            images=[image],
            padding=False, 
            return_tensors="pt"
        )
        
        # Squeeze batch dim
        input_ids = inputs.input_ids[0]
        attention_mask = inputs.attention_mask[0]
        pixel_values = inputs.pixel_values 
        image_grid_thw = inputs.image_grid_thw[0]
        
        # Labels for Text Loss (standard masking)
        labels = input_ids.clone()
        
        # Mask the user part (everything before assistant response)
        # Qwen2-VL template format: <|im_start|>user...<|im_end|><|im_start|>assistant...<|im_end|>
        # We want to mask everything before and including the <|im_start|>assistant token
        assistant_start = self.processor.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        if len(assistant_start) > 0:
            # Find first occurrence of assistant start token
            input_ids_list = input_ids.tolist()
            assistant_start_list = assistant_start
            for i in range(len(input_ids_list) - len(assistant_start_list) + 1):
                if input_ids_list[i:i+len(assistant_start_list)] == assistant_start_list:
                    # Mask everything before assistant (including the <|im_start|>assistant token itself)
                    # Then unmask the actual assistant content
                    labels[:i+len(assistant_start_list)] = -100
                    break
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
        }

class TextOnlyCollator:
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id
        
    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Pad input_ids and labels
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = [item['attention_mask'] for item in batch]
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        
        pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
        image_grid_thw = torch.stack([item['image_grid_thw'] for item in batch], dim=0)
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

