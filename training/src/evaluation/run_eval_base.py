import argparse
import os
import json
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.base_evaluator import BaseQwenEvaluator
from evaluation.adapters import get_adapter

def main():
    parser = argparse.ArgumentParser(description="Base Qwen2.5-VL Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model (e.g., Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., vstar)")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root directory or file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output JSONL")
    parser.add_argument("--task_name", type=str, default=None, help="Task name for BLINK dataset")
    parser.add_argument("--max_samples", type=int, default=None, help="Debug: limit samples")
    
    args = parser.parse_args()

    # 1. Initialize Adapter
    print(f"Loading dataset: {args.dataset} from {args.data_root}")
    try:
        adapter = get_adapter(args.dataset, args.data_root, task_name=args.task_name)
    except Exception as e:
        print(f"Failed to load adapter: {e}")
        return

    print(f"Found {len(adapter)} samples.")
    
    # 2. Initialize Evaluator
    print(f"Loading base model from {args.model_path}")
    evaluator = BaseQwenEvaluator(args.model_path)
    
    # 3. Evaluation Loop
    results = []
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # Open file for streaming write
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        count = 0
        for sample in tqdm(adapter):
            if args.max_samples and count >= args.max_samples:
                break
                
            # Generate
            output_text = evaluator.generate(
                sample['image_path'], 
                sample['prompt']
            )
            
            if output_text is not None:
                # Result Object
                res = {
                    "id": sample['id'],
                    "prompt": sample['prompt'],
                    "ground_truth": sample['ground_truth'],
                    "model_output": output_text,
                    "lvr_count": 0,  # Base model doesn't use <lvr> tokens
                    "meta": sample.get('meta', {})
                }
                
                # Write line
                f_out.write(json.dumps(res) + "\n")
                f_out.flush()
                count += 1
            else:
                print(f"Skipping sample {sample['id']} due to generation error.")

    print(f"\nEvaluation Complete.")
    print(f"Results saved to {args.output_file}")
    print(f"Processed {count} samples.")

if __name__ == "__main__":
    main()





