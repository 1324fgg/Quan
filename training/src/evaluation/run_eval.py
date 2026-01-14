import argparse
import os
import json
import re
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluator import LaViTEvaluator
from evaluation.adapters import get_adapter

def main():
    parser = argparse.ArgumentParser(description="LaViT Unified Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., mmvp, vsp, blink)")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root directory or file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output JSONL")
    parser.add_argument("--task_name", type=str, default=None, help="Task name for BLINK dataset (e.g., Counting, Jigsaw)")
    parser.add_argument("--max_samples", type=int, default=None, help="Debug: limit samples")
    parser.add_argument("--force_lvr", action="store_true", help="Force append <lvr> tokens (Prompt Injection)")
    parser.add_argument("--mask_lvr", action="store_true", help="Mask <lvr> tokens (Attention Masking)")
    
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
    print(f"Loading model from {args.checkpoint}")
    evaluator = LaViTEvaluator(args.checkpoint)
    
    # 3. Evaluation Loop
    results = []
    lvr_stats = 0
    
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
                sample['prompt'], 
                force_lvr=args.force_lvr,
                mask_lvr=args.mask_lvr
            )
            
            if output_text is not None:
                # Stats - Check for both <lvr> and numbered <lvr1>, <lvr2>, etc.
                lvr_pattern = r"<lvr\d*>|<lvr>"
                lvr_matches = re.findall(lvr_pattern, output_text)
                lvr_count = len(lvr_matches)
                if args.force_lvr:
                    lvr_count += 4 # Conceptually used
                
                if lvr_count > 0:
                    lvr_stats += 1
                
                # Result Object
                res = {
                    "id": sample['id'],
                    "prompt": sample['prompt'],
                    "ground_truth": sample['ground_truth'],
                    "model_output": output_text,
                    "lvr_count": lvr_count,
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
    if count > 0:
        print(f"LVR Usage Rate: {lvr_stats}/{count} ({lvr_stats/count*100:.2f}%)")
    else:
        print(f"No samples were successfully processed.")

if __name__ == "__main__":
    main()
