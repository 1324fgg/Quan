import json
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enrich_dataset(input_path, output_path, metadata_dir, attention_dir, vtop_root):
    logger.info(f"Loading input data: {input_path}")
    with open(input_path, 'r') as f:
        data_full = json.load(f)
        results = data_full['results']
    
    # 1. Group samples by dataset to process metadata efficiently
    dataset_map = {}
    for idx, item in enumerate(results):
        ds = item.get('dataset', 'unknown')
        if ds not in dataset_map:
            dataset_map[ds] = []
        dataset_map[ds].append(idx)
        
    logger.info(f"Found datasets: {list(dataset_map.keys())}")
    
    # helper
    def find_match(candidates, query_text):
        if not candidates: return None
        for cand in candidates:
            if cand['question'].strip() == query_text.strip():
                return cand
        q_norm = query_text.lower().replace('\n', ' ')
        best_cand = None
        max_overlap = 0
        for cand in candidates:
             c_q = cand['question'].lower().replace('\n', ' ')
             if c_q in q_norm or q_norm in c_q:
                 if len(c_q) > max_overlap:
                     max_overlap = len(c_q)
                     best_cand = cand
        return best_cand

    # 2. Iterate datasets, load metadata
    for ds_name, indices in dataset_map.items():
        logger.info(f"Processing dataset: {ds_name} ({len(indices)} samples)")
        
        meta_filename = f"{ds_name}_cot_train.jsonl"
        meta_path = os.path.join(metadata_dir, meta_filename)
        
        if not os.path.exists(meta_path):
            logger.warning(f"Metadata file missing for {ds_name}: {meta_path}. Skipping text enrichment.")
            continue
            
        meta_lookup = {}
        try:
            with open(meta_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        img = entry.get('image', '')
                        if img:
                            if img not in meta_lookup: meta_lookup[img] = []
                            meta_lookup[img].append(entry)
                    except: continue
        except Exception as e:
            logger.error(f"Failed to load {meta_path}: {e}")
            continue
            
        success_count = 0
        for idx in indices:
            item = results[idx]
            img_filename = os.path.basename(item['image_relative_path'])
            candidates = meta_lookup.get(img_filename, [])
            match = find_match(candidates, item['question'])
            
            if match:
                item['ground_truth_enriched'] = match.get('answer', '')
                item['metadata_source'] = meta_path
                success_count += 1
            else:
                logger.debug(f"No match for {img_filename}")
                
        logger.info(f"  Matched {success_count}/{len(indices)} samples.")
        del meta_lookup
        
    # 3. Add Paths & Validate
    valid_results = []
    missing_att = 0
    missing_vtop = 0
    
    logger.info("Verifying file paths...")
    for item in results:
        # V_top path: Assume v_top_layer_path is relative to 'vtop_root' 
        # But 'v_top_layer_path' in json is currently 'tensors/sample_...'.
        # And user structure was .../viscot_vtop_only/tensors.
        # If user passes vtop_root as .../viscot_vtop_only, we join properly.
        # We try to handle both absolute joining to handle flexibility.
        
        rel_path = item['v_top_layer_path'] 
        # If rel_path starts with tensors/, we can join.
        v_top_abs = os.path.join(vtop_root, os.path.basename(rel_path))
        # OR if structure matches json
        # v_top_abs = os.path.join(vtop_root, rel_path)
        
        # Check specific existence preference
        if not os.path.exists(v_top_abs):
             # Try nested 'tensors' dir just in case
             v_top_abs_alt = os.path.join(vtop_root, "tensors", os.path.basename(rel_path))
             if os.path.exists(v_top_abs_alt):
                 v_top_abs = v_top_abs_alt
        
        # Attention path
        q_id = item['question_id']
        att_filename = f"sample_{q_id:06d}_attention.json"
        att_abs = os.path.join(attention_dir, att_filename)
        
        if not os.path.exists(v_top_abs):
            missing_vtop += 1
            continue
        if not os.path.exists(att_abs):
            missing_att += 1
            continue
            
        item['v_top_path_abs'] = v_top_abs
        item['attention_path_abs'] = att_abs
        
        if 'ground_truth_enriched' not in item:
             item['ground_truth_enriched'] = item.get('ground_truth', '')
             
        valid_results.append(item)
        
    logger.info(f"Finished. Total Valid: {len(valid_results)}. Filtered (Vtop: {missing_vtop}, Att: {missing_att}).")
    
    data_full['results'] = valid_results
    with open(output_path, 'w') as f:
        json.dump(data_full, f, indent=2)
    logger.info(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich dataset json with metadata and absolute paths.")
    parser.add_argument("--input", type=str, required=True, help="Path to input trajectories.json")
    parser.add_argument("--output", type=str, required=True, help="Path to save enriched json")
    parser.add_argument("--metadata_root", type=str, default="/root/autodl-tmp/ViLR/data/Visual-CoT-full/metadata", help="Dir containing metadata jsonl files")
    parser.add_argument("--attention_dict", type=str, default="/root/autodl-tmp/ViLR/trajectories/bbox_attention", help="Dir containing attention json files")
    parser.add_argument("--vtop_root", type=str, default="/root/autodl-tmp/ViLR/trajectories/viscot_vtop_only/tensors", help="Dir containing vtop .pth files")
    
    args = parser.parse_args()
    
    enrich_dataset(
        input_path=args.input,
        output_path=args.output,
        metadata_dir=args.metadata_root,
        attention_dir=args.attention_dict,
        vtop_root=args.vtop_root
    )
