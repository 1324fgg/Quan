import os
import json
import csv
import pandas as pd
from abc import ABC, abstractmethod

class BaseDatasetAdapter(ABC):
    """
    Abstract base class for dataset adapters.
    Each adapter MUST yield a dictionary with:
    {
        "id": str/int,
        "image_path": str (absolute or valid relative path) OR PIL Image OR List[PIL Image],
        "prompt": str,
        "ground_truth": str (optional),
        "meta": dict (optional, original metadata)
    }
    Note: image_path can be a single image or a list of images for multi-image tasks.
    """
    def __init__(self, data_root, task_name=None):
        self.data_root = data_root
        self.task_name = task_name
        self.data = []
        self._load_data()

    @abstractmethod
    def _load_data(self):
        pass

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for item in self.data:
            yield item

class MMVPAdapter(BaseDatasetAdapter):
    """
    Adapter for MMVP Benchmark.
    Structure:
    - data_root/Questions.csv
    - data_root/MMVP Images/{Index}.jpg
    """
    def _load_data(self):
        csv_path = os.path.join(self.data_root, "Questions.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"MMVP Questions.csv not found at {csv_path}")
            
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = row["Index"]
                prompt = f"{row['Question']}\n{row['Options']}\nAnswer with the option letter from the given choices directly."
                image_path = os.path.join(self.data_root, "MMVP Images", f"{idx}.jpg")
                
                self.data.append({
                    "id": idx,
                    "image_path": image_path,
                    "prompt": prompt,
                    "ground_truth": row["Correct Answer"],
                    "meta": row
                })

class VSPAdapter(BaseDatasetAdapter):
    """
    Adapter for VSP (Visual Spatial Planning).
    """
    def _load_data(self):
        target_file = self.data_root
        if os.path.isdir(self.data_root):
             target_file = os.path.join(self.data_root, "test_direct.jsonl")
             
        if not os.path.exists(target_file):
             raise FileNotFoundError(f"VSP data file not found at {target_file}")

        with open(target_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                img_rel = item['image_input']
                if img_rel.startswith("./"):
                    img_rel = img_rel[2:]
                
                abs_image_path = os.path.abspath(os.path.join("/root/autodl-tmp/ViLR", img_rel))
                
                self.data.append({
                    "id": item["map_id"],
                    "image_path": abs_image_path,
                    "prompt": item["text_input"],
                    "ground_truth": item["map_desc"],
                    "meta": item
                })

class BLINKAdapter(BaseDatasetAdapter):
    """
    Adapter for BLINK Benchmark (Counting, Jigsaw, etc.).
    Structure:
    - data_root/{task_name}/val-00000-of-00001.parquet
    """
    def _load_data(self):
        if not self.task_name:
            raise ValueError("BLINKAdapter requires 'task_name' (e.g., Counting, Jigsaw).")
            
        # BLINK usually puts test data in 'val' or 'test' split. 
        # Checking local files shows 'val-00000-of-00001.parquet' exists.
        # We prefer 'val' for now as it usually has labels.
        
        parquet_path = os.path.join(self.data_root, self.task_name, "val-00000-of-00001.parquet")
        if not os.path.exists(parquet_path):
            # Fallback to test
            parquet_path = os.path.join(self.data_root, self.task_name, "test-00000-of-00001.parquet")
            
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"BLINK data not found for task {self.task_name} at {parquet_path}")
            
        print(f"Loading BLINK task '{self.task_name}' from {parquet_path}...")
        
        # Load Parquet
        df = pd.read_parquet(parquet_path)
        
        for idx, row in df.iterrows():
            # BLINK dataset structure:
            # - image_1, image_2, etc. (dict with 'bytes' and 'path')
            # - question: str
            # - prompt: str (formatted question with choices)
            # - answer: str (e.g., "(D)")
            # - choices: array of strings
            
            # Load ALL available images (image_1, image_2, etc.)
            # For tasks like Jigsaw, we need multiple images (e.g., 3 images)
            image_objs = []
            from io import BytesIO
            from PIL import Image
            
            for img_key in ['image_1', 'image_2', 'image_3', 'image_4']:
                if img_key in row and row[img_key] is not None:
                    image_data = row[img_key]
                    try:
                        # Handle dict format with 'bytes' key
                        if isinstance(image_data, dict) and 'bytes' in image_data:
                            image_obj = Image.open(BytesIO(image_data['bytes'])).convert("RGB")
                            image_objs.append(image_obj)
                        # Handle direct bytes
                        elif isinstance(image_data, bytes):
                            image_obj = Image.open(BytesIO(image_data)).convert("RGB")
                            image_objs.append(image_obj)
                    except Exception as e:
                        continue
            
            if len(image_objs) == 0:
                # Skip if no valid image found
                continue
            
            # For single image tasks, use single image; for multi-image tasks, use list
            # Most BLINK tasks need multiple images, so we use list
            # If only one image exists, we still use list for consistency
            image_path = image_objs if len(image_objs) > 1 else image_objs[0]

            # Use prompt if available, otherwise construct from question
            if 'prompt' in row and row['prompt']:
                prompt = row['prompt']
            else:
                # Construct prompt from question and choices
                choices = row.get('choices', [])
                prompt = row.get('question', '')
                
                if len(choices) > 0:
                    prompt += "\nOptions:"
                    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    for i, choice in enumerate(choices):
                        label = labels[i] if i < len(labels) else str(i)
                        prompt += f"\n({label}) {choice}"
                    prompt += "\nAnswer with the option letter from the given choices directly."
            
            # Ground Truth - extract letter from answer like "(D)" -> "D"
            answer_text = row.get('answer', "")
            ground_truth = answer_text
            
            # Extract letter from format like "(D)" or "D"
            if answer_text.startswith('(') and answer_text.endswith(')'):
                ground_truth = answer_text[1:-1]
            elif answer_text in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                ground_truth = answer_text
            else:
                # Try to map answer text to choice index
                choices = row.get('choices', [])
                if answer_text in choices:
                    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    idx_ans = list(choices).index(answer_text)
                    if idx_ans < len(labels):
                        ground_truth = labels[idx_ans]

            # Convert choices to list if it's a numpy array
            choices_list = row.get('choices', [])
            if hasattr(choices_list, 'tolist'):
                choices_list = choices_list.tolist()
            
            self.data.append({
                "id": row.get('idx', f"{self.task_name}_{idx}"),
                "image_path": image_path, # Pass PIL Image(s) directly (single or list)
                "prompt": prompt,
                "ground_truth": ground_truth,
                "meta": {"task": self.task_name, "choices": choices_list, "num_images": len(image_objs)}
            })

class MMStarAdapter(BaseDatasetAdapter):
    """
    Adapter for MMStar Benchmark.
    Structure:
    - data_root/mmstar.parquet
    
    Dataset fields:
    - index: int (sample index)
    - question: str (includes options)
    - answer: str (e.g., "A", "B", "C", "D")
    - category: str (coarse perception, fine-grained perception, etc.)
    - l2_category: str (detailed category)
    - image: bytes (image data)
    - meta_info: dict (source, split, image_path)
    """
    def _load_data(self):
        # Find parquet file
        if os.path.isdir(self.data_root):
            parquet_path = os.path.join(self.data_root, "mmstar.parquet")
        else:
            parquet_path = self.data_root
            
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"MMStar parquet not found at {parquet_path}")
        
        print(f"Loading MMStar from {parquet_path}...")
        
        # Load Parquet
        df = pd.read_parquet(parquet_path)
        
        from io import BytesIO
        from PIL import Image
        
        for idx, row in df.iterrows():
            # Load image from bytes
            image_data = row['image']
            try:
                if isinstance(image_data, bytes):
                    image_obj = Image.open(BytesIO(image_data)).convert("RGB")
                elif isinstance(image_data, dict) and 'bytes' in image_data:
                    image_obj = Image.open(BytesIO(image_data['bytes'])).convert("RGB")
                else:
                    print(f"Skipping sample {idx}: Unknown image format")
                    continue
            except Exception as e:
                print(f"Skipping sample {idx}: Failed to load image - {e}")
                continue
            
            # Question already includes options in MMStar
            question = row['question']
            
            # Ensure prompt ends with instruction
            if "Answer with" not in question:
                prompt = question + "\nAnswer with the option letter from the given choices directly."
            else:
                prompt = question
            
            # Ground truth answer
            answer = row['answer']
            
            # Meta info
            meta_info = row.get('meta_info', {})
            if isinstance(meta_info, str):
                try:
                    meta_info = json.loads(meta_info)
                except:
                    meta_info = {}
            
            self.data.append({
                "id": row['index'],
                "image_path": image_obj,  # Pass PIL Image directly
                "prompt": prompt,
                "ground_truth": answer,
                "meta": {
                    "category": row.get('category', ''),
                    "l2_category": row.get('l2_category', ''),
                    **meta_info
                }
            })
        
        print(f"Loaded {len(self.data)} samples from MMStar.")

class VStarAdapter(BaseDatasetAdapter):
    """
    Adapter for V*Bench (VStar) Benchmark.
    Uses HuggingFace datasets library to load from craigwu/vstar_bench.
    data_root parameter is ignored - dataset is loaded directly from HuggingFace.
    
    Dataset fields:
    - image: str (relative path to image in repo)
    - text: str (question with options and answer instruction)
    - category: str (e.g., 'direct_attributes', 'relative_position')
    - question_id: str
    - label: str (ground truth answer, e.g., 'A', 'B', 'C', 'D')
    """
    def _load_data(self):
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
        from PIL import Image
        
        print(f"Loading V*Bench dataset from HuggingFace (data_root parameter ignored)...")
        
        # Load dataset from HuggingFace
        dataset = load_dataset("craigwu/vstar_bench")
        test_data = dataset['test']
        
        repo_id = "craigwu/vstar_bench"
        
        for item in test_data:
            # Download image from HuggingFace Hub
            try:
                image_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=item['image'],
                    repo_type='dataset'
                )
                # Load image as PIL Image
                image_obj = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Skipping sample {item['question_id']}: Failed to load image - {e}")
                continue
            
            # Text already includes options and answer instruction
            prompt = item['text']
            
            # Ground truth answer
            ground_truth = item['label']
            
            self.data.append({
                "id": item['question_id'],
                "image_path": image_obj,  # Pass PIL Image directly
                "prompt": prompt,
                "ground_truth": ground_truth,
                "meta": {
                    "category": item.get('category', ''),
                }
            })
        
        print(f"Loaded {len(self.data)} samples from V*Bench.")


# Factory map
ADAPTER_MAP = {
    "mmvp": MMVPAdapter,
    "vsp": VSPAdapter,
    "blink": BLINKAdapter,
    "mmstar": MMStarAdapter,
    "vstar": VStarAdapter
}

def get_adapter(name, data_root, task_name=None):
    if name not in ADAPTER_MAP:
        raise ValueError(f"Unknown dataset name: {name}. Available: {list(ADAPTER_MAP.keys())}")
    return ADAPTER_MAP[name](data_root, task_name)
