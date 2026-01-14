#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只提取 V_top_layer 的脚本（不做视觉轨迹），用于 Visual-CoT 数据。

流程：
1. 用非 eager 模式的 Qwen2.5-VL 模型生成完整回答（使用LVR格式，不强制<step>标签）。
2. 可选：调用外部 LLM 做答案校验（validate_sample），不通过则跳过该样本。
3. 若校验通过：
   - 构造「图像 + 问题」的 LVR 标准格式 prompt；
   - 前向一次，使用 VTopLayerExtractor 在最后一层 Transformer 上挂 hook，
     抓取所有图像 patch 的顶层隐藏向量 V_top_layer (N_img × d)；
   - 将 V_top_layer 以二进制文件存盘（.pth），在 JSON 中只保存其相对路径。

注意：
- 本脚本不依赖 attn_implementation="eager"，生成阶段和 V_top_layer 抓取阶段都使用默认的注意力实现（通常是 SDPA/flash attention），显存占用更低。
- 暂不提取视觉轨迹 p_t，后续可以基于数据集中提供的 bbox 人工构造注意力分布。
- 使用 LVR 标准格式（与 VisualMindTraining 对齐），不强制使用 <step> 标签。
"""

import os
import json
import re
import argparse
import random
from typing import Dict, List, Any, Optional

from PIL import Image

# 在任何 torch/cuda 初始化之前设置显存分配策略，减少碎片导致的大块分配失败
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))) # For src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../extraction'))) # For v_top_layer_extractor

from src.lvr_prompt_utils import build_prompt_with_chat_template, build_teacher_generation_prompt
from v_top_layer_extractor import VTopLayerExtractor
from src.answer_validator import validate_sample


# --------------------------------------------------------------------
# 文本解析工具
# --------------------------------------------------------------------

def parse_question(conversation_value: str) -> str:
    """从 conversation 的 value 中提取问题，去除 <image> 前缀。"""
    question = conversation_value.strip()
    if question.startswith("<image>"):
        question = question[7:]
        if question.startswith("\n"):
            question = question[1:]
    return question.strip()


def extract_answer(conversation_value: str) -> str:
    """从 conversation 的 value 中提取 <answer>...</answer> 包围的内容。"""
    pattern = r"<answer>(.*?)</answer>"
    m = re.search(pattern, conversation_value, re.DOTALL)
    if m:
        return m.group(1).strip()
    return conversation_value.strip()


def load_viscot_data(json_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """加载 Visual-CoT 数据文件。"""
    print(f"[INFO] 加载数据文件: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]
    print(f"[INFO] 共加载 {len(data)} 个样本")
    return data


# --------------------------------------------------------------------
# 仅负责：生成 teacher steps + 抓取 V_top_layer 的提取器
# --------------------------------------------------------------------

class VTopOnlyExtractor:
    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer

        # 注意：这里不指定 attn_implementation，使用默认实现（通常是 SDPA/flash attn），
        # 只要 forward 能产出最后一层 hidden states，就可以用 VTopLayerExtractor 抓取 V_top_layer。
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        # V_top_layer 提取器（基于最后一层 hidden states 的 hook）
        self.vtop_extractor = VTopLayerExtractor(self.model, verbose=True)

        # 检测是否多 GPU，便于输入对齐到正确设备
        self.use_multi_gpu = hasattr(self.model, "hf_device_map") and self.model.hf_device_map is not None
        self._input_device = self._get_model_input_device()

    def _get_model_input_device(self) -> torch.device:
        """获取模型输入应该放到的设备（兼容 device_map=auto）。"""
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if not self.use_multi_gpu:
            return torch.device(self.device)

        # 多 GPU：尝试从 language_model 的 embedding 层拿设备
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "model"):
            lm = self.model.language_model.model
            if hasattr(lm, "embed_tokens") and hasattr(lm.embed_tokens, "weight"):
                dev = lm.embed_tokens.weight.device
                print(f"[INFO] 检测到模型输入设备: {dev}")
                return dev

        # 退化：用第一个参数的设备
        dev = next(self.model.parameters()).device
        print(f"[INFO] 使用第一个参数的设备作为输入设备: {dev}")
        return dev

    def _encode_with_image(self, image: Image.Image, text: str) -> Dict[str, torch.Tensor]:
        """将图像与文本编码为模型输入，并搬到正确设备。"""
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: (v.to(self._input_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        return inputs

    def _image_token_mask(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        针对 Qwen2.5-VL，找到输入序列中图像 token 的位置。
        图像 token 位于 <|vision_start|> 和 <|vision_end|> 之间。
        """
        input_ids_flat = input_ids.squeeze(0)
        vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        try:
            start_idx = (input_ids_flat == vision_start_token_id).nonzero(as_tuple=True)[0][0].item()
            end_idx = (input_ids_flat == vision_end_token_id).nonzero(as_tuple=True)[0][0].item()
        except IndexError:
            return None

        mask = torch.zeros_like(input_ids_flat, dtype=torch.bool)
        mask[start_idx + 1 : end_idx] = True
        return mask

    @torch.no_grad()
    def teacher_generate_steps(
        self,
        image: Image.Image,
        question: str,
        T: int = 4,
        max_new_tokens: int = 1024,
    ) -> List[str]:
        """
        让教师模型生成完整回答（使用 LVR 格式，不强制使用 <step> 标签）。
        这里不需要 attention，只是正常 generate。
        
        注意：T 参数被保留用于兼容性，但不再在 prompt 中强制指定步骤数。
        返回的是完整的生成文本，包装在单个元素的列表中以保持接口兼容性。
        """
        prompt = build_teacher_generation_prompt(self.processor, question, force_T=T)
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: (v.to(self._input_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]

        # 为了安全，限制最大生成长度，避免 KV cache 过大
        safe_max_new_tokens = min(max_new_tokens, 1024)
        if max_new_tokens > safe_max_new_tokens:
            print(f"[WARNING] max_new_tokens={max_new_tokens} 过大，已限制为 {safe_max_new_tokens}")

        out = self.model.generate(
            **inputs,
            max_new_tokens=safe_max_new_tokens,
            do_sample=False,
            temperature=0.01,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,
        )
        text = self.processor.decode(out[0][input_length:], skip_special_tokens=True)
        print("[INFO] teacher generate response:", text)

        # LVR 格式：直接返回完整生成文本，不解析 <step> 标签
        # 为了保持接口兼容性，仍然返回列表，但只包含一个元素
        return [text]

    @torch.no_grad()
    def capture_v_top_layer(
        self,
        image: Image.Image,
        question: str,
        T: int,
        save_path: str,
    ) -> str:
        """
        使用「图像 + 问题」的 LVR 标准格式做一次前向，
        并通过 VTopLayerExtractor 抓取 V_top_layer，保存到 save_path。
        返回实际保存的路径（可能追加后缀）。
        
        注意：T 参数被保留用于兼容性，但在 LVR 格式中不再使用。
        """
        prompt_initial = build_prompt_with_chat_template(
            self.processor, question, steps_prefix=[], force_T=T
        )
        inputs_initial = self._encode_with_image(image, prompt_initial)

        image_token_mask = self._image_token_mask(inputs_initial["input_ids"])
        if image_token_mask is None:
            raise RuntimeError("无法识别图像 token 位置，无法提取 V_top_layer")

        print("[INFO] 开始抓取 V_top_layer（LVR格式，非 eager 模式，基于最后一层 hidden states）...")
        v_path = self.vtop_extractor.capture_and_save(
            inputs=inputs_initial,
            image_token_mask=image_token_mask,
            save_path=save_path,
        )
        # 清理中间变量
        del inputs_initial, image_token_mask
        torch.cuda.empty_cache()
        import gc

        gc.collect()
        return v_path


# --------------------------------------------------------------------
# 单样本处理逻辑
# --------------------------------------------------------------------

def process_sample(
    sample: Dict,
    extractor: VTopOnlyExtractor,
    base_image_dir: str,
    output_dir: str,
    steps: int = 1,
    enable_validation: bool = True,
    validation_api_key: Optional[str] = None,
    validation_model: Optional[str] = "mimo-v2-flash",
) -> Dict[str, Any]:
    """
    处理单个样本：
    1) 生成教师多步推理回答；
    2) 可选：用外部 LLM 做答案校验；
    3) 若通过校验，则抓取并保存 V_top_layer。
    """
    try:
        # 1. 解析基础字段
        image_paths = sample.get("image", [])
        if not image_paths:
            raise ValueError("样本中没有 image 字段")

        image_relative_path = image_paths[0]
        image_full_path = os.path.join(base_image_dir, image_relative_path)
        if not os.path.exists(image_full_path):
            raise FileNotFoundError(f"图像文件不存在: {image_full_path}")

        conversations = sample.get("conversations", [])
        if len(conversations) < 2:
            raise ValueError("conversations 字段至少需要 2 个元素")

        question_raw = conversations[0].get("value", "")
        question = parse_question(question_raw)

        answer_raw = conversations[1].get("value", "")
        ground_truth = extract_answer(answer_raw)

        qid = sample.get("question_id", "unknown")
        print(f"[样本 {qid}] Ground Truth: {ground_truth}")

        # 2. 加载图像
        image = Image.open(image_full_path).convert("RGB")

        # 3. 生成教师完整回答（LVR 格式）
        print(f"[样本 {qid}] 生成教师回答（LVR格式）...")
        teacher_steps = extractor.teacher_generate_steps(
            image=image,
            question=question,
            T=steps,
            max_new_tokens=4096,
        )
        # teacher_steps 现在只包含一个元素（完整回答）
        teacher_full_response = teacher_steps[0] if teacher_steps else ""

        # 4. 可选：答案校验（不影响 v_top_layer 抓取，仅记录结果）
        is_answer_correct = True
        final_answer = None
        validation_explanation = None

        if enable_validation:
            print(f"[样本 {qid}] 校验答案...")
            is_answer_correct, final_answer, validation_explanation = validate_sample(
                teacher_response=teacher_full_response,
                ground_truth=ground_truth,
                question=question,
                api_key=validation_api_key,
                model=validation_model,
            )
            if is_answer_correct:
                print(f"[样本 {qid}] 答案校验通过")
            else:
                print(f"[样本 {qid}] 答案校验未通过（但仍会抓取 V_top_layer）")

        # 5. 生成 V_top_layer 的保存路径（无论验证结果如何都会抓取）
        tensors_dir = os.path.join(output_dir, "tensors")
        os.makedirs(tensors_dir, exist_ok=True)
        vtop_filename = f"sample_{int(qid):06d}_v_top_layer.pth" if isinstance(qid, int) or str(qid).isdigit() else f"sample_{qid}_v_top_layer.pth"
        vtop_save_path = os.path.join(tensors_dir, vtop_filename)
        vtop_relative_path = os.path.join("tensors", vtop_filename)

        # 6. 抓取并保存 V_top_layer
        print(f"[样本 {qid}] 抓取 V_top_layer ...")
        vtop_actual_path = extractor.capture_v_top_layer(
            image=image,
            question=question,
            T=steps,
            save_path=vtop_save_path,
        )
        # 如果 capture_and_save 自动调整了后缀，更新相对路径
        if vtop_actual_path != vtop_save_path:
            vtop_relative_path = os.path.relpath(vtop_actual_path, output_dir)

        # 7. 构建结果（仅包含元数据 + V_top_layer 路径）
        result: Dict[str, Any] = {
            "question_id": qid,
            "dataset": sample.get("dataset", None),
            "split": sample.get("split", None),
            "image_path": image_full_path,
            "image_relative_path": image_relative_path,
            "question": question,
            "ground_truth": ground_truth,
            "teacher_full_response": teacher_full_response,
            "final_answer": final_answer,
            "v_top_layer_path": vtop_relative_path,
            "parameters": {
                "steps": steps,
            },
        }

        if enable_validation:
            result["validation"] = {
                "is_correct": is_answer_correct,
                "explanation": validation_explanation,
            }

        print(f"[样本 {qid}] 完成 V_top_layer 抓取并保存: {vtop_relative_path}")
        return result

    except Exception as e:
        print(f"[样本 {sample.get('question_id', 'unknown')}] 处理失败: {e}")
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc

        gc.collect()
        return {
            "question_id": sample.get("question_id", None),
            "error": str(e),
            "image_path": image_full_path if "image_full_path" in locals() else None,
        }


# --------------------------------------------------------------------
# CLI 主入口
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="仅提取 Visual-CoT 样本的 V_top_layer（LVR格式，非 eager 模式，无视觉轨迹）"
    )

    # 数据相关
    parser.add_argument(
        "--data_file",
        type=str,
        default="./data/Visual-CoT-full/viscot_363k_lvr_formatted.json",
        help="Visual-CoT 数据文件路径",
    )
    parser.add_argument(
        "--base_image_dir",
        type=str,
        default="./data/Visual-CoT-full",
        help="图像文件基础目录",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="最大处理样本数（0 或负数表示处理全部）",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="起始样本索引（在随机打乱后应用）",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="随机数种子，用于固定随机顺序（默认使用该值）",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="是否随机打乱数据顺序（配合 random_seed 可复现）",
    )

    # 模型相关
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/my_qwen_model/Qwen/Qwen2.5-VL-32B-Instruct",
        help="Qwen2.5-VL 模型路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备（cuda 或 cpu），实际 forward 会根据 device_map=auto 分配",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="期望的推理步骤数 T（保留用于兼容性，LVR格式不在prompt中强制指定步骤数）",
    )

    # 校验相关
    parser.add_argument(
        "--enable_validation",
        action="store_true",
        default=True,
        help="是否启用答案校验（默认启用）",
    )
    parser.add_argument(
        "--disable_validation",
        action="store_false",
        dest="enable_validation",
        help="禁用答案校验",
    )
    parser.add_argument(
        "--validation_api_key",
        type=str,
        default=None,
        help="OpenRouter API 密钥（可选，默认使用内置配置）",
    )
    parser.add_argument(
        "--validation_model",
        type=str,
        default="mimo-v2-flash",
        help="用于答案校验的大模型名称",
    )

    # 输出相关
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trajectories/viscot_vtop_only",
        help="输出目录（将包含 trajectories.json 和 tensors/ 下的 V_top_layer 文件）",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="json",
        choices=["json", "jsonl"],
        help="保存格式：json（单文件）或 jsonl（每行一个样本）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="增量追加模式：加载已有 checkpoint，跳过已处理的样本，追加新结果",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # ----------------------------------------------------------------
    # [Safety Patch] Prevent accidental overwrite
    # ----------------------------------------------------------------
    potential_output_json = os.path.join(args.output_dir, "trajectories_vtop.json")
    potential_output_jsonl = os.path.join(args.output_dir, "trajectories_vtop.jsonl")
    
    target_output_file = potential_output_json if args.save_format == "json" else potential_output_jsonl
    
    if os.path.exists(target_output_file) and not args.resume:
        print(f"[ERROR] Output file already exists: {target_output_file}")
        print(f"[ERROR] Use --resume to append/continue, or delete the file to restart.")
        print(f"[ERROR] Aborting to prevent data loss.")
        return

    # 加载数据（先全部加载，再根据 shuffle / start_idx / max_samples 处理）
    data = load_viscot_data(args.data_file, max_samples=None)

    # 随机打乱数据顺序（如果启用）
    used_seed: Optional[int] = None
    if args.shuffle or args.random_seed is not None:
        if args.random_seed is not None:
            used_seed = args.random_seed
            random.seed(args.random_seed)
            print(f"[INFO] 使用随机数种子: {args.random_seed}")
        else:
            # 启用了 shuffle 但未显式提供种子，则使用当前时间戳
            import time

            used_seed = int(time.time())
            random.seed(used_seed)
            print(f"[INFO] 使用时间戳作为随机数种子: {used_seed}")

        indices = list(range(len(data)))
        random.shuffle(indices)
        data = [data[i] for i in indices]
        print(f"[INFO] 数据已随机打乱，共 {len(data)} 个样本")
    else:
        print(f"[INFO] 保持原始数据顺序，共 {len(data)} 个样本")

    # 应用起始索引和最大样本数限制
    if args.start_idx > 0:
        data = data[args.start_idx :]
        print(f"[INFO] 从索引 {args.start_idx} 开始处理，剩余 {len(data)} 个样本")

    if args.max_samples and args.max_samples > 0:
        original_count = len(data)
        data = data[: args.max_samples]
        print(f"[INFO] 限制处理数量为 {args.max_samples} 个样本（原始: {original_count}）")

    # 记录处理参数，便于复现与分析
    processing_params = {
        "random_seed": used_seed,
        "shuffled": bool(args.shuffle or args.random_seed is not None),
        "start_idx": args.start_idx,
        "max_samples": args.max_samples,
        "total_samples_loaded": len(data),
    }

    # 增量追加模式：加载已有 checkpoint
    existing_results: List[Dict[str, Any]] = []
    processed_qids: set = set()
    
    if args.resume:
        ckpt_path = os.path.join(args.output_dir, "vtop_checkpoint.json")
        if os.path.exists(ckpt_path):
            print(f"[INFO] 增量追加模式：加载已有 checkpoint: {ckpt_path}")
            with open(ckpt_path, "r", encoding="utf-8") as f:
                ckpt_data = json.load(f)
            existing_results = ckpt_data.get("results", [])
            # 提取已处理的 question_id
            for r in existing_results:
                qid = r.get("question_id")
                if qid is not None:
                    processed_qids.add(qid)
            print(f"[INFO] 已加载 {len(existing_results)} 个已处理样本，将跳过这些样本")
        else:
            print(f"[INFO] 增量追加模式：未找到已有 checkpoint，将从头开始处理")

    # 初始化提取器
    print(f"[INFO] 加载模型: {args.model_path}")
    extractor = VTopOnlyExtractor(
        model_path=args.model_path,
        device=args.device,
        dtype=torch.bfloat16,
    )
    print("[INFO] 模型加载完成")

    # 初始化结果列表（增量追加模式下，从已有结果开始）
    results: List[Dict[str, Any]] = existing_results.copy() if args.resume else []
    
    # 统计跳过和新处理的样本数
    n_skipped = 0
    n_newly_processed = 0

    for i, sample in enumerate(data):
        qid = sample.get('question_id', 'unknown')
        
        # 增量追加模式：跳过已处理的样本
        if args.resume and qid in processed_qids:
            n_skipped += 1
            if n_skipped <= 5 or n_skipped % 100 == 0:
                print(f"[SKIP] 样本 {i+1}/{len(data)} (question_id={qid}) 已处理，跳过")
            continue
        
        print(f"\n[MAIN] 处理样本 {i+1}/{len(data)} (question_id={qid})")
        res = process_sample(
            sample=sample,
            extractor=extractor,
            base_image_dir=args.base_image_dir,
            output_dir=args.output_dir,
            steps=args.steps,
            enable_validation=args.enable_validation,
            validation_api_key=args.validation_api_key,
            validation_model=args.validation_model,
        )
        # 所有样本按顺序存储
        results.append(res)
        n_newly_processed += 1

        # 每处理若干新样本，保存一次 checkpoint
        if n_newly_processed % 10 == 0:
            ckpt_path = os.path.join(args.output_dir, "vtop_checkpoint.json")
            checkpoint_data = {
                "processing_params": processing_params,
                "results": results,
            }
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            # 统计正负样本数量
            n_correct = sum(1 for r in results if r.get("validation", {}).get("is_correct", True) and not r.get("error"))
            n_incorrect = len(results) - n_correct
            print(f"[CHECKPOINT] 已保存 {len(results)} 个样本到 {ckpt_path} (正确: {n_correct}, 错误: {n_incorrect})")
    
    # 打印增量追加统计
    if args.resume:
        print(f"\n[INFO] 增量追加统计: 跳过 {n_skipped} 个已处理样本，新处理 {n_newly_processed} 个样本")

    # 保存最终结果（所有样本按顺序保存到一个文件）
    print(f"\n[MAIN] 正在保存最终结果到 {args.output_dir} ...")
    if args.save_format == "json":
        out_path = os.path.join(args.output_dir, "trajectories_vtop.json")
        output_data = {
            "processing_params": processing_params,
            "results": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"[MAIN] 已保存 {len(results)} 个样本到 {out_path}")
    else:
        out_path = os.path.join(args.output_dir, "trajectories_vtop.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "metadata", "processing_params": processing_params}, ensure_ascii=False) + "\n")
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[MAIN] 已保存 {len(results)} 个样本到 {out_path}")
    
    # 打印统计信息
    total = len(results)
    n_correct = sum(1 for r in results if r.get("validation", {}).get("is_correct", True) and not r.get("error"))
    n_error = sum(1 for r in results if r.get("error"))
    n_incorrect = total - n_correct - n_error
    
    print(f"\n[MAIN] 处理完成统计:")
    print(f"  - 总样本数: {total}")
    print(f"  - 验证通过: {n_correct} ({n_correct/total*100:.1f}%)" if total > 0 else "  - 验证通过: 0")
    print(f"  - 验证未通过: {n_incorrect} ({n_incorrect/total*100:.1f}%)" if total > 0 else "  - 验证未通过: 0")
    print(f"  - 处理出错: {n_error} ({n_error/total*100:.1f}%)" if total > 0 else "  - 处理出错: 0")


if __name__ == "__main__":
    main()


