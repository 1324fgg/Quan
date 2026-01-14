#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual-CoT 数据集视觉轨迹提取脚本

支持两种提取方法：
1. gradient: 梯度归因方法（teacher_traj_extractor_stepwise.py）
2. attention: 注意力权重方法（teacher_traj_extractor_attention.py）
"""

import os
import json
import re
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from PIL import Image

# 在任何 torch/cuda 初始化之前设置显存分配策略，减少碎片导致的大块分配失败
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))) # For src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../extraction'))) # For extractors

from teacher_traj_extractor_stepwise import TeacherTrajectoryExtractor
from teacher_traj_extractor_attention import AttentionBasedTrajectoryExtractor
from src.answer_validator import validate_sample


def parse_question(conversation_value: str) -> str:
    """从conversation的value中提取问题，去除<image>\n前缀"""
    question = conversation_value.strip()
    # 去除开头的 <image>\n 或 <image>
    if question.startswith("<image>"):
        question = question[7:]  # 移除 "<image>"
        if question.startswith("\n"):
            question = question[1:]  # 移除换行符
    return question.strip()


def extract_answer(conversation_value: str) -> str:
    """从conversation的value中提取<answer>...</answer>包围的内容"""
    # 使用正则表达式匹配 <answer>...</answer>
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, conversation_value, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # 如果没有找到，返回整个value（可能格式不同）
        return conversation_value.strip()


def load_viscot_data(json_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """加载Visual-CoT数据文件"""
    print(f"正在加载数据文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]
    
    print(f"加载了 {len(data)} 个样本")
    return data


def process_sample(
    sample: Dict,
    extractor: Union[TeacherTrajectoryExtractor, AttentionBasedTrajectoryExtractor],
    base_image_dir: str,
    output_dir: str,
    extraction_method: str = "gradient",  # "gradient" 或 "attention"
    steps: int = 1,
    tau_grad: float = 0.01,
    skip_markup: bool = True,
    downsample: int = 2,
    max_tokens_per_step: int = 50,
    max_grad_calls_per_step: int = 10,
    aggregation_method: str = "simple_avg",
    attribution_threshold: float = 0.1,
    min_entropy: float = 0.5,
    topk: int = 8,  # 用于attention方法
    enable_validation: bool = True,
    validation_api_key: Optional[str] = None,
    validation_model: Optional[str] = None,
    enable_v_top_layer: bool = True,
) -> Dict[str, Any]:
    """
    处理单个样本，提取视觉轨迹
    
    Returns:
        包含提取结果的字典
    """
    try:
        # 1. 解析数据
        image_paths = sample.get("image", [])
        if not image_paths:
            raise ValueError("样本中没有image字段")
        
        # 使用第一个图像（如果有多张图）
        image_relative_path = image_paths[0]
        image_full_path = os.path.join(base_image_dir, image_relative_path)
        
        if not os.path.exists(image_full_path):
            raise FileNotFoundError(f"图像文件不存在: {image_full_path}")
        
        # 解析question和ground_truth
        conversations = sample.get("conversations", [])
        if len(conversations) < 2:
            raise ValueError("conversations字段至少需要2个元素")
        
        question_raw = conversations[0].get("value", "")
        question = parse_question(question_raw)
        
        answer_raw = conversations[1].get("value", "")
        ground_truth = extract_answer(answer_raw)
        
        # 打印 ground truth
        print(f"  [样本 {sample.get('question_id', 'unknown')}] Ground Truth: {ground_truth}")
        
        # 2. 加载图像
        image = Image.open(image_full_path).convert("RGB")
        
        # 3. 生成教师步骤
        print(f"  [样本 {sample.get('question_id', 'unknown')}] 生成教师步骤...")
        teacher_steps = extractor.teacher_generate_steps(
            image, question, T=steps, max_new_tokens=4096
        )
        teacher_full_response = "\n".join(teacher_steps)
        
        # 3.5. 答案校验（如果启用）
        is_answer_correct = True
        final_answer = None
        validation_explanation = None
        
        if enable_validation:
            print(f"  [样本 {sample.get('question_id', 'unknown')}] 校验答案...")
            is_answer_correct, final_answer, validation_explanation = validate_sample(
                teacher_response=teacher_full_response,
                ground_truth=ground_truth,
                question=question,
                api_key=validation_api_key,
                model=validation_model,
            )
            
            if is_answer_correct:
                print(f"  [样本 {sample.get('question_id', 'unknown')}] 答案校验通过")
            else:
                print(f"  [样本 {sample.get('question_id', 'unknown')}] 答案校验未通过，跳过轨迹提取")
                return {
                    "question_id": sample.get("question_id", "unknown"),
                    "dataset": sample.get("dataset", None),
                    "split": sample.get("split", None),
                    "image_path": image_full_path,
                    "image_relative_path": image_relative_path,
                    "question": question,
                    "ground_truth": ground_truth,
                    "teacher_full_response": teacher_full_response,
                    "final_answer": final_answer,
                    "validation": {
                        "is_correct": False,
                        "explanation": validation_explanation,
                    },
                    "skipped": True,
                    "skip_reason": "答案校验未通过",
                }
        
        # 4. 生成 V_top_layer 的保存路径
        question_id = sample.get("question_id", "unknown")
        # 创建 tensors 子目录用于存储二进制文件
        tensors_dir = os.path.join(output_dir, "tensors")
        os.makedirs(tensors_dir, exist_ok=True)
        # 生成唯一的文件名
        v_top_layer_filename = f"sample_{question_id:06d}_v_top_layer.pth"
        v_top_layer_save_path = os.path.join(tensors_dir, v_top_layer_filename)
        # 相对路径（用于 JSON 中存储）
        v_top_layer_relative_path = os.path.join("tensors", v_top_layer_filename)
        
        # 5. 提取视觉轨迹
        print(f"  [样本 {question_id}] 提取视觉轨迹（方法: {extraction_method}）...")
        
        if extraction_method == "gradient":
            # 梯度归因方法
            trajectory = extractor.extract_pt_per_step(
                image=image,
                question=question,
                teacher_steps=teacher_steps,
                tau_grad=tau_grad,
                skip_markup=skip_markup,
                downsample=downsample,
                max_tokens_per_step=max_tokens_per_step,
                max_grad_calls_per_step=max_grad_calls_per_step,
                aggregation_method=aggregation_method,
                attribution_threshold=attribution_threshold,
                min_entropy=min_entropy,
                v_top_layer_save_path=v_top_layer_save_path if enable_v_top_layer else None,
                enable_v_top_layer=enable_v_top_layer,
            )
        elif extraction_method == "attention":
            # 注意力权重方法
            trajectory = extractor.extract_pt_per_step(
                image=image,
                question=question,
                teacher_steps=teacher_steps,
                topk=topk,
                v_top_layer_save_path=v_top_layer_save_path if enable_v_top_layer else None,
                enable_v_top_layer=enable_v_top_layer,
            )
        else:
            raise ValueError(f"未知的提取方法: {extraction_method}，支持的方法: 'gradient', 'attention'")
        
        # 6. 构建结果
        result = {
            "question_id": question_id,
            "dataset": sample.get("dataset", None),
            "split": sample.get("split", None),
            "image_path": image_full_path,
            "image_relative_path": image_relative_path,
            "question": question,
            "ground_truth": ground_truth,
            "teacher_full_response": teacher_full_response,
            "final_answer": final_answer,
            "v_top_layer_path": v_top_layer_relative_path,  # 在根目录保存路径（相对路径）
            "trajectory": {
                "steps": trajectory.steps,
            },
            "parameters": {
                "extraction_method": extraction_method,
                "steps": steps,
                "tau_grad": tau_grad if extraction_method == "gradient" else None,
                "skip_markup": skip_markup if extraction_method == "gradient" else None,
                "downsample": downsample if extraction_method == "gradient" else None,
                "max_tokens_per_step": max_tokens_per_step if extraction_method == "gradient" else None,
                "max_grad_calls_per_step": max_grad_calls_per_step if extraction_method == "gradient" else None,
                "aggregation_method": aggregation_method if extraction_method == "gradient" else None,
                "attribution_threshold": attribution_threshold if extraction_method == "gradient" else None,
                "min_entropy": min_entropy if extraction_method == "gradient" else None,
                "topk": topk if extraction_method == "attention" else None,
            },
        }
        
        # 添加校验信息（如果启用）
        if enable_validation:
            result["validation"] = {
                "is_correct": is_answer_correct,
                "explanation": validation_explanation,
            }
        
        print(f"  [样本 {sample.get('question_id', 'unknown')}] 完成！")
        return result
        
    except Exception as e:
        print(f"  [样本 {sample.get('question_id', 'unknown')}] 处理失败: {str(e)}")
        # 异常时清理显存（支持多 GPU）
        if torch.cuda.is_available():
            if hasattr(extractor.model, 'hf_device_map') and extractor.model.hf_device_map is not None:
                # 多 GPU 模式：清理所有 GPU
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
            else:
                # 单 GPU 模式：只清理当前设备
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()
        return {
            "question_id": sample.get("question_id", None),
            "error": str(e),
            "image_path": image_full_path if 'image_full_path' in locals() else None,
        }


def main():
    parser = argparse.ArgumentParser(
        description="从Visual-CoT数据集中提取视觉轨迹"
    )
    
    # 数据相关参数
    parser.add_argument("--data_file", type=str, default="./data/Visual-CoT-full/viscot_363k_lvr_formatted.json", help="Visual-CoT数据文件路径")
    parser.add_argument("--base_image_dir", type=str, default="./data/Visual-CoT-full", help="图像文件的基础目录")
    parser.add_argument("--max_samples", type=int, default=5, help="最大处理样本数量（None表示处理所有样本）")
    parser.add_argument("--start_idx", type=int, default=0, help="起始样本索引（在随机打乱后应用）")
    parser.add_argument("--random_seed", type=int, default=5, help="随机数种子，用于固定随机顺序（默认不使用随机）")
    parser.add_argument("--shuffle", action="store_true", default=True, help="随机打乱数据顺序（需要配合 --random_seed 使用以确保可复现）")
    
    # 模型相关参数
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/my_qwen_model/Qwen/Qwen2.5-VL-32B-Instruct", help="模型路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备（cuda或cpu）")
    
    # 提取方法选择
    parser.add_argument("--method", type=str, default="attention", choices=["gradient", "attention"], 
                        help="提取方法: 'gradient' (梯度归因) 或 'attention' (注意力权重)")
    
    # 提取参数（梯度归因方法）
    parser.add_argument("--steps", type=int, default=1, help="推理步骤数")
    parser.add_argument("--tau_grad", type=float, default=0.01, help="梯度归因的温度参数（仅用于gradient方法）")
    parser.add_argument("--skip_markup", action="store_true", help="跳过包含<>的标记token（默认启用，仅用于gradient方法）")
    parser.add_argument("--no_skip_markup", action="store_false", dest="skip_markup", help="禁用跳过标记token")
    parser.set_defaults(skip_markup=True)
    parser.add_argument("--downsample", type=int, default=1, help="token抽样间隔（仅用于gradient方法）")
    parser.add_argument("--max_tokens_per_step", type=int, default=50, help="每步最大处理token数量（仅用于gradient方法）")
    parser.add_argument("--max_grad_calls_per_step", type=int, default=10, help="每步最大梯度计算次数（仅用于gradient方法）")
    parser.add_argument("--aggregation_method", type=str, default="weighted_by_strength", choices=["simple_avg", "weighted_by_strength", "filter_by_threshold", "entropy_weighted"], help="聚合策略（仅用于gradient方法）")
    parser.add_argument("--attribution_threshold", type=float, default=0.1, help="归因强度阈值（用于filter_by_threshold，仅用于gradient方法）")
    parser.add_argument("--min_entropy", type=float, default=0.5, help="最小熵阈值（用于entropy_weighted，仅用于gradient方法）")
    
    # 提取参数（注意力权重方法）
    parser.add_argument("--topk", type=int, default=8, help="返回top-k个最重要的图像token索引（仅用于attention方法）")
    
    # 输出相关参数
    parser.add_argument("--output_dir", type=str, default="./trajectories/viscot", help="输出目录")
    parser.add_argument("--save_format", type=str, default="json", choices=["json", "jsonl"], help="保存格式：json（单个文件）或jsonl（每行一个样本）")
    
    # 校验相关参数
    parser.add_argument("--enable_validation", action="store_true", default=True, help="启用答案校验（默认启用）")
    parser.add_argument("--disable_validation", action="store_false", dest="enable_validation", help="禁用答案校验")
    parser.add_argument("--validation_api_key", type=str, default=None, help="OpenRouter API 密钥（默认使用内置密钥）")
    parser.add_argument("--validation_model", type=str, default="mimo-v2-flash", help="用于校验的模型名称")
    parser.add_argument("--save_only_correct", action="store_true", default=False, help="只保存校验通过的样本（默认保存所有样本）")
    
    # V_top_layer 相关参数
    parser.add_argument("--enable_v_top_layer", action="store_true", default=True, help="启用 V_top_layer 捕获（默认启用）")

    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 加载数据
    data = load_viscot_data(args.data_file, max_samples=None)  # 先加载全部数据，稍后再限制数量
    
    # 随机打乱数据顺序（如果启用）
    used_seed = None
    if args.shuffle or args.random_seed is not None:
        if args.random_seed is not None:
            used_seed = args.random_seed
            random.seed(args.random_seed)
            print(f"使用随机数种子: {args.random_seed}")
        else:
            # 如果启用了 shuffle 但没有指定种子，使用当前时间作为种子
            import time
            used_seed = int(time.time())
            random.seed(used_seed)
            print(f"使用时间戳作为随机数种子: {used_seed}")
        
        # 创建索引列表并打乱
        indices = list(range(len(data)))
        random.shuffle(indices)
        # 按照打乱后的顺序重新排列数据
        data = [data[i] for i in indices]
        print(f"数据已随机打乱，共 {len(data)} 个样本")
    else:
        print(f"保持原始数据顺序，共 {len(data)} 个样本")
    
    # 应用起始索引和最大样本数限制
    if args.start_idx > 0:
        data = data[args.start_idx:]
        print(f"从索引 {args.start_idx} 开始处理，剩余 {len(data)} 个样本")
    
    if args.max_samples is not None and args.max_samples > 0:
        original_count = len(data)
        data = data[:args.max_samples]
        print(f"限制处理数量为 {args.max_samples} 个样本（原始: {original_count}）")
    
    # 保存处理参数（用于记录和复现）
    processing_params = {
        "extraction_method": args.method,
        "random_seed": used_seed,
        "shuffled": args.shuffle or args.random_seed is not None,
        "start_idx": args.start_idx,
        "max_samples": args.max_samples,
        "total_samples_loaded": len(data),
    }
    
    # 初始化提取器
    print(f"正在加载模型: {args.model_path}")
    print(f"使用提取方法: {args.method}")
    
    # 诊断：检查GPU数量和显存
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"[GPU诊断] 检测到 {gpu_count} 个GPU:")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3  # GB
            print(f"  GPU {i}: {props.name}, 总显存: {total_memory:.2f}GB")
    else:
        print("[GPU诊断] 未检测到CUDA设备")
    
    if args.method == "gradient":
        extractor = TeacherTrajectoryExtractor(
            model_path=args.model_path,
            device=args.device,
            dtype=torch.bfloat16
        )
    elif args.method == "attention":
        extractor = AttentionBasedTrajectoryExtractor(
            model_path=args.model_path,
            device=args.device,
            dtype=torch.bfloat16
        )
    else:
        raise ValueError(f"未知的提取方法: {args.method}")
    
    print("模型加载完成！")
    
    # 诊断：检查模型实际使用的GPU
    if torch.cuda.is_available():
        print(f"[GPU诊断] 模型加载后的GPU使用情况:")
        if hasattr(extractor.model, 'hf_device_map') and extractor.model.hf_device_map is not None:
            print(f"  设备分布: {extractor.model.hf_device_map}")
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1024**3
                reserv = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  GPU {i}: 已分配={alloc:.2f}GB, 已保留={reserv:.2f}GB")
        else:
            print("  警告：模型未使用多GPU，只使用了单GPU模式")
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserv = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU 0: 已分配={alloc:.2f}GB, 已保留={reserv:.2f}GB")
    
    # 处理所有样本
    results = []
    for i, sample in enumerate(data):
        print(f"\n[{i+1}/{len(data)}] 处理样本...")
        
        # 诊断：打印处理前的显存状态
        if torch.cuda.is_available():
            if hasattr(extractor.model, 'hf_device_map') and extractor.model.hf_device_map is not None:
                print(f"[显存状态-处理前] 所有GPU显存使用:")
                for gpu_id in range(torch.cuda.device_count()):
                    alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    reserv = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    print(f"  GPU {gpu_id}: 已分配={alloc:.2f}GB, 已保留={reserv:.2f}GB")
            else:
                alloc = torch.cuda.memory_allocated() / 1024**3
                reserv = torch.cuda.memory_reserved() / 1024**3
                print(f"[显存状态-处理前] GPU 0: 已分配={alloc:.2f}GB, 已保留={reserv:.2f}GB")
        
        result = process_sample(
            sample=sample,
            extractor=extractor,
            base_image_dir=args.base_image_dir,
            output_dir=args.output_dir,
            extraction_method=args.method,
            steps=args.steps,
            tau_grad=args.tau_grad,
            skip_markup=args.skip_markup,
            downsample=args.downsample,
            max_tokens_per_step=args.max_tokens_per_step,
            max_grad_calls_per_step=args.max_grad_calls_per_step,
            aggregation_method=args.aggregation_method,
            attribution_threshold=args.attribution_threshold,
            min_entropy=args.min_entropy,
            topk=args.topk,
            enable_validation=args.enable_validation,
            validation_api_key=args.validation_api_key,
            validation_model=args.validation_model,
            enable_v_top_layer=args.enable_v_top_layer,
        )
        results.append(result)
        
        # 处理完样本后，强制清理显存
        if torch.cuda.is_available():
            if hasattr(extractor.model, 'hf_device_map') and extractor.model.hf_device_map is not None:
                # 多 GPU 模式：清理所有 GPU
                for gpu_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
            else:
                # 单 GPU 模式：只清理当前设备
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()
        
        # 每处理10个样本保存一次检查点（防止数据丢失）
        if (i + 1) % 10 == 0:
            checkpoint_file = os.path.join(args.output_dir, "trajectories_checkpoint.json")
            checkpoint_data = {
                "processing_params": processing_params,
                "results": results,
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            print(f"已保存检查点到 {checkpoint_file} ({len(results)} 个样本)")
    
    # 保存最终结果
    print(f"\n正在保存结果到 {args.output_dir}...")
    
    # 根据 --save_only_correct 选项过滤结果
    if args.save_only_correct and args.enable_validation:
        # 只保存校验通过的样本
        filtered_results = [r for r in results if r.get("validation", {}).get("is_correct", False)]
        print(f"  过滤前: {len(results)} 个样本")
        print(f"  过滤后: {len(filtered_results)} 个样本（仅校验通过的）")
    else:
        # 保存所有样本
        filtered_results = results
    
    if args.save_format == "json":
        # 保存为单个JSON文件
        output_file = os.path.join(args.output_dir, "trajectories.json")
        output_data = {
            "processing_params": processing_params,
            "results": filtered_results,
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(filtered_results)} 个样本到 {output_file}")
    else:
        # 保存为JSONL文件（每行一个样本）
        output_file = os.path.join(args.output_dir, "trajectories.jsonl")
        # JSONL 格式：第一行保存处理参数，后续每行一个样本
        with open(output_file, 'w', encoding='utf-8') as f:
            # 第一行：处理参数
            f.write(json.dumps({"type": "metadata", "processing_params": processing_params}, ensure_ascii=False) + '\n')
            # 后续行：样本数据
            for result in filtered_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"已保存 {len(filtered_results)} 个样本到 {output_file}")
    
    # 统计信息
    successful = sum(1 for r in results if "error" not in r and r.get("skipped", False) == False)
    failed = sum(1 for r in results if "error" in r)
    skipped = sum(1 for r in results if r.get("skipped", False) == True)
    total = len(results)
    print(f"\n处理完成！")
    print(f"  成功（已保存轨迹）: {successful}")
    print(f"  跳过（答案校验未通过）: {skipped}")
    print(f"  失败（处理错误）: {failed}")
    print(f"  总计: {total}")
    
    # 如果启用了校验，显示校验统计
    if args.enable_validation:
        validated_correct = sum(1 for r in results if r.get("validation", {}).get("is_correct", False))
        validated_incorrect = sum(1 for r in results if "validation" in r and not r.get("validation", {}).get("is_correct", True))
        print(f"\n校验统计:")
        print(f"  校验通过: {validated_correct}")
        print(f"  校验未通过: {validated_incorrect}")


if __name__ == "__main__":
    main()

