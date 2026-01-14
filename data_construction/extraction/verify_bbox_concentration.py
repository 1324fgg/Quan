#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校验教师轨迹注意力在 Ground Truth BBox 中的权重占比
用于验证教师轨迹是否真的关注到了目标区域
"""

import os
import json
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image

def estimate_grid_size(
    width: int,
    height: int,
    patch_size: int = 14,
    merge_size: int = 2,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
) -> Tuple[int, int]:
    """估算 Qwen2.5-VL 的 grid 尺寸"""
    pixels = width * height
    scale = 1.0
    if pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
    elif pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
    
    new_width = max(int(width * scale) // (patch_size * merge_size) * (patch_size * merge_size), patch_size * merge_size)
    new_height = max(int(height * scale) // (patch_size * merge_size) * (patch_size * merge_size), patch_size * merge_size)
    
    grid_w = new_width // patch_size // merge_size
    grid_h = new_height // patch_size // merge_size
    return grid_h, grid_w

def bbox_to_patch_indices(
    bbox: List[float],
    grid_h: int,
    grid_w: int,
) -> List[int]:
    """将归一化 bbox [x1, y1, x2, y2] 转换为 patch 索引"""
    x1, y1, x2, y2 = bbox
    
    # 按照 traj_vis.py 逻辑，使用 (min_dim, max_dim) 作为 reshape 后的 grid
    reshape_rows, reshape_cols = (grid_h, grid_w) if grid_h <= grid_w else (grid_w, grid_h)
    
    row_start = max(0, int(y1 * reshape_rows))
    row_end = min(reshape_rows, int(math.ceil(y2 * reshape_rows)))
    col_start = max(0, int(x1 * reshape_cols))
    col_end = min(reshape_cols, int(math.ceil(x2 * reshape_cols)))
    
    indices = []
    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            indices.append(r * reshape_cols + c)
    return indices

def calculate_concentration(
    attention_weights: List[float],
    bboxes: List[List[float]],
    image_size: Tuple[int, int]
) -> float:
    """计算注意力在 bboxes 中的总权重占比"""
    if not attention_weights or not bboxes:
        return 0.0
    
    n_tokens = len(attention_weights)
    h, w = image_size
    grid_h, grid_w = estimate_grid_size(w, h)
    
    # 校准 grid 尺寸以匹配 token 数
    if grid_h * grid_w != n_tokens:
        # 简单校准：如果是转置关系
        if grid_w * grid_h == n_tokens:
            pass # 后面逻辑会处理 reshape
        else:
            # 暴力反推
            ratio = w / h
            grid_h = int(math.sqrt(n_tokens / ratio))
            grid_w = n_tokens // grid_h
            while grid_h * grid_w != n_tokens:
                if grid_h * grid_w < n_tokens: grid_w += 1
                else: grid_h -= 1; grid_w = n_tokens // grid_h

    all_bbox_indices = set()
    for bbox in bboxes:
        indices = bbox_to_patch_indices(bbox, grid_h, grid_w)
        all_bbox_indices.update(indices)
    
    weights = np.array(attention_weights)
    inside_weight = sum(weights[idx] for idx in all_bbox_indices if idx < len(weights))
    total_weight = weights.sum()
    
    return float(inside_weight / (total_weight + 1e-10))

def verify_sample(traj_path: str, bboxes: List[List[float]], image_path: Optional[str] = None) -> float:
    """验证单个轨迹文件的注意力集中度"""
    with open(traj_path, 'r') as f:
        data = json.load(f)
    
    # 提取第一步的注意力
    if 'steps' not in data or not data['steps']:
        return 0.0
    
    attention = data['steps'][0].get('p_t', [])
    if not attention:
        return 0.0
    
    # 获取图像尺寸
    if 'metadata' in data and 'image_size' in data['metadata']:
        image_size = tuple(data['metadata']['image_size'])
    elif image_path and os.path.exists(image_path):
        with Image.open(image_path) as img:
            image_size = (img.height, img.width)
    else:
        # 默认假设
        image_size = (1000, 1000)
    
    return calculate_concentration(attention, bboxes, image_size)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", type=str, required=True, help="轨迹 JSON 目录")
    parser.add_argument("--data_file", type=str, required=True, help="包含 bbox 的原始数据文件")
    parser.add_argument("--output", type=str, default="concentration_report.json")
    args = parser.parse_args()
    
    with open(args.data_file, 'r') as f:
        dataset = json.load(f)
    
    results = []
    concentrations = []
    
    for item in dataset:
        qid = item.get('question_id')
        bboxes = item.get('bboxes', [])
        if not bboxes: continue
        
        # 匹配轨迹文件
        traj_filename = f"sample_{qid:06d}_attention.json" if isinstance(qid, int) else f"sample_{qid}_attention.json"
        traj_path = os.path.join(args.traj_dir, traj_filename)
        
        if os.path.exists(traj_path):
            concentration = verify_sample(traj_path, bboxes)
            concentrations.append(concentration)
            results.append({
                "question_id": qid,
                "concentration": concentration,
                "is_valid": concentration > 0.1 # 阈值可调
            })
    
    report = {
        "average_concentration": np.mean(concentrations) if concentrations else 0,
        "median_concentration": np.median(concentrations) if concentrations else 0,
        "min_concentration": np.min(concentrations) if concentrations else 0,
        "max_concentration": np.max(concentrations) if concentrations else 0,
        "total_samples": len(results),
        "valid_samples": sum(1 for r in results if r['is_valid']),
        "details": results
    }
    
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Verified {len(results)} samples.")
    print(f"Average Concentration: {report['average_concentration']:.4f}")
    print(f"Report saved to {args.output}")
