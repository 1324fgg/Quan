#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并轨迹（Attention）和 VTOP 数据，生成最终训练集。
包含：
1. 正确性校验（is_correct）
2. 注意力集中度校验（concentration > threshold）
3. 设置绝对路径供 dataset.py 使用
"""

import os
import json
import argparse
import sys
from typing import List, Dict, Any

# 添加搜索路径以导入 verify_bbox_concentration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'extraction')))
from verify_bbox_concentration import verify_sample

def prepare_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_file", type=str, required=True, help="轨迹 JSON 文件 (attention)")
    parser.add_argument("--vtop_file", type=str, required=True, help="VTOP JSON 文件")
    parser.add_argument("--original_data", type=str, required=True, help="包含 bbox 的原始数据文件")
    parser.add_argument("--output_file", type=str, required=True, help="输出的训练集文件")
    parser.add_argument("--concentration_threshold", type=float, default=0.1, help="注意力集中度阈值")
    parser.add_argument("--traj_root", type=str, required=True, help="轨迹文件根目录（用于校验和设置绝对路径）")
    parser.add_argument("--vtop_root", type=str, required=True, help="VTOP文件根目录")
    args = parser.parse_args()

    print(f"Loading original data from {args.original_data}...")
    with open(args.original_data, 'r') as f:
        original_list = json.load(f)
    original_map = {item.get('question_id'): item for item in original_list}

    print(f"Loading trajectory results from {args.traj_file}...")
    with open(args.traj_file, 'r') as f:
        traj_data = json.load(f)
    traj_results = {item.get('question_id'): item for item in traj_data.get('results', [])}

    print(f"Loading VTOP results from {args.vtop_file}...")
    with open(args.vtop_file, 'r') as f:
        vtop_data = json.load(f)
    vtop_results = {item.get('question_id'): item for item in vtop_data.get('results', [])}

    final_results = []
    stats = {
        "total": len(original_list),
        "found_in_traj": 0,
        "found_in_vtop": 0,
        "correct": 0,
        "concentrated": 0,
        "final": 0
    }

    for qid, item in original_map.items():
        traj_item = traj_results.get(qid)
        vtop_item = vtop_results.get(qid)

        if not traj_item or not vtop_item:
            continue
        
        stats["found_in_traj"] += 1
        stats["found_in_vtop"] += 1

        # 1. 正确性校验
        is_correct = traj_item.get('validation', {}).get('is_correct', False)
        if not is_correct:
            continue
        stats["correct"] += 1

        # 2. 注意力集中度校验
        bboxes = item.get('bboxes', [])
        if not bboxes:
            continue
        
        # 寻找实际的轨迹文件路径
        traj_filename = f"sample_{qid:06d}_attention.json" if isinstance(qid, int) else f"sample_{qid}_attention.json"
        actual_traj_path = os.path.join(args.traj_root, traj_filename)
        
        if os.path.exists(actual_traj_path):
            concentration = verify_sample(actual_traj_path, bboxes)
        else:
            # 如果没找到单独的 JSON，尝试从 traj_item 中计算
            from verify_bbox_concentration import calculate_concentration
            p_t = traj_item.get('trajectory', {}).get('steps', [{}])[0].get('p_t', [])
            image_size = traj_item.get('metadata', {}).get('image_size', (1000, 1000))
            concentration = calculate_concentration(p_t, bboxes, image_size)

        if concentration < args.concentration_threshold:
            continue
        stats["concentrated"] += 1

        # 合并数据并设置绝对路径
        merged = traj_item.copy()
        
        # 设置 dataset.py 要求的绝对路径
        vtop_rel = vtop_item.get('v_top_layer_path')
        if vtop_rel:
            merged['v_top_path_abs'] = os.path.abspath(os.path.join(args.vtop_root, vtop_rel))
        
        merged['attention_path_abs'] = os.path.abspath(actual_traj_path)
        merged['concentration'] = concentration
        
        # 确保包含训练所需的关键字段
        if "ground_truth_enriched" not in merged:
            merged["ground_truth_enriched"] = item.get("ground_truth", merged.get("ground_truth", ""))

        final_results.append(merged)

    stats["final"] = len(final_results)
    
    output_data = {
        "processing_params": {
            "concentration_threshold": args.concentration_threshold,
            "stats": stats
        },
        "results": final_results
    }

    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\nProcessing Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nFinal training data saved to {args.output_file}")

if __name__ == "__main__":
    prepare_data()
