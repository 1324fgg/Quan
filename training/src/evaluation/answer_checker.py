#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
答案检查工具
用于评估结果中答案正确性的判断
"""

import re
import json
from typing import Optional, Tuple, List, Dict


def extract_answer_letter(text: str) -> Optional[str]:
    """
    从文本中提取答案字母（用于选择题）
    
    采用更健壮的提取策略：
    1. 寻找明确的结论标记（如 "Answer:", "Therefore", "Final Answer:" 等）
    2. 优先考虑文本末尾出现的标记
    3. 寻找带括号的字母 (A)
    
    Args:
        text: 输入文本
        
    Returns:
        提取的字母（小写），如果未找到则返回 None
    """
    if not text:
        return None
        
    # 移除结束标记和空白
    text = re.sub(r'<\|im_end\|>$', '', str(text)).strip()
    
    # 1. 寻找带括号的字母，通常这是最可靠的信号
    # 我们寻找最后一个出现的 (A) 到 (H)
    matches = re.findall(r'\(([A-Ha-h])\)', text)
    if matches:
        return matches[-1].lower()

    # 2. 寻找结论性标记及其后的字母
    conclusion_patterns = [
        r'[Aa]nswer\s*:?\s*([A-Ha-h])\b',
        r'[Cc]hoice\s*:?\s*([A-Ha-h])\b',
        r'[Cc]onclusion\s*:?\s*([A-Ha-h])\b',
        r'[Ff]inal\s+answer\s*:?\s*([A-Ha-h])\b',
        r'[Tt]herefore.*?\bis\s+([A-Ha-h])\b',
        r'[Tt]he\s+correct\s+answer\s+is\s+([A-Ha-h])\b',
        r'([A-Ha-h])\s+is\s+the\s+correct\s+answer'
    ]
    
    for pattern in conclusion_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].lower()

    # 3. 寻找以字母开头的最后几行（通常是 "A. xxx"）
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        # 匹配 "A." 或 "A:" 或 "A)"
        match = re.match(r'^([A-Ha-h])[\.\:\)]', line)
        if match:
            # 注意：这可能是选项列表，只有在文本末尾时才比较可靠
            # 这里我们先记录下来，如果没有更好的再用
            pass
            
    # 4. 寻找最后一个出现的独立字母 (A-H)
    # 排除常见的干扰项如 'a' (冠词)
    matches = re.findall(r'\b([A-Ha-h])\b', text)
    if matches:
        # 过滤掉小写的 'a'，因为它太常用了
        filtered = [m for m in matches if m.upper() in 'ABCDEFGH' and m != 'a']
        if filtered:
            return filtered[-1].lower()
        # 如果只有 'a' 且它在最后，也勉强接受
        if matches[-1].lower() == 'a':
            return 'a'

    # 最后的兜底：原本的逻辑，但只匹配单个字母，且不再那么激进
    if len(text) < 10:
        match = re.search(r'([A-Ha-h])', text, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    return None


def check_answer_correct(
    predicted_answer: str,
    ground_truth: str,
    method: str = "regex"
) -> bool:
    """
    判断预测答案是否正确
    
    Args:
        predicted_answer: 模型预测的答案
        ground_truth: 标准答案
        method: 判断方法，目前支持 "regex"（正则表达式方法）
        
    Returns:
        True 如果答案正确，False 否则
    """
    if method == "regex":
        pred_letter = extract_answer_letter(predicted_answer)
        gt_letter = extract_answer_letter(ground_truth)
        
        if pred_letter is None or gt_letter is None:
            return False
        
        return pred_letter == gt_letter
    else:
        raise ValueError(f"Unsupported method: {method}")


def calculate_accuracy(
    results_file: str,
    method: str = "regex"
) -> Dict:
    """
    计算评估结果的准确率
    
    Args:
        results_file: JSONL 格式的评估结果文件路径
        method: 判断方法，目前支持 "regex"
        
    Returns:
        包含统计信息的字典:
        {
            "total": int,
            "correct": int,
            "wrong": int,
            "accuracy": float,
            "failed_extraction": int  # 无法提取答案的样本数
        }
    """
    total = 0
    correct = 0
    failed_extraction = 0
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            total += 1
            
            pred = data.get('model_output', '')
            gt = data.get('ground_truth', '')
            
            if check_answer_correct(pred, gt, method=method):
                correct += 1
            else:
                # 检查是否是因为无法提取答案
                pred_letter = extract_answer_letter(pred)
                gt_letter = extract_answer_letter(gt)
                if pred_letter is None or gt_letter is None:
                    failed_extraction += 1
    
    accuracy = correct / total * 100 if total > 0 else 0.0
    
    return {
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": accuracy,
        "failed_extraction": failed_extraction
    }


def analyze_results(
    results_file: str,
    method: str = "regex",
    show_examples: int = 5
) -> None:
    """
    分析评估结果并打印统计信息
    
    Args:
        results_file: JSONL 格式的评估结果文件路径
        method: 判断方法
        show_examples: 显示正确/错误示例的数量
    """
    stats = calculate_accuracy(results_file, method=method)
    
    print("=" * 60)
    print("评估结果统计")
    print("=" * 60)
    print(f"总样本数: {stats['total']}")
    print(f"正确数: {stats['correct']}")
    print(f"错误数: {stats['wrong']}")
    print(f"准确率: {stats['accuracy']:.2f}%")
    if stats['failed_extraction'] > 0:
        print(f"无法提取答案: {stats['failed_extraction']}")
    print("=" * 60)
    
    # 显示示例
    correct_examples = []
    wrong_examples = []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            pred = data.get('model_output', '')
            gt = data.get('ground_truth', '')
            
            is_correct = check_answer_correct(pred, gt, method=method)
            
            example = {
                'id': data.get('id', ''),
                'prompt': data.get('prompt', '').split('\n')[0],
                'gt': gt,
                'pred': pred[:100] + '...' if len(pred) > 100 else pred
            }
            
            if is_correct:
                if len(correct_examples) < show_examples:
                    correct_examples.append(example)
            else:
                if len(wrong_examples) < show_examples:
                    wrong_examples.append(example)
            
            if len(correct_examples) >= show_examples and len(wrong_examples) >= show_examples:
                break
    
    if correct_examples:
        print(f"\n正确示例 (前{len(correct_examples)}个):")
        for i, ex in enumerate(correct_examples, 1):
            print(f"\n  {i}. ID: {ex['id']}")
            print(f"     问题: {ex['prompt']}")
            print(f"     GT: {ex['gt']} | Pred: {ex['pred']}")
    
    if wrong_examples:
        print(f"\n错误示例 (前{len(wrong_examples)}个):")
        for i, ex in enumerate(wrong_examples, 1):
            print(f"\n  {i}. ID: {ex['id']}")
            print(f"     问题: {ex['prompt']}")
            print(f"     GT: {ex['gt']} | Pred: {ex['pred']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="计算评估结果的准确率")
    parser.add_argument("--results_file", type=str, required=True, help="评估结果 JSONL 文件路径")
    parser.add_argument("--method", type=str, default="regex", help="判断方法 (默认: regex)")
    parser.add_argument("--show_examples", type=int, default=5, help="显示示例数量")
    
    args = parser.parse_args()
    
    analyze_results(args.results_file, method=args.method, show_examples=args.show_examples)
