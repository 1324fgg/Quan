#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
答案校验模块
使用 LLM API 来判断教师模型的答案是否正确
"""

import re
from typing import Optional, Tuple
from openai import OpenAI


def extract_final_answer(teacher_response: str) -> Optional[str]:
    """
    从教师模型的响应中提取 FINAL ANSWER 部分
    
    Args:
        teacher_response: 教师模型的完整响应文本
        
    Returns:
        FINAL ANSWER 的内容，如果没有找到则返回 None
    """
    # 首先尝试找到 "FINAL ANSWER:" 标记
    final_answer_marker = re.search(
        r"FINAL\s+ANSWER\s*:?\s*\n?", 
        teacher_response, 
        re.IGNORECASE
    )
    
    if final_answer_marker:
        # 提取 FINAL ANSWER 标记后的所有内容
        start_pos = final_answer_marker.end()
        remaining_text = teacher_response[start_pos:].strip()
        
        if remaining_text:
            # 提取第一段非空内容（可能是多行）
            lines = remaining_text.split('\n')
            answer_lines = []
            for line in lines:
                line = line.strip()
                # 跳过空行和明显的格式标记
                if line and not line.startswith('<step') and not line.startswith('REASONING'):
                    answer_lines.append(line)
                    # 如果遇到明显的结束标记，停止
                    if line.startswith('---') or line.startswith('==='):
                        break
            
            if answer_lines:
                answer = ' '.join(answer_lines).strip()
                # 去除可能的额外格式标记
                answer = re.sub(r"^\s*\[|\]\s*$", "", answer)  # 去除方括号
                answer = re.sub(r"^\"|\"$", "", answer)  # 去除引号
                answer = answer.strip()
                if answer:
                    return answer
    
    # 如果没有找到 FINAL ANSWER，尝试提取最后一个非空行（作为备用方案）
    lines = teacher_response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('<step') and not line.startswith('REASONING') and 'FINAL' not in line.upper():
            return line
    
    return None


def validate_answer_with_llm(
    predicted_answer: str,
    ground_truth: str,
    question: Optional[str] = None,
    api_key: str = "sk-ssi02jvclw4pw34b7wkxngtmkwt0g5x6b107wok1f4p0khnc",
    model: str = "mimo-v2-flash",
    base_url: str = "https://api.xiaomimimo.com/v1",
) -> Tuple[bool, str]:
    """
    使用 LLM API 判断预测答案是否正确
    
    Args:
        predicted_answer: 教师 model 预测的答案
        ground_truth: 标准答案
        question: 问题（可选，用于提供上下文）
        api_key: API 密钥
        model: 使用的模型名称
        base_url: API 基础 URL
        
    Returns:
        (is_correct: bool, explanation: str) 元组
        is_correct: 答案是否正确
        explanation: LLM 的判断说明
    """
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    
    # 构建提示词（英文）
    prompt_parts = []
    
    # 添加问题（如果有）
    if question:
        prompt_parts.extend([
            "Question:",
            question,
            "",
        ])
    
    # 添加标准答案和预测答案
    prompt_parts.extend([
        "Ground Truth Answer:",
        ground_truth,
        "",
        "Teacher's Response:",
        predicted_answer,
        "",
        "Please carefully compare the teacher's response with the ground truth answer and determine if they express the same meaning or provide the same result.",
        "If the teacher's response contains reasoning steps, focus on the final answer or conclusion.",
        "Consider the following:",
        "- Do both answers address the same question?",
        "- Do they convey the same information or conclusion?",
        "- Are they semantically equivalent (even if worded differently)?",
        "- If the teacher's response contains multiple parts, does the final answer or conclusion match the ground truth?",
        "",
        "Your response must start with either 'YES' (if the answers are consistent) or 'NO' (if they are inconsistent),",
        "followed by a brief explanation of your judgment.",
    ])
    
    prompt = "\n".join(prompt_parts)
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,  # 使用较低的温度以获得更一致的判断
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # 解析响应，判断是否为 YES
        response_upper = response_text.upper()
        is_correct = response_upper.startswith("YES")
        
        return is_correct, response_text
        
    except Exception as e:
        # 如果 API 调用失败，返回错误信息
        error_msg = f"API call failed: {str(e)}"
        print(f"[WARNING] {error_msg}")
        return False, error_msg


def validate_sample(
    teacher_response: str,
    ground_truth: str,
    question: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    完整的样本校验流程
    
    Args:
        teacher_response: 教师模型的完整响应
        ground_truth: 标准答案
        question: 问题（可选）
        api_key: API 密钥（可选，使用默认值）
        model: 使用的模型名称（可选，使用默认值）
        base_url: API 基础 URL（可选，使用默认值）
        
    Returns:
        (is_valid: bool, final_answer: Optional[str], explanation: Optional[str]) 元组
        is_valid: 答案是否正确
        final_answer: 提取的最终答案
        explanation: LLM 的判断说明
    """
    # 1. 提取 FINAL ANSWER
    final_answer = extract_final_answer(teacher_response)
    
    # 2. 如果无法提取 FINAL ANSWER，使用完整的教师响应作为预测答案
    if final_answer is None:
        print("[WARNING] Failed to extract FINAL ANSWER, using full teacher response for validation")
        predicted_answer = teacher_response
        final_answer = None  # 保持为 None 以标识提取失败
    else:
        predicted_answer = final_answer
    
    # 3. 使用 LLM 判断答案是否正确
    if api_key is None:
        api_key = "sk-ssi02jvclw4pw34b7wkxngtmkwt0g5x6b107wok1f4p0khnc"
    if model is None:
        model = "mimo-v2-flash"
    if base_url is None:
        base_url = "https://api.xiaomimimo.com/v1"
    
    is_correct, explanation = validate_answer_with_llm(
        predicted_answer=predicted_answer,
        ground_truth=ground_truth,
        question=question,
        api_key=api_key,
        model=model,
        base_url=base_url,
    )
    
    return is_correct, final_answer, explanation


if __name__ == "__main__":
    # 测试代码
    test_response = """
REASONING:
<step>The image shows two individuals standing outdoors near a gate.</step>
<step>The person on the left has short hair, while the person on the right has longer, wavier hair.</step>

FINAL ANSWER:
The individual on the left has short hair, and the individual on the right has longer, wavy hair styled casually.
"""
    
    test_ground_truth = "The person on the left has short hair."
    
    is_valid, final_answer, explanation = validate_sample(
        teacher_response=test_response,
        ground_truth=test_ground_truth,
        question="What is the hair style of the person on the left?",
    )
    
    print(f"Final Answer: {final_answer}")
    print(f"Is Correct: {is_valid}")
    print(f"Explanation: {explanation}")
