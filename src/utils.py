# -*- coding: utf-8 -*-
"""
工具函数模块
包含重新设计的 build_prompt 函数，使用 processor.apply_chat_template 方法
"""

from typing import List, Dict, Any
import pdb
import torch

def build_prompt_with_chat_template(
    processor,
    question: str, 
    steps_prefix: List[str], 
    force_T: int
) -> str:
    """
    使用 apply_chat_template 重新构造提示词
    
    Args:
        processor: AutoProcessor 实例，用于应用聊天模板
        question: 用户问题
        steps_prefix: 已完成的步骤列表，用于复现模型回答轨迹
        force_T: 总共需要的推理步骤数
    
    Returns:
        str: 格式化后的提示词
    """
    
    system_instruction = (
        "As a professional maze solver, your task is to observe the maze and answer the designed question "

        
        f"You must solve the question whatever it is in step by step in exactly {force_T} reasoning steps. Follow this EXACT format:\n\n"
        "REASONING:\n" +
        ("\n".join(["<step>analyze one aspect of the problem</step>" for i in range(force_T)])) +
        "\n\nFINAL ANSWER:\n\n\n"
        
        "IMPORTANT RULES:\n"
        "- Each <step> tag must contain only ONE specific analysis or observation\n"
        "- Do NOT include any reasoning outside the <step> tags\n"
        "- After all steps, provide ONLY the action sequence in the final answer\n"
        "- Do not explain your final answer - just give the moves"
    )

    test_instruction = (
        "Please carefully observe the provided image and answer the question accordingly."

        f"You must solve the question whatever it is in step by step in exactly {force_T} reasoning steps. Follow this EXACT format:\n\n"
        "REASONING:\n" +
        ("\n".join(["<step>analyze one aspect of the problem</step>" for i in range(force_T)])) +
        "\n\nFINAL ANSWER:\n\n\n"

         "IMPORTANT RULES:\n"
        "- Each <step> tag must contain only ONE specific analysis or observation\n"
        "- Do NOT include any reasoning outside the <step> tags\n"
        "- After all steps, provide ONLY the action sequence in the final answer\n"
        "- Do not explain your final answer - just give the moves"
    )
    # 构建消息列表
    messages = [
        {
            "role": "system",
            "content": test_instruction
        },
        {
            "role": "user", 
            "content": [
                {"type": "image"},  # 图像位置
                {"type": "text", "text": f"Question: {question}"}
            ]
        }
    ]
    
    # 如果有已完成的步骤，将其作为 assistant 的回答添加到对话历史中
    if steps_prefix:
        assistant_response = "\n".join(steps_prefix)
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })  
        
    
    # 使用 processor 的 apply_chat_template 方法生成格式化的提示词
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    if steps_prefix:
        inputs = processor(text=prompt, return_tensors="pt").to("cuda")
        
        # 找到最后一个值为151645的元素并删除
        input_ids = inputs["input_ids"][0]  # 获取第一个批次的input_ids
        last_idx = -1
        for i in range(len(input_ids) - 1, -1, -1):  # 从后往前查找
            if input_ids[i] == 151645:
                last_idx = i
                break
        
        if last_idx != -1:
            # 删除找到的元素
            new_input_ids = torch.cat([input_ids[:last_idx], input_ids[last_idx+1:]])
            inputs["input_ids"] = new_input_ids.unsqueeze(0)  # 重新添加批次维度
        prompt = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)

    return prompt


def build_teacher_generation_prompt(processor, question: str, force_T: int) -> str:
    """
    为教师模型生成完整回答构建提示词
    
    Args:
        processor: AutoProcessor 实例
        question: 用户问题
        force_T: 总共需要的推理步骤数
    
    Returns:
        str: 格式化后的提示词
    """
    system_instruction = (
        "As a professional maze solver, your task is to observe the maze and answer the designed question "

        
        f"You must solve the question whatever it is in step by step in exactly {force_T} reasoning steps. Follow this EXACT format:\n\n"
        "REASONING:\n" +
        ("\n".join(["<step>analyze one aspect of the problem</step>" for i in range(force_T)])) +
        "\n\nFINAL ANSWER:\n\n\n"
        
        "IMPORTANT RULES:\n"
        "- Each <step> tag must contain only ONE specific analysis or observation\n"
        "- Do NOT include any reasoning outside the <step> tags\n"
        "- After all steps, provide ONLY the action sequence in the final answer\n"
        "- Do not explain your final answer - just give the moves"
    )
    test_instruction = (
        "Please carefully observe the provided image and answer the question accordingly."

        f"You must solve the question whatever it is in step by step in exactly {force_T} reasoning steps. Follow this EXACT format:\n\n"
        "REASONING:\n" +
        ("\n".join(["<step>analyze one aspect of the problem</step>" for i in range(force_T)])) +
        "\n\nFINAL ANSWER:\n\n\n"

         "IMPORTANT RULES:\n"
        "- Each <step> tag must contain only ONE specific analysis or observation\n"
        "- Do NOT include any reasoning outside the <step> tags\n"
        "- After all steps, provide ONLY the action sequence in the final answer\n"
        "- Do not explain your final answer - just give the moves"
    )
    messages = [
        {
            "role": "system",
            "content": test_instruction
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Question: {question}"}
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt
