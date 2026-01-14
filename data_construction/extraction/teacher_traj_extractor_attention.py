# teacher_traj_extractor_attention.py
# -*- coding: utf-8 -*-
"""
Attention-based visual trajectory extractor (LVR-style format).

This script uses attention weights instead of gradient-based attribution
to obtain per-patch importance scores with lower computational cost.

High-level algorithm:
1. Run a forward pass and collect attention weights from all layers.
2. Locate text tokens and image tokens in the input sequence.
3. Extract the sub-matrix of attention from text tokens to image tokens.
4. Average over layers, heads and text tokens to obtain a per-image-token
   attention score (visual trajectory).

LVR-format notes:
- Uses an LVR-style prompt format (aligned with VisualMindTraining).
- Does not require <step> tags; the teacher model is free to produce reasoning.
- Designed for single-step use with the ``--steps 1`` option.

Usage example:
  python teacher_traj_extractor_attention.py --image path/to/img.jpg --question "..." --steps 1 --save out.json
"""

from __future__ import annotations
import os, json, re, gc
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))) # For src

from src.lvr_prompt_utils import build_prompt_with_chat_template, build_teacher_generation_prompt
from v_top_layer_extractor import VTopLayerExtractor
from src.attention_hook_utils import AttentionHookExtractor

# ----------------------------
# Data Structures
# ----------------------------

@dataclass
class Trajectory:
    steps: List[Dict]  # each: { "p_t": list[float], "p_topk_idx": list[int], "method": str, ... }
    v_top_layer_path: Optional[str] = None  # 指向 V_top_layer 二进制文件的路径（.npy 或 .pt）
    image_path: Optional[str] = None  # 原始图像路径（用于可视化）
    question: Optional[str] = None  # 问题文本（用于可视化）

# ----------------------------
# Attention-based Trajectory Extractor
# ----------------------------

class AttentionBasedTrajectoryExtractor:
    """
    基于注意力权重的视觉轨迹提取器
    
    核心思想：
    - 从模型前向传播中提取注意力权重（output_attentions=True）
    - 识别文本token位置（image_token_mask的逆）
    - 提取文本token对图像token的注意力子矩阵
    - 对所有层、所有头、所有文本token求平均，得到 S 向量（长度为 N_img）
    """

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        """
        初始化提取器
        
        Args:
            model_path: 模型路径
            device: 设备（"cuda" 或 "cpu"）
            dtype: 模型数据类型
        """
        self.device = device
        
        # 加载 processor 和 model
        # 检查 checkpoint 中是否有 preprocessor_config.json
        preprocessor_path = os.path.join(model_path, "preprocessor_config.json")
        if not os.path.exists(preprocessor_path):
            print(f"[INFO] preprocessor_config.json not found in {model_path}. Loading from base model.")
            # 从基础模型加载 processor
            base_model_path = "/root/autodl-tmp/my_qwen_model/Qwen2.5-VL-3B-Instruct"
            if not os.path.exists(base_model_path):
                base_model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
            
            self.processor = AutoProcessor.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )
            
            # 从 checkpoint 加载 tokenizer（确保使用训练后的词汇表）
            from transformers import AutoTokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                self.processor.tokenizer = tokenizer
                print("[INFO] Successfully loaded tokenizer from checkpoint.")
            except Exception as e:
                print(f"[WARNING] Could not load tokenizer from checkpoint: {e}. Using base tokenizer.")
        else:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        
        self.tokenizer = self.processor.tokenizer

        # 检查GPU数量，如果有多GPU则强制使用多GPU
        max_memory = None
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # 计算每个GPU的可用显存（保留10%作为缓冲）
            max_memory = {}
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / 1024**3
                # 为每个GPU分配90%的显存，保留10%作为缓冲
                max_memory[i] = f"{int(total_gb * 0.9)}GiB"
            print(f"[INFO] 检测到 {torch.cuda.device_count()} 个GPU，强制使用多GPU模式")
            print(f"[INFO] 每个GPU的max_memory设置: {max_memory}")
        elif torch.cuda.is_available():
            print(f"[INFO] 只检测到1个GPU，使用单GPU模式")

        # 加载模型时指定attn_implementation='eager'以支持attention输出
        # 注意：虽然eager模式比默认模式更占显存，但为了获取attention权重是必须的
        # 我们会在生成阶段关闭output_attentions来节省显存，只在提取阶段开启
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
            max_memory=max_memory,  # 显式指定max_memory以强制使用多GPU
            trust_remote_code=True,
            attn_implementation="eager"  # 必须使用eager模式才能输出attention
        )
        
        # 注意：不要全局开启 output_attentions，这会导致 generate 阶段显存爆炸
        # 我们只在 compute_attention_based_attribution 中按需开启
        # if hasattr(self.model, 'config'):
        #     self.model.config.output_attentions = True
        # if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'config'):
        #     self.model.language_model.config.output_attentions = True
        
        self.model.eval()

        self.v_top_layer_extractor = VTopLayerExtractor(self.model, verbose=True)
        self.attention_hook_extractor = AttentionHookExtractor(self.model, verbose=True)
        
        # 检测是否使用多 GPU
        self.use_multi_gpu = hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None
        if self.use_multi_gpu:
            print(f"[INFO] 检测到多 GPU 模式，设备分布: {self.model.hf_device_map}")
        else:
            print(f"[INFO] 使用单 GPU 模式")
        
        # 获取模型的输入设备（第一层的设备）
        self._input_device = self._get_model_input_device()
    
    def _get_model_input_device(self) -> torch.device:
        """
        获取模型的输入设备（第一层的设备）。
        当使用 device_map="auto" 时，模型的第一层可能不在 cuda:0 上。
        """
        if not self.use_multi_gpu:
            return torch.device(self.device)
        
        # 多 GPU 模式：找到模型的第一层（通常是 embedding 层）
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
            if hasattr(self.model.language_model.model, 'embed_tokens'):
                embed_layer = self.model.language_model.model.embed_tokens
                if hasattr(embed_layer, 'weight'):
                    device = embed_layer.weight.device
                    print(f"[INFO] 检测到模型输入设备: {device}")
                    return device
        
        # 备用方案：获取第一个参数的设备
        first_param = next(self.model.parameters())
        device = first_param.device
        print(f"[INFO] 使用第一个参数的设备作为输入设备: {device}")
        return device

    # ----------------------------
    # 内部辅助：暂时开启/关闭 output_attentions
    # ----------------------------
    def _collect_attention_configs(self) -> List:
        """
        收集需要同步设置 output_attentions 的 config 对象（去重）。
        """
        configs = []
        candidates = [
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "model", None), "config", None),
            getattr(getattr(self.model, "language_model", None), "config", None),
            getattr(
                getattr(getattr(self.model, "model", None), "language_model", None),
                "config",
                None,
            ),
        ]
        seen = set()
        for cfg in candidates:
            if cfg is None or not hasattr(cfg, "output_attentions"):
                continue
            if id(cfg) in seen:
                continue
            seen.add(id(cfg))
            configs.append(cfg)
        return configs

    def _set_output_attentions_temporarily(self, enable: bool) -> List[Tuple[object, bool]]:
        """
        暂时修改所有相关 config 的 output_attentions，并返回原始状态以便恢复。
        """
        configs = self._collect_attention_configs()
        prev_flags = []
        for cfg in configs:
            prev_value = getattr(cfg, "output_attentions", False)
            prev_flags.append((cfg, prev_value))
            cfg.output_attentions = enable
        return prev_flags

    def _restore_output_attentions(self, prev_flags) -> None:
        for cfg, value in prev_flags:
            cfg.output_attentions = value
    
    def _empty_cache_all_gpus(self):
        """清理所有 GPU 的显存缓存"""
        if torch.cuda.is_available():
            if self.use_multi_gpu:
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @torch.no_grad()
    def teacher_generate_steps(self, image: Image.Image, question: str, T: int = 4, max_new_tokens: int = 4096) -> List[str]:
        """
        让教师模型生成完整回答（使用 LVR 格式，不强制使用 <step> 标签）
        
        Args:
            image: 输入图像
            question: 问题
            T: 步骤数量（保留用于兼容性，但在 LVR 格式中不强制使用）
            max_new_tokens: 最大生成token数
            
        Returns:
            List[str]: 包含完整回答的列表（单个元素）
        """
        prompt = build_teacher_generation_prompt(self.processor, question, force_T=T)
        inputs = self.processor(
            text=prompt, 
            images=image, 
            return_tensors="pt"
        )
        # 将输入放到模型的输入设备上
        inputs = {k: (v.to(self._input_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]
        
        # 限制max_new_tokens以避免显存溢出（32B模型建议不超过1024）
        # 对于32B模型，4096个token的KV cache会占用约40-50GB显存
        safe_max_new_tokens = min(max_new_tokens, 1024)
        if max_new_tokens > safe_max_new_tokens:
            print(f"[WARNING] max_new_tokens={max_new_tokens} 过大，已限制为 {safe_max_new_tokens} 以避免显存溢出")
        
        out = self.model.generate(
            **inputs,
            max_new_tokens=safe_max_new_tokens,
            do_sample=False,         
            temperature=0.01,        
            repetition_penalty=1.05,
            # 显存优化：使用更小的batch size和及时清理
            pad_token_id=self.tokenizer.eos_token_id,
            # 关键优化：生成阶段显式关闭 attention 输出，避免显存爆炸
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True, 
        )
        text = self.processor.decode(out[0][input_length:], skip_special_tokens=True)
        print("teacher generate response (LVR format):", text)
        
        # 生成后立即清理显存
        del inputs, out
        self._empty_cache_all_gpus()
        import gc
        gc.collect()
        
        # LVR 格式：直接返回完整生成文本，不解析 <step> 标签
        # 为了保持接口兼容性，仍然返回列表，但只包含一个元素
        return [text]

    def _encode_with_image(self, image: Image.Image, text: str) -> Dict[str, torch.Tensor]:
        """
        编码图像和文本为模型输入
        
        Args:
            image: 输入图像
            text: 输入文本
            
        Returns:
            Dict[str, torch.Tensor]: 模型输入字典
        """
        inputs = self.processor(
            text=text, 
            images=image, 
            return_tensors="pt"
        )
        # 将输入放到模型的输入设备上
        inputs = {k: (v.to(self._input_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        return inputs

    def _image_token_mask(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        识别输入序列中图像token的位置
        
        Args:
            input_ids: 输入token ID序列 (batch_size, seq_len)
            
        Returns:
            Optional[torch.Tensor]: 布尔掩码，True表示图像token位置 (seq_len,)
        """
        input_ids_flat = input_ids.squeeze(0)
        
        vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        
        try:
            # 找到 <|vision_start|> 和 <|vision_end|> 的索引
            start_idx = (input_ids_flat == vision_start_token_id).nonzero(as_tuple=True)[0][0].item()
            end_idx = (input_ids_flat == vision_end_token_id).nonzero(as_tuple=True)[0][0].item()

            # 创建一个布尔掩码，标记出实际的图像token
            mask = torch.zeros_like(input_ids_flat, dtype=torch.bool)
            mask[start_idx + 1:end_idx] = True
            return mask
        except IndexError:
            # 如果没有找到这些token，则返回None
            return None

    def _text_token_mask(self, image_token_mask: torch.Tensor) -> torch.Tensor:
        """
        获取文本token掩码（image_token_mask的逆）
        
        Args:
            image_token_mask: 图像token掩码 (seq_len,)
            
        Returns:
            torch.Tensor: 文本token掩码 (seq_len,)，True表示文本token位置
        """
        return ~image_token_mask

    @torch.no_grad()
    def compute_attention_based_attribution(
        self,
        inputs: Dict[str, torch.Tensor],
        image_token_mask: torch.Tensor,
        text_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算基于注意力权重的归因分布 S (使用 Hook 优化版)
        """
        # 如果没有提供文本token掩码，则从图像token掩码计算
        if text_token_mask is None:
            text_token_mask = self._text_token_mask(image_token_mask)
        
        # 使用 Hook 提取器
        # 这会自动注册 hooks -> 前向(截获并销毁attention) -> 移除 hooks -> 返回聚合结果
        # 显存占用极低 (O(1) 级别，只存当前层的矩阵)
        print("[INFO] 使用 Hook 机制流式提取 Attention...")
        prev_flags = self._set_output_attentions_temporarily(True)
        try:
            S = self.attention_hook_extractor.extract_optimized(
                inputs=inputs,
                image_token_mask=image_token_mask,
                text_token_mask=text_token_mask
            )
        finally:
            self._restore_output_attentions(prev_flags)
        
        return S

    def extract_pt_per_step(
        self,
        image: Image.Image,
        question: str,
        teacher_steps: List[str],
        topk: int = 8,
        v_top_layer_save_path: Optional[str] = None,
        enable_v_top_layer: bool = False,  # 注意力方法通常不需要V_top_layer
        image_path: Optional[str] = None,  # 原始图像路径（用于保存到轨迹中）
    ) -> Trajectory:
        """
        对每个步骤提取基于注意力权重的归因分布
        
        注意：在 LVR 格式下，teacher_steps 通常只包含一个元素（完整回答），
        因此 T=1，只会提取一次归因分布。
        
        Args:
            image: 输入图像
            question: 问题
            teacher_steps: 教师模型生成的步骤列表（LVR格式下通常为单元素列表）
            topk: 返回top-k个最重要的图像token索引
            v_top_layer_save_path: V_top_layer保存路径（可选）
            enable_v_top_layer: 是否启用V_top_layer捕获（通常不需要）
            
        Returns:
            Trajectory: 包含所有步骤归因分布的轨迹对象
        """
        T = len(teacher_steps)
        print(f"[INFO] 提取轨迹，共 {T} 个步骤（LVR格式）")
        traj = {"steps": []}
        
        # 添加内存监控
        def print_memory_usage(stage=""):
            if torch.cuda.is_available():
                if self.use_multi_gpu:
                    total_allocated = sum(
                        torch.cuda.memory_allocated(i) / 1024**3 
                        for i in range(torch.cuda.device_count())
                    )
                    total_reserved = sum(
                        torch.cuda.memory_reserved(i) / 1024**3 
                        for i in range(torch.cuda.device_count())
                    )
                    print(f"[Memory {stage}] Allocated: {total_allocated:.2f}GB (所有GPU), Reserved: {total_reserved:.2f}GB (所有GPU)")
                else:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"[Memory {stage}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        print_memory_usage("start")

        v_top_layer_path = None

        if enable_v_top_layer and v_top_layer_save_path is not None:
            print("[INFO] 开始抓取 V_top_layer（静态特征）...")

            prompt_initial = build_prompt_with_chat_template(
                self.processor, question, steps_prefix=[], force_T=T
            )
            inputs_initial = self._encode_with_image(image, prompt_initial)

            image_token_mask = self._image_token_mask(inputs_initial["input_ids"])
            if image_token_mask is None:
                print("[WARNING] 无法识别图像 token 位置，跳过 V_top_layer 提取")
            else:
                try:
                    v_top_layer = self.v_top_layer_extractor.capture(
                        inputs=inputs_initial,
                        image_token_mask=image_token_mask
                    )
                    print(f"[INFO] V_top_layer 抓取完成: shape={tuple(v_top_layer.shape)}, "
                          f"N_img={v_top_layer.shape[0]}, d={v_top_layer.shape[1]}")
                    v_top_layer_path = self.v_top_layer_extractor.save(
                        v_top_layer, v_top_layer_save_path
                    )
                    del v_top_layer
                except Exception as e:
                    print(f"[WARNING] V_top_layer 捕获失败，跳过: {str(e)}")

            del inputs_initial, prompt_initial
            self._empty_cache_all_gpus()
            gc.collect()
        elif not enable_v_top_layer:
            print("[INFO] V_top_layer 捕获已禁用")
        elif v_top_layer_save_path is None:
            print("[INFO] 未提供 V_top_layer 保存路径，跳过捕获")

        # 对每个步骤提取归因分布
        # 注意：在 LVR 格式下，通常 T=1，只会执行一次
        for t in range(1, T + 1):
            print(f"\n[INFO] 处理步骤 {t}/{T}")
            
            # 构造包含到第t步的完整提示
            # LVR格式：如果 t=1，steps_prefix=teacher_steps[:1]，即包含完整回答
            prompt_full = build_prompt_with_chat_template(
                self.processor, question, steps_prefix=teacher_steps[:t], force_T=T
            )
            
            # 编码输入
            inputs_full = self._encode_with_image(image, prompt_full)
            
            # 获取图像token掩码
            image_token_mask = self._image_token_mask(inputs_full["input_ids"])
            if image_token_mask is None:
                raise RuntimeError(f"步骤 {t}: 无法识别图像token位置")
            
            # 计算基于注意力权重的归因分布 S
            S = self.compute_attention_based_attribution(
                inputs=inputs_full,
                image_token_mask=image_token_mask
            )
            
            # S 是一个一维向量，长度为 N_img
            # 可以将其归一化为概率分布（可选）
            # 使用softmax进行归一化，温度参数可以控制分布的尖锐程度
            # 这里直接使用原始注意力强度，也可以选择归一化
            # 直接使用原始注意力强度，归一化到[0,1]范围
            p_t = (S - S.min()) / (S.max() - S.min() + 1e-10)
            
            # 或者直接使用原始注意力强度（归一化到[0,1]）
            # p_t = (S - S.min()) / (S.max() - S.min() + 1e-10)
            
            # 获取top-k索引
            topk_idx = torch.topk(p_t, k=min(topk, p_t.numel())).indices.tolist()
            
            # 保存步骤信息
            traj["steps"].append({
                "p_t": p_t.tolist(),
                "p_topk_idx": topk_idx,
                "tau_used": "attention",  # 标识使用注意力方法
                "method": "attention_based",
                "raw_attention_scores": S.tolist(),  # 保存原始注意力分数
            })
            
            print(f"[INFO] 步骤 {t}: 归因分布计算完成，N_img={len(p_t)}, "
                  f"最大注意力强度={S.max().item():.4f}, 平均注意力强度={S.mean().item():.4f}")
            
            # 清理内存
            del inputs_full, image_token_mask, S, p_t
            self._empty_cache_all_gpus()
            gc.collect()
            
            print_memory_usage(f"step_{t}")

        return Trajectory(
            steps=traj["steps"], 
            v_top_layer_path=v_top_layer_path,
            image_path=image_path,
            question=question
        )

# ----------------------------
# Example CLI
# ----------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="/root/autodl-tmp/ViLR/data/BLINK_output/test-00000-of-00001/test_Relative_Depth_1/124_1.jpg")
    parser.add_argument("--question", type=str, default="Which point is closer to the camera?")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--save", type=str, default="/root/autodl-tmp/ViLR/output/traj_1.json")
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--topk", type=int, default=8, help="返回top-k个最重要的图像token索引")
    parser.add_argument("--enable_v_top_layer", type=bool, default=False)
    parser.add_argument("--v_top_layer_save_path", type=str, default=None)
    args = parser.parse_args()

    # 设置内存优化环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    cache_dir = "../my_qwen_model"
    os.environ['HF_HOME'] = cache_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_model_directory = "/root/autodl-tmp/ViLR/training/checkpoints/single_stage_4lvr-1400steps_1/checkpoint-1000"
    
    extractor = AttentionBasedTrajectoryExtractor(
        local_model_directory, 
        device=device, 
        dtype=torch.bfloat16
    )

    img = Image.open(args.image).convert("RGB")

    # 1) 让教师模型生成T个步骤
    teacher_steps = extractor.teacher_generate_steps(img, args.question, T=args.steps)
    print("Teacher steps:\n", "\n".join(teacher_steps))
    
    # 2) 提取基于注意力权重的归因分布
    traj = extractor.extract_pt_per_step(
        img, args.question, teacher_steps,
        topk=args.topk,
        image_path=args.image  # 保存图像路径到轨迹中
    )

    # 3) 保存
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(traj.__dict__, f, ensure_ascii=False, indent=2)
    print(f"Saved trajectory to {args.save}")

if __name__ == "__main__":
    main()

