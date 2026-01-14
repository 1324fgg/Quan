# teacher_traj_extractor.py
# -*- coding: utf-8 -*-
"""
Multi-forward visual-trajectory extractor for Qwen2.5-VL-72B-Instruct on VSP Spatial Planning.
Method A: run a separate forward pass per step boundary, to avoid future-information leakage.

Usage (outline):
  python teacher_traj_extractor.py --image path/to/img.jpg --question "..." --steps 4 --save out.json

Requirements:
  pip install transformers accelerate pillow torch
  (and have enough GPU for Qwen2.5-VL-72B-Instruct; for prototyping, swap to a smaller Qwen-VL)
"""

from __future__ import annotations
import os, json, re, math, gc
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoProcessor,AutoModelForImageTextToText
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import build_prompt_with_chat_template, build_teacher_generation_prompt
# ----------------------------
# Utilities
# ----------------------------

# ---------- PROBE: auto-detect image-token module producing (B, N_img, d) ----------

def probe_image_token_module(model, processor, image, question, T=4, max_new_tokens=16):
    """
    Run one forward with global hooks to print modules whose outputs look like image tokens: (B, N_img, d).
    Returns a sorted list of candidates (name, shape). Use the last one (deepest) as target.
    """
    import torch
    import re
    model.eval()

    # 构造最短输入（无需先生成 <step>…）——只要能触发视觉分支
    prompt = f"Image:\n<image>\n\nQuestion:\n{question}\n"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    for k, v in inputs.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            inputs[k] = v.to(next(model.parameters()).device)

    candidates = []
    hooks = []

    def is_img_token_tensor(t: torch.Tensor):
        if not torch.is_tensor(t) or t.dim() != 3 or not torch.is_floating_point(t):
            return False
        B, N, D = t.shape
        if B != 1: return False
        if not (16 <= N <= 8192): return False
        if not (128 <= D <= 8192): return False
        return True

    def make_hook(name):
        def _hook(module, inp, out):
            def try_tensor(x):
                if torch.is_tensor(x) and is_img_token_tensor(x):
                    candidates.append((name, tuple(x.shape)))
            if torch.is_tensor(out):
                try_tensor(out)
            elif isinstance(out, (tuple, list)):
                for item in out:
                    try_tensor(item)
        return _hook

    # 在所有模块上挂只读 hook（开销只做一次）
    for name, m in model.named_modules():
        hooks.append(m.register_forward_hook(make_hook(name)))

    # 前向一次
    with torch.no_grad():
        _ = model(**{k: v.to(next(model.parameters()).device) if torch.is_tensor(v) else v for k, v in inputs.items()})

    # 移除 hook
    for h in hooks:
        h.remove()

    # 去重 & 排序：越深的模块名通常越靠后；同时按 N_img 从大到小看
    seen = set()
    uniq = []
    for name, shape in candidates:
        if (name, shape) not in seen:
            seen.add((name, shape))
            uniq.append((name, shape))

    # 简单排序：按名字长度（更深的路径）优先，再按 N_img
    uniq.sort(key=lambda x: (len(x[0]), x[1][1]), reverse=True)

    print("\n[PROBE] Candidate image-token modules (pick the top one as target):")
    for i, (n, shp) in enumerate(uniq[:20], 1):
        print(f"  {i:2d}. {n:60s}  shape={shp}")
    return uniq


STEP_PATTERN = re.compile(r"<step>\s*(.*?)\s*</step>", re.DOTALL | re.IGNORECASE)

def split_steps(text: str, T: int) -> List[str]:
    """解析T个<step>...</step>块从教师输出中，按出现顺序提取。"""
    matches = STEP_PATTERN.findall(text)
    if not matches:
        raise ValueError("No <step>...</step> format blocks found in teacher output.")
    
    # 提取所有匹配的内容，并重新包装为 <step>...</step> 格式
    steps = []
    for content in matches:
        steps.append(f"<step>{content}</step>")
    
    # 检查是否提取到足够数量的步骤
    if len(steps) < T:
        raise ValueError(f"Only found {len(steps)} steps, but need {T} steps.")
    
    # 如果提取的步骤多于所需，只返回前T个
    if len(steps) > T:
        print(f"[WARNING] Found {len(steps)} steps, but only need {T}. Returning first {T} steps.")
        steps = steps[:T]
    
    # 按顺序返回步骤
    return steps


def to_device(x, device):
    if isinstance(x, dict):
        return {k: (to_device(v, device)) for k, v in x.items()}
    return x.to(device) if hasattr(x, "to") else x

# ----------------------------
# Attention backend (concat)
# ----------------------------

class ConcatAttentionBackend:
    """
    Try to read attention from final transformer layers (self-attention)
    and aggregate attention paid from the last token to *image token positions*.

    This assumes the model concatenates image tokens into the same sequence.
    """

    def __init__(self, model, tokenizer, image_token_mask_getter=None):
        self.model = model
        self.tokenizer = tokenizer
        self.image_token_mask_getter = image_token_mask_getter
        self._attn_buffers = []  # collects attention per layer on forward

    def _hook_attn(self, module, input, output):
        # output.attn_probs or similar varies by impl; try common patterns
        if isinstance(output, tuple):
            # some HF modules return (hidden_states, attn_weights, ...)
            for item in output:
                if torch.is_tensor(item) and item.dim() >= 3:
                    self._attn_buffers.append(item.detach())  # (B, H, Q, K)
                    break
        elif hasattr(output, "attn_probs"):
            self._attn_buffers.append(output.attn_probs.detach())

    def _register_hooks(self):
        self._hooks = []
        for name, m in self.model.named_modules():
            # Heuristic: hook attention modules in the decoder
            if any(k in name.lower() for k in ["attn", "attention"]) and hasattr(m, "forward"):
                self._hooks.append(m.register_forward_hook(self._hook_attn))

    def _remove_hooks(self):
        for h in getattr(self, "_hooks", []):
            h.remove()
        self._hooks = []

    @torch.no_grad()
    def compute_pt(
        self,
        inputs: Dict[str, torch.Tensor],
        image_token_mask: torch.Tensor,
        target_pos: int,
        tau: float = 0.15,
        agg: str = "last2"  # aggregate last/last2 layers
    ) -> Optional[torch.Tensor]:
        """
        Run forward, gather attentions, and compute p_t over image token positions.
        Returns p_t as (N_img,) on CPU, or None if attention not available.
        """
        self._attn_buffers.clear()
        self._register_hooks()
        try:
            outputs = self.model(**inputs, output_attentions=True)
        finally:
            self._remove_hooks()

        if not self._attn_buffers:
            return None  # attention not exposed

        # Collect attn tensors: list of (B,H,Q,K). We need Q at target_pos.
        # Stack and aggregate over layers/heads, then restrict to image K positions.
        # NOTE: different shapes may occur; be defensive.
        attns = []
        for a in self._attn_buffers:
            if a.dim() == 4:  # (B,H,Q,K)
                attns.append(a)
        if not attns:
            return None

        A = torch.stack(attns, dim=0)  # (L,B,H,Q,K)
        A = A[:, 0]                    # (L,H,Q,K) assume batch=1
        if target_pos >= A.shape[2]:
            target_pos = A.shape[2]-1  # clamp

        # Select query row at target_pos
        A_q = A[:, :, target_pos, :]   # (L,H,K)

        # Aggregate layers/heads
        if agg == "last2" and A_q.shape[0] >= 2:
            A_agg = A_q[-2:].mean(dim=0)  # (H,K)
        else:
            A_agg = A_q[-1]               # (H,K)

        A_agg = A_agg.mean(dim=0)         # mean over heads -> (K,)

        # Mask to image token positions
        if image_token_mask is None or image_token_mask.sum() == 0:
            return None
        A_img = A_agg[image_token_mask.bool()]  # (N_img,)

        # Temperature softmax -> probability
        p = F.softmax(A_img / tau, dim=-1).detach().cpu()
        return p

# ----------------------------
# Gradient backend (fallback)
# ----------------------------

class GradBackend:
    """
    包裹 Qwen2.5-VL 的 get_image_features，使返回的视觉token成为叶子张量，
    该叶子既被模型后续使用，也被我们保存为 self._img_feat 以做梯度归因。
    """

    def __init__(self, model, verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self._img_feat = None
        self._last_layer_name = None
        self._wrapped = False
        self._orig_get_image_features = None  # 保存原函数
        # 检测是否使用多 GPU
        self.use_multi_gpu = hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None
    
    def _empty_cache_all_gpus(self):
        """清理所有 GPU 的显存缓存"""
        if torch.cuda.is_available():
            if self.use_multi_gpu:
                # 多 GPU 模式：清理所有 GPU
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
            else:
                # 单 GPU 模式：只清理当前设备
                torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ---------- 包裹 / 恢复 ----------

    def _enable_wrap(self):
        if self._wrapped:
            return
        core = getattr(self.model, "model", None)
        if core is None or not hasattr(core, "get_image_features"):
            raise RuntimeError("model.model.get_image_features 未找到")
        self._orig_get_image_features = core.get_image_features

        def wrapped_get_image_features(*args, **kwargs):
            out = self._orig_get_image_features(*args, **kwargs)

            if isinstance(out, tuple):
                # Qwen3-VL-MoE 返回两个值: (image_embeds, deepstack_image_embeds)
                if len(out) == 2:
                    # 处理第一个返回值 image_embeds
                    if torch.is_tensor(out[0]):
                        t = out[0]                                 # t: (49, 3584)（或 (B,N,D)）
                    elif isinstance(out[0], (list, tuple)) and torch.is_tensor(out[0][0]):
                        t = out[0][0]                                 # t: (64,2048)
                    else:
                        raise ValueError(f"[DEBUG] get_image_features returned non-tensor data: {type(out[0])}")

                    leaf = t.clone().detach().requires_grad_(True) # ★ 生成叶子；形状/类型不变
                    # 我们自己用 (1,N,D) 视图做归因（注意：只保存视图，不返回视图）
                    self._img_feat = leaf
                    self._last_layer_name = "model.get_image_features[0]"
                    try:
                        leaf.retain_grad()
                    except Exception:
                        pass
                    
                    # 保持原始数据结构：如果原来是 (list(tensor), list(tensor))，就返回 (list(leaf), list(tensor))
                    if isinstance(out[0], (list, tuple)):
                        # 原来是 list/tuple，保持 list/tuple 结构
                        modified_first = list(out[0])
                        modified_first[0] = leaf
                        out = (modified_first, out[1])
                    else:
                        # 原来是 tensor，直接替换
                        out = (leaf, out[1])
                    
                    return out
                elif len(out) == 1:
                    # 兼容只有一个返回值的情况（Qwen2.5-VL）
                    if torch.is_tensor(out[0]):
                        t = out[0]                                 # t: (49, 3584)（或 (B,N,D)）
                    elif torch.is_tensor(out[0][0]):
                        t = out[0][0]                                 # t: (64,2048)
                    else:
                        raise ValueError(f"[DEBUG] get_image_features returned non-tensor data: {type(out[0])}")

                    leaf = t.clone().detach().requires_grad_(True) # ★ 生成叶子；形状/类型不变
                    # 我们自己用 (1,N,D) 视图做归因（注意：只保存视图，不返回视图）
                    self._img_feat = leaf
                    self._last_layer_name = "model.get_image_features[0]"
                    try:
                        leaf.retain_grad()
                    except Exception:
                        pass
                    # 返回值必须保持原容器和原形状（tuple((49,3584)/(64,2048),)）
                    out = (leaf,)
                    print(f"[DEBUG] Visual feature tensor detected: {leaf.shape}, dtype={leaf.dtype}")
                    print(f"[DEBUG] Visual feature device: {leaf.device}")
                    # if self.verbose:
                    #     print(f"[GradBackend] wrap: tuple[0] -> leaf {tuple(leaf.shape)} {leaf.dtype} dev={leaf.device}")
                    return out
                else:
                    raise ValueError(f"[DEBUG] Unexpected tuple length: {len(out)}")

            # === 仅诊断打印开始 ===
            def _print_tensor(tag, t):
                print(f"[GF dbg] {tag}: shape={tuple(t.shape)} dtype={t.dtype} "
                    f"device={t.device} dim={t.dim()} req={t.requires_grad} "
                    f"leaf={t.is_leaf} grad_fn={type(t.grad_fn).__name__ if t.grad_fn else None}")

            print("[GF dbg] get_image_features type:", type(out).__name__)
            if torch.is_tensor(out):
                _print_tensor("out", out)
            elif isinstance(out, (tuple, list)):
                print("[GF dbg] len:", len(out))
                for i, it in enumerate(out):
                    if torch.is_tensor(it):
                        _print_tensor(f"out[{i}]", it)
                    else:
                        print(f"[GF dbg] out[{i}]:", type(it).__name__)
            elif isinstance(out, dict):
                print("[GF dbg] keys:", list(out.keys()))
                for k, v in out.items():
                    if torch.is_tensor(v):
                        _print_tensor(f"out['{k}']", v)
                    else:
                        print(f"[GF dbg] out['{k}']:", type(v).__name__)
            else:
                print("[GF dbg] (unrecognized container)")

            return out
        core.get_image_features = wrapped_get_image_features
        self._wrapped = True

    def _disable_wrap(self):
        """前向结束后恢复原函数，避免影响其他调用（如 teacher generate）。"""
        if not self._wrapped:
            return
        core = getattr(self.model, "model", None)
        if core is not None and self._orig_get_image_features is not None:
            core.get_image_features = self._orig_get_image_features
        self._orig_get_image_features = None
        self._wrapped = False

    def _clear(self):
        self._img_feat = None
        self._last_layer_name = None

    # ---------- 主入口：计算单步 p_t ----------

    def compute_pt(
        self,
        inputs: Dict[str, torch.Tensor],
        target_pos: int,
        tau: float = 0.05,
        target_id: Optional[int] = None,
        return_metadata: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        1) 启用函数包裹，前向一次，抓 self._img_feat = (B=1, N_img, D)
        2) 构造 target_pos 的 next-token NLL
        3) 对 self._img_feat 求梯度，聚合成 (N_img,) 概率 p_t
        
        Returns:
            p: 归因概率分布 (N_img,)
            如果 return_metadata=True，同时返回包含归因强度的元数据字典
        """
        self._clear()
        # 不要对 input_ids / attention_mask 开梯度；只对浮点输入开梯度（通常没必要）
        for k, v in inputs.items():
            if torch.is_tensor(v) and torch.is_floating_point(v):
                v.requires_grad_(True)

        # ★ 启用包裹
        self._enable_wrap()
        try:
            # import pdb; pdb.set_trace()
            outputs = self.model(**inputs, output_hidden_states=False, output_attentions=False)
        finally:
            # ★ 恢复原函数，避免影响其他前向（例如 teacher_generate_steps）
            self._disable_wrap()

        logits = outputs.logits.float()  # 用 fp32 算 NLL 更稳
        B, S, V = logits.shape
        assert B == 1, "当前实现假定 batch=1"
        target_pos = min(target_pos, S - 1)

        if self._img_feat is None:
            raise RuntimeError("get_image_features 未返回可用视觉token；请检查权重与processor输入。")

        if self.verbose:
            print(f"[GradBackend] Hooked image tokens at: {self._last_layer_name}, feat shape={tuple(self._img_feat.shape)}, "
                  f"requires_grad={self._img_feat.requires_grad}, grad_fn={type(self._img_feat.grad_fn).__name__ if self._img_feat.grad_fn else None}")

        # 局部 NLL（gold token）
        next_logits = logits[0, target_pos]                      # (V,)
        if target_id is None:
            target_id = int(next_logits.argmax().item())         # 自一致：走模型自己最可能的下一个
        nll = -F.log_softmax(next_logits, dim=-1)[target_id]
        # 对 (1, N_img, D) 求梯度
        grads = torch.autograd.grad(nll, self._img_feat, retain_graph=False, allow_unused=False)[0]  # (1, N_img, D)

        # 按维度聚合 -> (N_img,) -> Softmax(τ)
        g = grads.abs().mean(dim=-1).squeeze(0)  # (N_img,)
        # --- 加入下面这几行 ---
        print(f"--- 原始归因分数 (g) ---")
        print(f"最小值: {g.min().item():.6f}, 最大值: {g.max().item():.6f}, 标准差: {g.std().item():.6f}")
        # print(g.tolist()) # 可选：取消注释以查看所有值

        p = F.softmax(g / tau, dim=-1).detach().cpu()
        
        # 计算元数据（用于智能聚合）
        metadata = {}
        if return_metadata:
            g_cpu = g.detach().cpu()
            # 归因强度：最大归因值、平均归因值、归因总和
            metadata['attribution_max'] = g_cpu.max().item()
            metadata['attribution_mean'] = g_cpu.mean().item()
            metadata['attribution_sum'] = g_cpu.sum().item()
            # 归因分布的熵（熵越高，说明依赖越分散但可能更真实）
            p_for_entropy = p + 1e-10  # 避免log(0)
            entropy = -(p_for_entropy * torch.log(p_for_entropy)).sum().item()
            metadata['entropy'] = entropy
            # 最大归因值的归一化（归一化后的最大归因值，反映依赖的集中程度）
            metadata['max_normalized'] = g_cpu.max().item() / (g_cpu.sum().item() + 1e-10)

        # 更积极的内存清理
        self._clear()
        del outputs, logits, next_logits, nll, grads, g
        self._empty_cache_all_gpus()
        gc.collect()
        
        if return_metadata:
            return p, metadata
        return p


# ----------------------------
# Top Layer Backend (for V_top_layer extraction)
# ----------------------------

from v_top_layer_extractor import VTopLayerExtractor

# ----------------------------
# Main extractor
# ----------------------------

@dataclass
class Trajectory:
    steps: List[Dict]  # each: { "p_t": list[float], "p_topk_idx": list[int], "tau": float, ... }
    v_top_layer_path: Optional[str] = None  # 指向 V_top_layer 二进制文件的路径（.npy 或 .pt）

class TeacherTrajectoryExtractor:

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16, tau_grad: float = 0.01):
    # 注意：__init__ 的參數已經修改，移除了 cache_dir
        self.device = device
        # 直接從您的本地路徑加載 processor 和 model
        # 不再需要 cache_dir 參數
        self.processor = AutoProcessor.from_pretrained(
            model_path,  # <--- 修改點 1
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

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,  # <--- 修改點 2
            dtype=dtype,
            device_map="auto",
            max_memory=max_memory,  # 显式指定max_memory以强制使用多GPU
            trust_remote_code=True
        )
        self.model.eval()
        
        # 检测是否使用多 GPU（device_map="auto" 时）
        self.use_multi_gpu = hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None
        if self.use_multi_gpu:
            print(f"[INFO] 检测到多 GPU 模式，设备分布: {self.model.hf_device_map}")
        else:
            print(f"[INFO] 使用单 GPU 模式")
        
        # 获取模型的输入设备（第一层的设备）
        self._input_device = self._get_model_input_device()

        # 针对 MoE 模型的内存优化
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'config'):
            config = self.model.language_model.config
            if hasattr(config, 'num_experts') and config.num_experts > 1:
                print(f"[INFO] 检测到 MoE 模型，专家数量: {config.num_experts}")
                # 设置更保守的内存管理
                self._empty_cache_all_gpus()
                # 设置 PyTorch 内存分配策略
                os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

        # Backends
        self.attn_backend = ConcatAttentionBackend(self.model, self.tokenizer)
        self.grad_backend = GradBackend(self.model, verbose=True)
        self.v_top_layer_extractor = VTopLayerExtractor(self.model, verbose=True)
    
    def _get_model_input_device(self) -> torch.device:
        """
        获取模型的输入设备（第一层的设备）。
        当使用 device_map="auto" 时，模型的第一层可能不在 cuda:0 上。
        """
        if not self.use_multi_gpu:
            # 单 GPU 模式，返回指定的设备
            return torch.device(self.device)
        
        # 多 GPU 模式：找到模型的第一层（通常是 embedding 层）
        # 对于 Qwen2.5-VL，通常是 model.language_model.model.embed_tokens
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
    
    def _empty_cache_all_gpus(self):
        """清理所有 GPU 的显存缓存"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @torch.no_grad()
    def teacher_generate_steps(self, image: Image.Image, question: str, T: int = 4, max_new_tokens: int = 4096) -> List[str]:
        prompt = build_teacher_generation_prompt(self.processor, question, T)
        # processor类似以前的tokenizer，但是更强大，能处理多模态信息
        inputs = self.processor(
            text=prompt, 
            images=image, 
            return_tensors="pt"
        )
        # 多 GPU 模式：将输入放到模型的输入设备上（第一层的设备）
        # 单 GPU 模式：使用指定的设备
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
            # top_p=0.9,              
            repetition_penalty=1.05,
            # 显存优化：使用更小的batch size和及时清理
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.processor.decode(out[0][input_length:], skip_special_tokens=True)
        print("teacher generate steps:", text)
        steps = split_steps(text, T)
        
        # 生成后立即清理显存
        del inputs, out
        self._empty_cache_all_gpus()
        import gc
        gc.collect()
        
        return steps


    def _encode_with_image(self, image: Image.Image, text: str) -> Dict[str, torch.Tensor]:
        inputs = self.processor(
            text=text, 
            images=image, 
            return_tensors="pt"
        )
        # 多 GPU 模式：将输入放到模型的输入设备上（第一层的设备）
        # 单 GPU 模式：使用指定的设备
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
        im_start_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        try:
            # 找到 <|vision_start|> 和 <|vision_end|> 的索引
            start_idx = (input_ids_flat == vision_start_token_id).nonzero(as_tuple=True)[0][0].item()
            end_idx = (input_ids_flat == vision_end_token_id).nonzero(as_tuple=True)[0][0].item()

            # 创建一个布尔掩码，标记出实际的图像 token
            mask = torch.zeros_like(input_ids_flat, dtype=torch.bool)
            mask[start_idx + 1:end_idx] = True
            return mask
        except IndexError:
            # 如果没有找到这些 token，则返回 None
            return None

    def extract_pt_per_step(
        self,
        image: Image.Image,
        question: str,
        teacher_steps: List[str],
        tau_grad: float,
        tau_attn: float = 0.15,      # 未用到；保留以兼容接口 
        topk: int = 8,
        skip_markup: bool = True,    # 跳过诸如 "<stepX>" 这类包含 <> 的标记 token
        downsample: int = 2,         # 对很长段可设为 2/3 做抽样，增加默认值以减少计算量
        max_tokens_per_step: int = 50,  # 限制每步处理的token数量，避免内存溢出
        max_grad_calls_per_step: int = 10,  # 限制每步的梯度计算次数，这是真正的内存控制参数
        aggregation_method: str = "weighted_by_strength",  # 聚合策略: "simple_avg", "weighted_by_strength", "filter_by_threshold", "entropy_weighted"
        attribution_threshold: float = 0.1,  # 用于 filter_by_threshold：归一化后的最小归因强度阈值
        min_entropy: float = 0.5,  # 用于过滤：最小熵阈值（低于此值的token被认为依赖不够明显）
        v_top_layer_save_path: Optional[str] = None,  # V_top_layer 的保存路径（.npy 或 .pt 文件）
        enable_v_top_layer: bool = True,  # 是否启用 V_top_layer 捕获
    ) -> Trajectory:
        """
        对每一步"整段文本"做归因：对段内每个 token（可选跳过标记/抽样）分别计算图像归因分布 p_j，
        使用智能聚合策略得到该步的热图 p_t。

        聚合策略选项:
        - "simple_avg": 简单平均（原始方法）
        - "weighted_by_strength": 基于归因强度加权平均（推荐）
        - "filter_by_threshold": 过滤低依赖token后平均
        - "entropy_weighted": 基于熵加权（熵越高权重越大）
        
        需要 GradBackend.compute_pt 支持:
            compute_pt(inputs, target_pos, tau, target_id=None, return_metadata=True)
        其中 target_pos 的 logits 用来预测 "下一个 token"（即 target_id）。
        """
        import gc
        T = len(teacher_steps)
        traj = {"steps": []}
        
        # 添加内存监控（支持多 GPU）
        def print_memory_usage(stage=""):
            if torch.cuda.is_available():
                if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None:
                    # 多 GPU 模式：显示所有 GPU 的显存使用
                    total_allocated = sum(
                        torch.cuda.memory_allocated(i) / 1024**3 
                        for i in range(torch.cuda.device_count())
                    )
                    total_reserved = sum(
                        torch.cuda.memory_reserved(i) / 1024**3 
                        for i in range(torch.cuda.device_count())
                    )
                    print(f"[Memory {stage}] Allocated: {total_allocated:.2f}GB (所有GPU), Reserved: {total_reserved:.2f}GB (所有GPU)")
                    # 可选：显示每个 GPU 的详细信息
                    for i in range(torch.cuda.device_count()):
                        alloc = torch.cuda.memory_allocated(i) / 1024**3
                        reserv = torch.cuda.memory_reserved(i) / 1024**3
                        print(f"  GPU {i}: Allocated={alloc:.2f}GB, Reserved={reserv:.2f}GB")
                else:
                    # 单 GPU 模式
                    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    print(f"[Memory {stage}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        print_memory_usage("start")

        # ========== 步骤 0: 抓取静态的 V_top_layer（可选）==========
        # 在处理完 Image+Question 后、生成第一个答案 token 之前，抓取一次 V_top_layer
        # 这个矩阵在所有后续 step 中都是静态的（不会改变）
        v_top_layer_path = None
        
        if enable_v_top_layer and v_top_layer_save_path is not None:
            print("[INFO] 开始抓取 V_top_layer（静态特征）...")

            prompt_initial = build_prompt_with_chat_template(
                self.processor, question, [], force_T=T
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
                    print(f"[INFO] V_top_layer 将在所有 {T} 个 step 中复用（静态特征）")

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
        
        print_memory_usage("after_v_top_layer_extraction")

        # ========== 步骤 1-T: 对每个 step 提取动态的 p_t ==========
        for t in range(1, T + 1):
            # 1) 构造"到 t-1 步的前缀"和"包含完整第 t 步"的提示
            prompt_prefix = build_prompt_with_chat_template(
                self.processor, question, teacher_steps[:t-1], force_T=T
            )
            prompt_full = build_prompt_with_chat_template(
                self.processor, question, teacher_steps[:t],   force_T=T
            )

            # 2) 编码两份输入，用 prefix 的长度来界定第 t 步的 token 区间
            inputs_prefix = self._encode_with_image(image, prompt_prefix)
            inputs_full   = self._encode_with_image(image, prompt_full)

            ids_full = inputs_full["input_ids"][0]              # (S_full,)
            S_prefix = inputs_prefix["input_ids"].shape[1]
            S_full   = ids_full.shape[0]

            # 第 t 步在 token 序列中的闭开区间 [S_prefix, S_full)
            span_start, span_end = S_prefix, S_full

            # 3) 逐 token 归因并累加（使用智能聚合策略）
            token_attributions = []  # 存储 (p_j, metadata) 元组
            grad_calls_made = 0  # 跟踪实际的梯度计算次数
            
            # 限制处理的token数量，避免内存溢出
            tokens_to_process = min(span_end - span_start, max_tokens_per_step)
            print(f"[INFO] Step {t}: 处理 {tokens_to_process} 个 tokens (原始: {span_end - span_start})")

            for j in range(span_start, span_start + tokens_to_process):
                # 限制梯度计算次数，这是真正的内存控制
                if grad_calls_made >= max_grad_calls_per_step:
                    print(f"[WARNING] Step {t}: 达到最大梯度计算次数限制 ({max_grad_calls_per_step})，停止处理")
                    break
                # 抽样：downsample>1 时只取一部分 token 以降计算量
                if downsample > 1 and ((j - span_start) % downsample != 0):
                    continue

                # 可选：跳过可能属于标签/标点的 token（含 '<' 或 '>'）
                if skip_markup:
                    tok_str = self.tokenizer.decode(ids_full[j:j+1], skip_special_tokens=False)
                    if ("<" in tok_str) or (">" in tok_str):
                        continue

                prev_pos = j - 1                           # 用 logits[prev_pos] 预测 ids_full[j]
                target_id = int(ids_full[j].item())        # 显式指定"下一个 token"= 当前 token

                # ★ 单 token 归因（一次前向）：对图像 token 产生 p_j ∈ R^{N_img}
                # 在每次梯度计算前清理内存
                # 清理所有 GPU 的显存（如果是多 GPU 模式）
                if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None:
                    # 多 GPU 模式：清理所有 GPU
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                else:
                    # 单 GPU 模式：只清理当前设备
                    torch.cuda.empty_cache()
                gc.collect()
                
                # 添加内存检查，如果内存不足就跳过（检查所有 GPU 的总显存）
                if torch.cuda.is_available():
                    if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None:
                        # 多 GPU 模式：计算所有 GPU 的总显存
                        total_allocated = sum(
                            torch.cuda.memory_allocated(i) / 1024**3 
                            for i in range(torch.cuda.device_count())
                        )  # GB
                        if total_allocated > 90:  # 如果所有 GPU 总显存已使用超过90GB，跳过
                            print(f"[WARNING] 总显存使用过高 ({total_allocated:.1f}GB)，跳过token {j}")
                            continue
                    else:
                        # 单 GPU 模式
                        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                        if allocated > 90:  # 如果已使用超过90GB内存，跳过
                            print(f"[WARNING] 内存使用过高 ({allocated:.1f}GB)，跳过token {j}")
                            continue
                
                p_j, metadata_j = self.grad_backend.compute_pt(
                    inputs=inputs_full,
                    target_pos=prev_pos,
                    tau=tau_grad,
                    target_id=target_id,
                    return_metadata=True
                )

                token_attributions.append((p_j, metadata_j, j))
                grad_calls_made += 1
                
                print(f"[INFO] Step {t}: 完成梯度计算 {grad_calls_made}/{max_grad_calls_per_step}, "
                      f"归因强度={metadata_j['attribution_max']:.4f}, 熵={metadata_j['entropy']:.4f}")

            # 4) 若因为过滤导致一个都没用上，回退到"最后一个 token"
            if len(token_attributions) == 0:
                # 选择最后一 token
                j = span_end - 1
                prev_pos  = j - 1
                target_id = int(ids_full[j].item())
                
                # 回退情况下也清理内存
                # 清理所有 GPU 的显存（如果是多 GPU 模式）
                if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None:
                    # 多 GPU 模式：清理所有 GPU
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                else:
                    # 单 GPU 模式：只清理当前设备
                    torch.cuda.empty_cache()
                gc.collect()
                
                p_j, metadata_j = self.grad_backend.compute_pt(
                    inputs=inputs_full,
                    target_pos=prev_pos,
                    tau=tau_grad,
                    target_id=target_id,
                    return_metadata=True
                )
                token_attributions.append((p_j, metadata_j, j))

            # 5) 根据聚合策略计算最终热图 p_t
            p_t = self._aggregate_attributions(
                token_attributions,
                method=aggregation_method,
                threshold=attribution_threshold,
                min_entropy=min_entropy
            )

            # 6) 诊断/可视化辅助信息
            topk_idx = torch.topk(p_t, k=min(topk, p_t.numel())).indices.tolist()
            
            # 注意：v_top_layer 不再保存在每个 step 中，而是保存在单独的二进制文件中
            # 路径信息会保存在 Trajectory 的根目录下
            
            traj["steps"].append({
                "p_t": p_t.tolist(),
                "p_topk_idx": topk_idx,
                "tau_used": float(tau_grad),
                "method": f"grad_{aggregation_method}",
                "span_token_count": len(token_attributions),
                "downsample": int(downsample),
                "skip_markup": bool(skip_markup),
                "aggregation_method": aggregation_method,
            })

            # 7) 更积极的内存清理
            del inputs_prefix, inputs_full, token_attributions, p_t
            if 'inputs' in locals():
                del inputs
            # 清理所有 GPU 的显存（如果是多 GPU 模式）
            if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None:
                # 多 GPU 模式：清理所有 GPU
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
            else:
                # 单 GPU 模式：只清理当前设备
                torch.cuda.empty_cache()
            gc.collect()
            
            # 强制同步，确保清理完成
            torch.cuda.synchronize()
            
            # 监控内存使用
            print_memory_usage(f"step_{t}")

        return Trajectory(steps=traj["steps"], v_top_layer_path=v_top_layer_path)

    def _aggregate_attributions(
        self,
        token_attributions: List[Tuple[torch.Tensor, Dict, int]],
        method: str = "weighted_by_strength",
        threshold: float = 0.1,
        min_entropy: float = 0.5,
    ) -> torch.Tensor:
        """
        智能聚合多个token的归因分布。
        
        Args:
            token_attributions: List of (p_j, metadata_j, token_idx) tuples
            method: 聚合策略
            threshold: 用于 filter_by_threshold 的阈值
            min_entropy: 最小熵阈值
            
        Returns:
            p_t: 聚合后的归因分布 (N_img,)
        """
        if method == "simple_avg":
            # 简单平均（原始方法）
            P_sum = None
            for p_j, _, _ in token_attributions:
                P_sum = p_j if P_sum is None else (P_sum + p_j)
            return (P_sum / len(token_attributions)).cpu()
        
        elif method == "weighted_by_strength":
            # 基于归因强度加权：使用最大归因值作为权重
            weights = []
            p_list = []
            for p_j, metadata_j, _ in token_attributions:
                # 使用归一化后的最大归因值作为权重（反映依赖强度）
                weight = metadata_j['max_normalized']
                weights.append(weight)
                p_list.append(p_j)
            
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = weights / (weights.sum() + 1e-10)  # 归一化权重
            
            # 加权平均
            p_t = torch.zeros_like(p_list[0])
            for w, p_j in zip(weights, p_list):
                p_t += w * p_j
            return p_t.cpu()
        
        elif method == "filter_by_threshold":
            # 过滤低依赖token：只保留归因强度超过阈值的token
            filtered = []
            for p_j, metadata_j, token_idx in token_attributions:
                # 使用归一化后的最大归因值作为过滤标准
                if metadata_j['max_normalized'] >= threshold:
                    filtered.append(p_j)
                    print(f"[Filter] Token {token_idx}: 归因强度={metadata_j['max_normalized']:.4f} >= {threshold}, 保留")
                else:
                    print(f"[Filter] Token {token_idx}: 归因强度={metadata_j['max_normalized']:.4f} < {threshold}, 过滤")
            
            if len(filtered) == 0:
                print(f"[WARNING] 所有token都被过滤，使用原始平均")
                return self._aggregate_attributions(token_attributions, "simple_avg", threshold, min_entropy)
            
            # 对过滤后的token求平均
            P_sum = None
            for p_j in filtered:
                P_sum = p_j if P_sum is None else (P_sum + p_j)
            return (P_sum / len(filtered)).cpu()
        
        elif method == "entropy_weighted":
            # 基于熵加权：熵越高，说明依赖越明显（分布更集中或更有意义）
            weights = []
            p_list = []
            for p_j, metadata_j, _ in token_attributions:
                # 使用熵作为权重（熵越高权重越大）
                # 但也可以使用熵的softmax来避免极端值
                entropy = metadata_j['entropy']
                # 过滤掉熵太低的token（依赖不够明显）
                if entropy >= min_entropy:
                    weights.append(entropy)
                    p_list.append(p_j)
            
            if len(weights) == 0:
                print(f"[WARNING] 所有token的熵都低于{min_entropy}，使用原始平均")
                return self._aggregate_attributions(token_attributions, "simple_avg", threshold, min_entropy)
            
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = F.softmax(weights, dim=0)  # 使用softmax归一化，使权重更平滑
            
            # 加权平均
            p_t = torch.zeros_like(p_list[0])
            for w, p_j in zip(weights, p_list):
                p_t += w * p_j
            return p_t.cpu()
        
        else:
            raise ValueError(f"未知的聚合策略: {method}")

# ----------------------------
# Example CLI
# ----------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="./data/examples/images/level_3/316/map_3x3_step_0.png")
    parser.add_argument("--question", type=str, default="How can the character go to the gift and avoid obstacles?(e.g. water surface).")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--save", type=str, default="traj.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=200, help="每步最大处理token数量，用于控制内存使用")
    parser.add_argument("--downsample", type=int, default=1, help="token抽样间隔，减少计算量")
    parser.add_argument("--max_grad_calls", type=int, default=50, help="每步最大梯度计算次数，这是真正的内存控制参数")
    parser.add_argument("--tau_grad", type=float, default=0.01, help="梯度归因的温度参数")
    parser.add_argument("--aggregation_method", type=str, default="weighted_by_strength", 
                        choices=["simple_avg", "weighted_by_strength", "filter_by_threshold", "entropy_weighted"],
                        help="聚合策略: simple_avg(简单平均), weighted_by_strength(基于强度加权,推荐), filter_by_threshold(阈值过滤), entropy_weighted(熵加权)")
    parser.add_argument("--attribution_threshold", type=float, default=0.1, 
                        help="用于filter_by_threshold策略的归因强度阈值")
    parser.add_argument("--min_entropy", type=float, default=0.5, 
                        help="用于entropy_weighted策略的最小熵阈值")
    args = parser.parse_args()

    # 设置内存优化环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 用于调试内存问题
    
    cache_dir = "../my_qwen_model"
    os.environ['HF_HOME'] = cache_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # local_model_directory = "/root/autodl-tmp/my_qwen_model/qwen/Qwen3-VL-30B-A3B-Instruct"
    local_model_directory = "/root/autodl-tmp/my_qwen_model/Qwen/Qwen2.5-VL-32B-Instruct"
    extractor = TeacherTrajectoryExtractor(local_model_directory, device=device, dtype=torch.bfloat16)

    img = Image.open(args.image).convert("RGB")

    # 1) Have teacher produce exactly T steps
    teacher_steps = extractor.teacher_generate_steps(img, args.question, T=args.steps)
    print("Teacher steps:\n", "\n".join(teacher_steps))
    # 2) Extract p_t per step (Method A: multi-forward)
    traj = extractor.extract_pt_per_step(
        img, args.question, teacher_steps,
        max_tokens_per_step=args.max_tokens,
        downsample=args.downsample,
        max_grad_calls_per_step=args.max_grad_calls,
        tau_grad=args.tau_grad,
        aggregation_method=args.aggregation_method,
        attribution_threshold=args.attribution_threshold,
        min_entropy=args.min_entropy
    )

    # 3) Save
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(traj.__dict__, f, ensure_ascii=False, indent=2)
    print(f"Saved trajectory to {args.save}")

if __name__ == "__main__":
    main()
