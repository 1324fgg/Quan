import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple

class AttentionHookExtractor:
    """
    使用 PyTorch Hook 机制逐层提取 Attention 权重，
    避免一次性在显存中保存所有层的 Attention 矩阵。
    """
    
    def __init__(self, model, verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self.hooks = []
        self.layer_outputs = {}  # 暂存每层的处理结果
        self._text_token_mask = None
        self._image_token_mask = None
        self._current_device = None
        
        # 自动探测 attention 模块的名称路径
        self.attn_module_name = self._detect_attn_module_name()
        if self.verbose:
            print(f"[INFO] Detected attention module name: {self.attn_module_name}")

    def _detect_attn_module_name(self) -> str:
        """
        探测模型中 Attention 模块的命名规则。
        Qwen2.5-VL 通常是 'self_attn'。
        """
        for name, module in self.model.named_modules():
            if "layers.0" in name and "self_attn" in name:
                # 返回去掉了 'layers.0.' 和前缀的部分，只保留层内的相对路径
                # 例如: model.layers.0.self_attn -> self_attn
                return "self_attn"
        return "self_attn"  # Default fallback

    def _hook_fn(self, module, args, output, layer_idx):
        """
        Hook 函数：截获 attention_weights。
        注意：这依赖于 eager 模式下 forward 返回 (attn_output, attn_weights, ...)
        """
        # output 通常是 (attn_output, attn_weights, past_key_value)
        # 如果 output_attentions=True，attn_weights 是第二个元素
        # 但即使 output_attentions=False，在 eager 模式内部计算时，attn_weights 也是存在的
        # 我们需要确保 attn_implementation="eager" 并且 output_attentions=True 才会返回 weights
        # 或者，我们可以 hook 模块内部的 softmax 输出来强制获取
        
        # 对于 Qwen2.5-VL (eager mode)，self_attn 的 forward 返回:
        # (attn_output, attn_weights) if output_attentions=True
        
        attn_weights = None
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]
        
        if attn_weights is None:
            if self.verbose:
                print(f"[WARNING] Layer {layer_idx}: No attention weights found in output.")
            return

        # 立即处理 attention weights，不保存原始大矩阵
        processed_data = self._process_attention_weights(attn_weights)
        
        # 保存处理后的轻量结果
        # 使用 CPU 存储以节省显存，或者如果显存够大也可以留在 GPU
        self.layer_outputs[layer_idx] = processed_data.detach().cpu()
        
        # 显式删除引用，虽然 hook 结束后局部变量会被销毁，但为了保险
        del attn_weights

    def _process_attention_weights(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        处理单层的 attention weights: (B, H, L, L)
        
        改进：提取**答案区域的 token** 对图像 token 的注意力。
        答案区域 = image_token 之后的所有 text token (即问题+答案部分)
        这比使用最后一个 token 更稳健。
        """
        # attn_weights: (B, H, L, L)
        # 假设 B=1
        
        device = attn_weights.device
        
        if self._image_token_mask.device != device:
            self._image_token_mask = self._image_token_mask.to(device)
        if self._text_token_mask.device != device:
            self._text_token_mask = self._text_token_mask.to(device)
            
        image_mask = self._image_token_mask
        text_mask = self._text_token_mask
        
        # 找到图像区域结束后的 text token 位置
        # image_mask 的最后一个 True 的位置之后就是答案区域
        image_positions = torch.where(image_mask)[0]
        if len(image_positions) == 0:
            # 没有图像 token，返回空
            return torch.zeros(image_mask.sum(), device='cpu')
        
        last_image_pos = image_positions[-1].item()
        seq_len = attn_weights.shape[2]
        
        # 答案区域：从图像结束后到序列末尾
        # 这包括问题后缀和模型的答案
        answer_start = last_image_pos + 1
        answer_end = seq_len
        
        # 提取答案区域对图像的注意力
        # attn_weights: (1, H, L, L)
        # 选择 query 位置在答案区域，key 位置在图像区域
        answer_to_image = attn_weights[0, :, answer_start:answer_end, :][:, :, image_mask]
        # shape: (H, num_answer_tokens, num_image)
        
        # 对 heads 和 answer_tokens 求平均
        if answer_to_image.shape[1] > 0:
            avg_attention = answer_to_image.mean(dim=(0, 1))  # (num_image,)
        else:
            # 如果没有答案 token，回退到最后一个 token
            last_token_attn = attn_weights[0, :, -1, :]
            avg_attention = last_token_attn[:, image_mask].mean(dim=0)
        
        return avg_attention

    def register_hooks(self):
        """注册 hooks 到每一层"""
        self.hooks = []
        self.layer_outputs = {}
        
        # 遍历所有模块，寻找 attention 模块
        # 策略：
        # 1. 排除 'visual' 模块 (我们只关心文本部分的 attention 对图像 token 的关注)
        # 2. 匹配名称中包含 'attn' 或 'attention' 的模块
        # 3. 确保该模块是 Transformer 层的一部分 (通常在 'layers' 或 'blocks' 下)
        
        layer_count = 0
        
        for name, module in self.model.named_modules():
            # 排除视觉编码器
            if "visual" in name:
                continue
                
            # 匹配 Attention 模块
            # Qwen2.5-VL: language_model.layers.0.self_attn
            if "attn" in name.lower() or "attention" in name.lower():
                # 简单的层索引提取逻辑
                # 假设结构类似 "...layers.N.self_attn..."
                parts = name.split('.')
                layer_idx = -1
                
                # 尝试从名字中提取层号
                for i, part in enumerate(parts):
                    if part == "layers" or part == "blocks":
                        if i + 1 < len(parts) and parts[i+1].isdigit():
                            layer_idx = int(parts[i+1])
                            break
                
                if layer_idx >= 0:
                    # 确保是 self-attention (不是 cross-attention 或其他)
                    # 对于 decoder-only 模型，通常只有 self_attn
                    # 我们这里全挂，反正后面如果不输出 attention weights (None)，hook_fn 会忽略
                    
                    # 避免重复挂载 (比如 self_attn 可能有子模块也叫 attn)
                    # 我们只挂载叶子节点或者标准的 attention 块
                    # Qwen2_5_VLAttention 是我们要找的
                    if "Qwen2_5_VLAttention" in module.__class__.__name__ or \
                       "LlamaAttention" in module.__class__.__name__ or \
                       "SelfAttention" in module.__class__.__name__ or \
                       name.endswith(self.attn_module_name):
                        
                        if self.verbose:
                            print(f"[DEBUG] Hooking layer {layer_idx}: {name} ({module.__class__.__name__})")
                            
                        hook = module.register_forward_hook(
                            lambda m, inp, out, idx=layer_idx: self._hook_fn(m, inp, out, idx)
                        )
                        self.hooks.append(hook)
                        layer_count += 1
        
        if layer_count == 0:
            print("[WARNING] No attention modules hooked! Check model structure.")
            print(f"[DEBUG] Model modules: {[n for n, _ in self.model.named_modules()][:20]}...")

    def remove_hooks(self):
        """移除所有 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def extract(self, inputs, image_token_mask, text_token_mask):
        """
        执行提取流程
        """
        self._image_token_mask = image_token_mask
        self._text_token_mask = text_token_mask
        
        # 1. 注册 hooks
        self.register_hooks()
        
        try:
            # 2. 执行前向传播
            self.model(**inputs, output_attentions=True, output_hidden_states=False)
            
        finally:
            # 3. 移除 hooks
            self.remove_hooks()
            
        # 4. 聚合结果
        if not self.layer_outputs:
            # 如果 hook 没抓到（可能是 flash attn 没返回 weights），尝试降级方案
            # 或者直接抛出错误
            print("[ERROR] No attention data captured. Ensure attn_implementation='eager'.")
            raise RuntimeError("No attention data captured via hooks.")
            
        # 将所有层的平均值再求平均
        stacked_avgs = torch.stack(list(self.layer_outputs.values()), dim=0) 
        final_s = stacked_avgs.mean(dim=0) 
        
        return final_s

    def _hook_fn_modify(self, module, args, output, layer_idx):
        """
        修改版 Hook：处理完 attention 后，从输出中将其删除，防止占用显存。
        """
        # output 是 tuple: (attn_output, attn_weights, ...)
        if not isinstance(output, tuple):
            return output
            
        attn_output = output[0]
        attn_weights = output[1] if len(output) > 1 else None
        
        if attn_weights is not None:
            # 1. 处理
            processed = self._process_attention_weights(attn_weights)
            self.layer_outputs[layer_idx] = processed.detach().cpu()
            
            # 2. 销毁 weights (返回一个新的 tuple，把 weights 设为 None)
            new_output = (attn_output, None) + output[2:]
            return new_output
        elif self.verbose and layer_idx < 2:
            print(f"[DEBUG] Layer {layer_idx}: hook 触发但没有拿到 attention weights (output_attentions 未开启？)")
            
        return output

    def register_modifying_hooks(self):
        """注册修改返回值的 hooks"""
        self.hooks = []
        self.layer_outputs = {}
        
        layer_count = 0
        
        for name, module in self.model.named_modules():
            # 排除视觉编码器
            if "visual" in name:
                continue
                
            # 匹配 Attention 模块
            if "attn" in name.lower() or "attention" in name.lower():
                # 提取层号
                parts = name.split('.')
                layer_idx = -1
                for i, part in enumerate(parts):
                    if part == "layers" or part == "blocks":
                        if i + 1 < len(parts) and parts[i+1].isdigit():
                            layer_idx = int(parts[i+1])
                            break
                
                if layer_idx >= 0:
                    # 确认模块类型
                    # 只要名字匹配 attn 且在 layers 下，基本就是我们要的
                    # 更加宽松的匹配，确保捕获
                    if name.endswith("self_attn") or name.endswith("attention"):
                        if self.verbose and layer_count < 2: # 只打印前两个
                            print(f"[DEBUG] Hooking (modifying) layer {layer_idx}: {name}")
                            
                        hook = module.register_forward_hook(
                            lambda m, inp, out, idx=layer_idx: self._hook_fn_modify(m, inp, out, idx)
                        )
                        self.hooks.append(hook)
                        layer_count += 1
                        
        if layer_count == 0:
            print("[WARNING] No attention modules hooked! Check model structure.")

    @torch.no_grad()
    def extract_optimized(self, inputs, image_token_mask, text_token_mask):
        """
        优化版提取：使用 modifying hooks 节省显存
        
        改进：只使用最后 N 层的注意力（这些层包含更高级的语义信息）
        """
        self._image_token_mask = image_token_mask
        self._text_token_mask = text_token_mask
        
        self.register_modifying_hooks()
        
        try:
            # 前向传播
            # 必须开启 output_attentions=True 才能让 eager 模式计算 attention weights
            # 但因为 hook 把 weights 替换成了 None，最终返回的 outputs.attentions 将全是 None
            # 显存占用 O(1)
            self.model(**inputs, output_attentions=True, output_hidden_states=False)
        finally:
            self.remove_hooks()
            
        if not self.layer_outputs:
            if self.verbose:
                print("[WARNING] Hook 未捕获到任何 attention tensor，返回全零向量")
            return torch.zeros(image_token_mask.sum(), device='cpu') # Fallback
        
        # 只使用最后 N 层的注意力 (这些层包含更高级的语义信息)
        # 对于 Qwen2.5-VL-32B，有 64 层，我们使用最后 8 层
        layer_indices = sorted(self.layer_outputs.keys())
        n_layers = len(layer_indices)
        n_use = min(8, n_layers)  # 使用最后 8 层，或全部层（如果少于 8 层）
        
        # 选择最后 n_use 层
        selected_indices = layer_indices[-n_use:]
        selected_tensors = [self.layer_outputs[idx] for idx in selected_indices]
        
        if self.verbose:
            print(f"[INFO] 使用最后 {n_use} 层 (layers {selected_indices[0]}-{selected_indices[-1]}) 的注意力")
            
        stacked_avgs = torch.stack(selected_tensors, dim=0)
        final_s = stacked_avgs.mean(dim=0)
        
        return final_s

