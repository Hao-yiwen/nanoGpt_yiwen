# -*- coding: utf-8 -*-
"""
NanoGPT 模型定义

包含:
- NanoGPTConfig (HuggingFace 兼容配置)
- RotaryPositionalEmbedding (RoPE)
- CausalSelfAttention (GQA + KV Cache + Flash Attention)
- FeedForward
- Block
- GPTLanguageModel (继承自 PreTrainedModel)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


# ============================================================================
# NanoGPT 配置类 (HuggingFace 兼容)
# ============================================================================

class NanoGPTConfig(PretrainedConfig):
    """
    NanoGPT 模型配置类，继承自 HuggingFace PretrainedConfig。

    支持通过 config.json 保存和加载配置。
    """
    model_type = "nanogpt"

    def __init__(
        self,
        vocab_size: int = 10000,
        n_embed: int = 768,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        n_layers: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        # MoE 配置
        use_moe: bool = False,
        num_experts: int = 8,
        num_shared_experts: int = 1,
        moe_top_k: int = 1,
        moe_freq: int = 2,
        aux_loss_coef: float = 0.01,
        # YaRN 配置
        rope_scaling_type: str = 'none',  # 'none' | 'yarn'
        rope_scaling_factor: float = 1.0,
        yarn_beta_fast: int = 32,
        yarn_beta_slow: int = 1,
        yarn_original_max_seq_len: Optional[int] = None,
        # HF 兼容参数
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = True,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        self.num_hidden_layers = n_layers  # HF 兼容别名
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        # MoE 配置
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.moe_top_k = moe_top_k
        self.moe_freq = moe_freq
        self.aux_loss_coef = aux_loss_coef
        # YaRN 配置
        self.rope_scaling_type = rope_scaling_type
        self.rope_scaling_factor = rope_scaling_factor
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow
        self.yarn_original_max_seq_len = yarn_original_max_seq_len or max_seq_len


# ============================================================================
# RMSNorm
# ============================================================================

class RMSNorm(nn.Module):
    """RMSNorm: 比 LayerNorm 更快，效果相当"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 计算均方根
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 归一化并缩放
        x = x / rms
        return x * self.weight


# ============================================================================
# YaRN 旋转位置编码（兼容标准 RoPE）
# ============================================================================

class YaRNRotaryEmbedding(nn.Module):
    """
    YaRN (Yet another RoPE extensioN) 旋转位置编码

    支持两种模式:
    - scaling_type='none': 标准 RoPE
    - scaling_type='yarn': YaRN 动态位置编码扩展

    YaRN 核心思想:
    - 高频维度（短波长）: 保持原始频率，捕捉局部细节
    - 低频维度（长波长）: 应用插值缩放，支持更长序列
    - 中间维度: 平滑混合
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: int = 10000,
        scaling_type: str = 'none',
        scaling_factor: float = 1.0,
        original_max_seq_len: Optional[int] = None,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.original_max_seq_len = original_max_seq_len or max_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # 当前缓存的最大长度
        self.max_seq_len_cached = max_seq_len

        # 计算频率并缓存
        inv_freq = self._compute_inv_freq()
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # 预计算 cos/sin 缓存
        self._set_cos_sin_cache(max_seq_len)

    def _compute_inv_freq(self) -> torch.Tensor:
        """计算逆频率，根据 scaling_type 选择不同策略"""
        # 原始 RoPE 频率: 1 / base^(2i/d)
        inv_freq_original = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )

        if self.scaling_type == 'none' or self.scaling_factor <= 1.0:
            return inv_freq_original

        # YaRN: 基于波长的混合策略
        # 插值频率 (全拉伸)
        inv_freq_interpolated = inv_freq_original / self.scaling_factor

        # 计算波长: λ = 2π / freq
        wavelen = 2 * math.pi / inv_freq_original

        # 计算 ramp 混合比例
        # 波长越长(低频) -> ramp 越接近 1 -> 越需要插值
        # 波长越短(高频) -> ramp 越接近 0 -> 保持原始
        ramp = (wavelen / self.original_max_seq_len - self.beta_slow) / (
            self.beta_fast - self.beta_slow
        )
        ramp = torch.clamp(ramp, 0.0, 1.0)

        # 混合: (1-ramp)*原始 + ramp*插值
        inv_freq = (1 - ramp) * inv_freq_original + ramp * inv_freq_interpolated

        return inv_freq

    def _set_cos_sin_cache(self, seq_len: int):
        """预计算 cos/sin 缓存"""
        self.max_seq_len_cached = seq_len

        # 位置索引: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.inv_freq.device).float()

        # 频率矩阵: [seq_len, dim//2]
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)

        # 拼接为 [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)

        # 缓存 cos/sin
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取位置编码的 cos/sin

        Args:
            seq_len: 序列长度

        Returns:
            (cos, sin): 形状均为 [seq_len, dim]
        """
        # 动态扩展缓存（如果需要）
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


# 保持向后兼容的别名
RotaryPositionalEmbedding = YaRNRotaryEmbedding


# ============================================================================
# GQA 分组查询注意力 + Flash Attention
# ============================================================================

class CausalSelfAttention(nn.Module):
    """分组查询注意力 GQA（带 RoPE + Flash Attention）"""

    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        n_embd = config.n_embed
        n_head = config.n_heads
        n_kv_heads = config.n_kv_heads

        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_head
        self.head_dim = n_embd // n_head
        self.n_rep = n_head // self.n_kv_heads  # 每个 KV 头对应多少个 Q 头

        assert n_head % self.n_kv_heads == 0, "n_head 必须能被 n_kv_heads 整除"

        # GQA: Q 投影维度不变，K/V 投影维度按 n_kv_heads 缩小
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.resid_dropout = nn.Dropout(config.dropout)

        # YaRN RoPE 实例（支持动态位置编码扩展）
        self.rope = YaRNRotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            scaling_type=config.rope_scaling_type,
            scaling_factor=config.rope_scaling_factor,
            original_max_seq_len=config.yarn_original_max_seq_len,
            beta_fast=config.yarn_beta_fast,
            beta_slow=config.yarn_beta_slow,
        )

        # YaRN 温度缩放因子
        self.scaling_factor = config.rope_scaling_factor

        # Flash Attention 的 dropout 比率
        self.dropout = config.dropout

    def _repeat_kv(self, x, n_rep):
        """将 KV 头扩展以匹配 Q 头数量"""
        if n_rep == 1:
            return x
        B, n_kv_heads, T, head_dim = x.shape
        x = x[:, :, None, :, :].expand(B, n_kv_heads, n_rep, T, head_dim)
        return x.reshape(B, n_kv_heads * n_rep, T, head_dim)

    def forward(self, x, attention_mask=None, start_pos=0, kv_cache=None):
        """
        Args:
            x: 输入 (B, T, C)
            attention_mask: 注意力掩码 (B, S)，1 表示有效，0 表示忽略（padding）
                           S 是完整序列长度（包括 KV cache）
            start_pos: 当前位置索引（用于 RoPE 和 KV Cache）
            kv_cache: 缓存的 (k, v)，形状 (B, n_kv_heads, cache_len, head_dim)
        Returns:
            out: 输出
            new_kv_cache: 更新后的 (k, v) 缓存
        """
        B, T, C = x.shape

        # 分别计算 Q, K, V
        q = self.q_proj(x)  # (B, T, n_embd)
        k = self.k_proj(x)  # (B, T, n_kv_heads * head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_heads * head_dim)

        # 重塑为多头
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE（使用正确的位置索引）
        seq_len = start_pos + T
        cos, sin = self.rope(seq_len)
        cos = cos[start_pos:seq_len]
        sin = sin[start_pos:seq_len]
        q, k = self._apply_rope(q, k, cos, sin)

        # KV Cache 处理（兼容新版 DynamicCache）
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            if k_cache is not None and v_cache is not None:
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)

        # 保存新的缓存（未扩展的 K/V）
        new_kv_cache = (k, v)

        # 扩展 KV 头以匹配 Q 头数量
        k_expanded = self._repeat_kv(k, self.n_rep)
        v_expanded = self._repeat_kv(v, self.n_rep)

        # 构建注意力掩码
        S = k_expanded.shape[2]  # key 序列长度（可能包含 cache）
        attn_mask = None

        # 使用缓存时 Q 只有当前 token，不需要 causal mask
        use_causal = (kv_cache is None) and (T > 1)

        if attention_mask is not None:
            # attention_mask: (B, S) -> (B, 1, 1, S)
            # 1 表示有效，0 表示忽略
            # scaled_dot_product_attention 需要: True 表示屏蔽
            attn_mask = attention_mask[:, None, None, :S].to(dtype=x.dtype)
            attn_mask = (1.0 - attn_mask) * torch.finfo(x.dtype).min

            if use_causal:
                # 构建因果掩码并与 padding mask 合并
                causal_mask = torch.triu(
                    torch.ones(T, S, dtype=x.dtype, device=x.device),
                    diagonal=S - T + 1
                ) * torch.finfo(x.dtype).min
                causal_mask = causal_mask[None, None, :, :]  # (1, 1, T, S)
                attn_mask = attn_mask + causal_mask
                use_causal = False  # 已经手动处理了因果掩码

        # Flash Attention（PyTorch 2.0+）
        # YaRN 温度缩放: scale = 1 / sqrt(d_k * scaling_factor)
        # 当 scaling_factor > 1 时，降低注意力分数的方差，提升长序列效果
        attn_scale = 1.0 / math.sqrt(self.head_dim * self.scaling_factor)

        out = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal,
            scale=attn_scale,
        )

        # 合并多头: (B, n_head, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))

        return out, new_kv_cache

    def _apply_rope(self, q, k, cos, sin):
        """对多头 Q、K 应用 RoPE"""
        # q: (B, n_head, T, head_dim), k: (B, n_kv_heads, T, head_dim)
        # cos, sin: (T, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)  # (1, 1, T, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)

        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k


# ============================================================================
# 前馈网络
# ============================================================================

# class FeedForward(nn.Module):
#     """前馈网络：两层线性变换 + GELU 激活"""

#     def __init__(self, n_embd):
#         super().__init__()
#         self.c_fc = nn.Linear(n_embd, 4 * n_embd)
#         self.gelu = nn.GELU()
#         self.c_proj = nn.Linear(4 * n_embd, n_embd)
#         self.dropout = nn.Dropout(DROP_OUT)

#     def forward(self, x):
#         x = self.c_fc(x)
#         x = self.gelu(x)
#         x = self.c_proj(x)
#         x = self.dropout(x)
#         return x

# ============================================================================
# SwiGLU 前馈网络
# ============================================================================

class FeedForward(nn.Module):
    """前馈网络：SwiGLU 激活"""

    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        n_embd = config.n_embed
        # 隐藏层维度，用 8/3 保持参数量相近
        hidden_dim = int(8 / 3 * n_embd)

        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)  # 门控分支
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=False)  # 值分支
        self.w3 = nn.Linear(hidden_dim, n_embd, bias=False)  # 输出投影
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一步：门控分支，过 SiLU 激活
        gate = self.w1(x)
        gate = F.silu(gate)

        # 第二步：值分支，不过激活函数
        up = self.w2(x)

        # 第三步：门控相乘
        hidden = gate * up

        # 第四步：输出投影
        out = self.w3(hidden)
        out = self.dropout(out)

        return out


# ============================================================================
# MoE: Router 路由器
# ============================================================================

class Router(nn.Module):
    """Top-K 路由器，带负载均衡损失"""

    def __init__(self, n_embd, num_experts, moe_top_k):
        super().__init__()
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k
        self.gate = nn.Linear(n_embd, num_experts, bias=False)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape                                       # B=batch, T=seq_len, C=n_embd
        x_flat = x.view(-1, C)                                  # (B*T, C)

        # 计算路由分数
        logits = self.gate(x_flat)                              # (B*T, num_experts)
        probs = F.softmax(logits, dim=-1)                       # (B*T, num_experts)

        # Top-K 选择
        weights, indices = torch.topk(probs, self.moe_top_k, dim=-1)  # (B*T, top_k), (B*T, top_k)
        weights = weights / weights.sum(dim=-1, keepdim=True)   # (B*T, top_k) 归一化

        # 计算负载均衡损失
        # f_i: 每个专家被选中的 token 比例
        mask = F.one_hot(indices, self.num_experts).sum(dim=1)  # (B*T, num_experts)
        f = mask.float().mean(dim=0)                            # (num_experts,)
        # P_i: 每个专家的平均路由概率
        P = probs.mean(dim=0)                                   # (num_experts,)
        aux_loss = self.num_experts * (f * P).sum()             # scalar

        # 返回值:
        # weights: (B*T, top_k) 每个 token 选中的专家权重（归一化后）
        # indices: (B*T, top_k) 每个 token 选中的专家索引
        # aux_loss: scalar 负载均衡辅助损失
        return weights, indices, aux_loss


# ============================================================================
# MoE: MoELayer 混合专家层
# ============================================================================

class MoELayer(nn.Module):
    """
    Mixture of Experts 层 (带共享专家)

    架构:
    - 共享专家 (Shared Experts): 处理所有 token，学习通用知识
    - 路由专家 (Routed Experts): 根据 Router 选择性激活，学习专业知识

    输出 = 共享专家输出 + 路由专家加权输出
    """

    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        n_embd = config.n_embed
        num_experts = config.num_experts
        num_shared_experts = config.num_shared_experts
        moe_top_k = config.moe_top_k

        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.moe_top_k = moe_top_k

        # 路由专家 (Routed Experts)
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(num_experts)])
        self.router = Router(n_embd, num_experts, moe_top_k)

        # 共享专家 (Shared Experts) - 如果 num_shared_experts > 0
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config) for _ in range(num_shared_experts)
            ])
        else:
            self.shared_experts = None

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (B*T, C)

        # ========== 共享专家计算 ==========
        # 共享专家处理所有 token，不需要路由
        shared_output = torch.zeros_like(x_flat)  # (B*T, C)
        if self.shared_experts is not None:
            for shared_expert in self.shared_experts:
                shared_output += shared_expert(x_flat)  # 所有共享专家的输出相加

        # ========== 路由专家计算 ==========
        # 路由决定每个 token 去哪些专家
        weights, indices, aux_loss = self.router(x)  # weights/indices: (B*T, top_k)

        # 思路: 遍历每个专家，找到选择了该专家的 token，计算输出后加权累加
        #
        # 示例: 假设 top_k=2, num_experts=8, 有 4 个 token
        # indices = [[0, 3],    # token 0 选了专家 0 和 3
        #            [1, 0],    # token 1 选了专家 1 和 0
        #            [0, 2],    # token 2 选了专家 0 和 2
        #            [3, 1]]    # token 3 选了专家 3 和 1
        # weights = [[0.6, 0.4],
        #            [0.7, 0.3],
        #            [0.5, 0.5],
        #            [0.8, 0.2]]
        #
        # 当 i=0 (专家0) 时:
        #   mask = [True, True, True, False]  # token 0,1,2 选了专家0
        #   expert_weights = [0.6, 0.3, 0.5]  # 各 token 给专家0的权重

        routed_output = torch.zeros_like(x_flat)  # (B*T, C) 初始化路由专家输出
        for i, expert in enumerate(self.experts):
            # 找到选择了这个专家的 token (在 top_k 个选择中任意位置出现即可)
            mask = (indices == i).any(dim=-1)  # (B*T,) bool mask
            if mask.any():
                # 提取选中该专家的 token 输入
                expert_input = x_flat[mask]       # (n, C) n=选中该专家的token数
                expert_output = expert(expert_input)  # (n, C) 专家计算结果

                # 获取这个专家的权重
                # 1. 找到 indices 中等于 i 的位置，取对应的 weight，其他位置置 0
                expert_weights = torch.where(
                    indices == i,                 # (B*T, top_k) 条件
                    weights,                      # 满足条件取 weights
                    torch.zeros_like(weights)     # 不满足取 0
                )  # (B*T, top_k)
                # 2. 沿 top_k 维度求和（一个 token 最多在一个位置选中该专家）
                # 3. 只保留 mask 为 True 的行
                expert_weights = expert_weights.sum(dim=-1, keepdim=True)[mask]  # (n, 1)

                # 加权累加到输出（一个 token 可能选了多个专家，所以是累加）
                routed_output[mask] += expert_weights * expert_output  # (n, C)

        # ========== 合并输出 ==========
        # 最终输出 = 共享专家输出 + 路由专家输出
        output = shared_output + routed_output

        return output.view(B, T, C), aux_loss  # 恢复原始形状


# ============================================================================
# Transformer Block
# ============================================================================

class Block(nn.Module):
    """
    Transformer Block: 自注意力 + 前馈网络

    使用 Pre-LayerNorm 结构（先 LayerNorm 再计算）和残差连接
    支持 MoE（Mixture of Experts）
    """

    def __init__(self, config: NanoGPTConfig, layer_idx: int):
        super().__init__()
        # 根据 layer_idx 和 moe_freq 判断是否使用 MoE
        self.use_moe = config.use_moe and (layer_idx % config.moe_freq == 1)
        self.layer_idx = layer_idx

        self.sa = CausalSelfAttention(config)
        if self.use_moe:
            self.ff = MoELayer(config)
        else:
            self.ff = FeedForward(config)
        self.ln1 = RMSNorm(config.n_embed)
        self.ln2 = RMSNorm(config.n_embed)

    def forward(self, x, attention_mask=None, start_pos=0, kv_cache=None):
        # 残差连接：x = x + sublayer(LayerNorm(x))
        h, new_kv_cache = self.sa(self.ln1(x), attention_mask, start_pos, kv_cache)
        x = x + h
        if self.use_moe:
            ff_out, aux_loss = self.ff(self.ln2(x))
            x = x + ff_out
            return x, new_kv_cache, aux_loss
        else:
            x = x + self.ff(self.ln2(x))
            return x, new_kv_cache


# ============================================================================
# 主模型定义 (HuggingFace 兼容)
# ============================================================================

class GPTLanguageModel(PreTrainedModel, GenerationMixin):
    """
    GPT 风格的语言模型，继承自 HuggingFace PreTrainedModel 和 GenerationMixin

    特性：
    - GQA 分组查询注意力 + Flash Attention
    - KV Cache 推理加速
    - RoPE 旋转位置编码
    - GPT-2 风格权重初始化
    - 兼容 HuggingFace save_pretrained / from_pretrained
    - 兼容 HuggingFace generate()
    """
    config_class = NanoGPTConfig
    # base_model_prefix = "model"  # 移除无效的 prefix，因为是扁平结构
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: NanoGPTConfig):
        super().__init__(config)
        self.config = config

        # Token 嵌入 (使用 RoPE，不需要位置嵌入)
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)

        # Transformer 层（使用 GQA + KV Cache）
        self.blocks = nn.ModuleList([
            Block(config, i) for i in range(config.n_layers)
        ])
        self.ln_f = RMSNorm(config.n_embed)  # 最终的 RMSNorm

        # 输出头：将嵌入映射到词汇表大小
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # 权重共享：Embedding 与 LM Head 共用同一权重矩阵
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding_table.weight

        # 初始化权重
        self.post_init()

    def _init_weights(self, module):
        """GPT-2 风格的权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

        # 残差投影层特殊缩放（GPT-2 论文）
        if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight') and p is module.weight:
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def get_input_embeddings(self):
        return self.token_embedding_table

    def set_input_embeddings(self, value):
        self.token_embedding_table = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        HuggingFace 兼容的 forward 方法

        Args:
            input_ids: 输入 token ids (B, T)
            attention_mask: 注意力掩码 (B, S)，1 表示有效，0 表示忽略（padding）
                           S 是完整序列长度（包括 KV cache 中的历史 token）
            past_key_values: KV Cache，List[Tuple[k, v]]
            labels: 标签 token ids (B, T)，用于计算损失
            use_cache: 是否返回 KV Cache
            output_attentions: 是否返回注意力权重（暂不支持）
            output_hidden_states: 是否返回隐藏状态（暂不支持）
            return_dict: 是否返回 dict 格式

        Returns:
            CausalLMOutputWithPast 包含 loss, logits, past_key_values
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        B, T = input_ids.shape

        # 计算 start_pos（用于 RoPE 和 KV Cache）
        start_pos = 0
        if past_key_values is not None and len(past_key_values) > 0:
            first_layer = past_key_values[0]
            # 兼容新版 DynamicCache：检查 first_layer 是否有效
            if first_layer is not None and len(first_layer) > 0 and first_layer[0] is not None:
                start_pos = first_layer[0].shape[2]  # cache_len

        # 嵌入 (RoPE 在注意力层内部应用)
        tok_emb = self.token_embedding_table(input_ids)  # (B, T, n_embed)
        x = tok_emb

        # Transformer 处理（逐层传递 KV Cache）
        new_kv_caches = [] if use_cache else None
        total_aux_loss = 0.0

        for i, block in enumerate(self.blocks):
            # 安全获取 kv_cache（兼容 DynamicCache）
            kv_cache = None
            if past_key_values is not None and i < len(past_key_values):
                kv_cache = past_key_values[i]
            if block.use_moe:
                x, new_kv_cache, aux_loss = block(x, attention_mask, start_pos, kv_cache)
                total_aux_loss += aux_loss
            else:
                x, new_kv_cache = block(x, attention_mask, start_pos, kv_cache)

            if use_cache:
                new_kv_caches.append(new_kv_cache)

        x = self.ln_f(x)

        # 输出 logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 计算损失（如果提供 labels）
        loss = None
        if labels is not None:
            # Shift 以计算下一个 token 预测损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

            # 仅当使用 MoE 时添加辅助损失
            if self.config.use_moe and total_aux_loss > 0:
                loss = ce_loss + self.config.aux_loss_coef * total_aux_loss
            else:
                loss = ce_loss

        if not return_dict:
            output = (logits, new_kv_caches)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_kv_caches,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        为 HuggingFace generate() 准备输入

        当使用 KV Cache 时，只需要传入最后一个 token
        """
        # 如果有 past_key_values，只取最后一个 token
        if past_key_values is not None and len(past_key_values) > 0:
            first_layer = past_key_values[0]
            # 检查 first_layer 是否有效（兼容新版 DynamicCache）
            if first_layer is not None and len(first_layer) > 0 and first_layer[0] is not None:
                # 检查缓存长度，避免超过 max_seq_len
                cache_len = first_layer[0].shape[2]
                if cache_len >= self.config.max_seq_len:
                    # 截断缓存
                    past_key_values = [
                        (k[:, :, 1:, :], v[:, :, 1:, :]) for k, v in past_key_values
                    ]
                input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "attention_mask": attention_mask,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        为 beam search 重排 KV Cache
        """
        reordered_past = []
        for layer_past in past_key_values:
            if layer_past is None:
                reordered_past.append(None)
            else:
                reordered_past.append(
                    tuple(
                        past_state.index_select(0, beam_idx.to(past_state.device))
                        if past_state is not None else None
                        for past_state in layer_past
                    )
                )
        return reordered_past
