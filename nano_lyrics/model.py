# -*- coding: utf-8 -*-
"""
NanoGPT 模型定义

包含:
- RotaryPositionalEmbedding (RoPE)
- CausalSelfAttention (GQA + KV Cache + Flash Attention)
- FeedForward
- Block
- GPTLanguageModel
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from config import (N_EMBED, N_HEADS, N_KV_HEADS, N_LAYERS, DROP_OUT, MAX_SEQ_LEN,
                    USE_MOE, NUM_EXPERTS, NUM_SHARED_EXPERTS, TOP_K, AUX_LOSS_COEF)


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
# RoPE 旋转位置编码
# ============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """RoPE 旋转位置编码"""

    def __init__(self, dim, max_seq_len=MAX_SEQ_LEN, base=10000):
        super().__init__()
        # 计算频率 θ_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # 预计算 cos 和 sin（加速）
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)  # [seq_len, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)        # [seq_len, dim]
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


# ============================================================================
# GQA 分组查询注意力 + Flash Attention
# ============================================================================

class CausalSelfAttention(nn.Module):
    """分组查询注意力 GQA（带 RoPE + Flash Attention）"""

    def __init__(self, n_embd, n_head, n_kv_heads=None):
        super().__init__()
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

        self.resid_dropout = nn.Dropout(DROP_OUT)

        # 单个共享的 RoPE 实例
        self.rope = RotaryPositionalEmbedding(self.head_dim)

        # Flash Attention 的 dropout 比率
        self.dropout = DROP_OUT

    def _repeat_kv(self, x, n_rep):
        """将 KV 头扩展以匹配 Q 头数量"""
        if n_rep == 1:
            return x
        B, n_kv_heads, T, head_dim = x.shape
        x = x[:, :, None, :, :].expand(B, n_kv_heads, n_rep, T, head_dim)
        return x.reshape(B, n_kv_heads * n_rep, T, head_dim)

    def forward(self, x, start_pos=0, kv_cache=None):
        """
        Args:
            x: 输入 (B, T, C)
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

        # KV Cache 处理
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # 保存新的缓存（未扩展的 K/V）
        new_kv_cache = (k, v)

        # 扩展 KV 头以匹配 Q 头数量
        k_expanded = self._repeat_kv(k, self.n_rep)
        v_expanded = self._repeat_kv(v, self.n_rep)

        # Flash Attention（PyTorch 2.0+）
        # 使用缓存时 Q 只有当前 token，不需要 causal mask
        is_causal = (kv_cache is None) and (T > 1)
        out = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # 合并多头: (B, n_head, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))

        return out, new_kv_cache

    def _apply_rope(self, q, k, cos, sin):
        """对多头 Q、K 应用 RoPE"""
        # q: (B, n_head, T, head_dim), k: (B, n_kv_heads, T, head_dim)
        # cos, sin: (T, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)

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

    def __init__(self, n_embd):
        super().__init__()
        # 隐藏层维度，用 8/3 保持参数量相近
        hidden_dim = int(8 / 3 * n_embd)
        
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)  # 门控分支
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=False)  # 值分支
        self.w3 = nn.Linear(hidden_dim, n_embd, bias=False)  # 输出投影
        self.dropout = nn.Dropout(DROP_OUT)

    def forward(self, x):
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

    def __init__(self, n_embd, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(n_embd, num_experts, bias=False)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape                                       # B=batch, T=seq_len, C=n_embd
        x_flat = x.view(-1, C)                                  # (B*T, C)

        # 计算路由分数
        logits = self.gate(x_flat)                              # (B*T, num_experts)
        probs = F.softmax(logits, dim=-1)                       # (B*T, num_experts)

        # Top-K 选择
        weights, indices = torch.topk(probs, self.top_k, dim=-1)  # (B*T, top_k), (B*T, top_k)
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

    def __init__(self, n_embd, num_experts, num_shared_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        
        # 路由专家 (Routed Experts)
        self.experts = nn.ModuleList([FeedForward(n_embd) for _ in range(num_experts)])
        self.router = Router(n_embd, num_experts, top_k)
        
        # 共享专家 (Shared Experts) - 如果 num_shared_experts > 0
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(n_embd) for _ in range(num_shared_experts)
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

    def __init__(self, n_embd, n_head, n_kv_heads=None, use_moe=False):
        super().__init__()
        self.use_moe = use_moe
        self.sa = CausalSelfAttention(n_embd, n_head, n_kv_heads)
        if use_moe:
            self.ff = MoELayer(n_embd, NUM_EXPERTS, NUM_SHARED_EXPERTS, TOP_K)
        else:
            self.ff = FeedForward(n_embd)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x, start_pos=0, kv_cache=None):
        # 残差连接：x = x + sublayer(LayerNorm(x))
        h, new_kv_cache = self.sa(self.ln1(x), start_pos, kv_cache)
        x = x + h
        if self.use_moe:
            ff_out, aux_loss = self.ff(self.ln2(x))
            x = x + ff_out
            return x, new_kv_cache, aux_loss
        else:
            x = x + self.ff(self.ln2(x))
            return x, new_kv_cache


# ============================================================================
# 主模型定义
# ============================================================================

class GPTLanguageModel(nn.Module):
    """
    GPT 风格的语言模型

    特性：
    - GQA 分组查询注意力 + Flash Attention
    - KV Cache 推理加速
    - RoPE 旋转位置编码
    - GPT-2 风格权重初始化
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.use_moe = USE_MOE  # 保存配置供 forward 使用

        # Token 嵌入 (使用 RoPE，不需要位置嵌入)
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)

        # Transformer 层（使用 GQA + KV Cache）
        # MoE: 交替层使用（奇数层：1, 3, 5, 7, 9, 11）
        from config import MOE_FREQ
        self.blocks = nn.ModuleList([
            Block(N_EMBED, N_HEADS, N_KV_HEADS,
                  use_moe=(USE_MOE and i % MOE_FREQ == 1))
            for i in range(N_LAYERS)
        ])
        self.ln_f = RMSNorm(N_EMBED)  # 最终的 RMSNorm

        # 输出头：将嵌入映射到词汇表大小
        self.lm_head = nn.Linear(N_EMBED, vocab_size, bias=False)

        # 应用 GPT-2 风格权重初始化
        self.apply(self._init_weights)

        # 权重共享：Embedding 与 LM Head 共用同一权重矩阵
        self.lm_head.weight = self.token_embedding_table.weight

        # 残差投影层特殊缩放（GPT-2 论文）
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYERS))

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

    def forward(self, idx, targets=None, start_pos=0, kv_caches=None):
        """
        Args:
            idx: 输入 token ids (B, T)
            targets: 目标 token ids (B, T)，用于计算损失
            start_pos: 当前位置索引（用于 KV Cache）
            kv_caches: List[Tuple[k, v]]，每层一个缓存
        Returns:
            logits: 输出 logits
            loss: 损失（如果提供 targets）
            new_kv_caches: 更新后的 KV 缓存列表
        """
        B, T = idx.shape

        # 嵌入 (RoPE 在注意力层内部应用)
        tok_emb = self.token_embedding_table(idx)  # (B, T, N_EMBED)
        x = tok_emb

        # Transformer 处理（逐层传递 KV Cache）
        new_kv_caches = []
        total_aux_loss = 0.0

        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            if block.use_moe:
                x, new_kv_cache, aux_loss = block(x, start_pos, kv_cache)
                total_aux_loss += aux_loss
            else:
                x, new_kv_cache = block(x, start_pos, kv_cache)
            new_kv_caches.append(new_kv_cache)

        x = self.ln_f(x)

        # 输出 logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 计算损失（如果需要）
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            ce_loss = F.cross_entropy(logits_flat, targets_flat)

            # 仅当使用 MoE 时添加辅助损失
            if self.use_moe:
                loss = ce_loss + AUX_LOSS_COEF * total_aux_loss
            else:
                loss = ce_loss

        return logits, loss, new_kv_caches

    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=0.9, use_kv_cache=True):
        """
        生成文本，使用 Top-p (Nucleus) Sampling

        Args:
            idx: 输入 token ids (B, T)
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数 (越高越随机)
            top_p: nucleus sampling 阈值
            use_kv_cache: 是否使用 KV Cache 加速推理
        """
        kv_caches = None

        for i in range(max_new_tokens):
            if use_kv_cache:
                if i == 0:
                    # 第一次：处理整个输入序列
                    idx_cond = idx[:, -MAX_SEQ_LEN:]
                    start_pos = 0
                    kv_caches = None
                else:
                    # 后续：只处理最后一个 token
                    idx_cond = idx[:, -1:]
                    start_pos = min(idx.shape[1] - 1, MAX_SEQ_LEN - 1)

                    # 如果超过 MAX_SEQ_LEN，需要截断缓存
                    if idx.shape[1] > MAX_SEQ_LEN:
                        kv_caches = [(k[:, :, 1:, :], v[:, :, 1:, :]) for k, v in kv_caches]
            else:
                # 不使用缓存：每次处理整个序列
                idx_cond = idx[:, -MAX_SEQ_LEN:]
                start_pos = 0
                kv_caches = None

            # 前向传播
            logits, _, kv_caches = self(idx_cond, start_pos=start_pos, kv_caches=kv_caches)
            logits = logits[:, -1, :]  # 最后一个位置 (B, vocab_size)

            # 应用温度
            logits = logits / temperature

            # Top-p Sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 移除累积概率超过 top_p 的 token
            sorted_indices_to_remove = cumulative_probs > top_p
            # 保留第一个超过阈值的 token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # 将移除的 token 设为 -inf
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
