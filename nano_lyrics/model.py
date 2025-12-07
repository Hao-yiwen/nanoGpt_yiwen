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

from config import N_EMBED, N_HEADS, N_KV_HEADS, N_LAYERS, DROP_OUT, MAX_SEQ_LEN


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
# Transformer Block
# ============================================================================

class Block(nn.Module):
    """
    Transformer Block: 自注意力 + 前馈网络

    使用 Pre-LayerNorm 结构（先 LayerNorm 再计算）和残差连接
    """

    def __init__(self, n_embd, n_head, n_kv_heads=None):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head, n_kv_heads)
        self.ff = FeedForward(n_embd)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x, start_pos=0, kv_cache=None):
        # 残差连接：x = x + sublayer(LayerNorm(x))
        h, new_kv_cache = self.sa(self.ln1(x), start_pos, kv_cache)
        x = x + h
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

        # Token 嵌入 (使用 RoPE，不需要位置嵌入)
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)

        # Transformer 层（使用 GQA + KV Cache）
        self.blocks = nn.ModuleList([Block(N_EMBED, N_HEADS, N_KV_HEADS) for _ in range(N_LAYERS)])
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
        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches is not None else None
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
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

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
