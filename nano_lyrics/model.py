# -*- coding: utf-8 -*-
"""
NanoGPT 模型定义

包含:
- RotaryPositionalEmbedding (RoPE)
- CausalSelfAttention (Flash Attention)
- FeedForward
- Block
- GPTLanguageModel
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from config import N_EMBED, N_HEADS, N_LAYERS, DROP_OUT, BLOCK_SIZE


# ============================================================================
# RoPE 旋转位置编码
# ============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """RoPE 旋转位置编码"""

    def __init__(self, dim, max_seq_len=BLOCK_SIZE, base=10000):
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
# 批量化多头注意力 + Flash Attention
# ============================================================================

class CausalSelfAttention(nn.Module):
    """批量化多头注意力（带 RoPE + Flash Attention）"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # 合并 Q、K、V 投影为单个 Linear
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.resid_dropout = nn.Dropout(DROP_OUT)

        # 单个共享的 RoPE 实例
        self.rope = RotaryPositionalEmbedding(self.head_dim)

        # Flash Attention 的 dropout 比率
        self.dropout = DROP_OUT

    def forward(self, x):
        B, T, C = x.shape

        # 一次性计算所有 Q、K、V
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=-1)

        # 重塑为多头: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 应用 RoPE（对 Q 和 K）
        cos, sin = self.rope(T)
        q, k = self._apply_rope(q, k, cos, sin)

        # Flash Attention（PyTorch 2.0+）
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,  # 自动应用因果掩码
        )

        # 合并多头: (B, n_head, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))

        return out

    def _apply_rope(self, q, k, cos, sin):
        """对多头 Q、K 应用 RoPE"""
        # q, k: (B, n_head, T, head_dim)
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

class FeedForward(nn.Module):
    """前馈网络：两层线性变换 + GELU 激活"""

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(DROP_OUT)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ============================================================================
# Transformer Block
# ============================================================================

class Block(nn.Module):
    """
    Transformer Block: 自注意力 + 前馈网络

    使用 Pre-LayerNorm 结构（先 LayerNorm 再计算）和残差连接
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 残差连接：x = x + sublayer(LayerNorm(x))
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ============================================================================
# 主模型定义
# ============================================================================

class GPTLanguageModel(nn.Module):
    """
    GPT 风格的语言模型

    特性：
    - 批量化多头注意力 + Flash Attention
    - RoPE 旋转位置编码
    - GPT-2 风格权重初始化
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

        # Token 嵌入 (使用 RoPE，不需要位置嵌入)
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)

        # Transformer 层
        self.blocks = nn.Sequential(*[Block(N_EMBED, N_HEADS) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(N_EMBED)  # 最终的 LayerNorm

        # 输出头：将嵌入映射到词汇表大小
        self.lm_head = nn.Linear(N_EMBED, vocab_size, bias=False)

        # 应用 GPT-2 风格权重初始化
        self.apply(self._init_weights)

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
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 嵌入 (RoPE 在注意力层内部应用)
        tok_emb = self.token_embedding_table(idx)  # (B, T, N_EMBED)
        x = tok_emb

        # Transformer 处理
        x = self.blocks(x)
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

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=0.9):
        """
        生成文本，使用 Top-p (Nucleus) Sampling

        Args:
            idx: 输入 token ids (B, T)
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数 (越高越随机)
            top_p: nucleus sampling 阈值
        """
        for _ in range(max_new_tokens):
            # 截取最后 BLOCK_SIZE 个 token
            idx_cond = idx[:, -BLOCK_SIZE:]

            # 前向传播
            logits, _ = self(idx_cond)
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
