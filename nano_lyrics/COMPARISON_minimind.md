# nano_lyrics vs minimind 预训练实现对比

本文档详细对比 nano_lyrics 与 minimind 两个项目在模型架构和预训练实现上的差异。

---

## 一、模型架构对比

### 1.1 基础配置

| 配置项 | nano_lyrics | minimind |
|--------|-------------|----------|
| 隐藏维度 | 768 | 512 (可配置) |
| 注意力头数 | 12 | 8 |
| KV头数 (GQA) | 4 | 2 |
| 层数 | 12 | 8 |
| 上下文长度 | 1024 | 32768 |
| Dropout | 0.1 | 0.0 |

### 1.2 归一化层

**nano_lyrics: LayerNorm**
```python
self.ln1 = nn.LayerNorm(n_embd)
self.ln2 = nn.LayerNorm(n_embd)
```

**minimind: RMSNorm**
```python
class RMSNorm(torch.nn.Module):
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
```

**差异说明**: RMSNorm 省去了 LayerNorm 中的均值计算，训练速度更快，Llama系列模型均使用RMSNorm。

---

### 1.3 前馈网络 (FFN)

**nano_lyrics: 标准MLP**
```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = self.c_fc(x)      # 扩展 4x
        x = self.gelu(x)      # GELU激活
        x = self.c_proj(x)    # 投影回原维度
        return x
```

**minimind: SwiGLU**
```python
class FeedForward(nn.Module):
    def __init__(self, config):
        # intermediate_size ≈ hidden_size * 8/3
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = nn.SiLU()  # Swish激活

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

**差异说明**:
- SwiGLU 使用门控机制 `silu(gate) * up`，在 PaLM、Llama 中被证明效果更好
- SwiGLU 参数量略多 (3个投影 vs 2个)，但中间维度较小 (8/3x vs 4x)

---

### 1.4 旋转位置编码 (RoPE)

**nano_lyrics: 基础实现**
```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024, base=10000):
        # θ_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        # 预计算 cos/sin
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
```

**minimind: YaRN扩展支持**
```python
def precompute_freqs_cis(dim, end=32*1024, rope_base=1e6, rope_scaling=None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))

    if rope_scaling is not None:  # YaRN: 支持上下文长度外推
        # f'(i) = f(i) * ((1-γ) + γ/s)
        low = math.floor(inv_dim(beta_fast))
        high = math.ceil(inv_dim(beta_slow))
        ramp = torch.clamp((torch.arange(dim//2) - low) / (high - low), 0, 1)
        freqs = freqs * (1 - ramp + ramp / factor)
```

**差异说明**:
| 参数 | nano_lyrics | minimind |
|------|-------------|----------|
| rope_base | 10000 | 1000000 |
| 最大长度 | 1024 | 32768 |
| YaRN支持 | 无 | 有 (factor=16) |

minimind 可将 2048 的训练长度外推到 32768。

---

### 1.5 注意力机制

两者都使用 **GQA (Grouped Query Attention)** + **Flash Attention**:

**共同点**:
- Q投影保持原维度，K/V投影按KV头数缩小
- 使用 `repeat_kv` 将KV头扩展匹配Q头数
- 调用 `F.scaled_dot_product_attention` 实现Flash Attention
- 支持 KV Cache 加速推理

**差异点**:

| 特性 | nano_lyrics | minimind |
|------|-------------|----------|
| Flash Attention | 始终启用 | 可选 (`flash_attn`参数) |
| 无Flash时的实现 | 无 | 手动计算 scores+mask |
| attention_mask | 不支持 | 支持 (处理padding) |
| RoPE应用方式 | 在Attention内部 | 在forward外部传入 |

---

### 1.6 MOE (混合专家)

**nano_lyrics**: 不支持

**minimind**: 完整支持
```python
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)  # 专家路由
        self.shared_experts = nn.ModuleList([...])  # 共享专家

    def forward(self, x):
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 每个token选择top-k个专家
        # aux_loss用于平衡专家负载
```

MOE配置:
- `n_routed_experts=4`: 总专家数
- `num_experts_per_tok=2`: 每token选2个专家
- `n_shared_experts=1`: 共享专家数
- `aux_loss_alpha=0.01`: 辅助损失系数

---

### 1.7 权重共享

**nano_lyrics**: 无权重共享
```python
self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
self.lm_head = nn.Linear(N_EMBED, vocab_size, bias=False)
```

**minimind**: Embedding与LM Head权重共享
```python
self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
self.embed_tokens.weight = self.lm_head.weight  # 权重绑定
```

**差异说明**: 权重共享可减少约 `vocab_size * hidden_size` 个参数。

---

## 二、训练实现对比

### 2.1 训练配置

| 配置项 | nano_lyrics | minimind |
|--------|-------------|----------|
| 分布式训练 | 不支持 | DDP支持 |
| 混合精度 | BF16 (固定) | BF16/FP16 (可选) |
| 梯度累积 | 不支持 | 支持 |
| 学习率 | 3e-4 → 3e-5 | 5e-4 |
| 权重衰减 | 0.1 | 无显式设置 |
| 梯度裁剪 | 1.0 | 1.0 |
| Batch Size | 64 | 32 |
| 最大序列长度 | 1024 | 512 |

---

### 2.2 学习率调度

**nano_lyrics: Cosine Annealing with Warmup**
```python
def get_lr(iter_num):
    if iter_num < WARMUP_ITERS:
        return LEARNING_RATE * (iter_num + 1) / WARMUP_ITERS

    decay_ratio = (iter_num - WARMUP_ITERS) / (TRAIN_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)
```

**minimind**: 类似实现，在 `trainer_utils.py` 中

---

### 2.3 优化器

**nano_lyrics: AdamW + 参数分组**
```python
# 区分需要weight_decay的参数
decay_params = []      # 权重矩阵
no_decay_params = []   # bias, LayerNorm, Embedding

optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=LEARNING_RATE)
```

**minimind: AdamW 无分组**
```python
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

---

### 2.4 损失计算

**nano_lyrics: 简单交叉熵**
```python
loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
```

**minimind: 带mask的交叉熵**
```python
loss_fct = nn.CrossEntropyLoss(reduction='none')
loss = loss_fct(logits.view(-1, vocab), targets.view(-1)).view(bsz, seq_len)
loss = (loss * loss_mask).sum() / loss_mask.sum()  # 过滤padding
loss += res.aux_loss  # 加上MOE辅助损失
loss = loss / accumulation_steps  # 梯度累积
```

---

### 2.5 数据集

**nano_lyrics: 简单滑动窗口**
```python
class LyricsDataset(Dataset):
    def __getitem__(self, idx):
        x = self.data[idx : idx + block_size]
        y = self.data[idx + 1 : idx + block_size + 1]
        return x, y
```

**minimind: 带loss_mask**
```python
class PretrainDataset(Dataset):
    def __getitem__(self, idx):
        # 返回 X, Y, loss_mask
        # loss_mask用于标记哪些位置参与loss计算
        return X, Y, loss_mask
```

---

### 2.6 Checkpoint

**nano_lyrics**
```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'config': {...}
}
torch.save(checkpoint, path)
```

**minimind**
```python
# 支持epoch级和step级恢复
# 支持wandb_id恢复
lm_checkpoint(config, weight, model, optimizer, scaler, epoch, step, wandb, save_dir)
```

---

## 三、配置管理对比

**nano_lyrics: Python模块变量**
```python
# config.py
N_EMBED = 768
N_HEADS = 12
N_LAYERS = 12
```

**minimind: HuggingFace PretrainedConfig**
```python
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(self, hidden_size=512, num_hidden_layers=8, ...):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        ...
```

**优势对比**:
- nano_lyrics: 简单直接，适合学习
- minimind: 支持JSON序列化、HuggingFace生态集成、版本管理

---

## 四、总结

| 维度 | nano_lyrics | minimind |
|------|-------------|----------|
| **定位** | 教学/实验 | 生产级 |
| **复杂度** | 简单 (~400行核心代码) | 完整 (~1000行) |
| **性能优化** | 基础 | 全面 (RMSNorm, SwiGLU, MOE) |
| **扩展性** | 有限 | 高 (DDP, MOE, YaRN) |
| **HuggingFace兼容** | 无 | 完整 |
| **学习曲线** | 低 | 中等 |

### 建议

- **学习GPT原理**: 选择 nano_lyrics，代码简洁易懂
- **生产训练**: 选择 minimind，功能完整、性能优化好
- **长上下文需求**: 选择 minimind，支持YaRN外推
- **大规模训练**: 选择 minimind，支持DDP分布式

---

*文档生成时间: 2025-12-07*
