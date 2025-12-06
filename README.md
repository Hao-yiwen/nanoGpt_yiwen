# NanoGPT 中文歌词版

基于 Andrej Karpathy 的 NanoGPT，针对中文歌词生成进行优化的系列版本。

## 版本说明

### v0 系列 - 字符级（早期版本）

| 文件 | 特性 | 说明 |
|------|------|------|
| `v0_char_english.py` | 字符级 | 英文版，使用 input.txt |
| `v0_char_chinese.py` | 字符级 | 中文歌词版 |
| `v0_char_rope.py` | 字符级 + RoPE | 中文歌词 + 旋转位置编码 |

### v1-v6 系列 - BPE 分词（推荐）

| 文件 | 特性 | 模型规格 | 说明 |
|------|------|----------|------|
| `v1_bpe_rope.py` | BPE + RoPE | 384d, 6L, 6H | BPE 分词 + 旋转位置编码 |
| `v2_cosine_lr.py` | + Cosine LR | 512d, 6L, 8H | 新增：余弦退火学习率调度 |
| `v3_gpt2_amp.py` | + AMP + Compile | 768d, 12L, 12H | GPT-2 Small 架构，混合精度训练 |
| `v4_flash_optim.py` | + Flash Attention | 768d, 12L, 12H | 多项优化：Flash Attention, Weight Decay 等 |
| `v5_checkpoint.py` | + Checkpoint | 768d, 12L, 12H | 新增：模型保存与恢复，定期预测 |
| `v6_wandb.py` | + Wandb | 768d, 12L, 12H | 新增：实时 loss 曲线，训练可视化 |

## 各版本详细特性

### v0 系列 - 字符级（早期实验）
- **v0_char_english.py**: 英文字符级，基础 Transformer
- **v0_char_chinese.py**: 中文歌词字符级
- **v0_char_rope.py**: 中文歌词 + RoPE 位置编码

> 字符级词汇表小（~5000字符），但序列更长。BPE 版本效率更高。

### v1_bpe_rope.py - 基础版本
- SentencePiece BPE 分词（vocab_size=8000）
- RoPE 旋转位置编码
- 基础 Transformer 架构

### v2_cosine_lr.py - 学习率优化
- 新增：余弦退火学习率调度（带 warmup）
- 模型规格提升：512 dim, 8 heads

### v3_gpt2_amp.py - GPT-2 Small
- GPT-2 Small 架构：768 dim, 12 layers, 12 heads
- 混合精度训练（AMP）
- torch.compile 加速

### v4_flash_optim.py - 多项优化
- **批量化多头注意力**：合并 Q/K/V 投影，减少循环开销
- **Flash Attention**：`F.scaled_dot_product_attention`，2-4x 加速
- **Weight Decay**：参数分组，embedding/bias 不衰减
- **Gradient Clipping**：梯度裁剪，训练更稳定
- **DataLoader 多进程**：4 workers + pin_memory + prefetch
- **GPT-2 权重初始化**：正态分布 N(0, 0.02)，残差层缩放

### v5_checkpoint.py - 完整版
- 包含 v4 所有优化
- **Checkpoint 保存**：定期保存、最佳模型、断点恢复
- **训练预测**：每 1000 步生成一次文本预览

### v6_wandb.py - Wandb 可视化（推荐）
- 包含 v5 所有功能
- **Wandb 集成**：实时训练监控
- **Loss 曲线**：train/test loss 实时可视化
- **学习率曲线**：跟踪 LR 变化
- **超参数记录**：自动记录所有配置
- **生成样本**：每 1000 步记录生成的文本样本

## 快速开始

```bash
# 推荐使用最新版本（带 wandb 可视化）
python v6_wandb.py

# 或不使用 wandb
python v5_checkpoint.py
```

训练输出保存在 `checkpoints/` 目录：
- `best.pt` - 最佳模型（最低 test loss）
- `latest.pt` - 最新模型（用于恢复训练）
- `ckpt_1000.pt` 等 - 定期保存点

## 配置参数

编辑文件顶部的超参数：

```python
# 训练相关
BATCH_SIZE = 64          # 根据显存调整
BLOCK_SIZE = 1024        # 上下文长度
TRAIN_ITERS = 5000       # 训练步数

# 模型架构 (GPT-2 Small)
N_EMBED = 768
N_HEADS = 12
N_LAYERS = 12
```

## 数据

使用 `ChineseLyrics/` 目录下的中文歌词数据集（约 10 万首歌曲，20M tokens）。

---

## 模型架构

这是一个 **Decoder-only Transformer**，与 GPT-2 架构类似：

```
输入序列 → Token Embedding (RoPE 在 Attention 内部应用)
                        ↓
              ┌─────────────────┐
              │   Transformer   │ × N_LAYERS
              │     Block       │
              │  ┌───────────┐  │
              │  │ LayerNorm │  │
              │  │ Multi-Head│  │
              │  │ Attention │  │
              │  │ + Residual│  │
              │  ├───────────┤  │
              │  │ LayerNorm │  │
              │  │ FeedForward│ │
              │  │ + Residual│  │
              │  └───────────┘  │
              └─────────────────┘
                        ↓
                   LayerNorm
                        ↓
                   Linear Head
                        ↓
              输出 (vocab_size 维度)
```

## 训练原理

### 自回归语言模型 (Autoregressive LM)

模型的核心任务是：**给定前面的 token，预测下一个 token**。

### 并行训练机制

**一次前向传播** = 同时计算 `batch_size × sequence_length` 个预测任务
- 例如：`batch_size=64, block_size=1024` → 一次计算 **65,536** 个预测

### Causal Mask（因果掩码）

通过 `is_causal=True` 参数（Flash Attention）或下三角掩码矩阵，确保每个位置只能 attend 到它之前的位置。

## 依赖

```bash
pip install torch sentencepiece wandb
```

## 硬件要求

| 版本 | 显存需求 |
|------|----------|
| v1/v2 | 8GB+ |
| v3/v4/v5/v6 | 24GB+（可调整 BATCH_SIZE） |

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [nanoGPT by Karpathy](https://github.com/karpathy/nanoGPT) - 本项目参考
- [RoFormer](https://arxiv.org/abs/2104.09864) - RoPE 旋转位置编码
- [FlashAttention](https://arxiv.org/abs/2205.14135) - Flash Attention 论文
- [Weights & Biases](https://wandb.ai/) - 实验跟踪平台
