# NanoGPT - 字符级语言模型

一个简单的 GPT 风格语言模型实现，用于学习 Transformer 架构和自回归语言模型的训练原理。

## 模型架构

这是一个 **Decoder-only Transformer**，与 GPT-2 架构类似：

```
输入序列 → Token Embedding + Position Embedding
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

```
输入:    [h,   e,   l,   l,   o]
          ↓    ↓    ↓    ↓    ↓
目标:    [e,   l,   l,   o,   _]
```

### 并行训练机制

这是理解 Transformer 训练的关键点！

**不是**用完整序列预测一个字符，而是**并行地**对每个位置预测下一个字符：

| 位置 | 可见上下文 | 预测目标 |
|------|-----------|----------|
| 0 | h | e |
| 1 | h, e | l |
| 2 | h, e, l | l |
| 3 | h, e, l, l | o |
| 4 | h, e, l, l, o | (下一个) |

### Causal Mask（因果掩码）

通过下三角掩码矩阵，确保每个位置只能 attend 到它之前的位置：

```
        h    e    l    l    o
    ┌─────────────────────────┐
 h  │  ✓    ✗    ✗    ✗    ✗  │  位置0只能看位置0
 e  │  ✓    ✓    ✗    ✗    ✗  │  位置1能看0,1
 l  │  ✓    ✓    ✓    ✗    ✗  │  位置2能看0,1,2
 l  │  ✓    ✓    ✓    ✓    ✗  │  位置3能看0,1,2,3
 o  │  ✓    ✓    ✓    ✓    ✓  │  位置4能看所有
    └─────────────────────────┘
```

代码实现：
```python
self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
```

### 为什么这样高效？

- **一次前向传播** = 同时计算 `batch_size × sequence_length` 个预测任务
- 例如：`batch_size=64, block_size=256` → 一次计算 **16,384** 个预测
- 相比 RNN 逐步计算，Transformer 可以完全并行化

## 字符级 vs 子词级

| 模型 | 分词方式 | 词汇表大小 | 示例 |
|------|----------|------------|------|
| 本项目 | 字符级 | ~65 | `h` `e` `l` `l` `o` |
| GPT-2 | BPE 子词 | 50,257 | `hello` 或 `hel` `lo` |

字符级模型更简单，但需要更长的序列来表达相同的文本。

## 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `N_EMBED` | 384 | 嵌入维度 |
| `N_LAYERS` | 6 | Transformer Block 层数 |
| `N_HEADS` | 6 | 注意力头数 |
| `BLOCK_SIZE` | 256 | 最大上下文长度 |
| `BATCH_SIZE` | 64 | 批次大小 |
| `LEARNING_RATE` | 3e-4 | 学习率 |
| `DROP_OUT` | 0.2 | Dropout 比率 |

## 使用方法

### 1. 准备数据

将训练文本放在 `input.txt` 文件中。

### 2. 运行训练

```bash
python v2.py
```

### 3. 输出示例

训练过程会输出：
- 每 200 步的训练/测试 loss
- 训练前后的文本生成对比

## 文件结构

```
.
├── README.md       # 本文档
├── v2.py           # 主训练脚本
└── input.txt       # 训练数据
```

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [nanoGPT by Karpathy](https://github.com/karpathy/nanoGPT) - 本项目参考
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
