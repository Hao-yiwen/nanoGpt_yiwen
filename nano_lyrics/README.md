# NanoGPT 中文歌词生成器

基于 GPT-2 架构的中文歌词生成模型，使用 RoPE 位置编码和 Flash Attention。

## 目录结构

```
final_train/
├── config.py      # 超参数配置
├── model.py       # 模型定义 (GPT + RoPE + Flash Attention)
├── tokenizer.py   # BPE 分词器
├── train.py       # 训练脚本
├── inference.py   # 推理脚本
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch sentencepiece swanlab
```

### 2. 登录 Swanlab（可选，用于可视化）

```bash
swanlab login
```

### 3. 训练

使用 tmux 在后台运行（推荐）：

```bash
# 创建新的 tmux 会话
tmux new -s train

# 进入目录并开始训练
cd final_train
python train.py

# 按 Ctrl+B 然后按 D 分离会话（训练继续在后台运行）
```

tmux 常用命令：
```bash
tmux ls              # 查看所有会话
tmux attach -t train # 重新连接会话
tmux kill-session -t train  # 终止会话
```

训练输出：
- `checkpoints/best.pt` - 最佳模型
- `checkpoints/latest.pt` - 最新模型
- `chinese_lyrics_bpe.model` - BPE 分词器

### 4. 推理

```bash
# 单次生成
python inference.py --prompt "今夜我"

# 交互模式
python inference.py
```

## 配置参数

编辑 `config.py` 修改超参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 64 | 批大小（根据显存调整） |
| `MAX_SEQ_LEN` | 1024 | 最大序列长度 |
| `LEARNING_RATE` | 3e-4 | 最大学习率 |
| `TRAIN_ITERS` | 2000 | 训练迭代次数 |
| `N_EMBED` | 768 | 嵌入维度 |
| `N_HEADS` | 12 | 注意力头数 |
| `N_LAYERS` | 12 | Transformer 层数 |

## 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | checkpoints/best.pt | 模型路径 |
| `--prompt` | - | 输入提示词（不指定则进入交互模式） |
| `--max_tokens` | 200 | 最大生成长度 |
| `--temperature` | 1.0 | 温度（越高越随机） |
| `--top_p` | 0.9 | Top-p 采样阈值 |

### 示例

```bash
# 基础生成
python inference.py --prompt "如果爱"

# 更随机
python inference.py --prompt "想起你" --temperature 1.2

# 更保守
python inference.py --prompt "那一年" --top_p 0.8 --temperature 0.8

# 更长输出
python inference.py --prompt "我们的" --max_tokens 500
```

## 模型架构

- **架构**: Decoder-only Transformer (GPT-2 Small)
- **参数量**: ~85M
- **位置编码**: RoPE (旋转位置编码)
- **注意力**: Flash Attention
- **归一化**: Pre-LayerNorm
- **激活函数**: GELU

## 训练特性

- 混合精度训练 (AMP BF16/FP16)
- torch.compile 加速
- 余弦退火学习率 + Warmup
- Weight Decay 参数分组
- 梯度裁剪
- Checkpoint 自动保存/恢复
- Swanlab 实时可视化

## 硬件要求

| 配置 | 显存需求 |
|------|----------|
| BATCH_SIZE=64, MAX_SEQ_LEN=1024 | ~24GB |
| BATCH_SIZE=32, MAX_SEQ_LEN=1024 | ~16GB |
| BATCH_SIZE=16, MAX_SEQ_LEN=512 | ~8GB |

## 数据

使用 `ChineseLyrics/` 目录下的中文歌词数据集（约 10 万首歌曲，20M tokens）。

## 关闭 Swanlab

如不需要可视化，在 `config.py` 中设置：

```python
USE_SWANLAB = False
```
