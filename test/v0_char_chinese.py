# -*- coding: utf-8 -*-
"""
NanoGPT v3 - 中文歌词版本

基于 v2.py，使用中文歌词数据训练字符级语言模型。
数据来源：ChineseLyrics
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import re

# ============================================================================
# 超参数配置
# ============================================================================

# 数据相关
LYRICS_DIR = 'ChineseLyrics'
MAX_SONGS = 5000  # 加载的歌曲数量
RANDOM_SEED = 1337

# 训练相关
BATCH_SIZE = 64          # 每批样本数
BLOCK_SIZE = 256         # 最大上下文长度（序列长度）
LEARNING_RATE = 3e-4     # 学习率
TRAIN_ITERS = 5000       # 训练迭代次数
EVAL_ITERS = 200         # 评估时的迭代次数
EVAL_INTERVAL = 200      # 每隔多少步评估一次

# 模型架构相关
N_EMBED = 384            # 嵌入维度
N_HEADS = 6              # 注意力头数
N_LAYERS = 6             # Transformer Block 层数
DROP_OUT = 0.2           # Dropout 比率

# 生成相关
INITIAL_GENERATE_TOKENS = 100   # 训练前生成的 token 数
FINAL_GENERATE_TOKENS = 300     # 训练后生成的 token 数

# 设备选择
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ============================================================================
# 数据加载与预处理
# ============================================================================

def load_lyrics(max_songs=MAX_SONGS):
    """加载歌词文件，限制歌曲数量"""
    all_lyrics = []
    for i in range(1, 6):
        if len(all_lyrics) >= max_songs:
            break
        filepath = f'{LYRICS_DIR}/lyrics{i}.json'
        with open(filepath, 'r', encoding='utf-8') as f:
            songs = json.load(f)
            remaining = max_songs - len(all_lyrics)
            all_lyrics.extend(songs[:remaining])
    return all_lyrics


def preprocess_lyric(lyric_lines):
    """
    预处理单首歌的歌词

    过滤掉：
    - 元信息（作词、作曲、编曲等）
    - 结构标记（主歌、副歌、间奏等）
    - 单字符行
    """
    skip_patterns = [
        r'^作词', r'^作曲', r'^编曲', r'^演唱', r'^和声', r'^后期', r'^混音',
        r'^制作', r'^录音', r'^吉他', r'^钢琴', r'^贝斯', r'^鼓',
        r'^主歌\d*$', r'^副歌\d*$', r'^过渡\d*$', r'^间奏', r'^结尾',
        r'^verse', r'^chorus', r'^bridge', r'^\[', r'^【', r'^（.*）$',
        r'^\(.*\)$', r'^demo$', r'^intro', r'^outro',
    ]

    result = []
    for line in lyric_lines:
        line = line.strip()
        if not line:
            continue
        # 检查是否匹配跳过模式
        skip = any(re.match(p, line, re.IGNORECASE) for p in skip_patterns)
        if not skip and len(line) > 1:  # 过滤单字符行
            result.append(line)

    return '\n'.join(result)


# 加载并处理歌词
print("正在加载歌词...")
lyrics_data = load_lyrics()
print(f"加载了 {len(lyrics_data)} 首歌曲")

# 合并所有歌词为文本
text = '\n\n'.join(
    preprocess_lyric(song['lyric'])
    for song in lyrics_data
    if song.get('lyric')
)

print(f"数据集字符数: {len(text):,}")
print(f"数据预览:\n{text[:500]}...")

# 构建字符级词汇表
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"\n词汇表大小: {vocab_size}")
print(f"字符集预览: {''.join(chars[:100])}...")

# 创建编码/解码映射
stoi = {ch: i for i, ch in enumerate(chars)}  # 字符 -> 索引
itos = {i: ch for i, ch in enumerate(chars)}  # 索引 -> 字符

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 数据转为张量并分割
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]
print(f"\n训练集大小: {len(train_data):,} | 测试集大小: {len(test_data):,}")

# 设置随机种子以保证可复现性
torch.manual_seed(RANDOM_SEED)


# ============================================================================
# 数据批次生成
# ============================================================================

def get_batch(split):
    """
    生成一个训练/测试批次

    返回:
        x: 输入序列 (BATCH_SIZE, BLOCK_SIZE)
        y: 目标序列 (BATCH_SIZE, BLOCK_SIZE)，是 x 右移一位的结果
    """
    data_split = train_data if split == 'train' else test_data
    ix = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_split[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_split[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ============================================================================
# 模型组件定义
# ============================================================================

class Head(nn.Module):
    """单个注意力头"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        # 因果掩码：下三角矩阵，确保每个位置只能看到之前的位置
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROP_OUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # 计算注意力分数，并缩放
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)

        # 应用因果掩码：将未来位置设为 -inf，softmax 后变为 0
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # 加权聚合 value
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """多头注意力：并行运行多个注意力头，然后拼接结果"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)  # 投影层
        self.dropout = nn.Dropout(DROP_OUT)

    def forward(self, x):
        # 并行计算所有头，然后在最后一维拼接
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """前馈网络：两层线性变换 + ReLU 激活"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 扩展到 4 倍维度
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # 压缩回原维度
            nn.Dropout(DROP_OUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer Block: 自注意力 + 前馈网络

    使用 Pre-LayerNorm 结构（先 LayerNorm 再计算）和残差连接
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
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

class BigramLanguageModel(nn.Module):
    """
    GPT 风格的语言模型（中文歌词版本）

    使用字符级编码，适配中文歌词数据。
    """

    def __init__(self):
        super().__init__()
        # Token 和位置嵌入
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)

        # Transformer 层
        self.blocks = nn.Sequential(*[Block(N_EMBED, N_HEADS) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(N_EMBED)  # 最终的 LayerNorm

        # 输出头：将嵌入映射到词汇表大小
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        """
        前向传播

        Args:
            idx: 输入 token 索引 (B, T)
            targets: 目标 token 索引 (B, T)，训练时使用

        Returns:
            logits: 预测分布 (B, T, vocab_size) 或 (B*T, vocab_size)
            loss: 交叉熵损失（如果提供了 targets）
        """
        B, T = idx.shape

        # 嵌入
        tok_emb = self.token_embedding_table(idx)  # (B, T, N_EMBED)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))  # (T, N_EMBED)
        x = tok_emb + pos_emb  # (B, T, N_EMBED) - 广播相加

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
            logits = logits.view(B * T, C)      # 展平为 (B*T, vocab_size)
            targets = targets.view(B * T)       # 展平为 (B*T,)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        自回归生成文本

        Args:
            idx: 起始 token 序列 (B, T)
            max_new_tokens: 要生成的新 token 数量

        Returns:
            生成的完整序列 (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # 截断到最大上下文长度
            idx_cond = idx[:, -BLOCK_SIZE:]

            # 获取预测
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # 只取最后一个位置 (B, vocab_size)

            # 采样下一个 token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 拼接到序列
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ============================================================================
# 评估函数
# ============================================================================

@torch.no_grad()
def estimate_loss():
    """在训练集和测试集上估计平均损失"""
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ============================================================================
# 训练流程
# ============================================================================

if __name__ == '__main__':
    # 创建模型
    model = BigramLanguageModel().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {num_params:,}")

    # 测试初始状态
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    print(f"初始 loss: {loss:.4f}")

    # 训练前生成
    print("\n" + "=" * 50)
    print("训练前生成的文本:")
    print("=" * 50)
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(model.generate(context, max_new_tokens=INITIAL_GENERATE_TOKENS)[0].tolist()))

    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    print("\n" + "=" * 50)
    print("开始训练")
    print("=" * 50)

    for iter_num in range(TRAIN_ITERS):
        # 定期评估
        if iter_num % EVAL_INTERVAL == 0:
            losses = estimate_loss()
            print(f"Step {iter_num:5d} | train loss: {losses['train']:.4f} | test loss: {losses['test']:.4f}")

        # 获取批次并训练
        xb, yb = get_batch('train')
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

    # 最终评估
    losses = estimate_loss()
    print(f"\n训练完成!")
    print(f"最终 train loss: {losses['train']:.4f} | test loss: {losses['test']:.4f}")

    # 训练后生成
    print("\n" + "=" * 50)
    print("训练后生成的文本:")
    print("=" * 50)
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(model.generate(context, max_new_tokens=FINAL_GENERATE_TOKENS)[0].tolist()))
