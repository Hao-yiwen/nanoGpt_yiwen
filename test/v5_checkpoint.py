# -*- coding: utf-8 -*-
"""
NanoGPT v9 - 中文歌词版本 (优化版 + Checkpoint)

基于 v8.py，新增以下功能：
1. 批量化多头注意力 (CausalSelfAttention)
2. Flash Attention (F.scaled_dot_product_attention)
3. Weight Decay 参数分组
4. Gradient Clipping
5. DataLoader 多进程加载
6. GPT-2 风格权重初始化
7. Checkpoint 保存与恢复（v9 新增）
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import json
import re
import os
import math
import sentencepiece as spm

# ============================================================================
# 超参数配置
# ============================================================================

# 数据相关
LYRICS_DIR = 'ChineseLyrics'
BPE_VOCAB_SIZE = 10000             # BPE 词汇表大小
BPE_MODEL_PREFIX = 'chinese_lyrics_bpe'  # BPE 模型文件前缀
MAX_SONGS = 102197  # 加载的歌曲数量
RANDOM_SEED = 1337

# 训练相关
BATCH_SIZE = 64          # 每批样本数
BLOCK_SIZE = 1024        # 最大上下文长度（序列长度）
LEARNING_RATE = 3e-4     # 最大学习率
MIN_LR = 3e-5            # 最小学习率 (LEARNING_RATE 的 1/10)
WARMUP_ITERS = 500       # 预热迭代次数（5000步的10%）
TRAIN_ITERS = 5000       # 训练迭代次数
EVAL_ITERS = 200         # 评估时的迭代次数
EVAL_INTERVAL = 200      # 每隔多少步评估一次

# 训练稳定性
WEIGHT_DECAY = 0.1       # 权重衰减（AdamW 正则化）
GRAD_CLIP = 1.0          # 梯度裁剪阈值

# DataLoader 配置
NUM_WORKERS = 4          # 数据加载进程数
PIN_MEMORY = True        # 锁页内存，加速 GPU 传输
PREFETCH_FACTOR = 2      # 每个 worker 预取的 batch 数

# v9 新增：Checkpoint 配置
CHECKPOINT_DIR = 'checkpoints'
SAVE_INTERVAL = 1000     # 每 1000 步保存一次
SAVE_BEST = True         # 保存最佳模型
RESUME = True            # 是否尝试从 checkpoint 恢复

# 模型架构相关 (GPT-2 Small 规格)
N_EMBED = 768            # 嵌入维度
N_HEADS = 12             # 注意力头数 (768 / 12 = 64, 偶数满足 RoPE)
N_LAYERS = 12            # Transformer Block 层数
DROP_OUT = 0.1           # Dropout 比率（GPT-2 用 0.1）

# 生成相关
INITIAL_GENERATE_TOKENS = 100   # 训练前生成的 token 数
FINAL_GENERATE_TOKENS = 300     # 训练后生成的 token 数

# 设备选择
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# 混合精度训练配置
USE_AMP = True  # 开启混合精度训练
USE_COMPILE = True  # 开启 torch.compile 加速（需要 PyTorch 2.0+）

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
print(f"原始唯一字符数: {len(set(text)):,}")
print(f"数据预览:\n{text[:500]}...")

# ============================================================================
# BPE Tokenizer 训练与加载
# ============================================================================

def train_bpe_tokenizer(text, vocab_size, model_prefix):
    """
    训练 SentencePiece BPE tokenizer

    如果模型文件已存在，则直接加载；否则训练新模型。
    """
    model_file = f'{model_prefix}.model'

    # 检查是否已有训练好的模型
    if os.path.exists(model_file):
        print(f"加载已有 BPE 模型: {model_file}")
    else:
        print(f"训练 BPE tokenizer (vocab_size={vocab_size})...")
        # 保存文本到临时文件
        temp_corpus = 'temp_corpus.txt'
        with open(temp_corpus, 'w', encoding='utf-8') as f:
            f.write(text)

        # 训练 SentencePiece 模型
        spm.SentencePieceTrainer.train(
            input=temp_corpus,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=0.9995,  # 中文建议高覆盖率
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
        print(f"BPE 模型训练完成，保存至: {model_file}")

        # 清理临时文件
        if os.path.exists(temp_corpus):
            os.remove(temp_corpus)

    # 加载并返回 tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    return sp


# 训练/加载 BPE tokenizer
sp = train_bpe_tokenizer(text, BPE_VOCAB_SIZE, BPE_MODEL_PREFIX)
vocab_size = sp.get_piece_size()
print(f"\nBPE 词汇表大小: {vocab_size}")

# BPE 编码/解码函数
encode = lambda s: sp.encode(s, out_type=int)
decode = lambda l: sp.decode(list(l) if hasattr(l, '__iter__') else [l])

# 测试 tokenizer
test_text = "我爱你"
test_encoded = encode(test_text)
test_decoded = decode(test_encoded)
print(f"分词测试: '{test_text}' -> {test_encoded} -> '{test_decoded}'")

# 数据转为张量并分割
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]
print(f"\n训练集大小: {len(train_data):,} tokens | 测试集大小: {len(test_data):,} tokens")

# 设置随机种子以保证可复现性
torch.manual_seed(RANDOM_SEED)


# ============================================================================
# Dataset 和 DataLoader
# ============================================================================

class LyricsDataset(Dataset):
    """歌词数据集"""
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


# 创建数据集和加载器
train_dataset = LyricsDataset(train_data, BLOCK_SIZE)
test_dataset = LyricsDataset(test_data, BLOCK_SIZE)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    prefetch_factor=PREFETCH_FACTOR,
    drop_last=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    prefetch_factor=PREFETCH_FACTOR,
    drop_last=True,
)


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
# 模型组件定义
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
    GPT 风格的语言模型（中文歌词版本 v9）

    优化：
    - 批量化多头注意力 + Flash Attention
    - GPT-2 风格权重初始化
    """

    def __init__(self):
        super().__init__()
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
# v9 新增：Checkpoint 保存与恢复
# ============================================================================

def save_checkpoint(model, optimizer, scaler, iter_num, best_val_loss, path):
    """保存训练检查点"""
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 处理 compiled model
    model_state = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()

    checkpoint = {
        'model': model_state,
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': {
            'vocab_size': vocab_size,
            'n_embed': N_EMBED,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'block_size': BLOCK_SIZE,
            'batch_size': BATCH_SIZE,
        }
    }
    torch.save(checkpoint, path)
    print(f"  -> Checkpoint 已保存: {path}")


def load_checkpoint(path, model, optimizer=None, scaler=None):
    """加载训练检查点"""
    print(f"加载 Checkpoint: {path}")
    checkpoint = torch.load(path, map_location=DEVICE)

    # 处理 compiled model
    if hasattr(model, '_orig_mod'):
        model._orig_mod.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])

    iter_num = checkpoint.get('iter_num', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    print(f"  -> 恢复自: iter={iter_num}, best_val_loss={best_val_loss:.4f}")
    return iter_num, best_val_loss


# ============================================================================
# 评估函数
# ============================================================================

@torch.no_grad()
def estimate_loss():
    """在训练集和测试集上估计平均损失"""
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('test', test_loader)]:
        losses = []
        for i, (X, Y) in enumerate(loader):
            if i >= EVAL_ITERS:
                break
            X = X.to(DEVICE, non_blocking=True)
            Y = Y.to(DEVICE, non_blocking=True)
            _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses) if losses else 0.0
    model.train()
    return out


def get_lr(iter_num):
    """
    余弦退火学习率调度（带预热）

    1. 预热阶段：线性增加到 LEARNING_RATE
    2. 余弦退火：从 LEARNING_RATE 逐渐降到 MIN_LR
    """
    # 预热阶段：线性增加
    if iter_num < WARMUP_ITERS:
        return LEARNING_RATE * (iter_num + 1) / WARMUP_ITERS

    # 余弦退火阶段
    decay_iters = TRAIN_ITERS - WARMUP_ITERS
    iter_after_warmup = iter_num - WARMUP_ITERS

    # 余弦衰减系数 [0, 1]
    decay_ratio = iter_after_warmup / decay_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# ============================================================================
# 训练流程
# ============================================================================

if __name__ == '__main__':
    # 确保 checkpoint 目录存在
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 创建模型
    model = GPTLanguageModel().to(DEVICE)

    # torch.compile 加速（PyTorch 2.0+）
    if USE_COMPILE and hasattr(torch, 'compile'):
        print("启用 torch.compile 加速...")
        model = torch.compile(model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {num_params:,}")

    # 创建优化器（参数分组，区分 weight decay）
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or 'ln' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    print(f"参数分组: decay={len(decay_params)}, no_decay={len(no_decay_params)}")

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=LEARNING_RATE)

    # 混合精度训练 (AMP)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"混合精度训练: {'BF16' if amp_dtype == torch.bfloat16 else 'FP16' if USE_AMP else '关闭'}")

    # v9: 尝试从 checkpoint 恢复
    start_iter = 0
    best_val_loss = float('inf')
    resume_path = f'{CHECKPOINT_DIR}/latest.pt'

    if RESUME and os.path.exists(resume_path):
        start_iter, best_val_loss = load_checkpoint(resume_path, model, optimizer, scaler)
        start_iter += 1  # 从下一个迭代开始
    else:
        # 测试初始状态
        test_iter = iter(train_loader)
        xb, yb = next(test_iter)
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits, loss = model(xb, yb)
        print(f"初始 loss: {loss:.4f}")

        # 训练前生成
        print("\n" + "=" * 50)
        print("训练前生成的文本:")
        print("=" * 50)
        context = torch.tensor([[sp.bos_id()]], dtype=torch.long, device=DEVICE)
        print(decode(model.generate(context, max_new_tokens=INITIAL_GENERATE_TOKENS)[0].tolist()))

    # 训练循环
    print("\n" + "=" * 50)
    print(f"开始训练 (从 iter {start_iter} 到 {TRAIN_ITERS})")
    print("=" * 50)

    # 使用无限迭代器
    train_iter = iter(train_loader)

    for iter_num in range(start_iter, TRAIN_ITERS):
        # 更新学习率（余弦退火）
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 定期评估
        if iter_num % EVAL_INTERVAL == 0:
            losses = estimate_loss()
            print(f"Step {iter_num:5d} | train loss: {losses['train']:.4f} | test loss: {losses['test']:.4f} | lr: {lr:.2e}")

            # v9: 保存最佳模型
            if SAVE_BEST and losses['test'] < best_val_loss:
                best_val_loss = losses['test']
                save_checkpoint(model, optimizer, scaler, iter_num, best_val_loss,
                              f'{CHECKPOINT_DIR}/best.pt')

            # v9: 定期保存
            if iter_num > 0 and iter_num % SAVE_INTERVAL == 0:
                save_checkpoint(model, optimizer, scaler, iter_num, best_val_loss,
                              f'{CHECKPOINT_DIR}/ckpt_{iter_num}.pt')

                # v9: 每 1000 步生成一次预测
                model.eval()
                with torch.no_grad():
                    context = torch.tensor([[sp.bos_id()]], dtype=torch.long, device=DEVICE)
                    generated = model.generate(context, max_new_tokens=100)[0].tolist()
                    print(f"\n--- Step {iter_num} 生成预览 ---")
                    print(decode(generated))
                    print("-" * 30 + "\n")
                model.train()

        # 获取批次（循环重用）
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 混合精度前向传播
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=amp_dtype):
            logits, loss = model(xb, yb)

        # 混合精度反向传播
        scaler.scale(loss).backward()

        # 梯度裁剪（在 unscale 后进行）
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

    # v9: 保存最终模型
    save_checkpoint(model, optimizer, scaler, TRAIN_ITERS - 1, best_val_loss,
                   f'{CHECKPOINT_DIR}/latest.pt')

    # 最终评估
    losses = estimate_loss()
    print(f"\n训练完成!")
    print(f"最终 train loss: {losses['train']:.4f} | test loss: {losses['test']:.4f}")
    print(f"最佳 test loss: {best_val_loss:.4f}")

    # 训练后生成
    print("\n" + "=" * 50)
    print("训练后生成的文本:")
    print("=" * 50)
    context = torch.tensor([[sp.bos_id()]], dtype=torch.long, device=DEVICE)
    print(decode(model.generate(context, max_new_tokens=FINAL_GENERATE_TOKENS)[0].tolist()))

    print(f"\nCheckpoint 保存位置: {CHECKPOINT_DIR}/")
    print(f"  - best.pt: 最佳模型 (test loss: {best_val_loss:.4f})")
    print(f"  - latest.pt: 最新模型")
