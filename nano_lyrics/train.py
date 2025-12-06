# -*- coding: utf-8 -*-
"""
NanoGPT 训练脚本

训练中文歌词生成模型
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import re
import os
import math

from config import (
    LYRICS_DIR, MAX_SONGS, RANDOM_SEED,
    BATCH_SIZE, BLOCK_SIZE, LEARNING_RATE, MIN_LR, WARMUP_ITERS, TRAIN_ITERS,
    EVAL_ITERS, EVAL_INTERVAL, WEIGHT_DECAY, GRAD_CLIP,
    NUM_WORKERS, PIN_MEMORY, PREFETCH_FACTOR,
    USE_COMPILE,
    CHECKPOINT_DIR, SAVE_INTERVAL, SAVE_BEST, RESUME,
    SWANLAB_PROJECT, SWANLAB_RUN_NAME, USE_SWANLAB,
    INITIAL_GENERATE_TOKENS, FINAL_GENERATE_TOKENS, GENERATE_PROMPT,
    DEVICE,
)

# 延迟导入 swanlab（仅在需要时）
swanlab = None
from model import GPTLanguageModel
from tokenizer import train_bpe_tokenizer, encode, decode, get_bos_id


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


# ============================================================================
# Checkpoint 保存与恢复
# ============================================================================

def save_checkpoint(model, optimizer, iter_num, best_val_loss, vocab_size, path):
    """保存训练检查点"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_state = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()

    checkpoint = {
        'model': model_state,
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': {
            'vocab_size': vocab_size,
            'n_embed': 768,
            'n_heads': 12,
            'n_layers': 12,
            'block_size': BLOCK_SIZE,
            'batch_size': BATCH_SIZE,
        }
    }
    torch.save(checkpoint, path)
    print(f"  -> Checkpoint 已保存: {path}")


def load_checkpoint(path, model, optimizer=None):
    """加载训练检查点"""
    print(f"加载 Checkpoint: {path}")
    checkpoint = torch.load(path, map_location=DEVICE)

    if hasattr(model, '_orig_mod'):
        model._orig_mod.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    iter_num = checkpoint.get('iter_num', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    print(f"  -> 恢复自: iter={iter_num}, best_val_loss={best_val_loss:.4f}")
    return iter_num, best_val_loss


# ============================================================================
# 学习率调度
# ============================================================================

def get_lr(iter_num):
    """余弦退火学习率调度（带预热）"""
    if iter_num < WARMUP_ITERS:
        return LEARNING_RATE * (iter_num + 1) / WARMUP_ITERS

    decay_iters = TRAIN_ITERS - WARMUP_ITERS
    iter_after_warmup = iter_num - WARMUP_ITERS
    decay_ratio = iter_after_warmup / decay_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# ============================================================================
# 训练主函数
# ============================================================================

def main():
    print(f"Using device: {DEVICE}")

    # 确保 checkpoint 目录存在
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ========== 数据加载 ==========
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

    # ========== 训练/加载 tokenizer ==========
    sp = train_bpe_tokenizer(text)
    vocab_size = sp.get_piece_size()
    print(f"\nBPE 词汇表大小: {vocab_size}")

    # 测试 tokenizer
    test_text = "我爱你"
    test_encoded = encode(test_text, sp)
    test_decoded = decode(test_encoded, sp)
    print(f"分词测试: '{test_text}' -> {test_encoded} -> '{test_decoded}'")

    # 数据转为张量并分割
    data = torch.tensor(encode(text, sp), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data = data[n:]
    print(f"\n训练集大小: {len(train_data):,} tokens | 测试集大小: {len(test_data):,} tokens")

    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)

    # ========== 创建 DataLoader ==========
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

    # ========== 初始化 swanlab ==========
    global swanlab
    if USE_SWANLAB:
        import swanlab as _swanlab
        swanlab = _swanlab

        run_name = SWANLAB_RUN_NAME or f"NanoGPT-Epoch-{TRAIN_ITERS}-Batch-{BATCH_SIZE}-LR-{LEARNING_RATE}"
        swanlab.init(
            project=SWANLAB_PROJECT,
            experiment_name=run_name,
            config={
                'batch_size': BATCH_SIZE,
                'block_size': BLOCK_SIZE,
                'learning_rate': LEARNING_RATE,
                'min_lr': MIN_LR,
                'warmup_iters': WARMUP_ITERS,
                'train_iters': TRAIN_ITERS,
                'n_embed': 768,
                'n_heads': 12,
                'n_layers': 12,
                'dropout': 0.1,
                'weight_decay': WEIGHT_DECAY,
                'grad_clip': GRAD_CLIP,
                'vocab_size': vocab_size,
                'train_tokens': len(train_data),
                'test_tokens': len(test_data),
            }
        )
        print(f"Swanlab 初始化完成: {run_name}")

    # ========== 创建模型 ==========
    model = GPTLanguageModel(vocab_size).to(DEVICE).to(torch.bfloat16)
    print("训练精度: BF16")

    # torch.compile 加速（PyTorch 2.0+）
    if USE_COMPILE and hasattr(torch, 'compile'):
        print("启用 torch.compile 加速...")
        model = torch.compile(model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {num_params:,}")

    if USE_SWANLAB and swanlab:
        swanlab.log({'num_params': num_params})

    # ========== 创建优化器 ==========
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

    # ========== 评估函数 ==========
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

    # ========== 尝试从 checkpoint 恢复 ==========
    start_iter = 0
    best_val_loss = float('inf')
    resume_path = f'{CHECKPOINT_DIR}/latest.pt'

    if RESUME and os.path.exists(resume_path):
        start_iter, best_val_loss = load_checkpoint(resume_path, model, optimizer)
        start_iter += 1
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
        context = torch.tensor([encode(GENERATE_PROMPT, sp)], dtype=torch.long, device=DEVICE)
        print(decode(model.generate(context, max_new_tokens=INITIAL_GENERATE_TOKENS)[0].tolist(), sp))

    # ========== 训练循环 ==========
    print("\n" + "=" * 50)
    print(f"开始训练 (从 iter {start_iter} 到 {TRAIN_ITERS})")
    print("=" * 50)

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

            # 记录到 swanlab
            if USE_SWANLAB and swanlab:
                swanlab.log({
                    'train_loss': losses['train'],
                    'test_loss': losses['test'],
                    'learning_rate': lr,
                    'iter': iter_num,
                })

            # 保存最佳模型
            if SAVE_BEST and losses['test'] < best_val_loss:
                best_val_loss = losses['test']
                save_checkpoint(model, optimizer, iter_num, best_val_loss, vocab_size,
                              f'{CHECKPOINT_DIR}/best.pt')

            # 定期保存 + 生成预测
            if iter_num > 0 and iter_num % SAVE_INTERVAL == 0:
                save_checkpoint(model, optimizer, iter_num, best_val_loss, vocab_size,
                              f'{CHECKPOINT_DIR}/ckpt_{iter_num}.pt')

                # 每 1000 步生成一次预测
                model.eval()
                with torch.no_grad():
                    context = torch.tensor([encode(GENERATE_PROMPT, sp)], dtype=torch.long, device=DEVICE)
                    generated = model.generate(context, max_new_tokens=100)[0].tolist()
                    generated_text = decode(generated, sp)
                    print(f"\n--- Step {iter_num} 生成预览 ---")
                    print(generated_text)
                    print("-" * 30 + "\n")

                    # 记录生成文本到 swanlab
                    if USE_SWANLAB and swanlab:
                        swanlab.log({
                            'generated_text': swanlab.Text(generated_text),
                            'iter': iter_num,
                        })
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

        # 前向传播
        _, loss = model(xb, yb)

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

    # ========== 保存最终模型 ==========
    save_checkpoint(model, optimizer, TRAIN_ITERS - 1, best_val_loss, vocab_size,
                   f'{CHECKPOINT_DIR}/latest.pt')

    # 最终评估
    losses = estimate_loss()
    print(f"\n训练完成!")
    print(f"最终 train loss: {losses['train']:.4f} | test loss: {losses['test']:.4f}")
    print(f"最佳 test loss: {best_val_loss:.4f}")

    # 记录最终结果到 swanlab
    if USE_SWANLAB and swanlab:
        swanlab.log({
            'final_train_loss': losses['train'],
            'final_test_loss': losses['test'],
            'best_test_loss': best_val_loss,
        })

    # 训练后生成
    print("\n" + "=" * 50)
    print("训练后生成的文本:")
    print("=" * 50)
    context = torch.tensor([encode(GENERATE_PROMPT, sp)], dtype=torch.long, device=DEVICE)
    final_generated = decode(model.generate(context, max_new_tokens=FINAL_GENERATE_TOKENS)[0].tolist(), sp)
    print(final_generated)

    # 记录最终生成到 swanlab
    if USE_SWANLAB and swanlab:
        swanlab.log({'final_generated_text': swanlab.Text(final_generated)})
        swanlab.finish()
        print("\nSwanlab 日志已完成")

    print(f"\nCheckpoint 保存位置: {CHECKPOINT_DIR}/")
    print(f"  - best.pt: 最佳模型 (test loss: {best_val_loss:.4f})")
    print(f"  - latest.pt: 最新模型")


if __name__ == '__main__':
    main()
