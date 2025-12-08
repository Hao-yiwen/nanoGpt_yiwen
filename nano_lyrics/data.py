# -*- coding: utf-8 -*-
"""
数据加载与预处理模块

包含:
- load_lyrics: 加载歌词文件
- preprocess_lyric: 预处理歌词
- LyricsDataset: PyTorch Dataset 实现
"""

import torch
from torch.utils.data import Dataset
import json
import re

from config import LYRICS_DIR, MAX_SONGS


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


class LyricsDataset(Dataset):
    """歌词数据集（按歌曲切分，支持 loss_mask 和 attention_mask）"""

    def __init__(self, samples, block_size, pad_id):
        """
        Args:
            samples: List[List[int]], 每首歌的 token ids
            block_size: 最大序列长度
            pad_id: padding token id
        """
        self.samples = samples
        self.block_size = block_size
        self.pad_id = pad_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx].copy()

        # 截断或填充到 block_size + 1（因为需要 x 和 y 各 block_size 长度）
        if len(tokens) > self.block_size + 1:
            tokens = tokens[:self.block_size + 1]
        else:
            tokens = tokens + [self.pad_id] * (self.block_size + 1 - len(tokens))

        tokens = torch.tensor(tokens, dtype=torch.long)
        x = tokens[:-1]  # (block_size,)
        y = tokens[1:]   # (block_size,)

        # loss_mask: 1 表示计算 loss，0 表示忽略（padding 位置）
        loss_mask = (y != self.pad_id).long()

        # attention_mask: 1 表示有效 token，0 表示 padding（用于注意力计算）
        attention_mask = (x != self.pad_id).long()

        return x, y, loss_mask, attention_mask
