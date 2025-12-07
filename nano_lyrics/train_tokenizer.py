# -*- coding: utf-8 -*-
"""
单独训练 Tokenizer 脚本

训练 BPE tokenizer 并生成 HuggingFace Hub 兼容文件
"""

import json
from config import LYRICS_DIR, MAX_SONGS, BPE_VOCAB_SIZE
from tokenizer import train_bpe_tokenizer, encode, decode, get_vocab_size
import re


def load_lyrics(max_songs=MAX_SONGS):
    """加载歌词文件"""
    all_lyrics = []
    for i in range(1, 6):
        if len(all_lyrics) >= max_songs:
            break
        filepath = f'{LYRICS_DIR}/lyrics{i}.json'
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                songs = json.load(f)
                remaining = max_songs - len(all_lyrics)
                all_lyrics.extend(songs[:remaining])
        except FileNotFoundError:
            print(f"文件不存在: {filepath}")
            continue
    return all_lyrics


def preprocess_lyric(lyric_lines):
    """预处理歌词，过滤元信息"""
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
        skip = any(re.match(p, line, re.IGNORECASE) for p in skip_patterns)
        if not skip and len(line) > 1:
            result.append(line)

    return '\n'.join(result)


def main():
    print("=" * 50)
    print("训练 BPE Tokenizer")
    print("=" * 50)

    # 加载歌词
    print("\n正在加载歌词...")
    lyrics_data = load_lyrics()
    print(f"加载了 {len(lyrics_data)} 首歌曲")

    # 预处理
    processed_lyrics = [
        preprocess_lyric(song['lyric'])
        for song in lyrics_data
        if song.get('lyric')
    ]
    processed_lyrics = [lyric for lyric in processed_lyrics if lyric.strip()]

    # 合并文本
    text = '\n\n'.join(processed_lyrics)
    print(f"有效歌曲数: {len(processed_lyrics):,}")
    print(f"总字符数: {len(text):,}")
    print(f"唯一字符数: {len(set(text)):,}")

    # 训练 tokenizer
    print(f"\n训练 BPE tokenizer (vocab_size={BPE_VOCAB_SIZE})...")
    tokenizer = train_bpe_tokenizer(text)

    # 显示结果
    print(f"\n词汇表大小: {get_vocab_size(tokenizer)}")

    # 测试
    test_texts = ["今夜我", "爱情", "Hello World", "🎵 音乐"]
    print("\n分词测试:")
    for t in test_texts:
        encoded = encode(t, tokenizer)
        decoded = decode(encoded, tokenizer)
        print(f"  '{t}' -> {encoded} -> '{decoded}'")

    print("\n" + "=" * 50)
    print("训练完成！")
    print("Hub 文件保存在: chinese_lyrics_bpe/")
    print("=" * 50)


if __name__ == '__main__':
    main()
