# -*- coding: utf-8 -*-
"""
NanoGPT 分词器

使用 SentencePiece BPE 分词
"""

import os
import sentencepiece as spm

from config import BPE_VOCAB_SIZE, BPE_MODEL_PREFIX


# 全局 tokenizer 实例
_sp = None


def train_bpe_tokenizer(text, vocab_size=BPE_VOCAB_SIZE, model_prefix=BPE_MODEL_PREFIX):
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


def load_tokenizer(model_prefix=BPE_MODEL_PREFIX):
    """
    加载已训练的 tokenizer

    Returns:
        SentencePieceProcessor 实例
    """
    global _sp
    if _sp is None:
        model_file = f'{model_prefix}.model'
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"BPE 模型不存在: {model_file}，请先运行 train.py")
        _sp = spm.SentencePieceProcessor()
        _sp.load(model_file)
        print(f"加载 BPE 模型: {model_file}")
    return _sp


def encode(text, sp=None):
    """
    编码文本为 token ids

    Args:
        text: 输入文本
        sp: SentencePieceProcessor 实例（可选，不提供则使用全局实例）

    Returns:
        token ids 列表
    """
    if sp is None:
        sp = load_tokenizer()
    return sp.encode(text, out_type=int)


def decode(ids, sp=None):
    """
    解码 token ids 为文本

    Args:
        ids: token ids（列表或张量）
        sp: SentencePieceProcessor 实例（可选）

    Returns:
        解码后的文本
    """
    if sp is None:
        sp = load_tokenizer()
    # 支持多种输入类型
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    elif hasattr(ids, '__iter__'):
        ids = list(ids)
    else:
        ids = [ids]
    return sp.decode(ids)


def get_vocab_size(sp=None):
    """获取词汇表大小"""
    if sp is None:
        sp = load_tokenizer()
    return sp.get_piece_size()


def get_bos_id(sp=None):
    """获取 BOS token id"""
    if sp is None:
        sp = load_tokenizer()
    return sp.bos_id()


def get_eos_id(sp=None):
    """获取 EOS token id"""
    if sp is None:
        sp = load_tokenizer()
    return sp.eos_id()


def get_pad_id(sp=None):
    """获取 PAD token id"""
    if sp is None:
        sp = load_tokenizer()
    return sp.pad_id()
