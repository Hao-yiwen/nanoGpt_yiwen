# -*- coding: utf-8 -*-
"""
NanoGPT 分词器

使用 HuggingFace tokenizers 库进行 BPE 分词
支持 ChatML 格式的 chat template
"""

import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from config import BPE_VOCAB_SIZE, BPE_MODEL_PATH, SPECIAL_TOKENS, MAX_SEQ_LEN


# 特殊 token 定义
PAD_TOKEN = SPECIAL_TOKENS["pad_token"]      # <|pad|>
EOS_TOKEN = SPECIAL_TOKENS["eos_token"]      # <|endoftext|>
IM_START_TOKEN = SPECIAL_TOKENS["im_start_token"]  # <|im_start|>
IM_END_TOKEN = SPECIAL_TOKENS["im_end_token"]      # <|im_end|>

# 特殊 token 列表（按 ID 顺序）
SPECIAL_TOKENS_LIST = [PAD_TOKEN, EOS_TOKEN, IM_START_TOKEN, IM_END_TOKEN]

# 全局 tokenizer 实例
_tokenizer = None


def train_bpe_tokenizer(text, vocab_size=BPE_VOCAB_SIZE, model_path=BPE_MODEL_PATH):
    """
    训练 HuggingFace BPE tokenizer

    如果模型文件已存在，则直接加载；否则训练新模型。

    Args:
        text: 训练文本
        vocab_size: 词汇表大小
        model_path: 模型保存路径

    Returns:
        Tokenizer 实例
    """
    # 检查是否已有训练好的模型
    if os.path.exists(model_path):
        print(f"加载已有 BPE 模型: {model_path}")
        tokenizer = Tokenizer.from_file(model_path)
        return tokenizer

    print(f"训练 BPE tokenizer (vocab_size={vocab_size})...")

    # 创建 BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token=None))

    # Byte-level 预分词器（确保任何字符都能编码，无 UNK）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Byte-level 解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 训练器配置
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS_LIST,
        show_progress=True,
        min_frequency=2,
    )

    # 保存文本到临时文件
    temp_corpus = 'temp_corpus.txt'
    with open(temp_corpus, 'w', encoding='utf-8') as f:
        f.write(text)

    # 训练
    tokenizer.train([temp_corpus], trainer)

    # 保存模型
    tokenizer.save(model_path)
    print(f"BPE 模型训练完成，保存至: {model_path}")

    # 清理临时文件
    if os.path.exists(temp_corpus):
        os.remove(temp_corpus)

    # 同时保存 HuggingFace Hub 兼容格式
    save_dir = os.path.splitext(model_path)[0]  # 去掉 .json 后缀作为目录名
    save_for_hub(tokenizer, save_dir)

    return tokenizer


def load_tokenizer(model_path=BPE_MODEL_PATH):
    """
    加载已训练的 tokenizer

    Returns:
        Tokenizer 实例
    """
    global _tokenizer
    if _tokenizer is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BPE 模型不存在: {model_path}，请先运行 train.py")
        _tokenizer = Tokenizer.from_file(model_path)
        print(f"加载 BPE 模型: {model_path}")
    return _tokenizer


def encode(text, tokenizer=None):
    """
    编码文本为 token ids

    Args:
        text: 输入文本
        tokenizer: Tokenizer 实例（可选，不提供则使用全局实例）

    Returns:
        token ids 列表
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()
    encoding = tokenizer.encode(text)
    return encoding.ids


def decode(ids, tokenizer=None):
    """
    解码 token ids 为文本

    Args:
        ids: token ids（列表或张量）
        tokenizer: Tokenizer 实例（可选）

    Returns:
        解码后的文本
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()
    # 支持多种输入类型
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    elif hasattr(ids, '__iter__'):
        ids = list(ids)
    else:
        ids = [ids]
    return tokenizer.decode(ids)


def get_vocab_size(tokenizer=None):
    """获取词汇表大小"""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.get_vocab_size()


def get_pad_id(tokenizer=None):
    """获取 PAD token id"""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.token_to_id(PAD_TOKEN)


def get_eos_id(tokenizer=None):
    """获取 EOS token id (即 <|endoftext|>)"""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.token_to_id(EOS_TOKEN)


def get_bos_id(tokenizer=None):
    """
    获取 BOS token id

    注意：ChatML 格式没有传统的 BOS，这里返回 <|im_start|> 作为替代
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.token_to_id(IM_START_TOKEN)


def get_im_start_id(tokenizer=None):
    """获取 <|im_start|> token id"""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.token_to_id(IM_START_TOKEN)


def get_im_end_id(tokenizer=None):
    """获取 <|im_end|> token id"""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.token_to_id(IM_END_TOKEN)


def apply_chat_template(messages, tokenizer=None, add_generation_prompt=False):
    """
    应用 ChatML 模板格式化对话

    ChatML 格式:
    <|im_start|>system
    你是一个歌词创作助手。<|im_end|>
    <|im_start|>user
    写一首关于爱情的歌<|im_end|>
    <|im_start|>assistant
    今夜我独自徘徊...<|im_end|>

    Args:
        messages: 对话消息列表
            [{"role": "system/user/assistant", "content": "..."}]
        tokenizer: tokenizer 实例
        add_generation_prompt: 是否添加 assistant 开头提示（用于生成）

    Returns:
        token ids 列表
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()

    # 构建 ChatML 格式文本
    text_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text_parts.append(f"{IM_START_TOKEN}{role}\n{content}{IM_END_TOKEN}\n")

    # 如果需要添加生成提示
    if add_generation_prompt:
        text_parts.append(f"{IM_START_TOKEN}assistant\n")

    full_text = "".join(text_parts)

    # 编码
    return encode(full_text, tokenizer)


def format_chat_text(messages, add_generation_prompt=False):
    """
    格式化对话为 ChatML 文本（不编码）

    Args:
        messages: 对话消息列表
        add_generation_prompt: 是否添加 assistant 开头提示

    Returns:
        格式化后的文本字符串
    """
    text_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text_parts.append(f"{IM_START_TOKEN}{role}\n{content}{IM_END_TOKEN}\n")

    if add_generation_prompt:
        text_parts.append(f"{IM_START_TOKEN}assistant\n")

    return "".join(text_parts)


# ChatML Jinja2 模板
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def save_for_hub(tokenizer, save_dir="tokenizer_hub"):
    """
    保存 tokenizer 为 HuggingFace Hub 兼容格式

    使用 transformers.PreTrainedTokenizerFast 自动生成所有配置文件:
    - tokenizer.json
    - tokenizer_config.json
    - special_tokens_map.json

    Args:
        tokenizer: tokenizers.Tokenizer 实例
        save_dir: 保存目录
    """
    from transformers import PreTrainedTokenizerFast

    # 使用 PreTrainedTokenizerFast 包装 tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=IM_START_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=None,
        additional_special_tokens=[IM_START_TOKEN, IM_END_TOKEN],
        model_max_length=MAX_SEQ_LEN,
    )

    # 设置 chat template
    hf_tokenizer.chat_template = CHAT_TEMPLATE

    # 保存所有文件（自动生成 tokenizer.json, tokenizer_config.json, special_tokens_map.json）
    hf_tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer Hub 文件保存完成: {save_dir}/")
