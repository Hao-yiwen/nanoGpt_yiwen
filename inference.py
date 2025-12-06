# -*- coding: utf-8 -*-
"""
NanoGPT 中文歌词推理脚本

使用 hf_upload 目录中的模型和 tokenizer 进行文本生成
"""

import sys
import os
import argparse
import torch

# 添加 hf_upload 目录到路径
HF_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hf_upload')
sys.path.insert(0, HF_UPLOAD_DIR)

from model import GPTLanguageModel

import sentencepiece as spm

# ============================================================================
# 配置
# ============================================================================

MODEL_PATH = os.path.join(HF_UPLOAD_DIR, 'best.pt')
TOKENIZER_PATH = os.path.join(HF_UPLOAD_DIR, 'chinese_lyrics_bpe.model')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9

# ============================================================================
# Tokenizer
# ============================================================================

_sp = None

def load_tokenizer():
    """加载 SentencePiece tokenizer"""
    global _sp
    if _sp is None:
        _sp = spm.SentencePieceProcessor()
        _sp.load(TOKENIZER_PATH)
        print(f"加载 tokenizer: {TOKENIZER_PATH}")
    return _sp

def encode(text, sp=None):
    """编码文本为 token ids"""
    if sp is None:
        sp = load_tokenizer()
    return sp.encode(text, out_type=int)

def decode(ids, sp=None):
    """解码 token ids 为文本"""
    if sp is None:
        sp = load_tokenizer()
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    return sp.decode(ids)

# ============================================================================
# 模型加载
# ============================================================================

def load_model(checkpoint_path=MODEL_PATH):
    """加载训练好的模型"""
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # 获取 vocab_size
    vocab_size = checkpoint['config']['vocab_size']

    # 创建模型
    model = GPTLanguageModel(vocab_size).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model

# ============================================================================
# 文本生成
# ============================================================================

def generate(model, prompt, max_tokens=DEFAULT_MAX_TOKENS,
             temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P):
    """生成文本"""
    sp = load_tokenizer()
    input_ids = torch.tensor([encode(prompt, sp)], dtype=torch.long, device=DEVICE)

    print(f"Prompt: {prompt}")
    print(f"参数: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")
    print("-" * 50)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    return decode(output_ids[0].tolist(), sp)


def interactive_mode(model):
    """交互式生成模式"""
    print("\n" + "=" * 50)
    print("NanoGPT 中文歌词生成 - 交互模式")
    print("输入 prompt 开始生成，输入 'quit' 或 'exit' 退出")
    print("=" * 50 + "\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("退出")
                break
            if not prompt:
                continue

            result = generate(model, prompt)
            print("\n生成结果:")
            print(result)
            print("\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n退出")
            break


def main():
    parser = argparse.ArgumentParser(description='NanoGPT 中文歌词推理')
    parser.add_argument('--checkpoint', type=str, default=MODEL_PATH,
                        help='模型文件路径')
    parser.add_argument('--prompt', type=str, default=None,
                        help='输入提示词（不指定则进入交互模式）')
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help=f'最大生成 token 数 (默认: {DEFAULT_MAX_TOKENS})')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help=f'温度参数 (默认: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--top_p', type=float, default=DEFAULT_TOP_P,
                        help=f'Top-p 采样阈值 (默认: {DEFAULT_TOP_P})')

    args = parser.parse_args()

    # 加载模型
    model = load_model(args.checkpoint)

    if args.prompt:
        result = generate(
            model,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("\n生成结果:")
        print(result)
    else:
        interactive_mode(model)


if __name__ == '__main__':
    main()
