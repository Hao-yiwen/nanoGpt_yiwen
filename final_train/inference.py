# -*- coding: utf-8 -*-
"""
NanoGPT 推理脚本

使用训练好的模型生成文本
支持 Top-p (Nucleus) Sampling
"""

import argparse
import torch

from config import (
    CHECKPOINT_DIR, DEVICE,
    DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P,
)
from model import GPTLanguageModel
from tokenizer import load_tokenizer, encode, decode, get_vocab_size


def load_model(checkpoint_path):
    """
    加载训练好的模型

    Args:
        checkpoint_path: checkpoint 文件路径

    Returns:
        加载好的模型（eval 模式）
    """
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # 获取 vocab_size
    if 'config' in checkpoint and 'vocab_size' in checkpoint['config']:
        vocab_size = checkpoint['config']['vocab_size']
    else:
        vocab_size = get_vocab_size()

    # 创建模型
    model = GPTLanguageModel(vocab_size).to(DEVICE)

    # 加载权重
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model


def generate(model, prompt, max_tokens=DEFAULT_MAX_TOKENS,
             temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P):
    """
    生成文本

    Args:
        model: GPTLanguageModel 实例
        prompt: 输入提示词
        max_tokens: 最大生成 token 数
        temperature: 温度参数 (越高越随机)
        top_p: nucleus sampling 阈值

    Returns:
        生成的完整文本（包含 prompt）
    """
    # 编码 prompt
    sp = load_tokenizer()
    input_ids = torch.tensor([encode(prompt, sp)], dtype=torch.long, device=DEVICE)

    print(f"Prompt: {prompt}")
    print(f"参数: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")
    print("-" * 50)

    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    # 解码
    generated_text = decode(output_ids[0].tolist(), sp)
    return generated_text


def interactive_mode(model):
    """
    交互式生成模式

    用户可以不断输入 prompt 进行生成
    """
    print("\n" + "=" * 50)
    print("交互式生成模式")
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
    parser = argparse.ArgumentParser(description='NanoGPT 推理脚本')
    parser.add_argument('--checkpoint', type=str, default=f'{CHECKPOINT_DIR}/best.pt',
                        help='Checkpoint 文件路径')
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
        # 单次生成模式
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
        # 交互式模式
        interactive_mode(model)


if __name__ == '__main__':
    main()
