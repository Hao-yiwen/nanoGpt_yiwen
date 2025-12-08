# -*- coding: utf-8 -*-
"""
NanoGPT 推理脚本

使用训练好的模型生成文本
支持 HuggingFace generate() API
"""

import argparse
import torch

from transformers import PreTrainedTokenizerFast

from config import (
    CHECKPOINT_DIR, DEVICE,
    DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_REPETITION_PENALTY,
)
from model import GPTLanguageModel, NanoGPTConfig


def load_model(model_path):
    """
    从 HuggingFace 格式加载模型

    Args:
        model_path: 模型目录路径

    Returns:
        model: 加载好的模型（eval 模式）
        tokenizer: HuggingFace tokenizer
    """
    print(f"加载模型: {model_path}")

    # 加载模型
    model = GPTLanguageModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    model.eval()

    # 加载 tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

    print(f"模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"词汇表大小: {tokenizer.vocab_size}")

    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=DEFAULT_MAX_TOKENS,
             temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P,
             repetition_penalty=DEFAULT_REPETITION_PENALTY):
    """
    生成文本

    Args:
        model: GPTLanguageModel 实例
        tokenizer: HuggingFace tokenizer
        prompt: 输入提示词
        max_tokens: 最大生成 token 数
        temperature: 温度参数 (越高越随机)
        top_p: nucleus sampling 阈值

    Returns:
        生成的完整文本（包含 prompt）
    """
    # 编码 prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)

    print(f"Prompt: {prompt}")
    print(f"参数: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}, repetition_penalty={repetition_penalty}")
    print("-" * 50)

    # 使用 HuggingFace generate() API
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    return generated_text


def interactive_mode(model, tokenizer):
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

            result = generate(model, tokenizer, prompt)
            print("\n生成结果:")
            print(result)
            print("\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n退出")
            break


def main():
    parser = argparse.ArgumentParser(description='NanoGPT 推理脚本')
    parser.add_argument('--model', type=str, default=f'{CHECKPOINT_DIR}/best',
                        help='模型目录路径')
    parser.add_argument('--prompt', type=str, default=None,
                        help='输入提示词（不指定则进入交互模式）')
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help=f'最大生成 token 数 (默认: {DEFAULT_MAX_TOKENS})')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help=f'温度参数 (默认: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--top_p', type=float, default=DEFAULT_TOP_P,
                        help=f'Top-p 采样阈值 (默认: {DEFAULT_TOP_P})')
    parser.add_argument('--repetition_penalty', type=float, default=DEFAULT_REPETITION_PENALTY,
                        help=f'重复惩罚系数 (默认: {DEFAULT_REPETITION_PENALTY})')

    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_model(args.model)

    if args.prompt:
        # 单次生成模式
        result = generate(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print("\n生成结果:")
        print(result)
    else:
        # 交互式模式
        interactive_mode(model, tokenizer)


if __name__ == '__main__':
    main()
