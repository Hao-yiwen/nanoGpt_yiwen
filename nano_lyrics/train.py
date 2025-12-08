# -*- coding: utf-8 -*-
"""
NanoGPT 训练脚本

训练中文歌词生成模型
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import math

from transformers import PreTrainedTokenizerFast, GenerationConfig

from config import (
    RANDOM_SEED,
    BATCH_SIZE, MAX_SEQ_LEN, LEARNING_RATE, MIN_LR, WARMUP_ITERS, TRAIN_ITERS,
    EVAL_ITERS, EVAL_INTERVAL, WEIGHT_DECAY, GRAD_CLIP,
    NUM_WORKERS, PIN_MEMORY, PREFETCH_FACTOR,
    USE_COMPILE,
    CHECKPOINT_DIR, SAVE_INTERVAL, SAVE_BEST, RESUME,
    SWANLAB_PROJECT, SWANLAB_RUN_NAME, USE_SWANLAB,
    INITIAL_GENERATE_TOKENS, FINAL_GENERATE_TOKENS, GENERATE_PROMPT,
    DEVICE,
    # 模型配置
    N_EMBED, N_HEADS, N_KV_HEADS, N_LAYERS, DROP_OUT,
    USE_MOE, NUM_EXPERTS, NUM_SHARED_EXPERTS, TOP_K, MOE_FREQ, AUX_LOSS_COEF,
)

from model import GPTLanguageModel, NanoGPTConfig
from tokenizer import (
    train_bpe_tokenizer, encode, decode, get_vocab_size, get_pad_id,
    get_eos_id, get_im_start_id, CHAT_TEMPLATE,
    PAD_TOKEN, EOS_TOKEN, IM_START_TOKEN, IM_END_TOKEN,
)
from data import load_lyrics, preprocess_lyric, LyricsDataset

# 延迟导入 swanlab（仅在需要时）
swanlab = None


def get_generate_model(model):
    """获取可用于生成的模型（处理 torch.compile 包装）"""
    return model._orig_mod if hasattr(model, '_orig_mod') else model


# ============================================================================
# 模型保存与加载 (HuggingFace 格式)
# ============================================================================

def save_model_hf(model, tokenizer, save_path, optimizer_state=None, iter_num=None, best_val_loss=None):
    """
    保存模型为 HuggingFace 格式

    Args:
        model: GPTLanguageModel 实例
        tokenizer: tokenizers.Tokenizer 实例
        save_path: 保存目录
        optimizer_state: 优化器状态（可选，用于恢复训练）
        iter_num: 当前迭代次数
        best_val_loss: 最佳验证损失
    """
    os.makedirs(save_path, exist_ok=True)

    # 获取原始模型（如果使用了 torch.compile）
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model

    # 保存模型和配置（使用 safetensors 格式）
    model_to_save.save_pretrained(save_path, safe_serialization=True)

    # 创建 HF tokenizer 并保存
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=IM_START_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=None,
        additional_special_tokens=[IM_START_TOKEN, IM_END_TOKEN],
        model_max_length=MAX_SEQ_LEN,
    )
    hf_tokenizer.chat_template = CHAT_TEMPLATE
    hf_tokenizer.save_pretrained(save_path)

    # 保存生成配置
    gen_config = GenerationConfig(
        max_new_tokens=200,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        pad_token_id=get_pad_id(tokenizer),
        eos_token_id=get_eos_id(tokenizer),
        bos_token_id=get_im_start_id(tokenizer),
    )
    gen_config.save_pretrained(save_path)

    # 保存训练状态（用于恢复训练）
    if optimizer_state is not None or iter_num is not None:
        training_state = {
            'optimizer': optimizer_state,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
        }
        torch.save(training_state, os.path.join(save_path, 'training_state.pt'))

    print(f"  -> 模型已保存: {save_path}")


def load_model_hf(load_path, device=DEVICE, dtype=torch.bfloat16):
    """
    从 HuggingFace 格式加载模型

    Args:
        load_path: 模型目录
        device: 设备
        dtype: 数据类型

    Returns:
        model: GPTLanguageModel 实例
        training_state: 训练状态字典（如果存在）
    """
    print(f"加载模型: {load_path}")

    # 加载模型
    model = GPTLanguageModel.from_pretrained(
        load_path,
        torch_dtype=dtype,
    ).to(device)

    # 加载训练状态（如果存在）
    training_state_path = os.path.join(load_path, 'training_state.pt')
    training_state = None
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location=device)
        print(f"  -> 恢复训练状态: iter={training_state.get('iter_num', 0)}, "
              f"best_val_loss={training_state.get('best_val_loss', float('inf')):.4f}")

    return model, training_state


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

    # 预处理每首歌的歌词
    processed_lyrics = [
        preprocess_lyric(song['lyric'])
        for song in lyrics_data
        if song.get('lyric')
    ]
    # 过滤空歌词
    processed_lyrics = [lyric for lyric in processed_lyrics if lyric.strip()]

    # 合并所有歌词用于训练 tokenizer
    text = '\n\n'.join(processed_lyrics)

    print(f"有效歌曲数: {len(processed_lyrics):,}")
    print(f"数据集字符数: {len(text):,}")
    print(f"原始唯一字符数: {len(set(text)):,}")
    print(f"数据预览:\n{text[:500]}...")

    # ========== 训练/加载 tokenizer ==========
    tokenizer = train_bpe_tokenizer(text)
    vocab_size = get_vocab_size(tokenizer)
    pad_id = get_pad_id(tokenizer)
    print(f"\nBPE 词汇表大小: {vocab_size}")
    print(f"PAD token id: {pad_id}")

    # 测试 tokenizer
    test_text = "我爱你"
    test_encoded = encode(test_text, tokenizer)
    test_decoded = decode(test_encoded, tokenizer)
    print(f"分词测试: '{test_text}' -> {test_encoded} -> '{test_decoded}'")

    # 将每首歌单独编码为 token 列表
    all_samples = [encode(lyric, tokenizer) for lyric in processed_lyrics]

    # 统计歌曲长度分布
    lengths = [len(s) for s in all_samples]
    print(f"\n歌曲长度统计: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

    # 分割训练集和测试集（按歌曲数量）
    n = int(0.9 * len(all_samples))
    train_samples = all_samples[:n]
    test_samples = all_samples[n:]
    train_tokens = sum(len(s) for s in train_samples)
    test_tokens = sum(len(s) for s in test_samples)
    print(f"训练集: {len(train_samples):,} 首歌 ({train_tokens:,} tokens)")
    print(f"测试集: {len(test_samples):,} 首歌 ({test_tokens:,} tokens)")

    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)

    # ========== 创建 DataLoader ==========
    train_dataset = LyricsDataset(train_samples, MAX_SEQ_LEN, pad_id)
    test_dataset = LyricsDataset(test_samples, MAX_SEQ_LEN, pad_id)

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
                'max_seq_len': MAX_SEQ_LEN,
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
                'train_tokens': train_tokens,
                'test_tokens': test_tokens,
            }
        )
        print(f"Swanlab 初始化完成: {run_name}")

    # ========== 创建模型配置 ==========
    config = NanoGPTConfig(
        vocab_size=vocab_size,
        n_embed=N_EMBED,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        n_layers=N_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROP_OUT,
        # MoE 配置
        use_moe=USE_MOE,
        num_experts=NUM_EXPERTS,
        num_shared_experts=NUM_SHARED_EXPERTS,
        top_k=TOP_K,
        moe_freq=MOE_FREQ,
        aux_loss_coef=AUX_LOSS_COEF,
        # HF 兼容参数
        pad_token_id=pad_id,
        bos_token_id=get_im_start_id(tokenizer),
        eos_token_id=get_eos_id(tokenizer),
    )

    # ========== 创建模型 ==========
    model = GPTLanguageModel(config).to(DEVICE).to(torch.bfloat16)
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
        """在训练集和测试集上估计平均损失（带 loss_mask）"""
        out = {}
        model.eval()
        for split, loader in [('train', train_loader), ('test', test_loader)]:
            losses = []
            for i, (X, Y, loss_mask, attn_mask) in enumerate(loader):
                if i >= EVAL_ITERS:
                    break
                X = X.to(DEVICE, non_blocking=True)
                Y = Y.to(DEVICE, non_blocking=True)
                loss_mask = loss_mask.to(DEVICE, non_blocking=True)
                attn_mask = attn_mask.to(DEVICE, non_blocking=True)

                # 使用 HF 兼容的 forward 接口
                outputs = model(input_ids=X, attention_mask=attn_mask, use_cache=False)
                logits = outputs.logits
                B, T, C = logits.shape
                loss_per_token = F.cross_entropy(
                    logits.view(B * T, C), Y.view(B * T), reduction='none'
                ).view(B, T)
                loss = (loss_per_token * loss_mask).sum() / loss_mask.sum()
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses) if losses else 0.0
        model.train()
        return out

    # ========== 尝试从 checkpoint 恢复 ==========
    start_iter = 0
    best_val_loss = float('inf')
    resume_path = f'{CHECKPOINT_DIR}/latest'

    if RESUME and os.path.exists(resume_path):
        # 使用 HF 格式加载
        loaded_model, training_state = load_model_hf(resume_path, DEVICE, torch.bfloat16)
        # 复制权重到当前模型
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(loaded_model.state_dict())
        else:
            model.load_state_dict(loaded_model.state_dict())
        del loaded_model

        if training_state:
            start_iter = training_state.get('iter_num', 0) + 1
            best_val_loss = training_state.get('best_val_loss', float('inf'))
            if training_state.get('optimizer'):
                optimizer.load_state_dict(training_state['optimizer'])
    else:
        # 测试初始状态
        test_iter = iter(train_loader)
        xb, yb, loss_mask_init, attn_mask_init = next(test_iter)
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        loss_mask_init = loss_mask_init.to(DEVICE)
        attn_mask_init = attn_mask_init.to(DEVICE)
        outputs = model(input_ids=xb, attention_mask=attn_mask_init, use_cache=False)
        logits = outputs.logits
        B, T, C = logits.shape
        loss_per_token = F.cross_entropy(
            logits.view(B * T, C), yb.view(B * T), reduction='none'
        ).view(B, T)
        loss = (loss_per_token * loss_mask_init).sum() / loss_mask_init.sum()
        print(f"初始 loss: {loss:.4f}")

        # 训练前生成（使用 HF generate）
        print("\n" + "=" * 50)
        print("训练前生成的文本:")
        print("=" * 50)
        model.eval()
        context = torch.tensor([encode(GENERATE_PROMPT, tokenizer)], dtype=torch.long, device=DEVICE)
        gen_model = get_generate_model(model)
        generated = gen_model.generate(
            context,
            max_new_tokens=INITIAL_GENERATE_TOKENS,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            pad_token_id=pad_id,
        )
        print(decode(generated[0].tolist(), tokenizer))
        model.train()

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
                save_model_hf(
                    model, tokenizer, f'{CHECKPOINT_DIR}/best',
                    optimizer_state=optimizer.state_dict(),
                    iter_num=iter_num,
                    best_val_loss=best_val_loss
                )

            # 定期保存 + 生成预测
            if iter_num > 0 and iter_num % SAVE_INTERVAL == 0:
                save_model_hf(
                    model, tokenizer, f'{CHECKPOINT_DIR}/ckpt_{iter_num}',
                    optimizer_state=optimizer.state_dict(),
                    iter_num=iter_num,
                    best_val_loss=best_val_loss
                )

                # 每 1000 步生成一次预测（使用 HF generate）
                model.eval()
                with torch.no_grad():
                    context = torch.tensor([encode(GENERATE_PROMPT, tokenizer)], dtype=torch.long, device=DEVICE)
                    gen_model = get_generate_model(model)
                    generated = gen_model.generate(
                        context,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.9,
                        pad_token_id=pad_id,
                    )
                    generated_text = decode(generated[0].tolist(), tokenizer)
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
            xb, yb, loss_mask, attn_mask = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb, loss_mask, attn_mask = next(train_iter)

        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        loss_mask = loss_mask.to(DEVICE, non_blocking=True)
        attn_mask = attn_mask.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 前向传播（使用 HF 兼容接口，带 attention_mask）
        outputs = model(input_ids=xb, attention_mask=attn_mask, use_cache=False)
        logits = outputs.logits
        B, T, C = logits.shape
        loss_per_token = F.cross_entropy(
            logits.view(B * T, C), yb.view(B * T), reduction='none'
        ).view(B, T)
        loss = (loss_per_token * loss_mask).sum() / loss_mask.sum()

        # 如果模型使用了 MoE，可能需要添加辅助损失（已在 forward 中处理）
        # 这里使用手动计算的 loss_mask 损失

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

    # ========== 保存最终模型 ==========
    save_model_hf(
        model, tokenizer, f'{CHECKPOINT_DIR}/latest',
        optimizer_state=optimizer.state_dict(),
        iter_num=TRAIN_ITERS - 1,
        best_val_loss=best_val_loss
    )

    # 最终评估
    losses = estimate_loss()
    print("\n训练完成!")
    print(f"最终 train loss: {losses['train']:.4f} | test loss: {losses['test']:.4f}")
    print(f"最佳 test loss: {best_val_loss:.4f}")

    # 记录最终结果到 swanlab
    if USE_SWANLAB and swanlab:
        swanlab.log({
            'final_train_loss': losses['train'],
            'final_test_loss': losses['test'],
            'best_test_loss': best_val_loss,
        })

    # 训练后生成（使用 HF generate）
    print("\n" + "=" * 50)
    print("训练后生成的文本:")
    print("=" * 50)
    model.eval()
    context = torch.tensor([encode(GENERATE_PROMPT, tokenizer)], dtype=torch.long, device=DEVICE)
    gen_model = get_generate_model(model)
    final_generated_ids = gen_model.generate(
        context,
        max_new_tokens=FINAL_GENERATE_TOKENS,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        pad_token_id=pad_id,
    )
    final_generated = decode(final_generated_ids[0].tolist(), tokenizer)
    print(final_generated)

    # 记录最终生成到 swanlab
    if USE_SWANLAB and swanlab:
        swanlab.log({'final_generated_text': swanlab.Text(final_generated)})
        swanlab.finish()
        print("\nSwanlab 日志已完成")

    print(f"\n模型保存位置: {CHECKPOINT_DIR}/")
    print(f"  - best/: 最佳模型 (test loss: {best_val_loss:.4f})")
    print(f"  - latest/: 最新模型")
    print("\n可使用以下方式加载模型:")
    print(f"  model = GPTLanguageModel.from_pretrained('{CHECKPOINT_DIR}/best')")
    print(f"  tokenizer = PreTrainedTokenizerFast.from_pretrained('{CHECKPOINT_DIR}/best')")


if __name__ == '__main__':
    main()
